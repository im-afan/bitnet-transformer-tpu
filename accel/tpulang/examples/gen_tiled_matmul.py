#!/usr/bin/env python3
"""gen_tiled_matmul.py — examples/tiled_matmul.tpu, written in pytpu instead.

The pytpu milestone-1 acceptance test (see ../pytpu.md §6).  It builds the same
streamed `C = A @ W` as the hand-written program, writes it to
`examples/out/tiled_matmul_gen.tpu`, and checks it three ways:

  1. behavioural — both programs run in the ISS on the same DRAM image and must
     leave byte-identical results (not word-for-word identical *code*: pytpu is
     free to pick different registers and fold arithmetic the human did not);
  2. size        — the generated program must stay within a few words of the
     hand-written ~60.  A blowout means staging leaked and something unrolled;
  3. toolchain   — `gen_vectors.py`'s independent Python A@W reference passes on
     the generated program, which it finds by the `.equ` names alone.

    python examples/gen_tiled_matmul.py            # build + check
    python examples/gen_tiled_matmul.py --print    # also dump the source

Afterwards the generated file is an ordinary .tpu, so these work on it too:

    python gen_vectors.py -p examples/out/tiled_matmul_gen.tpu
    python torch_ref.py   -p examples/out/tiled_matmul_gen.tpu
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

import gen_vectors as gv                                      # noqa: E402
from iss import TPU                                           # noqa: E402
from pytpu import Program                                     # noqa: E402

HAND = os.path.join(HERE, "tiled_matmul.tpu")
OUT = os.path.join(HERE, "out", "tiled_matmul_gen.tpu")


def build() -> Program:
    """The whole kernel.  Compare with tiled_matmul.tpu lines 43-191."""
    p = Program("tiled matmul: C = A @ W, streamed one tile at a time")

    # ---- array / problem geometry (these .equ names are what gen_vectors.py and
    # torch_ref.py dispatch on — keep them) ----
    ROWS = p.equ("ROWS", 8)          # MXU contraction dim (K per tile)
    COLS = p.equ("COLS", 8)          # MXU output features (N per tile)
    T = p.equ("T", 4)                # tokens per matmul pass (cfg tlen)
    MTILES = p.equ("MTILES", 2)
    NTILES = p.equ("NTILES", 2)
    KTILES = p.equ("KTILES", 4)
    KLAST = p.equ("KLAST", KTILES - 1)

    # ---- tile byte sizes ----
    WCOL = p.equ("WCOL", (ROWS * 2) // 8)     # packed ternary bytes per column
    ATILE = p.equ("ATILE", T * ROWS)          # int8 A tile, row-major
    WTILE = p.equ("WTILE", COLS * WCOL)       # ternary W tile, col-major
    CTILE = p.equ("CTILE", T * COLS)          # int8 C tile (requant output)

    # ---- DRAM tile-strides (tile-major layout) ----
    ASTRIDE_M = p.equ("ASTRIDE_M", KTILES * ATILE)
    WSTRIDE_N = p.equ("WSTRIDE_N", KTILES * WTILE)

    # ---- DRAM addresses ----
    A = p.equ("A", 0x1000)           # A tiles [mt][kt]
    W = p.equ("W", 0x4000)           # W tiles [nt][kt]
    C = p.equ("C", 0x6000)           # C tiles [mt][nt]
    RQW = p.equ("RQW", 0x7000)       # requant {m0,n} word

    # ---- scratchpad buffers (fixed and tiny; they do not grow with the problem) ----
    ABUF = p.equ("ABUF", 0x0000)
    WBUF = p.equ("WBUF", 0x0100)
    CBUF = p.equ("CBUF", 0x0200)     # int32 accumulator; requant overlays its low bytes
    SRQW = p.equ("SRQW", 0x0300)

    p.cfg(tlen=T, scalar=SRQW)

    p.comment("stage the requant word once: DRAM -> scratch")
    p.cfg(len=4)
    p.rdmem(SRQW, RQW)

    def tile(kt, **kw):
        """One K tile: fill A and W, then one MXU pass.

        `kt` is a plain int for the peeled first/last tiles and a Reg inside the
        loop — the body is identical either way.  That is the whole point of the
        two-level staging: with an int, `kt * ATILE` folds and `+= arow`
        disappears entirely; with a Reg it becomes muls/adds.
        """
        at = kt * ATILE
        at += arow
        wt = kt * WTILE
        wt += wcol
        p.cfg(len=ATILE)
        p.rdmem(ABUF, at)
        p.cfg(len=WTILE)
        p.rdmem(WBUF, wt)
        p.matmul(CBUF, ABUF, WBUF, **kw)

    with p.loop(MTILES, name="mt") as mt:
        arow = mt * ASTRIDE_M                 # arow = A + mt*ASTRIDE_M
        arow += A

        with p.loop(NTILES, name="nt") as nt:
            wcol = nt * WSTRIDE_N             # wcol = W + nt*WSTRIDE_N
            wcol += W
            ctil = mt * NTILES                # ctil = C + (mt*NTILES + nt)*CTILE
            ctil += nt
            ctil *= CTILE
            ctil += C

            p.comment("K tile 0: initialise the int32 accumulator")
            tile(0)

            p.comment("K tiles 1..KLAST-1: accumulate")
            with p.loop(1, KLAST, name="kt") as kt:
                tile(kt, acc=True)

            p.comment("K tile KLAST: accumulate, then requant int32 -> int8")
            tile(KLAST, acc=True, rq=True)

            p.comment("spill this T x COLS int8 output tile: scratch -> DRAM")
            p.cfg(len=CTILE)
            p.wrmem(CBUF, ctil)

    p.halt()
    return p


def run_iss(consts: dict, words: list) -> tuple[dict, int, dict | None]:
    """Run in the ISS on this program's standard image.

    Returns the written-DRAM image, the mismatch count against gen_vectors'
    independent Python A@W, and the problem dimensions.
    """
    tpu = TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W, m0_w=gv.M0_W, n_w=gv.N_W)
    _in_img, ref, dims = gv.build_image(tpu, consts)
    tpu.run(words)
    out = {a: tpu.dram[a] for a in sorted(tpu.dram_written)}
    return out, (gv.check_reference(tpu, ref) if ref else 0), dims


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="dump", action="store_true",
                    help="print the generated tpulang source")
    ap.add_argument("-o", "--out", default=OUT, help="where to write the .tpu")
    args = ap.parse_args(argv)

    p = build()
    if args.dump:
        print(p.render())
    p.save(args.out)
    print(f"wrote   : {os.path.relpath(args.out, os.path.dirname(HERE))}")

    gen_words, gen_consts = gv.assemble_program(args.out)
    hand_words, hand_consts = gv.assemble_program(HAND)
    print(f"size    : {len(gen_words)} words generated vs {len(hand_words)} hand-written "
          f"(IMEM holds 1024)")

    gen_out, gen_bad, dims = run_iss(gen_consts, gen_words)
    hand_out, hand_bad, _ = run_iss(hand_consts, hand_words)
    shape = f"[{dims['M']}x{dims['K']}] @ [{dims['K']}x{dims['N']}]" if dims else "?"

    fails = 0
    if gen_bad or hand_bad:
        print(f"FAIL    : reference mismatches — generated {gen_bad}, hand-written {hand_bad}")
        fails += 1
    else:
        print(f"ref     : OK, both match the Python A@W reference  {shape}")

    if gen_out != hand_out:
        only_g = set(gen_out) - set(hand_out)
        only_h = set(hand_out) - set(gen_out)
        diff = [a for a in set(gen_out) & set(hand_out) if gen_out[a] != hand_out[a]]
        print(f"FAIL    : DRAM images differ — {len(diff)} byte(s) differ, "
              f"{len(only_g)} only generated, {len(only_h)} only hand-written")
        for a in sorted(diff)[:8]:
            print(f"          @{a:04x}: generated {gen_out[a]:#04x}  hand {hand_out[a]:#04x}")
        fails += 1
    else:
        print(f"equiv   : OK, byte-identical results ({len(gen_out)} output bytes)")

    if len(gen_words) > len(hand_words) + 8:
        print(f"FAIL    : {len(gen_words) - len(hand_words)} words larger than hand-written "
              "— something unrolled that should have stayed a loop")
        fails += 1

    print("PASS" if not fails else f"{fails} check(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
