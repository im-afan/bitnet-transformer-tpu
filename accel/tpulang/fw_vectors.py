#!/usr/bin/env python3
"""fw_vectors.py — golden vectors for a *firmware* kernel, from its command trace.

The `gen_vectors.py` of the CPU producer, and the last piece of phase 3
(`accel/tpu/docs/picorv32_migration.md` §8). Where `gen_vectors.py` assembles a
`.tpu` program and runs it through `iss.py`'s instruction decoder, this reads the
command trace a natively-compiled firmware kernel emitted and runs it through
`iss.py`'s *command* decoder — the same op bodies underneath, so the two paths
cannot disagree about numerics without disagreeing everywhere.

    make -C ../tpu/fw trace > matmul.trace.txt
    python fw_vectors.py -t matmul.trace.txt -o ../tpu/tb/vectors_fw

Three files come out, all `$readmemh`-able:

    fw_dram_in.hex    the operand image the testbench seeds DRAM with
    fw_dram_exp.hex   every DRAM byte the run wrote, and what it should be
    fw_cmds.hex       the expected command trace: 5 words per command
                      (unit, w0, w1, w2, w3), terminated by a 0xFFFFFFFF unit

The third is what makes this more than a rewrite of the old inline checks. The
testbench monitors the arbitrated command write inside `tpu_top` and diffs the
real PicoRV32's command stream against this one, so a failure says *which
command* diverged rather than only that the answer was wrong — which is the
capability the abandoned RV32IM interpreter was really being bought for
(§8.1).

**The operand formulas live here and nowhere else.** They used to be duplicated
between `fw_matmul_tb.sv` and `host/run_fw_matmul.py`, which is exactly the
drift this phase exists to remove.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iss import TPU, parse_trace  # noqa: E402

# Geometry. Must match the array the testbench instantiates and the numbers the
# firmware was built with (fw/Makefile takes the same three).
ROWS, COLS = 8, 8

A_ADDR, W_ADDR, C_ADDR = 0x0000, 0x2000, 0x4000


def a_val(m: int, k: int) -> int:
    """An int8 activation. Bounded to [-4, 4] so a chain of requants has somewhere
    to land instead of pinning at the clip."""
    return ((m * 3 + k * 5) % 9) - 4


def w_val(k: int, n: int) -> int:
    """An int4 weight in [-8, 7].

    Spans the **whole** grid rather than a ternary `%3 - 1`: -8 is the one value
    with no positive counterpart, so a run that never produces it cannot catch a
    nibble sign-extended as unsigned.
    """
    return ((k * 5 + n * 3) % 16) - 8


def w_hash(r: int, c: int, salt: int) -> int:
    """An int4 weight in [-8, 7] with no arithmetic structure in either index.

    A stand-in for a trained weight, for the one kernel whose operands are big
    enough that a `(a*r + b*c) % 16` pattern aliases with the tensor's own
    strides. Deterministic and platform-independent — plain 32-bit integer
    mixing, no RNG and no seeding.
    """
    v = (r * 2654435761 + c * 2246822519 + salt * 3266489917) & 0xFFFFFFFF
    v ^= v >> 15
    return ((v * 2654435761 >> 13) & 0xF) - 8


def put_rowmajor_i8(img: dict, base: int, rows: int, cols: int, stride: int,
                    fn) -> None:
    for r in range(rows):
        for c in range(cols):
            img[base + r * stride + c] = fn(r, c) & 0xFF


def put_rowmajor_i4(img: dict, base: int, rows: int, cols: int, fn) -> None:
    """An int4 weight block, row-major, two nibbles per byte, low nibble first.

    The single definition of the layout mxu.sv reads and iss.TPU._nib decodes.
    """
    wrow = (cols * 4) // 8
    for r in range(rows):
        for c in range(cols):
            addr = base + r * wrow + c // 2
            cur = img.get(addr, 0)
            nib = fn(r, c) & 0xF
            img[addr] = ((cur & 0xF0) | nib) if c % 2 == 0 else ((cur & 0x0F) | (nib << 4))


# =============================================================================
# Per-kernel operand images.
#
# One builder per firmware kernel, keyed by its name. This is where a kernel's
# DRAM layout is defined *once* — `host/run_fw_matmul.py` imports the matmul one
# rather than carrying its own copy, and the testbench gets it as a file. A new
# kernel adds a function here and nothing else.
# =============================================================================
def operands_matmul(args) -> dict:
    """matmul.c / matmul_loop.c: A[M][K] int8, W[K][N] int4, both at their own bases."""
    k, n = args.ktiles * ROWS, args.ntiles * COLS
    img: dict = {}
    put_rowmajor_i8(img, A_ADDR, args.M, k, k, a_val)
    put_rowmajor_i4(img, W_ADDR, k, n, w_val)
    return img


def operands_ffn(args) -> dict:
    """ffn.c: X[T][D] int8, W1[D][F] int4, W2[F][D] int4."""
    T, D, F = 8, 8, 16
    img: dict = {}
    put_rowmajor_i8(img, 0x0000, T, D, D, a_val)
    put_rowmajor_i4(img, 0x0400, D, F, w_val)
    # A different salt so a builder mix-up shows up as a wrong answer, not a
    # coincidentally-equal one.
    put_rowmajor_i4(img, 0x0800, F, D, lambda r, c: ((r * 3 + c * 7) % 16) - 8)
    return img


def operands_mha(args) -> dict:
    """mha.c: X[T][D] int8 and three [D][DH] int4 projections, distinct salts."""
    T, D, DH = 8, 8, 8
    img: dict = {}
    put_rowmajor_i8(img, 0x0000, T, D, D, a_val)
    put_rowmajor_i4(img, 0x0400, D, DH, w_val)
    put_rowmajor_i4(img, 0x0500, D, DH, lambda r, c: ((r * 3 + c * 7) % 16) - 8)
    put_rowmajor_i4(img, 0x0600, D, DH, lambda r, c: ((r * 11 + c * 5) % 16) - 8)
    return img


def operands_adder(args) -> dict:
    """adder.c: the whole model's DRAM image, with **synthetic** weights.

    Deliberately not a checkpoint. `make fw FWPROG=adder` is a datapath
    regression — does the CPU issue the right commands and does the array
    compute what the ISS says — and tying it to an untracked `.pt` would make it
    unrunnable the moment the model is retrained. `adder_export.py` stages the
    real thing into this same map.

    The map is adder.c's, and this is the only other place it is written down:

        0x00000 X0     [T][D]   int8    0x00800 mask  [T][T]  int8
        0x01400 W_fc   [D][16]  int4    0x01800 logits[T][16] int32 (out)
        0x02000 + L*0x6000: WQKV [D][3D], +0x1800 Wo [D][D],
                            +0x2000 W1 [D][F], +0x4000 W2 [F][D]

    Every tensor gets its own salt so a mis-addressed weight shows up as a wrong
    answer rather than as a coincidentally equal one, and the padding columns of
    W_fc carry **live** weights: staged as zeros they would agree with a kernel
    that strided its second output tile wrongly, because both would be zero.

    The weights come from :func:`w_hash` rather than the `(a*r + b*c) % 16`
    pattern the smaller kernels use, for a reason worth keeping: `Wq`, `Wk` and
    `Wv` are *column slices* of one fused [D][3D] block, 64 columns apart, and
    `c*b mod 16` has a period dividing 16 — so any linear pattern makes the
    three projections bit-identical and a Wk/Wv addressing bug invisible.
    """
    T, D, DFF, VPAD, LAYERS = 32, 64, 256, 16, 4
    img: dict = {}

    put_rowmajor_i8(img, 0x00000, T, D, D, a_val)
    for t in range(T):
        for s in range(T):
            # 0 where s <= t, -8 above. S is int4, so S-8 <= -1 for every S in
            # range and ReLU takes a masked entry to exactly zero.
            img[0x00800 + t * T + s] = (0 if s <= t else -8) & 0xFF
    put_rowmajor_i4(img, 0x01400, D, VPAD, lambda r, c: w_hash(r, c, 0))

    for l in range(LAYERS):
        base = 0x02000 + l * 0x06000
        put_rowmajor_i4(img, base + 0x0000, D, 3 * D,
                        lambda r, c, s=4 * l + 1: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x1800, D, D,
                        lambda r, c, s=4 * l + 2: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x2000, D, DFF,
                        lambda r, c, s=4 * l + 3: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x4000, DFF, D,
                        lambda r, c, s=4 * l + 4: w_hash(r, c, s))
    return img


OPERANDS = {
    "matmul": operands_matmul,
    "matmul_loop": operands_matmul,
    "ffn": operands_ffn,
    "mha": operands_mha,
    "adder": operands_adder,
}


def build_operands(tpu: TPU, args) -> dict:
    """Write the kernel's operand image into the model's DRAM; return it."""
    if args.kernel not in OPERANDS:
        raise SystemExit(f"no operand builder for kernel {args.kernel!r} — "
                         f"add one to OPERANDS in {__file__}")
    image = OPERANDS[args.kernel](args)
    for addr, byte in image.items():
        tpu.dram[addr] = byte
    return image


def operand_image(m_rows: int, k: int, n: int) -> dict:
    """The matmul kernels' image, for `host/run_fw_matmul.py`.

    Kept as a positional-argument shim because the host script knows its shape
    as three numbers, not as parsed args.
    """
    img: dict = {}
    put_rowmajor_i8(img, A_ADDR, m_rows, k, k, a_val)
    put_rowmajor_i4(img, W_ADDR, k, n, w_val)
    return img


def write_hex(path: str, byte_map: dict, header: str) -> None:
    """Sparse `$readmemh` byte image: an `@addr` directive per discontinuity."""
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        prev = None
        for addr in sorted(byte_map):
            if prev is None or addr != prev + 1:
                f.write(f"@{addr:05x}\n")
            f.write(f"{byte_map[addr]:02x}\n")
            prev = addr


def write_cmds(path: str, cmds: list, header: str) -> None:
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        for unit, w0, w1, w2, w3 in cmds:
            f.write(f"{unit:08x} {w0:08x} {w1:08x} {w2:08x} {w3:08x}\n")
        f.write("ffffffff 00000000 00000000 00000000 00000000\n")   # terminator


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--trace", required=True,
                    help="command trace from a -DTPU_TRACE firmware build")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("-k", "--kernel", default="matmul",
                    help="which kernel's operand image to build "
                         f"({', '.join(sorted(OPERANDS))})")
    ap.add_argument("-M", type=int, default=8, help="token rows")
    ap.add_argument("--ktiles", type=int, default=4)
    ap.add_argument("--ntiles", type=int, default=2)
    args = ap.parse_args()

    k, n = args.ktiles * ROWS, args.ntiles * COLS

    tpu = TPU(rows=ROWS, cols=COLS)
    dram_in = build_operands(tpu, args)

    with open(args.trace) as f:
        records = parse_trace(f.read())
    cmds = tpu.run_trace(records)

    # Everything the run spilled back to DRAM, straight out of the model's own
    # write tracking — no second reference implementation to keep in step.
    dram_exp = {a: tpu.dram[a] for a in sorted(tpu.dram_written)}

    os.makedirs(args.out, exist_ok=True)
    shape = f"{args.kernel}, int4 row-major weights"
    write_hex(os.path.join(args.out, "fw_dram_in.hex"), dram_in,
              f"firmware operands: {shape}")
    write_hex(os.path.join(args.out, "fw_dram_exp.hex"), dram_exp,
              f"firmware golden DRAM output (ISS-computed): {shape}")
    write_cmds(os.path.join(args.out, "fw_cmds.hex"), cmds,
               f"expected command trace, {len(cmds)} commands, from {args.trace}")

    print(f"{len(cmds)} commands, {len(dram_in)} operand bytes in, "
          f"{len(dram_exp)} bytes out -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
