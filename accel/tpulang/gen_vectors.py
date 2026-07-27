#!/usr/bin/env python3
"""gen_vectors.py — golden test-vector generator for tpu_top_tb.sv.

Ties the whole toolchain together end to end:

  1. generate deterministic sample tensors (int8 activations, ternary weights,
     the requant {m0,n} word) and lay them out in the external-DRAM image exactly
     where the program's address arithmetic expects them;
  2. assemble a ``.tpu`` program into machine words (assembler.py);
  3. run those words in the ISS (iss.py) — whose DMA now really moves bytes
     DRAM<->scratchpad — to get the golden final memory image;
  4. emit three ``$readmemh``-loadable files the SystemVerilog testbench loads:

       tpu_prog.hex      instruction words   (one 32-bit word per line, dense)
       tpu_spad_in.hex   input tensors       (one DRAM byte per line, sparse @addr)
       tpu_spad_exp.hex  expected outputs    (one DRAM byte per line, sparse @addr)

The testbench seeds DRAM with the input file, runs the program (its DMA fills the
scratchpad, computes, spills results back), and checks DRAM against the expected
file. The expected file therefore contains exactly the bytes the program spilled
with ``wrmem`` (tracked in ``tpu.dram_written``) — the host-visible outputs.

Two program shapes are supported, chosen automatically from the program's own
``.equ`` constants:

  * default (relu_layer, vector_add, softmax_row): fixed 8x8 geometry below.
  * tiled matmul (defines ``MTILES``): a general A@W streamed tile-by-tile; the
    inputs are built in the program's tile-major DRAM layout and the ISS result
    is checked against an independent Python reference matmul.

    python gen_vectors.py                             # default program + geometry
    python gen_vectors.py -p examples/tiled_matmul.tpu
"""

from __future__ import annotations

import argparse
import os

from assembler import Assembler
from iss import TPU, CFG_TLEN, CFG_VLEN, CFG_SCALAR, s8

# --- geometry — MUST match the DUT localparams in tpu_top_tb.sv ---------------
ROWS = 8          # MXU contraction dim d (activation vector length)
COLS = 8          # MXU output features
T = 4             # token rows
ADDR_W = 16
M0_W, N_W = 12, 4

# --- scratchpad address map (matches relu_layer.tpu / tpu_top_tb.sv) ----------
ACT = 0x0000      # A[t][i] int8, row-major
WGT = 0x1000      # W[i][j] ternary, column-major 2-bit packed
RQW = 0x7000      # requant {m0,n} word (int32)

# --- requant scale {m0=1, n=0}: identity + int8 clip -> plain integer matmul --
RQ_M0, RQ_N = 1, 0

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROG = os.path.join(HERE, "examples", "relu_layer.tpu")
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "tpu", "tb", "vectors"))


def clip8(v: int) -> int:
    return 127 if v > 127 else (-128 if v < -128 else v)


def pack_wcol(colvals: list, wcol_bytes: int) -> list:
    """Pack a ternary weight column ({-1,0,1}) into 2-bit codes (00=0,01=+1,11=-1)."""
    colint = 0
    for i, w in enumerate(colvals):
        code = 0 if w == 0 else (1 if w == 1 else 0b11)
        colint |= code << (2 * i)
    return [(colint >> (8 * b)) & 0xFF for b in range(wcol_bytes)]


def requant_word() -> int:
    return (RQ_N << M0_W) | RQ_M0


def assemble_program(path: str) -> tuple[list, dict]:
    """Assemble and also return the symbol table (.equ constants + labels)."""
    with open(path) as f:
        asm = Assembler()
        asm.scan(f.read())
    return asm.encode_all(), asm.consts


# =============================================================================
# Default fixed-geometry programs (relu_layer / vector_add / softmax_row).
# =============================================================================
def gen_tensors() -> tuple[list, list]:
    """Bounded, deterministic operands (identical to the old in-TB generator)."""
    act = [[((t * 3 + i) % 9) - 4 for i in range(ROWS)] for t in range(T)]     # [-4,4]
    wgt = [[((i + 2 * j) % 3) - 1 for j in range(COLS)] for i in range(ROWS)]   # {-1,0,1}
    return act, wgt


def build_default_image(tpu: TPU) -> dict:
    """Write the fixed-geometry tensors into DRAM; return {addr: byte} for the file."""
    act, wgt = gen_tensors()
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._a(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    # Activations: A[t][i] int8, row-major.
    for t in range(T):
        for i in range(ROWS):
            put(ACT + t * ROWS + i, act[t][i])

    # Weights: W[i][j] ternary, column-major 2-bit packed.
    wcol_bytes = (ROWS * 2) // 8
    for j in range(COLS):
        colvals = [wgt[i][j] for i in range(ROWS)]
        for b, byte in enumerate(pack_wcol(colvals, wcol_bytes)):
            put(WGT + j * wcol_bytes + b, byte)

    # Requant {m0,n} word: m0 in low M0_W bits, n above.
    word = requant_word()
    for b in range(4):
        put(RQW + b, (word >> (8 * b)) & 0xFF)

    tpu.cfg[CFG_TLEN] = T
    tpu.cfg[CFG_VLEN] = T * COLS
    tpu.cfg[CFG_SCALAR] = RQW
    return img


# =============================================================================
# Tiled matmul (examples/tiled_matmul.tpu): general A@W streamed tile-by-tile.
# Geometry and DRAM layout come straight from the program's .equ constants, so
# the two never drift.
# =============================================================================
def build_matmul_image(tpu: TPU, c: dict) -> tuple[dict, dict, dict]:
    """Lay A/W/RQW out in the program's tile-major DRAM layout; also return the
    independent reference C = clip(A @ W) and the C-tile address map."""
    if c["ROWS"] != ROWS or c["COLS"] != COLS:
        raise SystemExit(f"matmul geometry ROWS/COLS must be {ROWS}/{COLS} "
                         f"(DUT array size), got {c['ROWS']}/{c['COLS']}")
    Tt, MT, NT, KT = c["T"], c["MTILES"], c["NTILES"], c["KTILES"]
    ATILE, WTILE, CTILE, WCOL = c["ATILE"], c["WTILE"], c["CTILE"], c["WCOL"]
    A, W, Cbase = c["A"], c["W"], c["C"]
    M, K, N = MT * Tt, KT * ROWS, NT * COLS

    # Deterministic operands in [-4,4] (int8) and {-1,0,1} (ternary).
    a_full = [[((m * 3 + k) % 9) - 4 for k in range(K)] for m in range(M)]
    w_full = [[((k + 2 * n) % 3) - 1 for n in range(N)] for k in range(K)]

    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._a(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    # A tiles [mt][kt], each T x ROWS int8 row-major, tiles back to back.
    for mt in range(MT):
        for kt in range(KT):
            base = A + (mt * KT + kt) * ATILE
            for t in range(Tt):
                for i in range(ROWS):
                    put(base + t * ROWS + i, a_full[mt * Tt + t][kt * ROWS + i])

    # W tiles [nt][kt], each ROWS x COLS ternary, col-major 2-bit packed.
    for nt in range(NT):
        for kt in range(KT):
            base = W + (nt * KT + kt) * WTILE
            for j in range(COLS):
                colvals = [w_full[kt * ROWS + i][nt * COLS + j] for i in range(ROWS)]
                for b, byte in enumerate(pack_wcol(colvals, WCOL)):
                    put(base + j * WCOL + b, byte)

    # Requant {m0,n} word (identity: m0=1, n=0).
    word = requant_word()
    for b in range(4):
        put(c["RQW"] + b, (word >> (8 * b)) & 0xFF)

    # Reference C[m][n] = clip(Σ_k A[m][k]·W[k][n]) at its tile-major C address.
    ref: dict[int, int] = {}
    for mt in range(MT):
        for nt in range(NT):
            base = Cbase + (mt * NT + nt) * CTILE
            for t in range(Tt):
                for j in range(COLS):
                    m, n = mt * Tt + t, nt * COLS + j
                    acc = sum(a_full[m][k] * w_full[k][n] for k in range(K))
                    ref[base + t * COLS + j] = clip8(acc)

    tpu.cfg[CFG_TLEN] = Tt
    tpu.cfg[CFG_SCALAR] = c["RQW"]
    return img, ref, {"M": M, "N": N, "K": K}


def check_reference(tpu: TPU, ref: dict) -> int:
    """Compare the ISS DRAM result against the reference; return #mismatches."""
    bad = 0
    for addr in sorted(ref):
        got = s8(tpu.dram[tpu._a(addr)])
        if got != ref[addr]:
            bad += 1
            if bad <= 8:
                print(f"  MISMATCH @{addr:04x}: iss {got:4d}  ref {ref[addr]:4d}")
    return bad


# =============================================================================
# Emit.
# =============================================================================
def emit_words(path: str, words: list, note: str) -> None:
    with open(path, "w") as f:
        f.write(f"// {note}\n")
        for w in words:
            f.write(f"{w & 0xFFFFFFFF:08x}\n")


def emit_bytes(path: str, img: dict, note: str) -> None:
    """Sparse $readmemh byte image: `@addr` markers before each contiguous run."""
    with open(path, "w") as f:
        f.write(f"// {note}\n")
        prev = None
        for addr in sorted(img):
            if addr != prev:
                f.write(f"@{addr:04x}\n")
            f.write(f"{img[addr] & 0xFF:02x}\n")
            prev = addr + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--program", default=DEFAULT_PROG, help="tpulang .tpu source")
    ap.add_argument("-o", "--out-dir", default=OUT_DIR, help="output directory")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    words, consts = assemble_program(args.program)
    tpu = TPU(rows=ROWS, cols=COLS, addr_w=ADDR_W, m0_w=M0_W, n_w=N_W)

    ref = None
    if "MTILES" in consts:                       # the streaming tiled matmul
        in_img, ref, dims = build_matmul_image(tpu, consts)
        shape = f"[{dims['M']}x{dims['K']}] @ [{dims['K']}x{dims['N']}]"
    else:                                        # fixed-geometry default programs
        in_img = build_default_image(tpu)
        shape = None

    tpu.run(words)
    exp_img = {a: tpu.dram[a] for a in sorted(tpu.dram_written)}

    if ref is not None:
        bad = check_reference(tpu, ref)
        if bad:
            print(f"reference check FAILED: {bad} mismatch(es) for {shape}")
            return 1
        print(f"reference check OK: {shape} matches Python A@W "
              f"({len(ref)} output bytes)")

    prog_path = os.path.join(args.out_dir, "tpu_prog.hex")
    in_path = os.path.join(args.out_dir, "tpu_spad_in.hex")
    exp_path = os.path.join(args.out_dir, "tpu_spad_exp.hex")
    name = os.path.basename(args.program)
    emit_words(prog_path, words, f"instruction words assembled from {name}")
    emit_bytes(in_path, in_img, f"input tensors for {name} (DRAM image)")
    emit_bytes(exp_path, exp_img, f"golden outputs for {name} (ISS-computed, DRAM)")

    print(f"program : {len(words):4d} words         -> {prog_path}")
    print(f"inputs  : {len(in_img):4d} bytes         -> {in_path}")
    print(f"expected: {len(exp_img):4d} bytes written -> {exp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
