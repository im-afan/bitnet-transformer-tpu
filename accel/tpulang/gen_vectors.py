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

The input image is chosen automatically from the program's own ``.equ``
constants (:func:`program_kind`, dispatched by :func:`build_image`):

  * default (relu_layer, softmax_row): the fixed 8x8 geometry below.
  * vector_add (defines ``VEC_BYTES``): two int8 operand spans at A and B.
  * vpu_matmul (defines ``D_HEAD``): int8 Q and K for the attention-score block.
  * tiled matmul (defines ``MTILES``): a general A@W streamed tile-by-tile; the
    inputs are built in the program's tile-major DRAM layout and the ISS result
    is checked against an independent Python reference matmul.

A program that produces no reference result here can still be checked against
PyTorch — see ``torch_ref.py``, which decodes these same images into tensors.

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

# vpu_matmul's scores are int8 x int8 over 16 terms, so they run an order of
# magnitude wider than the ternary layers' and need a real rescale rather than
# the identity above. {m0=1, n=1} puts the bulk of the distribution across int8
# while leaving the strongest few scores to clip — so the requant's rounding,
# its arithmetic (floor) shift and its saturation are all exercised.
VPU_RQ_M0, VPU_RQ_N = 1, 1

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


def requant_word(m0: int = RQ_M0, n: int = RQ_N) -> int:
    return (n << M0_W) | m0


def assemble_program(path: str) -> tuple[list, dict]:
    """Assemble and also return the symbol table (.equ constants + labels)."""
    # Explicit UTF-8: the example sources use math notation in their comments
    # (Sigma, middle dot, section signs), and Python's default open() encoding
    # is the locale's — cp1252 on Windows, which cannot decode them.
    with open(path, encoding="utf-8") as f:
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
# vector_add (examples/vector_add.tpu): two int8 operand spans.
# =============================================================================
def build_vector_add_image(tpu: TPU, c: dict) -> dict:
    """Seed A and B at their own DRAM addresses; return {addr: byte}.

    ``vecadd`` is a VPU elementwise op, so despite the example's ``(int32)``
    comments it reads **int8** operands at stride 1 and widens to int32 on the
    way out — the operands are therefore ``VEC_BYTES`` plain bytes, of which the
    program's ``vlen = N`` uses the first N. The whole span is filled anyway so
    the DMA's ``len``-byte fill never reads DRAM the host left untouched.

    The values span most of int8 and the two sequences have coprime strides, so
    every lane sees a different (sign, sign) pairing and the sums overflow int8
    — which is the point of a widening add.
    """
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._a(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for k in range(c["VEC_BYTES"]):
        put(c["A"] + k, ((k * 7) % 251) - 125)          # [-125, 125]
        put(c["B"] + k, ((k * 13 + 61) % 251) - 125)    # [-125, 125]

    tpu.cfg[CFG_VLEN] = c["N"]
    return img


# =============================================================================
# vpu_matmul (examples/vpu_matmul.tpu): attention scores S = Q @ K^T.
# =============================================================================
def build_vpu_matmul_image(tpu: TPU, c: dict) -> dict:
    """Seed Q, K and the requant word; return {addr: byte}.

    Both operands are int8 *activations* here — unlike every other example there
    is no ternary tensor, which is the whole reason this kernel runs on the VPU
    instead of the MXU. They span [-8, 8] rather than the ternary layers' [-4, 4]
    so the 16-term dot products cover a wide enough range for the requant below
    to have something to do.

    The two sequences step through the period-17 sawtooth at different rates (3
    and 5, coprime to 17 and to each other), so Q's row `t` and K's row `s` line
    up only when ``3t = 5s (mod 17)`` — a minority of pairs. Those aligned pairs
    are the near-maximal scores, exactly as a real attention block has a few
    strong matches against a background of weak ones.
    """
    T, D = c["T"], c["D_HEAD"]
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._a(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for t in range(T):
        for d in range(D):
            put(c["Q"] + t * D + d, ((t * 3 + d) % 17) - 8)   # [-8, 8]
            put(c["K"] + t * D + d, ((t * 5 + d) % 17) - 8)   # [-8, 8]

    word = requant_word(VPU_RQ_M0, VPU_RQ_N)
    for b in range(4):
        put(c["RQW"] + b, (word >> (8 * b)) & 0xFF)

    tpu.cfg[CFG_VLEN] = D
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


# =============================================================================
# Dispatch: pick the input image from the program's own constants.
# =============================================================================
def program_kind(consts: dict) -> str:
    """Which input image a program needs, from its ``.equ`` symbol table.

    Keyed on constants rather than filename so a renamed or derived program
    still resolves. ``torch_ref.KERNELS`` identifies programs the same way.
    """
    if "MTILES" in consts:
        return "tiled_matmul"
    if "VEC_BYTES" in consts:
        return "vector_add"
    if "D_HEAD" in consts:
        return "vpu_matmul"
    return "default"                 # relu_layer, softmax_row: fixed 8x8


def build_image(tpu: TPU, consts: dict) -> tuple[dict, dict | None, dict | None]:
    """Seed ``tpu.dram`` for this program.

    Returns ``(in_img, ref, dims)`` — the ``{addr: byte}`` input image, plus the
    independent Python reference result and problem dimensions where the builder
    computes them (tiled matmul only; ``None`` otherwise).
    """
    kind = program_kind(consts)
    if kind == "tiled_matmul":
        return build_matmul_image(tpu, consts)
    if kind == "vector_add":
        return build_vector_add_image(tpu, consts), None, None
    if kind == "vpu_matmul":
        return build_vpu_matmul_image(tpu, consts), None, None
    return build_default_image(tpu), None, None


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

    in_img, ref, dims = build_image(tpu, consts)
    shape = (f"[{dims['M']}x{dims['K']}] @ [{dims['K']}x{dims['N']}]"
             if dims else None)

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
