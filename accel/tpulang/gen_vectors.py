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

  * default (relu_layer): the fixed 8x8 geometry below.
  * vector_add (defines ``VEC_BYTES``): two int8 operand spans at A and B.
  * vpu_matmul (defines ``D_HEAD``): int8 Q and K for the attention-score block.
  * tiled matmul (defines ``MTILES``): a general A@W streamed tile-by-tile; the
    inputs are built in the program's tile-major DRAM layout and the ISS result
    is checked against an independent Python reference matmul.
  * transpose_dma (defines ``VT``): a fused [T][3D] QKV block, checked against
    the transpose permutation written out directly.
  * highmem_dma (defines ``HI_HALF``): one int8 block in low DRAM; every output
    lands above 64 KB, which nothing else here touches.
  * adder_model (defines ``LSTRIDE``): the whole 4-layer transformer — X, the
    causal mask, ``fc``, and every layer's ternary weights and requant words.
    Operands go through :func:`_mix` rather than an index formula; see its
    docstring for why that matters at this scale.

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
ADDR_W = 16       # scratchpad byte address
MEM_ADDR_W = 19   # external SRAM (DRAM) byte address — a *wider* space
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
# Default fixed-geometry programs (relu_layer / vector_add).
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
        tpu.dram[tpu._d(addr)] = byte & 0xFF
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
        tpu.dram[tpu._d(addr)] = byte & 0xFF
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
        tpu.dram[tpu._d(addr)] = byte & 0xFF
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
        tpu.dram[tpu._d(addr)] = byte & 0xFF
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
# Strided matmul (examples/strided_matmul.tpu): one tile *in place* inside a
# larger matrix, exercising the arow/wcol/crow config strides rather than a
# tile-shaped copy of the operands.
# =============================================================================
def build_strided_image(tpu: TPU, c: dict) -> tuple[dict, dict, dict]:
    """Full M x K / K x N operands, plus a reference in which only the selected
    tile of C changes and every other element keeps its seeded value."""
    if c["ROWS"] != ROWS or c["COLS"] != COLS:
        raise SystemExit(f"strided geometry ROWS/COLS must be {ROWS}/{COLS} "
                         f"(DUT array size), got {c['ROWS']}/{c['COLS']}")
    M, K, N = c["M"], c["K"], c["N"]
    TI, TJ = c["TI"], c["TJ"]
    A, W, Cb = c["A"], c["W"], c["C"]
    AROW, WCOL, CROW = c["AROW"], c["WCOL"], c["CROW"]

    # Deterministic operands, distinct patterns from build_matmul_image's so a
    # copy-paste of the wrong builder is obvious.
    a_full = [[((m * 5 + k * 3) % 9) - 4 for k in range(K)] for m in range(M)]
    w_full = [[((k * 2 + n) % 3) - 1 for k in range(K)] for n in range(N)]
    # C is seeded non-zero: the tile the matmul does *not* touch must come back
    # byte-identical, which is what catches a stride that walks off the tile.
    c_full = [[((m * 7 + n * 11) % 61) - 30 for n in range(N)] for m in range(M)]

    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for m in range(M):                      # A: M x K int8, row-major
        for k in range(K):
            put(A + m * AROW + k, a_full[m][k])

    for n in range(N):                      # W: K x N trits, col-major 2b packed
        colvals = [w_full[n][k] for k in range(K)]
        for b, byte in enumerate(pack_wcol(colvals, WCOL)):
            put(W + n * WCOL + b, byte)

    for m in range(M):                      # C: M x N int32, row-major
        for n in range(N):
            v = c_full[m][n] & 0xFFFFFFFF
            for b in range(4):
                put(Cb + m * CROW + n * 4 + b, (v >> (8 * b)) & 0xFF)

    # Reference: C unchanged except the one tile, which is a plain int32 matmul
    # over the tile's own rows/columns (no requant — the program uses `matmul`).
    ref: dict[int, int] = {}
    for m in range(M):
        for n in range(N):
            in_tile = (TI * ROWS <= m < (TI + 1) * ROWS and
                       TJ * COLS <= n < (TJ + 1) * COLS)
            if in_tile:
                val = sum(a_full[m][TJ * ROWS + i] * w_full[n][TI * ROWS + i]
                          for i in range(ROWS))
            else:
                val = c_full[m][n]
            # check_reference() compares signed bytes (s8), so store the four
            # int32 bytes in that same convention rather than raw unsigned.
            val &= 0xFFFFFFFF
            for b in range(4):
                ref[Cb + m * CROW + n * 4 + b] = s8((val >> (8 * b)) & 0xFF)

    tpu.cfg[CFG_TLEN] = ROWS
    return img, ref, {"M": M, "N": N, "K": K}


# =============================================================================
# Hardware-tiled matmul (examples/tiled_matmul_hw.tpu): the whole M x K @ K x N
# contraction as one `matmul_t`, over a flat row-major layout.
# =============================================================================
def build_tiled_hw_image(tpu: TPU, c: dict) -> tuple[dict, dict, dict]:
    """Flat A/W/RQW operands, plus a reference for both output forms.

    The reference is a plain Python matmul over the *full* K, computed in one
    go — deliberately not tile-by-tile. Matching it is what shows the hardware's
    n-outer/k-inner decomposition reassembles the same contraction.
    """
    if c["ROWS"] != ROWS or c["COLS"] != COLS:
        raise SystemExit(f"tiled_hw geometry ROWS/COLS must be {ROWS}/{COLS} "
                         f"(DUT array size), got {c['ROWS']}/{c['COLS']}")
    M, K, N = c["M"], c["K"], c["N"]
    A, W, C32, C8 = c["A"], c["W"], c["C32"], c["C8"]
    AROW, WCOL, CROW = c["AROW"], c["WCOL"], c["CROW"]

    a_full = [[((m * 3 + k * 5) % 9) - 4 for k in range(K)] for m in range(M)]
    w_full = [[((k + 3 * n) % 3) - 1 for k in range(K)] for n in range(N)]

    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for m in range(M):                      # A: M x K int8, row-major
        for k in range(K):
            put(A + m * AROW + k, a_full[m][k])

    for n in range(N):                      # W: K x N trits, col-major 2b packed
        for b, byte in enumerate(pack_wcol([w_full[n][k] for k in range(K)], WCOL)):
            put(W + n * WCOL + b, byte)

    word = requant_word()                   # identity requant (m0=1, n=0)
    for b in range(4):
        put(c["RQW"] + b, (word >> (8 * b)) & 0xFF)

    ref: dict[int, int] = {}
    for m in range(M):
        for n in range(N):
            acc = sum(a_full[m][k] * w_full[n][k] for k in range(K))
            v = acc & 0xFFFFFFFF
            for b in range(4):              # int32 result
                ref[C32 + m * CROW + n * 4 + b] = s8((v >> (8 * b)) & 0xFF)
            ref[C8 + m * N + n] = clip8(acc)   # requantized int8 result

    tpu.cfg[CFG_TLEN] = M
    tpu.cfg[CFG_SCALAR] = c["RQW"]
    return img, ref, {"M": M, "N": N, "K": K}


# =============================================================================
# vecmatmul (examples/vecmatmul.tpu): an attention score block as one macro op.
# =============================================================================
def build_vecmatmul_image(tpu: TPU, c: dict) -> tuple[dict, dict, dict]:
    """Q/K operands plus the reference S = Q @ K^T, computed directly in Python."""
    TQ, TK, DH = c["TQ"], c["TK"], c["DH"]
    Q, K, S = c["Q"], c["K"], c["S"]
    QROW, KROW, SROW = c["QROW"], c["KROW"], c["SROW"]

    q_full = [[((t * 7 + d * 3) % 15) - 7 for d in range(DH)] for t in range(TQ)]
    k_full = [[((s * 5 + d * 2) % 13) - 6 for d in range(DH)] for s in range(TK)]

    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for t in range(TQ):
        for d in range(DH):
            put(Q + t * QROW + d, q_full[t][d])
    for s in range(TK):
        for d in range(DH):
            put(K + s * KROW + d, k_full[s][d])

    ref: dict[int, int] = {}
    for t in range(TQ):
        for s in range(TK):
            acc = sum(q_full[t][d] * k_full[s][d] for d in range(DH)) & 0xFFFFFFFF
            for b in range(4):
                ref[S + t * SROW + s * 4 + b] = s8((acc >> (8 * b)) & 0xFF)

    tpu.cfg[CFG_VLEN] = DH
    return img, ref, {"M": TQ, "N": TK, "K": DH}


# =============================================================================
def build_softmax_rows_image(tpu: TPU, c: dict) -> dict:
    """Seed the score rows and the requant word. No independent reference here:
    the result is a chain of LUT + fixed-point steps whose *definition* is the
    hardware's pass structure, so re-deriving it in Python would only restate
    iss.py. torch_ref.py checks it against PyTorch instead, which is the
    genuinely independent comparison."""
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    ROWS_, N = c["ROWS"], c["N"]
    for t in range(ROWS_):
        for i in range(N):
            # A spread wide enough that the row max really matters, and
            # different per row so a stuck max would show up.
            put(c["X"] + t * c["SROW"] + i, ((t * 11 + i * 7) % 41) - 20)

    # {m0=1, n=1}: halve then clip, landing x-max in the exp table's input scale
    # without collapsing the distribution to a couple of entries.
    word = requant_word(1, 1)
    for b in range(4):
        put(c["RQW"] + b, (word >> (8 * b)) & 0xFF)

    tpu.cfg[CFG_VLEN] = N
    return img


# =============================================================================
# transpose_dma (examples/transpose_dma.tpu): the DMA's transposing mode.
# =============================================================================
def build_transpose_image(tpu: TPU, c: dict) -> tuple[dict, dict, None]:
    """A fused [T][3D] QKV block, plus the two transposed slices it should yield.

    The reference is the permutation written out directly — element (t, d) of a
    slice belongs at ``base + d*T + t`` — rather than a second address generator,
    so it cannot share a bug with the one under test. Values are made to depend
    on both indices (and to differ between the Q and V slices) so a transfer that
    transposed the wrong slice, or skipped the row stride, cannot pass.
    """
    T, D, D3 = c["T"], c["D"], c["D3"]
    QKV, VT, QT = c["QKV"], c["VT"], c["QT"]

    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    # The whole row is seeded, including the K columns nothing should read.
    block = [[((t * 13 + j * 5) % 63) - 31 for j in range(D3)] for t in range(T)]
    for t in range(T):
        for j in range(D3):
            put(QKV + t * D3 + j, block[t][j])

    ref: dict[int, int] = {}
    for t in range(T):
        for d in range(D):
            ref[VT + d * T + t] = block[t][2 * D + d]   # V slice, transposed
            ref[QT + d * T + t] = block[t][d]           # Q slice, transposed
    return img, ref, None


# =============================================================================
# highmem_dma (examples/highmem_dma.tpu): DMA above the low 64 KB of DRAM.
# =============================================================================
def build_highmem_image(tpu: TPU, c: dict) -> dict:
    """Seed one [R][C] int8 block in *low* DRAM; every output lands high.

    No independent reference here — the program's outputs are a copy, a
    transpose and a round trip, which torch_ref.py checks against what those
    operations mean rather than against a second address calculation.

    Values depend on both indices and span most of int8, so a transfer that
    dropped the row stride, transposed the wrong way, or aliased a high address
    down into this seeded low block cannot accidentally agree.
    """
    R, C = c["R"], c["C"]
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    for r in range(R):
        for k in range(C):
            put(c["SRC"] + r * C + k, ((r * 7 + k * 3) % 251) - 125)
    return img


# =============================================================================
# adder_model (examples/adder_model.tpu): the whole transformer, 4 layers + head.
# =============================================================================
# Requant {m0, n} per layer, in the program's slot order (adder_kernel.md §4).
# A real checkpoint derives these from calibrated scales; here they are chosen
# only so no stage is degenerate — the ISS and the reference apply whatever
# words are in DRAM, so equality does not depend on the values, but a set that
# saturated everything to +-127 would make the comparison vacuous. The shifts
# below track the accumulator growth at each stage: ternary matmuls over K=128
# land around +-1e3, the two attention matmuls over 32 terms around +-5e4.
ADDER_RQ = [
    (1, 1),   # 0  RQ_Q    X @ Wq      128 terms, sd ~43
    (1, 1),   # 1  RQ_K    X @ Wk
    (1, 1),   # 2  RQ_V    X @ Wv
    (1, 6),   # 3  RQ_S    Q @ K^T     32 terms of ~21*21, sd ~2.5k
    (1, 0),   # 4  RQ_ID   mask clamp — must be identity
    (1, 0),   # 5  RQ_P    relu(S8), already in [0,127]
    (1, 7),   # 6  RQ_A    P @ V       32 terms; P is unsigned, so this one
              #                        carries a mean as well as a spread
    (1, 4),   # 7  RQ_O    A @ Wo      128 terms, sd ~530
    (1, 0),   # 8  RQ_XO   X + O       must be identity: the second add needs
              #                        both operands back on X's scale
    (1, 1),   # 9  RQ_X1   dyt(XO + X) — norm1; carries alpha, clips at +-127
    (1, 3),   # 10 RQ_H    X1 @ W1     128 terms, sd ~250
    (1, 0),   # 11 RQ_HR   relu(H)
    (1, 3),   # 12 RQ_F    HR @ W2
    (1, 1),   # 13 RQ_X2   dyt(X1 + F) — norm2
]


def _mix(x: int) -> int:
    """A cheap avalanche hash, for deterministic well-mixed test operands.

    The other builders use index formulas like ``(i*3 + j) % 9 - 4``, which are
    fine for an 8x8 tile but degenerate at model scale: a ternary weight of
    ``(k*31 + n*17) % 3`` is a period-3 lattice in both indices, and contracting
    it over K=128 against an equally regular activation makes ~97% of the
    products cancel. Mixing first keeps the operands deterministic without that
    structure, so the accumulators behave like the real thing.
    """
    x &= 0xFFFFFFFF
    x = (x * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    return x


def build_adder_model_image(tpu: TPU, c: dict) -> dict:
    """Seed X, the causal mask, `fc`, and all four layers' weights + requant words.

    No independent reference here: the pipeline is 4 layers of matmul/requant/
    LUT-free integer stages whose *definition* is the program, so restating it
    in this file would only duplicate iss.py. torch_ref.py checks it against a
    from-scratch PyTorch implementation instead, which is the comparison that
    means something.

    Weight trits mix all three indices so no two columns (or layers) are alike;
    a kernel that read the wrong layer block, the wrong projection out of the
    fused block, or the wrong column stride cannot accidentally agree.
    """
    T, D, D3 = c["T"], c["D"], c["D3"]
    VOCAB, LAYERS, WCOL = c["VOCAB"], c["LAYERS"], c["WCOL"]
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.dram[tpu._d(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    # X[t][d] int8 in [-8, 8], the embedded + positionally-encoded input.
    for t in range(T):
        for d in range(D):
            put(c["D_XIN"] + t * D + d, _mix(t * 4099 + d * 13 + 1) % 17 - 8)

    # Causal mask: 0 where key <= query, -128 above the diagonal.
    for t in range(T):
        for s in range(T):
            put(c["D_MASK"] + t * T + s, 0 if s <= t else -128)

    # fc.weight [VOCAB][D] int8 — nn.Linear stores [out, in] already.
    for v in range(VOCAB):
        for d in range(D):
            put(c["D_WFC"] + v * D + d, _mix(v * 65537 + d * 29 + 2) % 31 - 15)

    for L in range(LAYERS):
        base = c["D_L0"] + L * c["LSTRIDE"]

        # Fused [Wq|Wk|Wv]: [D][3D] trits, column-major, 2-bit packed. The salt
        # carries the layer and the block, so no two layers and no two of the
        # three projections share a weight — a kernel that read the wrong layer
        # block or the wrong third of the fused block cannot agree by accident.
        for n in range(D3):
            col = [_mix(L * 1000003 + n * 1021 + k + 3) % 3 - 1 for k in range(D)]
            for b, byte in enumerate(pack_wcol(col, WCOL)):
                put(base + c["O_WQKV"] + n * WCOL + b, byte)

        # Wo, W1, W2: [D][D] trits each, distinct salts.
        for off, salt in ((c["O_WO"], 11), (c["O_W1"], 22), (c["O_W2"], 33)):
            for n in range(D):
                col = [_mix(L * 1000003 + n * 1021 + k + salt * 7919)
                       % 3 - 1 for k in range(D)]
                for b, byte in enumerate(pack_wcol(col, WCOL)):
                    put(base + off + n * WCOL + b, byte)

        # This layer's 14 requant {m0,n} words.
        for k, (m0, nsh) in enumerate(ADDER_RQ):
            word = requant_word(m0, nsh)
            for b in range(4):
                put(base + c["O_RQW"] + 4 * k + b, (word >> (8 * b)) & 0xFF)

    return img


# =============================================================================
# Dispatch: pick the input image from the program's own constants.
# =============================================================================
def program_kind(consts: dict) -> str:
    """Which input image a program needs, from its ``.equ`` symbol table.

    Keyed on constants rather than filename so a renamed or derived program
    still resolves. ``torch_ref.KERNELS`` identifies programs the same way.
    """
    if "LSTRIDE" in consts:         # adder_model: the whole transformer
        return "adder_model"
    if "HI_HALF" in consts:         # highmem_dma: DRAM above the low 64 KB
        return "highmem_dma"
    if "VT" in consts:              # transpose_dma: the DMA's `.t` mode
        return "transpose_dma"
    if "MTILES" in consts:
        return "tiled_matmul"
    if "C32" in consts:             # tiled_matmul_hw: whole contraction in HW
        return "tiled_matmul_hw"
    if "TQ" in consts:              # vecmatmul: attention block as a macro op
        return "vecmatmul"
    if "AROW" in consts:            # strided_matmul: config-stride window
        return "strided_matmul"
    if "VEC_BYTES" in consts:
        return "vector_add"
    if "D_HEAD" in consts:
        return "vpu_matmul"
    return "default"                 # relu_layer: fixed 8x8


def build_image(tpu: TPU, consts: dict) -> tuple[dict, dict | None, dict | None]:
    """Seed ``tpu.dram`` for this program.

    Returns ``(in_img, ref, dims)`` — the ``{addr: byte}`` input image, plus the
    independent Python reference result and problem dimensions where the builder
    computes them (tiled matmul only; ``None`` otherwise).
    """
    kind = program_kind(consts)
    if kind == "adder_model":
        return build_adder_model_image(tpu, consts), None, None
    if kind == "highmem_dma":
        return build_highmem_image(tpu, consts), None, None
    if kind == "transpose_dma":
        return build_transpose_image(tpu, consts)
    if kind == "tiled_matmul":
        return build_matmul_image(tpu, consts)
    if kind == "tiled_matmul_hw":
        return build_tiled_hw_image(tpu, consts)
    if kind == "vecmatmul":
        return build_vecmatmul_image(tpu, consts)
    if kind == "strided_matmul":
        return build_strided_image(tpu, consts)
    if kind == "vector_add":
        return build_vector_add_image(tpu, consts), None, None
    if kind == "vpu_matmul":
        return build_vpu_matmul_image(tpu, consts), None, None
    return build_default_image(tpu), None, None


def check_reference(tpu: TPU, ref: dict) -> int:
    """Compare the ISS DRAM result against the reference; return #mismatches."""
    bad = 0
    for addr in sorted(ref):
        got = s8(tpu.dram[tpu._d(addr)])
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
    tpu = TPU(rows=ROWS, cols=COLS, addr_w=ADDR_W, mem_addr_w=MEM_ADDR_W,
              m0_w=M0_W, n_w=N_W)

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
        what = f"{shape} " if shape else ""
        print(f"reference check OK: {what}matches the Python reference "
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
