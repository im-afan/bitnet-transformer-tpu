#!/usr/bin/env python3
"""torch_ref.py — PyTorch references for the tpulang example kernels.

The point is to check the *hardware* against something that shares no code with
it. ``run_program.py`` already compares the FPGA's output bytes to a golden
image, but that image comes out of ``iss.py`` — the same simulator the assembler
and the testbench vectors are built from, so agreeing with it only proves the
device matches the model, not that either computes the right thing. This module
is the independent leg: it reads the *bytes actually sent to the board*, decodes
them into tensors, computes the kernel in plain PyTorch, and compares against the
*bytes actually read back*.

Nothing here calls the ISS. The only shared assumption is the DRAM layout, and
even that is taken from the program's own ``.equ`` constants (the assembler's
symbol table) rather than restated, so a layout change moves both sides at once.

Each reference is a function ``(consts, get_in, get_out) -> [Check, ...]`` where
``get_in``/``get_out`` fetch one byte of the input / output DRAM image. It
returns one :class:`Check` per output tensor the program produces.

Everything is computed in ``int64`` and compared **exactly** — these kernels are
integer end to end (int8 operands, int32 accumulate, fixed-point requant), so
there is no tolerance to allow. ``int64`` rather than ``int32`` only so the
``acc * m0`` intermediate inside :func:`requant8` cannot wrap.

Standalone (no board, no serial port) — assemble each example, run it in the
ISS, and check the ISS's DRAM against these references:

    python accel/tpulang/torch_ref.py
    python accel/tpulang/torch_ref.py -p examples/tiled_matmul.tpu

Against real hardware, ``accel/tpu/host/run_program.py`` calls :func:`verify`
after reading the results back.

Requires ``torch`` (CPU is fine).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch

# gen_vectors/iss/assembler import each other flatly; join them rather than
# turning tpulang into a package.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gen_vectors as gv                                      # noqa: E402
from iss import TPU, s8                                       # noqa: E402


class NoReference(Exception):
    """No PyTorch reference is registered for this program."""


class MissingByte(Exception):
    """An expected output byte is absent from the image being checked."""


# ---- byte-image readers -----------------------------------------------------

def input_reader(img: dict):
    """Read the DRAM *input* image. Absent bytes are 0 — DRAM starts zeroed, so
    a region the image never seeded really does read back as zeros."""
    return lambda addr: img.get(addr, 0)


def output_reader(img: dict):
    """Read the DRAM *output* image. An absent byte is an error, not a zero: it
    means the program never spilled a byte the reference expects, which is a
    real disagreement about what the kernel produces."""
    def get(addr: int) -> int:
        try:
            return img[addr]
        except KeyError:
            raise MissingByte(
                f"no output byte at 0x{addr:05x} — the program never wrmem'd it, "
                f"or the reference and the program disagree on the address map"
            ) from None
    return get


# ---- decode: DRAM bytes -> tensors ------------------------------------------

def i8_tensor(byte_at, base: int, shape: tuple) -> torch.Tensor:
    """``shape`` int8 elements at stride 1 from ``base``, widened to int64."""
    n = math.prod(shape)
    return torch.tensor([s8(byte_at(base + k)) for k in range(n)],
                        dtype=torch.int64).reshape(shape)


def i32_tensor(byte_at, base: int, shape: tuple) -> torch.Tensor:
    """``shape`` little-endian int32 elements at stride 4 from ``base``."""
    n = math.prod(shape)
    vals = []
    for k in range(n):
        u = sum((byte_at(base + 4 * k + b) & 0xFF) << (8 * b) for b in range(4))
        vals.append(u - (1 << 32) if u >= (1 << 31) else u)
    return torch.tensor(vals, dtype=torch.int64).reshape(shape)


def ternary_tensor(byte_at, base: int, rows: int, cols: int,
                   wcol_bytes: int) -> torch.Tensor:
    """A ternary weight tile ``W[i][j]``, stored column-major and 2-bit packed.

    Codes are ``00 = 0``, ``01 = +1``, ``11 = -1`` (scratchpad.md §2): bit 0 is
    the nonzero flag, bit 1 the sign.
    """
    w = [[0] * cols for _ in range(rows)]
    for j in range(cols):
        colint = sum((byte_at(base + j * wcol_bytes + b) & 0xFF) << (8 * b)
                     for b in range(wcol_bytes))
        for i in range(rows):
            code = (colint >> (2 * i)) & 0b11
            w[i][j] = 0 if not (code & 0b01) else (-1 if (code >> 1) & 1 else 1)
    return torch.tensor(w, dtype=torch.int64)


def requant_params(byte_at, addr: int) -> tuple[int, int]:
    """Unpack the requant ``{m0, n}`` word: ``m0`` in the low ``M0_W`` bits."""
    word = sum((byte_at(addr + b) & 0xFF) << (8 * b) for b in range(4))
    return word & ((1 << gv.M0_W) - 1), (word >> gv.M0_W) & ((1 << gv.N_W) - 1)


# ---- numerics ---------------------------------------------------------------

def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact ``[M,K] @ [K,N]`` over integers.

    ``torch.matmul``/``einsum`` route through BLAS on CPU and have no integer
    kernel, so contract by broadcasting instead. These matrices are tiny.
    """
    return (a[:, :, None] * b[None, :, :]).sum(dim=1)


def requant8(acc: torch.Tensor, m0: int, n: int) -> torch.Tensor:
    """``clip_int8((acc * m0 + (1 << (n-1))) >> n)`` — requant.sv / vpu.md.

    The shift is an *arithmetic* right shift, i.e. it floors; ``rounding_mode=
    "floor"`` is that, whereas torch's ``//`` on tensors and C's ``/`` truncate
    toward zero and would differ for negative accumulators.
    """
    rnd = 0 if n == 0 else (1 << (n - 1))
    shifted = torch.div(acc * m0 + rnd, 1 << n, rounding_mode="floor")
    return shifted.clamp(-128, 127)


# ---- results ----------------------------------------------------------------

@dataclass
class Check:
    """One output tensor: what PyTorch says it should be, what the device said."""

    name: str
    expected: torch.Tensor
    actual: torch.Tensor

    def __post_init__(self):
        if self.expected.shape != self.actual.shape:
            raise ValueError(f"{self.name}: reference {tuple(self.expected.shape)} "
                             f"vs device {tuple(self.actual.shape)}")

    @property
    def shape(self) -> tuple:
        return tuple(self.expected.shape)

    @property
    def total(self) -> int:
        return self.expected.numel()

    @property
    def wrong(self) -> torch.Tensor:
        return self.actual != self.expected

    @property
    def mismatches(self) -> int:
        return int(self.wrong.sum())

    @property
    def max_abs_diff(self) -> int:
        return int((self.actual - self.expected).abs().max()) if self.total else 0

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


@dataclass
class Report:
    kernel: str
    checks: list

    @property
    def mismatches(self) -> int:
        return sum(c.mismatches for c in self.checks)

    @property
    def total(self) -> int:
        return sum(c.total for c in self.checks)

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


# =============================================================================
# References, one per example.
# =============================================================================

def relu_layer(c: dict, get_in, get_out) -> list:
    """``Y = requant(A @ W)``, ``Z = relu(Y)`` — examples/relu_layer.tpu.

    Every tensor uses the same address in DRAM and scratchpad here, so the
    ``.equ`` addresses are directly the DRAM addresses. ``ROWS`` is not one of
    the program's constants; it falls out of the activation tile being
    ``T x ROWS`` int8.
    """
    T, COLS = c["T"], c["COLS"]
    ROWS = c["ACT_BYTES"] // T
    wcol_bytes = c["WGT_BYTES"] // COLS

    A = i8_tensor(get_in, c["ACT"], (T, ROWS))
    W = ternary_tensor(get_in, c["WGT"], ROWS, COLS, wcol_bytes)
    m0, n = requant_params(get_in, c["RQW"])

    # MXU: int8 x ternary, int32 accumulate, requantized int8 on store.
    Y = requant8(int_matmul(A, W), m0, n)
    # VPU relu: int8 in, int32 out, over the flattened NELEM layer output.
    Z = Y.reshape(-1).clamp(min=0)

    return [
        Check("Y = requant(A @ W)  int8", Y,
              i8_tensor(get_out, c["OUT"], (T, COLS))),
        Check("Z = relu(Y)         int32", Z,
              i32_tensor(get_out, c["RLU"], (c["NELEM"],))),
    ]


def vector_add(c: dict, get_in, get_out) -> list:
    """``C = A + B`` — examples/vector_add.tpu.

    ``vecadd`` is a VPU elementwise op, so it reads **int8** operands at stride 1
    and writes **int32** results at stride 4 (vpu.md §Datatypes) — the widening
    is the point, and the sums here overflow int8 on purpose.
    """
    N = c["N"]
    a = i8_tensor(get_in, c["A"], (N,))
    b = i8_tensor(get_in, c["B"], (N,))
    return [Check("C = A + B  int32", a + b, i32_tensor(get_out, c["C"], (N,)))]


def tiled_matmul(c: dict, get_in, get_out) -> list:
    """``C = requant(A @ W)`` streamed tile-by-tile — examples/tiled_matmul.tpu.

    The program never holds the whole problem, but the reference can: reassemble
    A and W from their tile-major DRAM layout into full ``[M,K]`` / ``[K,N]``
    matrices, do one dense matmul, and requantize once at the end. That is
    exactly the hardware's arithmetic because the requant only happens on the
    *last* K tile (``matmul.acc.rq``), after the full contraction has
    accumulated in int32.
    """
    T, ROWS, COLS = c["T"], c["ROWS"], c["COLS"]
    MT, NT, KT = c["MTILES"], c["NTILES"], c["KTILES"]
    ATILE, WTILE, CTILE, WCOL = c["ATILE"], c["WTILE"], c["CTILE"], c["WCOL"]
    M, K, N = MT * T, KT * ROWS, NT * COLS

    A = torch.zeros((M, K), dtype=torch.int64)
    for mt in range(MT):
        for kt in range(KT):                      # A tiles are [mt][kt]
            A[mt * T:(mt + 1) * T, kt * ROWS:(kt + 1) * ROWS] = \
                i8_tensor(get_in, c["A"] + (mt * KT + kt) * ATILE, (T, ROWS))

    W = torch.zeros((K, N), dtype=torch.int64)
    for nt in range(NT):
        for kt in range(KT):                      # W tiles are [nt][kt]
            W[kt * ROWS:(kt + 1) * ROWS, nt * COLS:(nt + 1) * COLS] = \
                ternary_tensor(get_in, c["W"] + (nt * KT + kt) * WTILE,
                               ROWS, COLS, WCOL)

    m0, n = requant_params(get_in, c["RQW"])
    expected = requant8(int_matmul(A, W), m0, n)

    actual = torch.zeros((M, N), dtype=torch.int64)
    for mt in range(MT):
        for nt in range(NT):                      # C tiles are [mt][nt]
            actual[mt * T:(mt + 1) * T, nt * COLS:(nt + 1) * COLS] = \
                i8_tensor(get_out, c["C"] + (mt * NT + nt) * CTILE, (T, COLS))

    return [Check(f"C = requant(A @ W)  [{M}x{K}] @ [{K}x{N}]  int8",
                  expected, actual)]


def vpu_matmul(c: dict, get_in, get_out) -> list:
    """``S = Q @ K^T`` on the VPU — examples/vpu_matmul.tpu.

    The program never forms K^T: it walks K's rows and contracts over the head
    dim with one ``vecdot`` per (query, key) pair. Here that is just a transpose,
    which is the point of checking it — if the kernel had the operand order or
    the row stride wrong, ``S`` would come back transposed or skewed and this
    would catch it, where a symmetric reference would not.

    Both stages are checked. Attention needs the int8 scores, but the raw int32
    block is spilled too, so a failure says whether the dot products or the
    requant is at fault instead of only that the end result is wrong.
    """
    T, D = c["T"], c["D_HEAD"]
    Q = i8_tensor(get_in, c["Q"], (T, D))
    K = i8_tensor(get_in, c["K"], (T, D))
    m0, n = requant_params(get_in, c["RQW"])

    raw = int_matmul(Q, K.t())                  # [T, T] int32
    scores = requant8(raw, m0, n)               # [T, T] int8

    return [
        Check("S32 = Q @ K^T        int32", raw,
              i32_tensor(get_out, c["S32"], (T, T))),
        Check("S8  = requant(S32)   int8", scores,
              i8_tensor(get_out, c["S8"], (T, T))),
    ]


def transpose_dma(c: dict, get_in, get_out) -> list:
    """``V^T`` and ``Q^T`` out of a fused QKV block — examples/transpose_dma.tpu.

    The device does this with two DMA dispatches (`wrmem.t` and `rdmem.t`); here
    it is ``.t()`` on a slice of the same block, so the check is against what a
    transpose *means* rather than against another index calculation.

    Both slices are checked, and they are read from opposite ends of the row: Q
    at column 0 exercises the fill direction with a zero source offset, V at
    column 2D exercises the spill direction with the row stride actually doing
    work. A transfer that ignored `tsrow` would return the first D columns of the
    flattened block for both and fail on V.
    """
    T, D, D3 = c["T"], c["D"], c["D3"]
    block = i8_tensor(get_in, c["QKV"], (T, D3))

    return [
        Check(f"V^T = QKV[:, 2D:3D].T   [{D}x{T}] int8",
              block[:, 2 * D:3 * D].t(), i8_tensor(get_out, c["VT"], (D, T))),
        Check(f"Q^T = QKV[:, 0:D].T     [{D}x{T}] int8",
              block[:, 0:D].t(), i8_tensor(get_out, c["QT"], (D, T))),
    ]


# ---- registry ---------------------------------------------------------------
# Programs are identified by their own .equ constants, the same way
# gen_vectors.program_kind picks an input-image builder — not by filename, so a
# renamed or derived program still resolves.

KERNELS = [
    ("transpose_dma", lambda c: "VT" in c,       transpose_dma),
    ("relu_layer",   lambda c: "RLU" in c,       relu_layer),
    ("vector_add",   lambda c: "VEC_BYTES" in c, vector_add),
    ("tiled_matmul", lambda c: "MTILES" in c,    tiled_matmul),
    ("vpu_matmul",   lambda c: "D_HEAD" in c,    vpu_matmul),
]

# Examples that deliberately have no reference, and why — a property of the .tpu
# source, not of this module.
UNSUPPORTED = [
    ("softmax_row", lambda c: "PROB" in c,
     "softmax_row.tpu does not currently compute softmax: `exp`, `redsum` and "
     "`sdiv` read int8 operands at stride 1, but the `sadd` and `exp` feeding "
     "them write int32 at stride 4. Two REQUANT ops (int32 -> int8 at the exp "
     "table's 1/16 in_scale, and again after exp) are needed first — see "
     "docs/vpu.md. The exp LUT itself is no longer the blocker: luts.py "
     "generates it, the ISS defaults to it and cmod_a7_top.sv loads it"),
]


def select(consts: dict):
    """Pick the reference for a program. Raises :class:`NoReference` if none."""
    for name, matches, fn in KERNELS:
        if matches(consts):
            return name, fn
    for name, matches, why in UNSUPPORTED:
        if matches(consts):
            raise NoReference(f"{name}: {why}")
    raise NoReference("no PyTorch reference is registered for this program "
                      f"(constants: {', '.join(sorted(consts)[:8])}...)")


def verify(consts: dict, in_img: dict, out_img: dict) -> Report:
    """Check ``out_img`` against the PyTorch reference for ``consts``' program.

    ``in_img`` is the DRAM image that was loaded (host ``W`` commands, or the
    testbench seed); ``out_img`` is what came back — device bytes from
    ``run_program.py``, or the ISS's DRAM when checking without a board.
    """
    name, fn = select(consts)
    return Report(name, fn(consts, input_reader(in_img), output_reader(out_img)))


# ---- reporting --------------------------------------------------------------

def print_report(rep: Report, limit: int = 8) -> int:
    """Print one line per output tensor. Returns the total mismatch count."""
    for chk in rep.checks:
        print(f"  [{'OK  ' if chk.ok else 'FAIL'}] {chk.name}  {chk.shape}  "
              f"{chk.mismatches}/{chk.total} differ, max|diff| = {chk.max_abs_diff}")
        if chk.ok:
            continue
        idx = chk.wrong.nonzero().tolist()
        for pos in idx[:limit]:
            at = "".join(f"[{i}]" for i in pos)
            print(f"         {at}: device {int(chk.actual[tuple(pos)]):6d}  "
                  f"torch {int(chk.expected[tuple(pos)]):6d}")
        if len(idx) > limit:
            print(f"         ... and {len(idx) - limit} more")
    return rep.mismatches


# ---- standalone: check every example against the ISS ------------------------

def run_in_iss(consts: dict, words: list) -> tuple[dict, dict]:
    """Seed DRAM and run the program in the ISS. Returns (in_img, out_img)."""
    tpu = TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W,
              m0_w=gv.M0_W, n_w=gv.N_W)
    in_img, _ref, _dims = gv.build_image(tpu, consts)
    tpu.run(words)
    return in_img, {a: tpu.dram[a] for a in sorted(tpu.dram_written)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--program", action="append", metavar="FILE",
                    help="tpulang source to check (repeatable; default: every "
                         "example that has a reference)")
    args = ap.parse_args(argv)

    progs = args.program or [os.path.join(HERE, "examples", f"{n}.tpu")
                             for n, _, _ in KERNELS]

    failed = 0
    for path in progs:
        print(f"\n{os.path.basename(path)}")
        words, consts = gv.assemble_program(path)
        try:
            select(consts)      # resolve the reference before paying for an ISS run
        except NoReference as exc:
            print(f"  [SKIP] {exc}")
            continue
        in_img, out_img = run_in_iss(consts, words)
        try:
            rep = verify(consts, in_img, out_img)
        except MissingByte as exc:
            print(f"  [FAIL] {exc}")
            failed += 1
            continue
        if print_report(rep):
            failed += 1

    print()
    if failed:
        print(f"FAILED: {failed} program(s) disagree with their PyTorch reference")
        return 1
    print("PASSED: every checked program matches its PyTorch reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
