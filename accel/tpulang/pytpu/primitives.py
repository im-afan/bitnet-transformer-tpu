#!/usr/bin/env python3
"""primitives.py — the Python face of ``lib/*.tpu``.

One wrapper per primitive. Each wrapper does three things and nothing else:

1. **checks the shapes and dtypes** against what the template's loop nest
   actually assumes, and fails with the reason rather than the assertion —
   ``KTILES >= 2`` is not an arbitrary rule, it is that the K loop has a distinct
   init tile and a distinct requant tile;
2. **allocates the primitive's private scratchpad buffers** out of a caller-
   supplied arena, so the composer never hand-places a temporary;
3. renders the template and appends it to the :class:`~program.Program`.

Scratch buffers are allocated fresh per instantiation rather than shared between
them. That wastes a little scratchpad (a few hundred bytes per block, against
64 KB) and removes a whole class of aliasing bug in exchange, which is the right
trade while instruction memory — not data memory — is the binding budget.

The **calling convention** every primitive obeys (PLAN.md §4.4):

* no register is live across a block — all state between primitives is in memory;
* no config register is live across a block — each sets ``tlen``/``vlen``/``len``
  before use, because stale config is this ISA's most common silent bug;
* inputs are read from DRAM and outputs written to DRAM, at one address that is
  valid in both spaces (so a fill is ``rdmem a, a``).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import template                                                 # noqa: E402
from memory import Arena, Tensor                                # noqa: E402
from template import Hex                                        # noqa: E402

MAX_VLEN = 1023         # cfg vlen is 10 bits (iss._vpu masks 0x3FF)
MAX_TLEN = 63           # cfg tlen is 6 bits  (iss._matmul masks 0x3F)
ROWS = COLS = 8         # MXU array


class PrimitiveError(Exception):
    """A primitive was instantiated with shapes its template cannot serve."""


def _check(cond, msg):
    if not cond:
        raise PrimitiveError(msg)


def _dtype(t: Tensor, want: str, role: str):
    _check(t.dtype == want,
           f"{role} tensor '{t.name}' is {t.dtype}, must be {want}")


def _chunk(n: int, name: str) -> int:
    """The largest divisor of ``n`` that fits in ``cfg vlen``."""
    if n <= MAX_VLEN:
        return n
    for c in range(MAX_VLEN, 0, -1):
        if n % c == 0:
            return c
    raise PrimitiveError(f"{name}: cannot chunk {n} elements under vlen<={MAX_VLEN}")


def _emit(prog, name: str, primitive: str, note: str, params: dict):
    text = template.load(primitive).render(name, params)
    return prog.add(name, primitive, text, note)


# =============================================================================
# MXU
# =============================================================================

def matmul_ternary(prog, scratch: Arena, name: str, a: Tensor, w: Tensor,
                   c: Tensor, rqw: Tensor, tt: int = 8):
    """``C[M,N] int8 = requant(A[M,K] int8 @ W[K,N] ternary)``."""
    _dtype(a, "int8", "activation")
    _dtype(w, "ternary", "weight")
    _dtype(c, "int8", "output")
    _dtype(rqw, "int32", "requant word")

    m, k = a.shape
    _check(w.shape == (k, c.shape[1]),
           f"{name}: A is [{m},{k}] and C is {list(c.shape)}, so W must be "
           f"[{k},{c.shape[1]}] — got {list(w.shape)}")
    n = c.shape[1]
    _check(c.shape[0] == m, f"{name}: C has {c.shape[0]} rows, A has {m}")
    _check(k % ROWS == 0, f"{name}: K = {k} must be a multiple of {ROWS}")
    _check(n % COLS == 0, f"{name}: N = {n} must be a multiple of {COLS}")
    _check(m % tt == 0, f"{name}: M = {m} must be a multiple of TT = {tt}")
    _check(tt <= MAX_TLEN, f"{name}: TT = {tt} exceeds cfg tlen's {MAX_TLEN}")
    _check(k // ROWS >= 2,
           f"{name}: K = {k} gives {k // ROWS} K-tile(s); the K loop needs at "
           f"least 2 (a distinct accumulator-init tile and a distinct requant "
           f"tile). Pad K to {2 * ROWS} or use a VPU matmul.")

    return _emit(prog, name, "matmul_ternary",
                 f"[{m},{k}] @ [{k},{n}] ternary -> int8", {
                     "A": a.a, "W": w.a, "C": c.a, "RQW": rqw.a,
                     "M": m, "K": k, "N": n, "TT": tt,
                     "ABUF": scratch.buffer(f"{name}.abuf", tt * ROWS),
                     "WBUF": scratch.buffer(f"{name}.wbuf", COLS * (ROWS * 2) // 8),
                     "CBUF": scratch.buffer(f"{name}.cbuf", tt * COLS * 4),
                 })


# =============================================================================
# VPU: matmul, transpose
# =============================================================================

def matmul_i8(prog, name: str, *, out: Tensor, heads: int, m: int, n: int,
              klen: int, a_addr: int, a_head: int, a_row: int,
              b_addr: int, b_head: int, b_row: int,
              o_head: int, o_row: int,
              stage: list, note: str = ""):
    """``O[h,m,n] int32 = sum_k A[h,m,k] * B[h,n,k]`` — batched, fully strided.

    ``stage`` is a list of ``(addr, nbytes)`` regions to fill from DRAM before
    the loops (one or two); the addresses are byte strides, not element strides,
    because the operands are generally sub-blocks of larger tensors.
    """
    _dtype(out, "int32", "output")
    _check(klen <= MAX_VLEN, f"{name}: contraction length {klen} exceeds vlen")
    _check(1 <= len(stage) <= 2, f"{name}: expected 1 or 2 staging regions")
    regions = list(stage) + [(stage[0][0], 0)] * (2 - len(stage))
    _check(out.nbytes == heads * m * n * 4,
           f"{name}: output '{out.name}' is {out.nbytes} B but [{heads},{m},{n}] "
           f"int32 needs {heads * m * n * 4} B")

    return _emit(prog, name, "matmul_i8",
                 note or f"[{heads},{m},{klen}] x [{heads},{n},{klen}]^T -> int32", {
                     "IN0": Hex(regions[0][0]), "IN0_BYTES": regions[0][1],
                     "IN1": Hex(regions[1][0]), "IN1_BYTES": regions[1][1],
                     "A": Hex(a_addr), "AH": a_head, "AM": a_row,
                     "B": Hex(b_addr), "BH": b_head, "BN": b_row,
                     "O": out.a, "OH": o_head, "OM": o_row,
                     "O_BYTES": out.nbytes,
                     "H": heads, "M": m, "N": n, "KLEN": klen,
                 })


def transpose_i8(prog, scratch: Arena, name: str, *, src_addr: int, dst: Tensor,
                 m: int, n: int, src_stride: int, dst_stride: int | None = None):
    """``B[n,m] = A[m,n]`` in DRAM, one byte per DMA pair."""
    _dtype(dst, "int8", "output")
    dst_stride = m if dst_stride is None else dst_stride
    _check(src_stride >= n, f"{name}: source row stride {src_stride} < N = {n}")
    _check(dst_stride >= m, f"{name}: dest row stride {dst_stride} < M = {m}")

    return _emit(prog, name, "transpose_i8", f"[{m},{n}] -> [{n},{m}] int8", {
        "SRC": Hex(src_addr), "DST": dst.a,
        "M": m, "N": n,
        "SRC_STRIDE": src_stride, "DST_STRIDE": dst_stride,
        "BUF": scratch.buffer(f"{name}.cell", 1, align=1),
    })


# =============================================================================
# VPU: elementwise blocks
# =============================================================================

def requant_block(prog, name: str, src: Tensor, dst: Tensor, rqw: Tensor):
    """``dst[N] int8 = requant(src[N] int32)``."""
    _dtype(src, "int32", "source")
    _dtype(dst, "int8", "output")
    _dtype(rqw, "int32", "requant word")
    n = dst.nelem
    _check(src.nelem == n,
           f"{name}: '{src.name}' has {src.nelem} elements, '{dst.name}' has {n}")
    chunk = _chunk(n, name)
    return _emit(prog, name, "requant_block", f"{n} elements int32 -> int8", {
        "SRC": src.a, "DST": dst.a, "RQW": rqw.a, "N": n, "CHUNK": chunk,
    })


def add_i8(prog, scratch: Arena, name: str, a: Tensor, b: Tensor, c: Tensor,
           rqw: Tensor, note: str = ""):
    """``C[N] int8 = requant(A[N] + B[N])`` — residual adds and the causal mask."""
    for t, role in ((a, "lhs"), (b, "rhs"), (c, "output")):
        _dtype(t, "int8", role)
    _dtype(rqw, "int32", "requant word")
    n = c.nelem
    _check(a.nelem == n and b.nelem == n,
           f"{name}: operands have {a.nelem}/{b.nelem} elements, output has {n}")
    chunk = _chunk(n, name)
    return _emit(prog, name, "add_i8", note or f"{n} elements int8 + int8", {
        "A": a.a, "B": b.a, "C": c.a, "RQW": rqw.a, "N": n, "CHUNK": chunk,
        "TMP": scratch.buffer(f"{name}.tmp", chunk * 4),
    })


def gelu_block(prog, scratch: Arena, name: str, src: Tensor, dst: Tensor,
               rqw: Tensor):
    """``dst[N] int8 = requant(gelu(src[N]))``. ``src`` must be at scale 1/16."""
    _dtype(src, "int8", "source")
    _dtype(dst, "int8", "output")
    _dtype(rqw, "int32", "requant word")
    n = dst.nelem
    _check(src.nelem == n,
           f"{name}: '{src.name}' has {src.nelem} elements, '{dst.name}' has {n}")
    _check(abs(src.scale - 1 / 16) < 1e-12,
           f"{name}: gelu's LUT assumes an input scale of 1/16 and never checks "
           f"it, but '{src.name}' is at {src.scale:g}. Have the producing "
           f"requant land it at 1/16 instead of rescaling here.")
    chunk = _chunk(n, name)
    return _emit(prog, name, "gelu_block", f"{n} elements, LUT in_scale 1/16", {
        "SRC": src.a, "DST": dst.a, "RQW": rqw.a, "N": n, "CHUNK": chunk,
        "TMP": scratch.buffer(f"{name}.tmp", chunk * 4),
    })


# =============================================================================
# VPU: composites
# =============================================================================

def softmax_rows(prog, scratch: Arena, name: str, src: Tensor, dst: Tensor,
                 rows: int, length: int):
    """Row-wise softmax: int8 at 1/16 (masked) -> int8 probabilities at 1/128."""
    _dtype(src, "int8", "source")
    _dtype(dst, "int8", "output")
    _check(src.nelem == rows * length and dst.nelem == rows * length,
           f"{name}: [{rows},{length}] needs {rows * length} elements, got "
           f"{src.nelem} in / {dst.nelem} out")
    _check(length <= MAX_VLEN, f"{name}: row length {length} exceeds vlen")
    _check(abs(src.scale - 1 / 16) < 1e-12,
           f"{name}: exp's LUT assumes an input scale of 1/16 and never checks "
           f"it, but '{src.name}' is at {src.scale:g}")

    b = lambda tag, nb, al=4: scratch.buffer(f"{name}.{tag}", nb, al)
    return _emit(prog, name, "softmax_rows", f"{rows} rows of {length}", {
        "SRC": src.a, "DST": dst.a, "R": rows, "L": length,
        "MAXA": b("maxa", 4), "NEGA": b("nega", 4), "DEN": b("den", 4),
        "SH32": b("sh32", length * 4), "SH8": b("sh8", length, 1),
        "E32": b("e32", length * 4), "E8": b("e8", length, 1),
        "P32": b("p32", length * 4),
        "RQ1": b("rq1", 4), "RQ2": b("rq2", 4),
    })


def layernorm_rows(prog, scratch: Arena, name: str, src: Tensor, dst: Tensor,
                   gamma: Tensor, beta: Tensor, rqy: Tensor, rqg: Tensor):
    """Row-wise LayerNorm with int8 gamma/beta. See lib/layernorm_rows.tpu."""
    for t, role in ((src, "source"), (dst, "output"),
                    (gamma, "gamma"), (beta, "beta")):
        _dtype(t, "int8", role)
    _dtype(rqy, "int32", "normalize requant word")
    _dtype(rqg, "int32", "gamma requant word")

    rows, length = src.shape
    _check(dst.shape == (rows, length),
           f"{name}: '{dst.name}' is {list(dst.shape)}, source is [{rows},{length}]")
    _check(gamma.nelem == length and beta.nelem == length,
           f"{name}: gamma/beta must have {length} elements, got "
           f"{gamma.nelem}/{beta.nelem}")
    _check(length <= MAX_VLEN, f"{name}: row length {length} exceeds vlen")

    b = lambda tag, nb, al=4: scratch.buffer(f"{name}.{tag}", nb, al)
    return _emit(prog, name, "layernorm_rows", f"{rows} rows of {length}", {
        "SRC": src.a, "DST": dst.a, "GAMMA": gamma.a, "BETA": beta.a,
        "R": rows, "L": length, "RQY": rqy.a, "RQG": rqg.a,
        "RQ1": b("rq1", 4), "TBL": b("tbl", 64),
        "C8": b("c8", length, 1), "T8": b("t8", length, 1),
        "T32": b("t32", length * 4),
        "W0": b("w0", 4), "NEGM": b("negm", 4), "RVAL": b("rval", 4),
    })
