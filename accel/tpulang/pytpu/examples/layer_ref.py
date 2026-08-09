#!/usr/bin/env python3
"""layer_ref.py — the exact integer reference for the generated layer.

This is the verification leg that matters. ``iss.py`` tells you the hardware and
the simulator agree; it cannot tell you the *program* is right, because the
program and the simulator can be wrong the same way. This module recomputes the
whole layer from the **bytes actually loaded into DRAM**, in plain Python
integers, sharing no code with ``iss.py``, and compares against the **bytes
actually spilled**. Every comparison is exact — the pipeline is integer end to
end, so there is no tolerance to allow.

What it does share with the composer is the address map and the ``{m0,n}`` words
(passed in as ``sites``), because those *are* the composer's declaration of what
the program means. What it does not share is a single line of arithmetic: the
requant, the reciprocal divide, the LUT lookups and the two scalar loops are all
re-derived here from the RTL's documented semantics.

The two scalar loops are worth reading against ``lib/layernorm_rows.tpu``:
``_trunc_div`` and ``_isqrt`` are what that template's binary-long-division and
bit-by-bit-sqrt loops compute, written the obvious way. If those two disagree
with the hardware's table walk, this reference catches it.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import luts                                                     # noqa: E402
import quant                                                    # noqa: E402

requant8 = quant.requant8
sdiv = quant.sdiv


def s8(b: int) -> int:
    b &= 0xFF
    return b - 0x100 if b >= 0x80 else b


# ---- the two scalar loops the ISA has no instructions for -------------------

def _trunc_div(v: int, d: int) -> int:
    """``trunc(v / d)`` — what the template's binary long division computes.

    The template works on ``|v|`` and re-signs, so this truncates toward zero,
    which is *not* what Python's ``//`` does for negative numerators.
    """
    q = abs(v) // d
    return -q if v < 0 else q


def _isqrt(v: int) -> int:
    """``floor(sqrt(v))``, clamped to at least 1 (a constant row divides by 1)."""
    if v <= 0:
        return 1
    r = 0
    bit = 1 << 15
    while bit:
        t = r + bit
        if t * t <= v:
            r = t
        bit >>= 1
    return max(r, 1)


# ---- op-level references, one per primitive ---------------------------------

def mm_ternary(a: list, w: list, m: int, k: int, n: int, rq: tuple) -> list:
    """``matmul_ternary``: int32 accumulate over all of K, one requant at the end."""
    m0, nsh = rq
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = sum(a[i][t] * w[t][j] for t in range(k))
            out[i][j] = requant8(acc, m0, nsh)
    return out


def softmax_row(row: list) -> list:
    """``softmax_rows``, step for step (see lib/softmax_rows.tpu)."""
    mx = max(row)
    sh8 = [requant8(x - mx, 1, 0) for x in row]              # clips at -128
    e32 = [s8(luts.EXP_LUT[v & 0xFF]) for v in sh8]          # the exp ROM
    e8 = [requant8(v, 1, 0) for v in e32]
    den = sum(e8)
    p32 = [sdiv(v, den) for v in e8]
    return [requant8(v, 1, 8) for v in p32]                  # -> scale 1/128


def layernorm_row(row: list, gamma: list, beta: list,
                  rqy: tuple, rqg: tuple) -> list:
    """``layernorm_rows``, step for step (see lib/layernorm_rows.tpu)."""
    length = len(row)
    mu = _trunc_div(sum(row), length)
    c8 = [requant8(x - mu, 1, 0) for x in row]               # clips at +-127
    r = _isqrt(sum(c * c for c in c8))
    y8 = [requant8(sdiv(c, r), *rqy) for c in c8]
    g8 = [requant8(y8[i] * gamma[i], *rqg) for i in range(length)]
    return [requant8(g8[i] + beta[i], 1, 0) for i in range(length)]


# ---- the whole layer --------------------------------------------------------

def run(cfg, t, sites: dict, get_in) -> dict:
    """Recompute every intermediate of the layer from the input DRAM image.

    ``t`` is the composer's tensor namespace (an object with one attribute per
    tensor) and ``sites`` maps a rescale site's name to its ``(m0, n)``.
    Returns ``{tensor_name: flat list of ints}`` for every tensor the program
    spills, ready to be compared byte for byte against the ISS's output.
    """
    T, d, f, hd, H = cfg.tokens, cfg.d, cfg.f, cfg.head_dim, cfg.heads

    def i8(base, rows, cols, stride=None):
        stride = cols if stride is None else stride
        return [[s8(get_in(base + r * stride + c)) for c in range(cols)]
                for r in range(rows)]

    X = i8(t.X.addr, T, d)
    Wqkv = quant.unpack_ternary(get_in, t.Wqkv.addr, d, 3 * d)
    Wo = quant.unpack_ternary(get_in, t.Wo.addr, d, d)
    W1 = quant.unpack_ternary(get_in, t.W1.addr, d, f)
    W2 = quant.unpack_ternary(get_in, t.W2.addr, f, d)
    g1 = i8(t.g1.addr, 1, d)[0]
    b1 = i8(t.b1.addr, 1, d)[0]
    g2 = i8(t.g2.addr, 1, d)[0]
    b2 = i8(t.b2.addr, 1, d)[0]
    mask = i8(t.mask.addr, H * T, T)

    out: dict = {}

    # ---- fused QKV projection (MXU) ----
    QKV = mm_ternary(X, Wqkv, T, d, 3 * d, sites["qkv"])
    out["QKV"] = _flat(QKV)

    # ---- scores: one vecdot per (head, query, key), contracting over head_dim.
    # Q and K are strided sub-blocks of QKV; the head axis is never materialised.
    S32 = [[[sum(QKV[q][h * hd + k] * QKV[s][d + h * hd + k] for k in range(hd))
             for s in range(T)] for q in range(T)] for h in range(H)]
    out["S32"] = [v for head in S32 for row in head for v in row]

    S8 = [requant8(v, *sites["scores"]) for v in out["S32"]]
    out["S8"] = S8

    flat_mask = [v for row in mask for v in row]
    SM8 = [requant8(S8[i] + flat_mask[i], 1, 0) for i in range(len(S8))]
    out["SM8"] = SM8

    P8 = []
    for r in range(H * T):
        P8 += softmax_row(SM8[r * T:(r + 1) * T])
    out["P8"] = P8

    # ---- V^T, by DMA byte moves ----
    Vt = [[QKV[tok][2 * d + j] for tok in range(T)] for j in range(d)]
    out["Vt"] = _flat(Vt)

    # ---- attention output: contract over keys, per head ----
    AO32 = [[0] * d for _ in range(T)]
    for h in range(H):
        for q in range(T):
            for j in range(hd):
                AO32[q][h * hd + j] = sum(
                    P8[(h * T + q) * T + s] * Vt[h * hd + j][s] for s in range(T))
    out["AO32"] = _flat(AO32)

    AO8 = [[requant8(v, *sites["ao"]) for v in row] for row in AO32]
    out["AO8"] = _flat(AO8)

    O = mm_ternary(AO8, Wo, T, d, d, sites["proj"])
    out["O"] = _flat(O)

    X1 = [[requant8(X[i][j] + O[i][j], *sites["res1"]) for j in range(d)]
          for i in range(T)]
    out["X1"] = _flat(X1)

    N1 = [layernorm_row(X1[i], g1, b1, sites["ln1_y"], sites["ln1_g"])
          for i in range(T)]
    out["N1"] = _flat(N1)

    Hh = mm_ternary(N1, W1, T, d, f, sites["ff1"])
    out["H"] = _flat(Hh)

    HG = [[requant8(s8(luts.GELU_LUT[v & 0xFF]), *sites["gelu"]) for v in row]
          for row in Hh]
    out["HG"] = _flat(HG)

    F = mm_ternary(HG, W2, T, f, d, sites["ff2"])
    out["F"] = _flat(F)

    X2 = [[requant8(N1[i][j] + F[i][j], *sites["res2"]) for j in range(d)]
          for i in range(T)]
    out["X2"] = _flat(X2)

    Y = [layernorm_row(X2[i], g2, b2, sites["ln2_y"], sites["ln2_g"])
         for i in range(T)]
    out["Y"] = _flat(Y)

    return out


def _flat(rows: list) -> list:
    return [v for row in rows for v in row]
