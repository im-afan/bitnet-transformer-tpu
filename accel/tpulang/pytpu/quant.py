#!/usr/bin/env python3
"""quant.py — the compile-time half of the numerics.

Every tensor in a generated program carries a **scale** (``real = int * scale``)
that exists only in Python. The hardware never sees it; what the hardware sees is
a ``{m0, n}`` word at each rescale site, and choosing those words so producer and
consumer agree is this module's whole job.

    dst[i] = clip_int8( (src[i] * m0 + (1 << (n-1))) >> n )

with ``m0`` in the low 12 bits of an int32 and ``n`` above (``M0_W``/``N_W`` in
``tpu_top.sv``). The shift is *arithmetic* — it floors, which differs from C's
truncation for negative accumulators, and the integer reference has to floor too.

Also here: quantizing float tensors to int8, and packing float weights to the
ternary layout the MXU reads (column-major, 2 bits per trit, tile-major
``[nt][kt]`` — the same layout ``gen_vectors.build_matmul_image`` produces, kept
in one place so the packer and ``lib/matmul_ternary.tpu``'s loop nest cannot
drift apart).
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Requant word field widths — tpu_top.sv M0_W / N_W, mirrored by gen_vectors.
M0_W, N_W = 12, 4
M0_MAX = (1 << M0_W) - 1        # 4095
N_MAX = (1 << N_W) - 1          # 15

ROWS = COLS = 8                 # MXU array (tpu_top_tb.sv DUT geometry)
WCOL_BYTES = (ROWS * 2) // 8    # packed ternary bytes per weight column (= 2)
WTILE_BYTES = COLS * WCOL_BYTES  # bytes per ternary tile (= 16)


class QuantError(Exception):
    """A scale ratio that no {m0, n} word can express."""


# ---- the requant word --------------------------------------------------------

def requant_word(m0: int, n: int) -> int:
    """Pack ``{m0, n}`` into the int32 the hardware reads."""
    if not (0 <= m0 <= M0_MAX):
        raise QuantError(f"m0 = {m0} does not fit in {M0_W} bits")
    if not (0 <= n <= N_MAX):
        raise QuantError(f"n = {n} does not fit in {N_W} bits")
    return (n << M0_W) | m0


def choose_requant(ratio: float) -> tuple[int, int]:
    """Pick ``{m0, n}`` with ``m0 / 2**n`` as close to ``ratio`` as possible.

    ``ratio`` is ``scale_in / scale_out``: the factor the requant must apply to
    turn the producer's integers into the consumer's. Maximizing ``n`` maximizes
    the precision of the fixed-point approximation, so take the largest ``n``
    whose ``m0`` still fits in ``M0_W`` bits.
    """
    if not (ratio > 0):
        raise QuantError(f"ratio must be positive, got {ratio}")
    for n in range(N_MAX, -1, -1):
        m0 = round(ratio * (1 << n))
        if 1 <= m0 <= M0_MAX:
            return m0, n
    if round(ratio) > M0_MAX:
        raise QuantError(
            f"ratio {ratio:g} needs m0 > {M0_MAX}: the producer's scale is more "
            f"than {M0_MAX}x the consumer's, so pick a coarser output scale"
        )
    raise QuantError(
        f"ratio {ratio:g} rounds to m0 = 0 even at n = {N_MAX} (< 2**-{N_MAX+1}): "
        f"the output scale is too coarse to represent this input at all"
    )


def requant_ratio(m0: int, n: int) -> float:
    """The factor ``{m0, n}`` actually applies (``choose_requant`` rounds)."""
    return m0 / (1 << n)


def requant8(acc: int, m0: int, n: int) -> int:
    """``clip_int8((acc*m0 + rnd) >> n)`` — arithmetic (flooring) shift."""
    rnd = 0 if n == 0 else (1 << (n - 1))
    v = (acc * m0 + rnd) >> n            # Python >> floors, like Verilog >>>
    return 127 if v > 127 else (-128 if v < -128 else v)


def layernorm_requant(length: int, scale_out: float) -> tuple[int, int]:
    """The ``{m0, n}`` that follows ``layernorm_rows``' ``sdiv``.

    The kernel normalizes by ``R = floor(sqrt(sum c^2))`` rather than by the
    standard deviation ``sqrt(sum c^2 / L)``, because dividing by ``L`` at run
    time would need a shift the scalar unit does not have. The leftover
    ``sqrt(L)`` is a compile-time constant, and this is where it goes:

        y32 = round(c8 * 2**15 / R)  =  z * 2**15 / sqrt(L)
        y8  = y32 * m0 / 2**n        =  z / scale_out
        =>  m0 / 2**n = sqrt(L) / (2**15 * scale_out)
    """
    return choose_requant(math.sqrt(length) / ((1 << 15) * scale_out))


def sdiv(a8: int, d: int) -> int:
    """``vpu.sv``'s reciprocal divide: ``round(a8 * 2**15 / d)``, bit-exact.

    The reciprocal is ``floor(2**31 / |d|)``, so the quantization of *that* is
    part of the result and any reference has to reproduce it rather than
    computing the exact quotient.
    """
    recip_q, div_q = 31, 15
    if d == 0:
        r = (1 << (recip_q + 1)) - 1          # divide-by-zero saturates
    else:
        r = (1 << recip_q) // abs(d)
    shift = recip_q - div_q
    q = (a8 * r + (1 << (shift - 1))) >> shift
    return -q if d < 0 else q


# ---- float -> int ------------------------------------------------------------

def scale_for(absmax: float, headroom: float = 1.0) -> float:
    """A symmetric int8 scale putting ``absmax`` at 127 (times ``headroom``)."""
    if absmax <= 0:
        return 1.0
    return absmax * headroom / 127.0


def quantize_i8(values, scale: float) -> list:
    """Round-to-nearest, clip to int8. ``values`` is any flat iterable of float."""
    out = []
    for v in values:
        q = math.floor(v / scale + 0.5)
        out.append(127 if q > 127 else (-128 if q < -128 else q))
    return out


def ternarize(w) -> tuple[list, float]:
    """BitNet-style ternary quantization of a 2-D float matrix.

    ``scale = mean(|W|)``; ``W_tern = clip(round(W / scale), -1, 1)``; the real
    matrix the hardware computes with is ``scale * W_tern``. This mirrors
    ``TernaryLinear.forward`` in ``model/transformer.py`` (which uses
    ``absmean + eps`` for the divisor and ``absmean`` for the reconstruction
    scale; the eps only matters for an all-zero matrix).
    """
    flat = [x for row in w for x in row]
    absmean = sum(abs(x) for x in flat) / max(len(flat), 1)
    if absmean == 0:
        return [[0] * len(w[0]) for _ in w], 1.0
    out = []
    for row in w:
        out.append([max(-1, min(1, math.floor(x / absmean + 0.5))) for x in row])
    return out, absmean


def pack_ternary(w_tern, k: int, n: int) -> list:
    """Pack a ``[K,N]`` ternary matrix into the MXU's byte layout.

    Tile-major ``[nt][kt]``, ``WTILE_BYTES`` per tile; within a tile,
    column-major with 2 bits per trit (``00 = 0, 01 = +1, 11 = -1`` —
    bit 0 is the nonzero flag, bit 1 the sign).
    """
    if k % ROWS or n % COLS:
        raise QuantError(f"ternary [{k},{n}] must tile by {ROWS}x{COLS}")
    ktiles, ntiles = k // ROWS, n // COLS
    out = [0] * (ktiles * ntiles * WTILE_BYTES)
    for nt in range(ntiles):
        for kt in range(ktiles):
            base = (nt * ktiles + kt) * WTILE_BYTES
            for j in range(COLS):
                colint = 0
                for i in range(ROWS):
                    t = w_tern[kt * ROWS + i][nt * COLS + j]
                    code = 0 if t == 0 else (1 if t > 0 else 0b11)
                    colint |= code << (2 * i)
                for b in range(WCOL_BYTES):
                    out[base + j * WCOL_BYTES + b] = (colint >> (8 * b)) & 0xFF
    return out


def unpack_ternary(byte_at, base: int, k: int, n: int) -> list:
    """Inverse of :func:`pack_ternary`, reading one byte at a time.

    Used by the reference to recover ``W`` from the *bytes actually loaded*,
    rather than from the Python object they were built from.
    """
    ktiles, ntiles = k // ROWS, n // COLS
    w = [[0] * n for _ in range(k)]
    for nt in range(ntiles):
        for kt in range(ktiles):
            tile = base + (nt * ktiles + kt) * WTILE_BYTES
            for j in range(COLS):
                colint = sum((byte_at(tile + j * WCOL_BYTES + b) & 0xFF) << (8 * b)
                             for b in range(WCOL_BYTES))
                for i in range(ROWS):
                    code = (colint >> (2 * i)) & 0b11
                    w[kt * ROWS + i][nt * COLS + j] = (
                        0 if not (code & 0b01) else (-1 if (code >> 1) & 1 else 1))
    return w


def int32_bytes(word: int) -> list:
    """Little-endian bytes of an int32 — how every scalar word is stored."""
    word &= 0xFFFFFFFF
    return [(word >> (8 * b)) & 0xFF for b in range(4)]
