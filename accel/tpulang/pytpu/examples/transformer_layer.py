#!/usr/bin/env python3
"""transformer_layer.py — one quantized transformer layer, generated end to end.

Builds the post-norm decoder layer of ``model/transformer.py``
(``Transformer.forward``) out of pytpu primitives:

    Y = norm2( X1 + W2 . gelu(W1 . X1) ),   X1 = norm1( X + Wo . attn(X) )

as a single tpulang program:

    QKV  = X . Wqkv                 matmul_ternary   MXU, fused Q|K|V
    S32  = Q . K^T   per head       matmul_i8        VPU (both operands int8)
    S8   = requant(S32)             requant_block    -> exp's 1/16 input scale
    SM8  = S8 + causal_mask         add_i8           mask is data, not an op
    P    = softmax(SM8)             softmax_rows     -> scale 1/128
    Vt   = V^T                      transpose_i8     DMA byte moves
    AO32 = P . Vt^T  per head       matmul_i8
    AO8  = requant(AO32)            requant_block
    O    = AO8 . Wo                 matmul_ternary
    X1'  = X + O                    add_i8           residual
    X1   = layernorm(X1', g1, b1)   layernorm_rows   norm1
    H    = X1 . W1                  matmul_ternary   -> 1/16 for the gelu LUT
    HG   = gelu(H)                  gelu_block
    F    = HG . W2                  matmul_ternary
    X2   = X1 + F                   add_i8           residual
    Y    = layernorm(X2, g2, b2)    layernorm_rows   norm2

Scales are **calibrated**, not guessed: the float layer runs first, and each
tensor's int8 scale is its observed absmax over 127. The ``{m0,n}`` word at every
rescale site then follows from the producer's and consumer's scales
(``quant.choose_requant``). Two scales are *not* free — the ``gelu`` and ``exp``
LUTs assume an input scale of 1/16 and never check it — so the requant that
feeds each of them is made to land there, which costs nothing because that
requant had to happen anyway.

Deliberate simplifications, all stated rather than hidden:

* **Demo geometry** (d=32, f=64, 2 heads, 8 tokens), not the shipped
  ``adder_ternary_vanilla`` (d=128, f=512, 4 heads). The real layer's ternary
  weights alone are 48 KB packed, against the ISS's 64 KB DRAM space. Nothing
  here is specialized to the small config.
* **No linear bias.** ``TernaryLinear`` has one; the MXU has no bias path, so
  adding it would cost four more ``add_i8`` blocks out of a 1024-word budget.
* **No dropout** (inference).

    python accel/tpulang/pytpu/examples/transformer_layer.py
    python accel/tpulang/pytpu/examples/transformer_layer.py --emit-vectors
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import layer_ref                                                # noqa: E402
import primitives as P                                          # noqa: E402
import quant                                                    # noqa: E402
from memory import Arena                                        # noqa: E402
from program import Program                                     # noqa: E402
from template import Hex                                        # noqa: E402

LUT_SCALE = 1 / 16          # gelu / exp input scale — a hardware constant
PROB_SCALE = 1 / 128        # softmax_rows' pinned output scale
NORM_SCALE = 1 / 32         # scale of the normalized (pre-gamma) LayerNorm value
EPS = 1e-5                  # nn.LayerNorm default


@dataclass
class Config:
    tokens: int = 8
    d: int = 32
    heads: int = 2
    head_dim: int = 16
    f: int = 64

    def __post_init__(self):
        if self.heads * self.head_dim != self.d:
            raise SystemExit(f"heads*head_dim ({self.heads}*{self.head_dim}) "
                             f"must equal d ({self.d})")
        if self.d & (self.d - 1):
            raise SystemExit(
                f"d = {self.d} must be a power of two: layernorm_rows divides the "
                f"row sum by the row length in a scalar loop whose cost assumes it")


# =============================================================================
# 1. The float layer — the thing being approximated, and the calibration source.
# =============================================================================

def build_weights(cfg: Config, seed: int = 0) -> SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    return SimpleNamespace(
        X=rnd(cfg.tokens, cfg.d),
        Wqkv=rnd(cfg.d, 3 * cfg.d) / math.sqrt(cfg.d),
        Wo=rnd(cfg.d, cfg.d) / math.sqrt(cfg.d),
        W1=rnd(cfg.d, cfg.f) / math.sqrt(cfg.d),
        W2=rnd(cfg.f, cfg.d) / math.sqrt(cfg.f),
        g1=1.0 + 0.1 * rnd(cfg.d), b1=0.1 * rnd(cfg.d),
        g2=1.0 + 0.1 * rnd(cfg.d), b2=0.1 * rnd(cfg.d),
    )


def ternary_pair(w: torch.Tensor) -> tuple:
    """``(trits, beta)`` with ``beta * trits`` the matrix the hardware computes."""
    tern, beta = quant.ternarize(w.tolist())
    return torch.tensor(tern, dtype=torch.float32), beta


def float_reference(cfg: Config, w: SimpleNamespace) -> SimpleNamespace:
    """Run the layer in float over the *ternarized* weights.

    Ternarizing first is deliberate: the weights are a hardware fact, not a
    quantization error, so the float reference should carry them too. What this
    reference then measures is the error the *activation* path adds.
    """
    T, d, hd, H = cfg.tokens, cfg.d, cfg.head_dim, cfg.heads
    Tqkv, bqkv = ternary_pair(w.Wqkv)
    To, bo = ternary_pair(w.Wo)
    T1, b1w = ternary_pair(w.W1)
    T2, b2w = ternary_pair(w.W2)

    QKV = w.X @ (bqkv * Tqkv)
    Q, K, V = QKV[:, :d], QKV[:, d:2 * d], QKV[:, 2 * d:]

    # [H, T, T] scores, one head at a time — the same slicing the strided
    # matmul_i8 does with byte offsets.
    S = torch.stack([Q[:, h * hd:(h + 1) * hd] @ K[:, h * hd:(h + 1) * hd].t()
                     for h in range(H)]) / math.sqrt(hd)
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    Pr = torch.softmax(S + causal, dim=-1)
    AO = torch.cat([Pr[h] @ V[:, h * hd:(h + 1) * hd] for h in range(H)], dim=1)

    O = AO @ (bo * To)
    X1r = w.X + O
    N1 = F.layer_norm(X1r, (d,), weight=w.g1, bias=w.b1, eps=EPS)
    Hh = N1 @ (b1w * T1)
    HG = F.gelu(Hh)
    Ff = HG @ (b2w * T2)
    X2 = N1 + Ff
    Y = F.layer_norm(X2, (d,), weight=w.g2, bias=w.b2, eps=EPS)

    return SimpleNamespace(
        trits=SimpleNamespace(Wqkv=Tqkv, Wo=To, W1=T1, W2=T2),
        betas=SimpleNamespace(Wqkv=bqkv, Wo=bo, W1=b1w, W2=b2w),
        QKV=QKV, S=S, Pr=Pr, AO=AO, O=O, X1r=X1r, N1=N1,
        H=Hh, HG=HG, F=Ff, X2=X2, Y=Y,
    )


def absmax_scale(x: torch.Tensor) -> float:
    return quant.scale_for(float(x.abs().max()))


# =============================================================================
# 2. Memory map, quantization, and the {m0,n} word at every rescale site.
# =============================================================================

def lay_out(cfg: Config, w: SimpleNamespace, ref: SimpleNamespace):
    """Allocate every tensor, quantize the inputs, and pick every requant word.

    Returns ``(dram, scratch, t, sites, scales, clips)``.
    """
    T, d, f, hd, H = cfg.tokens, cfg.d, cfg.f, cfg.head_dim, cfg.heads

    # Two disjoint address ranges, both below 0x8000 so every address `li`s to a
    # positive register value. Tensors use one address valid in *both* DRAM and
    # scratchpad (the identical-address convention); scratch buffers are
    # scratchpad-only and no DMA ever touches them.
    dram = Arena("tensor", 0x0000, 0x5000)
    scratch = Arena("scratch", 0x5000, 0x8000)

    # ---- calibrated activation scales (absmax / 127) ----
    s = SimpleNamespace(
        x=absmax_scale(w.X),
        qkv=absmax_scale(ref.QKV),
        score=LUT_SCALE,                 # pinned: exp's LUT input scale
        prob=PROB_SCALE,                 # pinned: softmax_rows' output
        ao=absmax_scale(ref.AO),
        x1=absmax_scale(ref.X1r),
        n1=absmax_scale(ref.N1),
        hid=LUT_SCALE,                   # pinned: gelu's LUT input scale
        hg=absmax_scale(ref.HG),
        x2=absmax_scale(ref.X2),
        y=absmax_scale(ref.Y),
        g1=absmax_scale(w.g1), g2=absmax_scale(w.g2),
    )
    # The residual adds require both operands in one scale, so the projection's
    # output is forced onto X's scale and the FF's onto N1's.
    s.o, s.ff = s.x, s.n1

    t = SimpleNamespace()

    def tensor(name, shape, dtype="int8", scale=1.0):
        obj = dram.tensor(name, shape, dtype, scale)
        setattr(t, name, obj)
        return obj

    # inputs
    tensor("X", (T, d), scale=s.x)
    tensor("Wqkv", (d, 3 * d), "ternary")
    tensor("Wo", (d, d), "ternary")
    tensor("W1", (d, f), "ternary")
    tensor("W2", (f, d), "ternary")
    tensor("g1", (d,), scale=s.g1)
    tensor("b1", (d,), scale=s.n1)
    tensor("g2", (d,), scale=s.g2)
    tensor("b2", (d,), scale=s.y)
    tensor("mask", (H * T, T), scale=s.score)
    # intermediates
    tensor("QKV", (T, 3 * d), scale=s.qkv)
    tensor("S32", (H, T, T), "int32")
    tensor("S8", (H * T, T), scale=s.score)
    tensor("SM8", (H * T, T), scale=s.score)
    tensor("P8", (H * T, T), scale=s.prob)
    tensor("Vt", (d, T), scale=s.qkv)
    tensor("AO32", (T, d), "int32")
    tensor("AO8", (T, d), scale=s.ao)
    tensor("O", (T, d), scale=s.o)
    tensor("X1", (T, d), scale=s.x1)
    tensor("N1", (T, d), scale=s.n1)
    tensor("H", (T, f), scale=s.hid)
    tensor("HG", (T, f), scale=s.hg)
    tensor("F", (T, d), scale=s.ff)
    tensor("X2", (T, d), scale=s.x2)
    tensor("Y", (T, d), scale=s.y)

    # ---- one {m0,n} word per rescale site ----
    # Each is the factor turning the producer's integers into the consumer's:
    # ratio = scale_in / scale_out, with scale_in the product of the operand
    # scales for a matmul.
    sites = {
        "qkv":    quant.choose_requant(s.x * ref.betas.Wqkv / s.qkv),
        # the /sqrt(head_dim) of attention lives here, folded into the requant
        "scores": quant.choose_requant(s.qkv * s.qkv / math.sqrt(hd) / s.score),
        "mask":   (1, 0),                       # both operands already at 1/16
        "ao":     quant.choose_requant(s.prob * s.qkv / s.ao),
        "proj":   quant.choose_requant(s.ao * ref.betas.Wo / s.o),
        "res1":   quant.choose_requant(s.x / s.x1),
        "ln1_y":  quant.layernorm_requant(d, NORM_SCALE),
        "ln1_g":  quant.choose_requant(NORM_SCALE * s.g1 / s.n1),
        "ff1":    quant.choose_requant(s.n1 * ref.betas.W1 / s.hid),
        "gelu":   quant.choose_requant(LUT_SCALE / s.hg),
        "ff2":    quant.choose_requant(s.hg * ref.betas.W2 / s.ff),
        "res2":   quant.choose_requant(s.n1 / s.x2),
        "ln2_y":  quant.layernorm_requant(d, NORM_SCALE),
        "ln2_g":  quant.choose_requant(NORM_SCALE * s.g2 / s.y),
    }
    for name, (m0, n) in sites.items():
        rq = dram.tensor(f"rq_{name}", (1,), "int32")
        rq.data = quant.int32_bytes(quant.requant_word(m0, n))
        setattr(t, f"rq_{name}", rq)

    # ---- host data ----
    t.X.data = _bytes(quant.quantize_i8(_flat(w.X), s.x))
    t.g1.data = _bytes(quant.quantize_i8(_flat(w.g1), s.g1))
    t.b1.data = _bytes(quant.quantize_i8(_flat(w.b1), s.n1))
    t.g2.data = _bytes(quant.quantize_i8(_flat(w.g2), s.g2))
    t.b2.data = _bytes(quant.quantize_i8(_flat(w.b2), s.y))
    for name in ("Wqkv", "Wo", "W1", "W2"):
        tern = getattr(ref.trits, name)
        rows, cols = tern.shape
        getattr(t, name).data = _bytes(
            quant.pack_ternary([[int(v) for v in row] for row in tern.tolist()],
                               rows, cols))

    # Causal mask as data: 0 where a key is visible, -128 (= -8 at scale 1/16)
    # where it is not. Adding -128 drives the entry to the requant's clip floor,
    # which is exactly where the exp table stores 0.
    mask = []
    for _h in range(H):
        for q in range(T):
            mask += [0 if k <= q else -128 for k in range(T)]
    t.mask.data = _bytes(mask)

    # How much the fixed LUT scales cost us, measured rather than assumed.
    clips = {
        "scores vs exp LUT domain":
            _clip_count(ref.S.reshape(-1) / 1.0, LUT_SCALE),
        "ff hidden vs gelu LUT domain":
            _clip_count(ref.H.reshape(-1), LUT_SCALE),
    }
    return dram, scratch, t, sites, s, clips


def _flat(x: torch.Tensor) -> list:
    return x.reshape(-1).tolist()


def _bytes(vals) -> list:
    return [v & 0xFF for v in vals]


def _clip_count(x: torch.Tensor, scale: float) -> tuple:
    """(clipped, total, observed absmax, representable absmax)."""
    limit = 127 * scale
    return int((x.abs() > limit).sum()), x.numel(), float(x.abs().max()), limit


# =============================================================================
# 3. Compose the program.
# =============================================================================

def compose(cfg: Config, t, sites, scratch: Arena) -> Program:
    T, d, f, hd, H = cfg.tokens, cfg.d, cfg.f, cfg.head_dim, cfg.heads
    prog = Program(name="transformer_layer")
    prog.notes = [
        f"one quantized transformer layer: T={T} d={d} f={f} "
        f"heads={H} head_dim={hd}",
        "post-norm: Y = norm2(X1 + W2.gelu(W1.X1)), X1 = norm1(X + Wo.attn(X))",
        "ternary weights (MXU), int8 activations, VPU attention, LUT gelu/exp",
    ]

    # ---- attention ----
    P.matmul_ternary(prog, scratch, "qkv", t.X, t.Wqkv, t.QKV, t.rq_qkv, tt=T)

    # Q and K are strided sub-blocks of QKV: a head's rows are head_dim
    # contiguous bytes at row stride 3d, which is all vecdot needs.
    P.matmul_i8(prog, "scores", out=t.S32, heads=H, m=T, n=T, klen=hd,
                a_addr=t.QKV.addr, a_head=hd, a_row=3 * d,
                b_addr=t.QKV.addr + d, b_head=hd, b_row=3 * d,
                o_head=T * T * 4, o_row=T * 4,
                stage=[(t.QKV.addr, t.QKV.nbytes)],
                note="S = Q K^T per head, over the head dim")

    P.requant_block(prog, "scores_rq", t.S32, t.S8, t.rq_scores)
    P.add_i8(prog, scratch, "mask", t.S8, t.mask, t.SM8, t.rq_mask,
             note="causal mask: +(-128) drives a key to exp's zero")
    P.softmax_rows(prog, scratch, "softmax", t.SM8, t.P8, rows=H * T, length=T)

    # V^T, so the P.V contraction (over keys) has both operands contiguous.
    P.transpose_i8(prog, scratch, "vt", src_addr=t.QKV.addr + 2 * d, dst=t.Vt,
                   m=T, n=d, src_stride=3 * d, dst_stride=T)

    P.matmul_i8(prog, "attn", out=t.AO32, heads=H, m=T, n=hd, klen=T,
                a_addr=t.P8.addr, a_head=T * T, a_row=T,
                b_addr=t.Vt.addr, b_head=hd * T, b_row=T,
                o_head=hd * 4, o_row=d * 4,
                stage=[(t.P8.addr, t.P8.nbytes), (t.Vt.addr, t.Vt.nbytes)],
                note="AO = P V^T per head, over the key axis")

    P.requant_block(prog, "attn_rq", t.AO32, t.AO8, t.rq_ao)
    P.matmul_ternary(prog, scratch, "proj", t.AO8, t.Wo, t.O, t.rq_proj, tt=T)

    # ---- residual + norm1 ----
    P.add_i8(prog, scratch, "res1", t.X, t.O, t.X1, t.rq_res1, note="residual")
    P.layernorm_rows(prog, scratch, "norm1", t.X1, t.N1, t.g1, t.b1,
                     t.rq_ln1_y, t.rq_ln1_g)

    # ---- feed-forward ----
    P.matmul_ternary(prog, scratch, "ff1", t.N1, t.W1, t.H, t.rq_ff1, tt=T)
    P.gelu_block(prog, scratch, "gelu", t.H, t.HG, t.rq_gelu)
    P.matmul_ternary(prog, scratch, "ff2", t.HG, t.W2, t.F, t.rq_ff2, tt=T)

    # ---- residual + norm2 ----
    P.add_i8(prog, scratch, "res2", t.N1, t.F, t.X2, t.rq_res2, note="residual")
    P.layernorm_rows(prog, scratch, "norm2", t.X2, t.Y, t.g2, t.b2,
                     t.rq_ln2_y, t.rq_ln2_g)
    return prog


# =============================================================================
# 4. Verify.
# =============================================================================

CHECKED = [  # tensor name -> dtype, in program order
    ("QKV", "int8"), ("S32", "int32"), ("S8", "int8"), ("SM8", "int8"),
    ("P8", "int8"), ("Vt", "int8"), ("AO32", "int32"), ("AO8", "int8"),
    ("O", "int8"), ("X1", "int8"), ("N1", "int8"), ("H", "int8"),
    ("HG", "int8"), ("F", "int8"), ("X2", "int8"), ("Y", "int8"),
]


def read_out(out_img: dict, addr: int, n: int, dtype: str) -> list:
    """Decode ``n`` elements from the spilled DRAM image. A missing byte is an
    error, not a zero: it means the program never wrote something we expect."""
    def byte(a):
        try:
            return out_img[a]
        except KeyError:
            raise AssertionError(
                f"no output byte at 0x{a:04x} — the program never wrmem'd it"
            ) from None
    if dtype == "int8":
        return [layer_ref.s8(byte(addr + k)) for k in range(n)]
    vals = []
    for k in range(n):
        u = sum(byte(addr + 4 * k + b) << (8 * b) for b in range(4))
        vals.append(u - (1 << 32) if u >= (1 << 31) else u)
    return vals


def check_integer(t, expect: dict, out_img: dict, limit: int = 6) -> int:
    print("\ninteger reference (exact, no tolerance)")
    bad_total = 0
    for name, dtype in CHECKED:
        tens = getattr(t, name)
        want = expect[name]
        got = read_out(out_img, tens.addr, len(want), dtype)
        wrong = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
        bad_total += len(wrong)
        flag = "OK  " if not wrong else "FAIL"
        print(f"  [{flag}] {name:<5s} {dtype:<5s} {str(list(tens.shape)):<12s} "
              f"{len(wrong)}/{len(want)} differ")
        for i in wrong[:limit]:
            print(f"         [{i}]: device {got[i]:6d}   reference {want[i]:6d}")
        if len(wrong) > limit:
            print(f"         ... and {len(wrong) - limit} more")
    return bad_total


def check_float(t, out_img: dict, ref: SimpleNamespace, scales) -> None:
    """Compare the dequantized result to the float layer.

    Reported, never asserted. int8 activations, a 32-wide integer LayerNorm and
    a 256-entry exp table do not reproduce float arithmetic, and a threshold here
    would only encode whatever error today's seed happens to give. The integer
    check above is what says the program is *correct*; this says what
    quantization *costs*.
    """
    y = read_out(out_img, t.Y.addr, t.Y.nelem, "int8")
    got = torch.tensor(y, dtype=torch.float32).reshape(ref.Y.shape) * scales.y
    err = (got - ref.Y).abs()
    denom = ref.Y.abs().mean()
    print("\nfloat reference (reported, not asserted)")
    print(f"  Y  max |err| = {float(err.max()):.4f}   mean |err| = "
          f"{float(err.mean()):.4f}   mean |Y| = {float(denom):.4f}   "
          f"relative = {float(err.mean() / denom):.1%}")
    corr = torch.corrcoef(torch.stack([got.reshape(-1), ref.Y.reshape(-1)]))[0, 1]
    print(f"  correlation with the float layer: {float(corr):.4f}")


# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=os.path.join(_HERE, "out",
                                                     "transformer_layer.tpu"),
                    help="where to write the generated .tpu")
    ap.add_argument("--emit-vectors", action="store_true",
                    help="also write tpu_prog/spad_in/spad_exp .hex for the testbench")
    ap.add_argument("--vectors-dir", default=os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "tpu", "tb", "vectors_layer")))
    ap.add_argument("--map", action="store_true", help="print the memory map")
    args = ap.parse_args(argv)

    cfg = Config()
    weights = build_weights(cfg, args.seed)
    ref = float_reference(cfg, weights)
    dram, scratch, t, sites, scales, clips = lay_out(cfg, weights, ref)

    prog = compose(cfg, t, sites, scratch)
    prog.track(dram)
    prog.track(scratch)
    words = prog.finalize()
    path = prog.emit_source(args.source)

    print(f"generated {path}")
    print(prog.report())
    print(f"\ntensor memory: {dram.used} B of "
          f"0x{dram.base:04x}..0x{dram.limit:04x};  scratch buffers: "
          f"{scratch.used} B of 0x{scratch.base:04x}..0x{scratch.limit:04x}")
    if args.map:
        print()
        print(prog.memory_map())

    print("\nrequant sites  (m0/2^n applied at each rescale)")
    for name, (m0, n) in sites.items():
        print(f"  {name:<8s} m0={m0:5d} n={n:2d}  ->  "
              f"x{quant.requant_ratio(m0, n):.6g}")

    print("\nfixed LUT input scales cost (1/16 covers +-7.94)")
    for what, (clipped, total, seen, limit) in clips.items():
        print(f"  {what:<32s} absmax {seen:6.3f} vs {limit:5.3f}  "
              f"-> {clipped}/{total} clipped")

    in_img = prog.host_image()
    out_img, _tpu = prog.run_iss(in_img)
    print(f"\nran in the ISS: {len(in_img)} input bytes, "
          f"{len(out_img)} bytes spilled to DRAM")

    expect = layer_ref.run(cfg, t, sites, lambda a: in_img.get(a, 0))
    bad = check_integer(t, expect, out_img)
    check_float(t, out_img, ref, scales)

    if args.emit_vectors:
        paths = prog.emit_vectors(args.vectors_dir, in_img, out_img)
        print("\nvectors for tpu_top_tb.sv:")
        for k, p in paths.items():
            print(f"  {k:<5s} {p}")

    print()
    if bad:
        print(f"FAILED: {bad} value(s) disagree with the integer reference")
        return 1
    print(f"PASSED: {len(words)} words, every intermediate matches the integer "
          f"reference exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
