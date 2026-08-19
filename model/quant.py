"""quant.py — benchmark a checkpoint under int8 activations, hardware-exact.

`TernaryLinear.quantize_activations` fake-quantizes in float: it rounds an
activation onto an int8 grid and immediately multiplies the scale back in, so
everything downstream still runs in float32. That answers "how much does int8
*rounding* cost", which is not the same question as "what would the accelerator
produce", because the accelerator has no floats at all. Between the host's
embedding lookup and the host's final argmax, every value in
`accel/tpu` is an int8 in a scratchpad or an int32 in an accumulator, and the
only thing that ever narrows one to the other is

    requant(acc) = clip_int8((acc * m0 + 2**(n-1)) >> n)      m0 < 4096, n <= 15

with `>>` an arithmetic (flooring) shift. This module runs the model that way:
integers end to end, the same requant word per site the kernel would load from
DRAM, the same clipping, the same flooring. The reported accuracy is what the
hardware would score, not an estimate of it.

    python -m model.quant                                   # 256 problems
    python -m model.quant --arch ternary_vanilla \
        --model-path model/saved/colab_ternary_mha_small_dyt.pt -n 512
    python -m model.quant --check --dynamic-range           # diagnostics

Four stages, each usable on its own from Python:

  1. :func:`instrumented_forward` — `Model.forward` rewritten inline so every
     intermediate is visible. :func:`check_instrumented` asserts it agrees with
     the real thing, which is what makes it safe to calibrate from.
  2. :func:`calibrate` — absmax at every site the integer pipeline needs a
     scale, over a calibration set disjoint from the evaluation set.
  3. :func:`prepare` — weights to trits/int8, and the requant word per site.
  4. :func:`int_forward` — the integer pipeline, batched.

Relationship to `accel/tpulang/adder_export.py`: that script does the same job
for one frozen checkpoint against `examples/adder_model.tpu`, and its integer
forward is verified bit-exact against both the ISS and the RTL. This module is
the same arithmetic generalized to whatever `Model` you hand it — GQA, ternary
or float weights, with or without DyT — and lives next to the model so it can
follow it as it changes. The fixed-point helpers here are deliberately
identical to that script's so the two remain comparable.

**What is host-side, and why** (unchanged from `adder_kernel.md` §1): the token
embedding, because the ISA has no gather; and the final argmax, because
`redmax` returns a maximum and not its index. Everything between them is the
device's. There is no longer a positional encoding to add.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import model.numbers_data as numbers_data
import model.transformer as transformer
from model.transformer import DyT, TernaryLinear

# =============================================================================
# The hardware's fixed point. These four numbers are facts about the RTL
# (`requant` in vpu.sv, mirrored in iss.requant8), not tunables.
# =============================================================================
M0_W, N_W = 12, 4
M0_MAX = (1 << M0_W) - 1          # 4095
N_MAX = (1 << N_W) - 1            # 15
INT8_MIN, INT8_MAX = -128, 127

Word = tuple[int, int]            # (m0, n), the requant multiplier m0 / 2**n

# The requant sites, in dataflow order. One word each, per layer.
# "KT"/"VT" are `tquant` rather than `requant` — the narrow that rounds the int8
# K and V onto {-1, 0, 1} so the MXU can take them as weight operands. The word
# format is identical, so nothing downstream of the table cares which op reads
# it; only the clip differs (+-1 instead of +-127).
SLOTS = ["Q", "K", "KT", "V", "VT", "S", "ID", "P", "A", "O", "XO",
         "X1", "H", "HR", "F", "X2"]

# The calibration sites. "xo" and "o" are the two attention-residual addends;
# they get recorded even though their scales are *pinned* (see derive_layer),
# because the amount by which they overflow the scale they are pinned to is the
# diagnostic that says whether the residual path is the problem.
SITES = ["x", "q", "k", "kt", "v", "vt", "p", "a", "o", "xo",
         "x1", "h", "hr", "f", "x2"]

# Sites whose grid is {-1, 0, 1}, not int8. Their scale is the tensor's
# **absmean**, which is BitNet's rule and the one TernaryLinear applies to a
# weight — absmax/1 would round every element but the largest to zero.
TERNARY_SITES = {"kt", "vt"}


def choose_word(m: float) -> Word:
    """Best `{m0, n}` with `m0/2**n ~= m`, `1 <= m0 <= 4095`, `0 <= n <= 15`.

    A larger shift gives finer relative resolution, so the largest feasible `n`
    wins. A multiplier outside the representable band (2**-15 to 4095)
    saturates rather than silently wrapping; :func:`report_words` flags how far
    each word missed by, which is the check that the saturation never bit.
    """
    if not (m > 0) or not math.isfinite(m):
        # Negative multipliers are unrepresentable: m0 is an unsigned field.
        # The only way one arises here is a negative DyT alpha (see dyt_alpha).
        raise ValueError(f"requant multiplier must be finite and positive, got {m!r}")
    for n in range(N_MAX, -1, -1):
        m0 = int(round(m * (1 << n)))
        if 1 <= m0 <= M0_MAX:
            return m0, n
    return (M0_MAX, 0) if m > 1 else (1, N_MAX)


def _rescale(acc: torch.Tensor, word: Word) -> torch.Tensor:
    """`(acc*m0 + 2**(n-1)) >> n`, unclipped. The shift floors, as `>>>` does."""
    m0, n = word
    rnd = 0 if n == 0 else (1 << (n - 1))
    return torch.div(acc * m0 + rnd, 1 << n, rounding_mode="floor")


def requant8(acc: torch.Tensor, word: Word) -> torch.Tensor:
    """`clip_int8((acc*m0 + 2**(n-1)) >> n)` — vpu.sv's VOP_REQUANT."""
    return _rescale(acc, word).clamp(INT8_MIN, INT8_MAX)


def dyt8(acc: torch.Tensor, word: Word) -> torch.Tensor:
    """The same rescale clipped to +-127 — vpu.sv's VOP_DYT, i.e. DyT.

    `DyT(x) = hardtanh(alpha*x, -1, 1)`, and with the output scale pinned to
    1/127 the saturating narrow *is* the hardtanh: the word carries
    `alpha * s_in * 127` and every value the clip catches is one hardtanh would
    have flattened. The clip has to be symmetric because hardtanh is odd —
    int8's -128 would put the saturated end at -1.0079 instead of -1, which is
    why this is a separate device op and not `requant` with a chosen scale.
    """
    return _rescale(acc, word).clamp(-INT8_MAX, INT8_MAX)


def tquant8(acc: torch.Tensor, word: Word) -> torch.Tensor:
    """`clip_pm1((acc*m0 + 2**(n-1)) >> n)` — vpu.sv's VOP_TQUANT.

    The same fixed point as :func:`requant8`, clipped to a single level per
    sign. The device writes the result 2 bits wide (00 = 0, 01 = +1, 11 = -1)
    straight into the column-major layout `matmul_t` reads its weights from, so
    this op is both the quantization *and* the packing.
    """
    return _rescale(acc, word).clamp(-1, 1)


def clip_rate(acc: torch.Tensor, word: Word, lo: int = INT8_MIN,
              hi: int = INT8_MAX) -> float:
    """Fraction of a narrow's inputs that saturate, i.e. that it discards.

    `lo`/`hi` are the op's clips: int8's [-128, 127] for `requant`, [-127, 127]
    for `dyt`, [-1, 1] for `tquant`. For a DyT site this is not a loss rate —
    saturating is the nonlinearity doing its job — and for a ternary site it is
    not one either, since clipping is most of what ternarizing *is*; but it is
    still the statistic that says how much of the tensor is living in the flat
    region.
    """
    shifted = _rescale(acc, word)
    return float(((shifted > hi) | (shifted < lo)).double().mean())


# =============================================================================
# Exact integer linear algebra.
#
# torch has no fast integer matmul, and the obvious int64 broadcast-and-sum
# materializes [M, K, N], which at model scale is gigabytes. Everything here is
# therefore evaluated in float64 and converted back — which is *exact*, not
# approximate, and the guard below is what makes that a checked claim rather
# than an assumption:
#
#   float64 represents every integer below 2**53 exactly, and the sum or
#   product of two exactly-represented integers is itself exact whenever the
#   result is also below 2**53. Every partial sum in these contractions is an
#   integer bounded by (max|a| * max|b| * K), so if that bound clears 2**53 the
#   result is exact whatever order BLAS accumulates in.
#
# The bounds here are not close: the widest is a ternary projection at
# 128 * 127 * 1 ~= 1.6e4, twelve orders of magnitude below the limit. Every
# weight in the ternary config is now a trit, the output head included.
# =============================================================================
_EXACT_LIMIT = 1 << 53


def _guard_exact(bound: float, what: str) -> None:
    if bound >= _EXACT_LIMIT:
        raise OverflowError(
            f"{what}: worst-case accumulator {bound:.3e} reaches float64's exact "
            f"integer range ({_EXACT_LIMIT:.3e}); this contraction would no longer "
            f"be bit-exact. Fall back to an int64 accumulation here."
        )


def ieinsum(eq: str, *ops: torch.Tensor, terms: int, what: str = "ieinsum") -> torch.Tensor:
    """Exact integer einsum over `terms` contracted elements. See above."""
    bound = float(terms)
    for o in ops:
        bound *= max(float(o.abs().max()), 1.0)
    _guard_exact(bound, what)
    out = torch.einsum(eq, *[o.to(torch.float64) for o in ops])
    return out.to(torch.int64)


def imatmul(a: torch.Tensor, b: torch.Tensor, what: str = "imatmul") -> torch.Tensor:
    """Exact integer `[..., K] @ [K, N]`. See above."""
    return ieinsum("...k,kn->...n", a, b, terms=a.shape[-1], what=what)


# =============================================================================
# Weights.
# =============================================================================
@dataclass
class QLinear:
    """A linear layer as the device holds it.

    `kind` decides which unit runs it: ternary weights go to the MXU's
    `matmul_t` (2 bits each, `scale` is the checkpoint's absmean and is never a
    tensor — it is just a factor in the following requant's multiplier), and
    int8 weights go to the VPU's `vecmatmul`. Either way `w` is `[in, out]`,
    which is the orientation `TernaryLinear` already stores and the transpose
    of `nn.Linear`'s.

    The ternary configs have no `int8` weight left — the output head became a
    `TernaryLinear` too, so `vecmatmul` has no caller in the shipped kernel and
    every matmul in the model runs on the array. The kind still exists because
    the float `adder_vanilla` / `adder_gqa` configs keep an `nn.Linear` head.
    """

    kind: str                         # "ternary" | "int8"
    w: torch.Tensor                   # [in, out] int64 holding trits or int8
    scale: float                      # absmean (ternary) or absmax/127 (int8)
    bias: torch.Tensor | None = None  # [out] float, quantized once s_out is known

    @property
    def wmax(self) -> float:
        return 1.0 if self.kind == "ternary" else float(INT8_MAX)


def quantize_i8(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric per-tensor int8: `real ~= int * scale`, `scale = absmax/127`."""
    a = float(t.detach().abs().max())
    s = a / 127.0 if a > 0 else 1.0
    return torch.round(t.detach() / s).clamp(-127, 127).to(torch.int64), s


def trits_and_absmean(lin: TernaryLinear) -> tuple[torch.Tensor, float]:
    """`TernaryLinear` -> ({-1,0,1} matrix, absmean). Mirrors its forward().

    Note the `+ eps` inside the division but *not* in the returned absmean —
    that asymmetry is `TernaryLinear.forward`'s, and reproducing it is the
    whole point.
    """
    absmean = float(lin.w.abs().mean())
    t = (lin.w.detach() / (absmean + lin.eps)).round().clamp(-1, 1)
    return t.to(torch.int64), absmean


def quantize_linear(lin: nn.Module) -> QLinear:
    """Any linear in the model -> the form the device runs."""
    if isinstance(lin, TernaryLinear):
        w, s = trits_and_absmean(lin)
        bias = None if lin.bias is None else lin.bias.detach().clone()
        return QLinear("ternary", w, s, bias)
    if isinstance(lin, nn.Linear):
        # nn.Linear stores [out, in]; the device contracts over `in`.
        w, s = quantize_i8(lin.weight.t().contiguous())
        bias = None if lin.bias is None else lin.bias.detach().clone()
        return QLinear("int8", w, s, bias)
    raise TypeError(f"cannot quantize {type(lin).__name__}")


def dyt_alpha(norm: nn.Module) -> float | None:
    """DyT's alpha, or None if this norm slot is a passthrough.

    `DyT(x) = hardtanh(alpha*x, -1, 1)`, and a hardtanh is *exactly* what a
    saturating narrow already does — so DyT costs the device nothing. It fuses
    into the narrow that was going to follow the residual add anyway: pick the
    output scale 1/127 and the clip saturates at exactly +-1. That narrow is
    `dyt` (vpu.sv VOP_DYT) rather than `requant` only because hardtanh is odd
    and int8's clip is not; see :func:`dyt8` and :func:`derive_layer`.
    """
    if isinstance(norm, DyT):
        a = float(norm.alpha.detach().reshape(-1)[0])
        if a <= 0:
            raise ValueError(
                f"DyT alpha = {a:.4g} <= 0; requant's m0 is an unsigned field, so a "
                f"negative scaling cannot be expressed as a requant word. The device "
                f"would need a separate negation pass."
            )
        return a
    if norm is None or isinstance(norm, nn.Identity):
        return None
    raise TypeError(f"unsupported norm {type(norm).__name__}")


# =============================================================================
# 1-2. The instrumented float forward, and calibration.
# =============================================================================
class Calibration:
    """Running absmax at every site the integer pipeline needs a scale."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.m = [{s: 0.0 for s in SITES} for _ in range(n_layers)]
        # Running [sum|t|, count] beside the absmax: a ternary site is scaled by
        # the absmean, so both statistics have to be collected in the one pass.
        self.a = [{s: [0.0, 0] for s in SITES} for _ in range(n_layers)]

    def see(self, layer: int, site: str, t: torch.Tensor) -> None:
        if t.numel() == 0:
            return
        a = t.detach().abs()
        v = float(a.max())
        if v > self.m[layer][site]:
            self.m[layer][site] = v
        acc = self.a[layer][site]
        acc[0] += float(a.sum())
        acc[1] += a.numel()

    def scale(self, layer: int, site: str) -> float:
        """absmax/127, or the absmean at a ternary site (see TERNARY_SITES)."""
        if site in TERNARY_SITES:
            total, n = self.a[layer][site]
            return (total / n) if (n and total > 0) else 1.0
        a = self.m[layer][site]
        return (a / 127.0) if a > 0 else 1.0


@torch.no_grad()
def instrumented_forward(
    model: transformer.Model,
    tokens: torch.Tensor,
    cal: Calibration | None = None,
    keep: dict | None = None,
) -> torch.Tensor:
    """`Model.forward` written inline so every intermediate is reachable.

    This mirrors `model/transformer.py` as it stands, including two things that
    a reader of the class hierarchy would not predict and that the integer
    pipeline has to reproduce exactly:

    * **`MultiHeadAttention.forward` returns `O + X`, and `Transformer.forward`
      then adds `X` again** — so the attention residual is `2X + O`, not
      `X + O`. That extra copy is why there is an `XO` requant slot (§ the
      device can only add two int8 operands at a time).
    * **the padding mask is ignored.** `MultiHeadAttention` builds its own
      causal mask and never reads the `attn_mask` it is handed, so the only
      mask in play is the causal one and the device needs no other.

    Dropout is off (the caller eval()s the model); attention is ReLU, not
    softmax. :func:`check_instrumented` is what keeps this honest.
    """
    B, T = tokens.shape
    dev = next(model.parameters()).device
    X = model.embedding(tokens)          # no positional encoding

    kv, g, dh = model.kv_heads, model.q_heads // model.kv_heads, model.head_dim
    causal = torch.triu(torch.ones(T, T, device=dev) * -1e9, diagonal=1)

    def rec(li, site, t):
        if cal is not None:
            cal.see(li, site, t)
        if keep is not None:
            keep.setdefault((li, site), []).append(t.detach().flatten().float())

    for li, layer in enumerate(model.layers):
        rec(li, "x", X)
        at = layer.attention
        Q, K, V = at.Wq(X), at.Wk(X), at.Wv(X)
        rec(li, "q", Q); rec(li, "k", K); rec(li, "v", V)
        # K and V are ternarized on top of their int8 narrow, so the same
        # tensor is seen twice - once for the int8 scale, once for the ternary
        # one, which is an absmean rather than an absmax (TERNARY_SITES).
        rec(li, "kt", K); rec(li, "vt", V)

        Qh = Q.reshape(B, T, kv, g, dh)
        Kh = K.reshape(B, T, kv, dh)
        Vh = V.reshape(B, T, kv, dh)
        S = torch.einsum("btkgh,bskh->btskg", Qh, Kh) / math.sqrt(dh)
        P = torch.relu(S + causal[None, :, :, None, None])
        # Only the surviving (post-ReLU) scores matter — see derive_layer (a).
        rec(li, "p", P[P > 0])
        A = torch.einsum("btskg,bskh->btkgh", P, Vh).reshape(B, T, kv * g * dh)
        rec(li, "a", A)

        O = at.Wo(A)
        rec(li, "o", O)
        XO = X + O                      # MultiHeadAttention's own residual
        rec(li, "xo", XO)
        X1 = layer.norm1(X + XO)        # Transformer's residual, then DyT
        rec(li, "x1", X1)

        H = layer.ff[0](X1)
        rec(li, "h", H)
        HR = torch.relu(H)
        rec(li, "hr", HR)
        F = layer.ff[2](HR)
        rec(li, "f", F)
        X = layer.norm2(X1 + F)
        rec(li, "x2", X)

    return model.fc(X)


@torch.no_grad()
def check_instrumented(model: transformer.Model, tokens: torch.Tensor,
                       masks: torch.Tensor) -> float:
    """Max abs difference between the inline forward and `Model.forward`.

    If this is not ~0 the calibration is describing a different network from
    the one being benchmarked, and every number downstream is meaningless.
    """
    return float((instrumented_forward(model, tokens) - model(tokens, masks)).abs().max())


@torch.no_grad()
def calibrate(model: transformer.Model, tokens: torch.Tensor) -> Calibration:
    """One instrumented float forward over the calibration set."""
    cal = Calibration(len(model.layers))
    instrumented_forward(model, tokens, cal)
    return cal


class QATCalibration(Calibration):
    """The scales the model was *trained* with, not ones measured after the fact.

    A QAT checkpoint carries an :class:`~model.transformer.ActQuant` per requant
    site whose scale was learned by LSQ, and the weights were fitted against
    exactly those scales — so re-deriving them by absmax is not a neutral
    substitution, it moves every rounding grid the model was trained on. This
    reads the learned scale out instead and presents it through the same
    ``scale(layer, site)`` interface, so :func:`derive_layer` is unchanged.

    ``.abs()`` mirrors ``ActQuant.forward``: the scale is a free parameter that
    LSQ can drive negative (layer 1's ``q_v`` in ``ternary_mha.pt`` is -0.00405),
    and the forward pass uses its magnitude.

    The absmax statistics are still collected, because :func:`report_clips` and
    the dynamic-range diagnostic are about the tensors, not the scales.
    """

    # site -> where the owning ActQuant lives, given (model, li).
    _SITES = {
        "q": lambda m, l: m.layers[l].attention.q_q,
        "k": lambda m, l: m.layers[l].attention.q_k,
        "kt": lambda m, l: m.layers[l].attention.q_kt,
        "v": lambda m, l: m.layers[l].attention.q_v,
        "vt": lambda m, l: m.layers[l].attention.q_vt,
        "p": lambda m, l: m.layers[l].attention.q_p,
        "a": lambda m, l: m.layers[l].attention.q_a,
        # q_o / q_xo *are* the residual-stream site (x_quant), shared by
        # construction in MultiHeadAttention.__init__.
        "x": lambda m, l: m.layers[l].attention.q_o,
        "o": lambda m, l: m.layers[l].attention.q_o,
        "xo": lambda m, l: m.layers[l].attention.q_xo,
        "x1": lambda m, l: m.layers[l].q_x1,
        "h": lambda m, l: m.layers[l].q_h,
        "hr": lambda m, l: m.layers[l].q_hr,
        "f": lambda m, l: m.layers[l].q_f,
        "x2": lambda m, l: m.layers[l].q_x2,
    }

    def __init__(self, model: transformer.Model):
        super().__init__(len(model.layers))
        # A site whose `initialized` flag is clear never saw a tensor, so its
        # `scale` is still the nn.Parameter's 1.0 and means nothing. That is
        # what a checkpoint predating a site looks like after a strict=False
        # load — the ternary q_kt/q_vt against any pre-`tquant` checkpoint —
        # and taking the 1.0 would silently quantize with a garbage scale
        # instead of falling back to the calibration set.
        self.learned = [
            {s: (float(get(model, li).scale.detach().abs())
                 if bool(get(model, li).initialized) else None)
             for s, get in self._SITES.items()}
            for li in range(len(model.layers))
        ]

    def scale(self, layer: int, site: str) -> float:
        s = self.learned[layer].get(site)
        return s if (s is not None and s > 0) else super().scale(layer, site)


def has_qat_scales(model: transformer.Model) -> bool:
    """True if this checkpoint was trained with the ActQuant sites in place."""
    return any(isinstance(m, transformer.ActQuant) for m in model.modules())


# =============================================================================
# 3. Derive the requant words.
# =============================================================================
DYT_OUT_SCALE = 1.0 / 127.0    # DyT's range is analytically [-1, 1]


@dataclass
class QuantModel:
    """Everything the device would be loaded with, plus the bookkeeping."""

    layers: list[dict] = field(default_factory=list)
    fc: QLinear | None = None
    fc_bias_i32: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    s_x0: float = 1.0
    kv: int = 1
    g: int = 1
    dh: int = 1
    n_tokens: int = 32
    cal: "Calibration | None" = None


def causal_mask_i8(T: int) -> torch.Tensor:
    """0 where key <= query, -128 above the diagonal.

    Exact, not a tolerance: `S8` is int8, so `S8 - 128 <= -1` for every value it
    can hold, and the ReLU after it takes any negative to exactly zero. The
    mask is data, not an instruction.
    """
    t = torch.arange(T)
    return torch.where(t[None, :] <= t[:, None], 0, -128).to(torch.int64)


def residual_scales(model: transformer.Model, cal: Calibration,
                    dyt_fixed: bool) -> list[float]:
    """The residual-stream scale at each layer's input.

    Single source of truth, because it is used three times and every use has to
    agree: to quantize `X0` before layer 0, as `s_x` inside layer `li`'s words,
    and as the *target* of layer `li-1`'s `RQ_X2`.

    With DyT in place the choice is nearly free. `norm2` bounds its output to
    [-1, 1] analytically, so `s = 1/127` needs no calibration at all and makes
    the `dyt` op's +-127 clip coincide with the hardtanh exactly — the model
    renormalizes the residual stream every layer, which is precisely what
    `adder_kernel.md` §8.1 found missing in the norm-free checkpoint. Layer 0
    is the exception: its input is the raw embedding, which nothing bounds, so
    it stays calibrated.
    """
    out = []
    for li, layer in enumerate(model.layers):
        if li == 0:
            out.append(cal.scale(0, "x"))
        else:
            prev = model.layers[li - 1]
            a = dyt_alpha(prev.norm2)
            out.append(DYT_OUT_SCALE if (a is not None and dyt_fixed)
                       else cal.scale(li, "x"))
    return out


def derive_layer(model: transformer.Model, li: int, cal: Calibration,
                 s_x: float, s_x_next: float, dyt_fixed: bool) -> dict:
    """The requant words for one layer, and the scales they were built from."""
    layer = model.layers[li]
    at = layer.attention
    qw = {n: quantize_linear(m) for n, m in
          (("Wq", at.Wq), ("Wk", at.Wk), ("Wv", at.Wv), ("Wo", at.Wo),
           ("W1", layer.ff[0]), ("W2", layer.ff[2]))}

    a1 = dyt_alpha(layer.norm1)
    a2 = dyt_alpha(layer.norm2)
    s = {k: cal.scale(li, k) for k in SITES}
    s["x"] = s_x
    s["x2"] = s_x_next                     # must equal the next layer's s_x
    if a1 is not None and dyt_fixed:
        s["x1"] = DYT_OUT_SCALE

    # --- the score scale is set by what SURVIVES, not by the widest score ----
    # S8 feeds `relu(S8 + mask)`, so a large *negative* score contributes
    # nothing: it clips to -128 and the ReLU takes it to exactly 0, which is the
    # correct answer. Calibrating on |S| would therefore spend most of the int8
    # range representing values that are then discarded. Pinning s_s to the
    # post-ReLU range is also what collapses RQ_P to an exact identity: the
    # multiplier comes out at 1.0, and requant with any m0 = 2**n is a no-op on
    # an integer accumulator.
    s["s"] = s["p"]

    dh = model.head_dim
    mult = {
        "Q":  s["x"] * qw["Wq"].scale / s["q"],
        "K":  s["x"] * qw["Wk"].scale / s["k"],
        # `tquant` re-rounds the int8 K and V onto {-1, 0, 1} so the MXU can
        # take them as weight operands, which is what moves both attention
        # matmuls off `vecmatmul`. From here on the attention scales are s_kt
        # and s_vt, not s_k and s_v.
        "KT": s["k"] / s["kt"],
        "V":  s["x"] * qw["Wv"].scale / s["v"],
        "VT": s["v"] / s["vt"],
        "S":  s["q"] * s["kt"] / (math.sqrt(dh) * s["s"]),
        "ID": 1.0,                          # mask add: narrow and clip only
        "P":  s["s"] / s["p"],
        "A":  s["p"] * s["vt"] / s["a"],
        # --- the three pinned words ----------------------------------------
        # `vecadd` adds two int8 operands, so a residual is only meaningful if
        # both sides already share a scale. That pins RQ_O and RQ_XO to s_x and
        # RQ_F to s_x1: those outputs do not get to choose their own scale, so
        # they are where clipping is expected and where it is measured.
        "O":  s["a"] * qw["Wo"].scale / s["x"],
        "XO": 1.0,
        # --- DyT, fused into the narrow that follows the residual add -------
        # X1 = hardtanh(alpha1 * (X + XO) * s_x). The multiplier carries alpha;
        # `dyt`'s +-127 clip *is* the hardtanh when s_x1 = 1/127.
        "X1": s["x"] * (a1 if a1 is not None else 1.0) / s["x1"],
        "H":  s["x1"] * qw["W1"].scale / s["h"],
        "HR": s["h"] / s["hr"],
        "F":  s["hr"] * qw["W2"].scale / s["x1"],
        "X2": s["x1"] * (a2 if a2 is not None else 1.0) / s["x2"],
    }
    words = {k: ((1, 0) if k in ("ID", "XO") else choose_word(v))
             for k, v in mult.items()}

    # Biases are added in int8 *after* the requant, not in int32 before it —
    # `vecadd` with a row stride of 0 is the broadcast the device has. The
    # bias-free configs (adder_ternary_vanilla) skip this entirely.
    out_scale = {"Wq": s["q"], "Wk": s["k"], "Wv": s["v"], "Wo": s["x"],
                 "W1": s["h"], "W2": s["x1"]}
    bias_i8 = {}
    for name, ql in qw.items():
        if ql.bias is None:
            bias_i8[name] = None
        else:
            bias_i8[name] = (torch.round(ql.bias / out_scale[name])
                             .clamp(INT8_MIN, INT8_MAX).to(torch.int64))

    return {"w": qw, "bias_i8": bias_i8, "words": words, "scales": s, "mult": mult,
            "dyt": (a1, a2)}


@torch.no_grad()
def prepare(model: transformer.Model, calib_tokens: torch.Tensor,
            dyt_fixed: bool = True, use_qat: bool | None = None) -> QuantModel:
    """Checkpoint + calibration set -> everything the device would be loaded with.

    ``use_qat`` picks where the activation scales come from: the checkpoint's
    learned LSQ scales (the default whenever it has them) or absmax over the
    calibration set. The calibration forward runs either way — the saturation
    and dynamic-range diagnostics need the tensors regardless.
    """
    model.eval()
    if use_qat is None:
        use_qat = has_qat_scales(model)
    cal = calibrate(model, calib_tokens)
    if use_qat:
        qcal = QATCalibration(model)
        qcal.m = cal.m                 # keep the measured absmax for the report
        qcal.a = cal.a                 # ...and the absmean accumulator
        cal = qcal
    L = len(model.layers)
    s_x_all = residual_scales(model, cal, dyt_fixed)

    qm = QuantModel(
        kv=model.kv_heads,
        g=model.q_heads // model.kv_heads,
        dh=model.head_dim,
        n_tokens=calib_tokens.shape[1],
    )
    for li in range(L):
        # Layer li's output scale IS layer li+1's input scale. The last layer
        # only feeds the head, which an argmax makes scale-invariant.
        if li + 1 < L:
            s_next = s_x_all[li + 1]
        else:
            a2 = dyt_alpha(model.layers[li].norm2)
            s_next = DYT_OUT_SCALE if (a2 is not None and dyt_fixed) else cal.scale(li, "x2")
        qm.layers.append(derive_layer(model, li, cal, s_x_all[li], s_next, dyt_fixed))

    qm.fc = quantize_linear(model.fc)
    s_out = qm.layers[-1]["scales"]["x2"]
    if qm.fc.bias is not None:
        # The head's output is int32 and nothing narrows it before the host's
        # argmax, so this bias can go into the accumulator exactly.
        qm.fc_bias_i32 = torch.round(qm.fc.bias / (s_out * qm.fc.scale)).to(torch.int64)
    qm.mask = causal_mask_i8(calib_tokens.shape[1])
    qm.s_x0 = s_x_all[0]
    qm.cal = cal
    return qm


# =============================================================================
# 4. The integer pipeline.
# =============================================================================
@torch.no_grad()
def quantize_input(model: transformer.Model, tokens: torch.Tensor,
                   s_x0: float) -> torch.Tensor:
    """embedding -> [B, T, D] int8, the device's input.

    There is no positional encoding any more (`Model.forward`), so the host's
    whole share of the front end is one embedding lookup.
    """
    X = model.embedding(tokens)
    return torch.round(X / s_x0).clamp(INT8_MIN, INT8_MAX).to(torch.int64)


def _linear(x: torch.Tensor, ql: QLinear, bias_i8: torch.Tensor | None,
            word: Word, stats: dict | None, tag: str) -> torch.Tensor:
    """One matmul + its requant: `matmul_t.rq` (ternary) or `vecmatmul` (int8)."""
    acc = imatmul(x, ql.w, what=tag)
    if stats is not None:
        stats.setdefault(tag, []).append(clip_rate(acc, word))
    out = requant8(acc, word)
    if bias_i8 is not None:
        out = requant8(out + bias_i8, (1, 0))
    return out


@torch.no_grad()
def int_forward(qm: QuantModel, X0: torch.Tensor,
                stats: dict | None = None) -> torch.Tensor:
    """`[B, T, D]` int8 in -> `[B, T, vocab]` int32 logits out.

    Every line below is one device dispatch, in order. Nothing here is float
    and nothing here is approximate: the only lossy steps are the requants,
    and they lose exactly what the hardware's requant loses.
    """
    B, T, _ = X0.shape
    kv, g, dh = qm.kv, qm.g, qm.dh
    X = X0
    mask = qm.mask[None, :, :, None, None]

    for li, L in enumerate(qm.layers):
        w, b, W = L["w"], L["bias_i8"], L["words"]

        Q = _linear(X, w["Wq"], b["Wq"], W["Q"], stats, "Q")
        K8 = _linear(X, w["Wk"], b["Wk"], W["K"], stats, "K")
        V8 = _linear(X, w["Wv"], b["Wv"], W["V"], stats, "V")

        # `tquant`: int8 -> {-1, 0, 1}, written 2 bits wide into the layout the
        # MXU reads a weight column from. Q and P stay int8 because they are the
        # *activation* operands; K and V become the weights, which is the whole
        # reason both attention matmuls can now run on the array.
        K = tquant8(K8, W["KT"])
        V = tquant8(V8, W["VT"])
        if stats is not None:
            stats.setdefault("KT", []).append(clip_rate(K8, W["KT"], -1, 1))
            stats.setdefault("VT", []).append(clip_rate(V8, W["VT"], -1, 1))

        Qh = Q.reshape(B, T, kv, g, dh)
        Kh = K.reshape(B, T, kv, dh)
        Vh = V.reshape(B, T, kv, dh)

        S32 = ieinsum("btkgh,bskh->btskg", Qh, Kh, terms=dh, what="QK")
        S8 = requant8(S32, W["S"])           # 1/sqrt(head_dim) folds in here
        S8 = requant8(S8 + mask, W["ID"])    # vecadd, then narrow: masked -> -128
        P8 = requant8(S8.clamp(min=0), W["P"])   # relu; RQ_P is the {1,0} identity
        A32 = ieinsum("btskg,bskh->btkgh", P8, Vh, terms=T, what="PV")
        A = requant8(A32, W["A"]).reshape(B, T, kv * g * dh)
        if stats is not None:
            stats.setdefault("A", []).append(clip_rate(A32, W["A"]))

        # A layer with DyT narrows its residuals with the device's `dyt`
        # (symmetric clip); one without falls back to a plain `requant`, which
        # is the same instruction with int8's asymmetric clip.
        a1, a2 = L["dyt"]
        norm1 = dyt8 if a1 is not None else requant8
        norm2 = dyt8 if a2 is not None else requant8
        lo1 = -INT8_MAX if a1 is not None else INT8_MIN
        lo2 = -INT8_MAX if a2 is not None else INT8_MIN

        O = _linear(A, w["Wo"], b["Wo"], W["O"], stats, "O")
        # MultiHeadAttention returns O + X, and Transformer adds X again. The
        # device adds two int8 operands at a time, so `2X + O` is two vecadds;
        # doing X+O first (rather than X+X) is what keeps the intermediate in
        # range, and RQ_XO is pinned to {1,0} because the second add needs both
        # operands back at s_x.
        XO = requant8(X + O, W["XO"])
        if stats is not None:
            stats.setdefault("XO", []).append(clip_rate(X + O, W["XO"]))
        X1 = norm1(XO + X, W["X1"])          # norm1: DyT, fused into the narrow
        if stats is not None:
            stats.setdefault("X1", []).append(clip_rate(XO + X, W["X1"], lo1))

        H = _linear(X1, w["W1"], b["W1"], W["H"], stats, "H")
        HR = requant8(H.clamp(min=0), W["HR"])
        F = _linear(HR, w["W2"], b["W2"], W["F"], stats, "F")
        X = norm2(X1 + F, W["X2"])           # norm2: DyT, fused into the narrow
        if stats is not None:
            stats.setdefault("X2", []).append(clip_rate(X1 + F, W["X2"], lo2))

    # The head is `matmul_t` against a ternary weight like everything above it,
    # and its output is never narrowed: an argmax does not care about scale, so
    # the device spills the int32 accumulator and the host reads it.
    logits = imatmul(X, qm.fc.w, what="fc")
    if qm.fc_bias_i32 is not None:
        logits = logits + qm.fc_bias_i32
    return logits


# =============================================================================
# Cross-check: the same scheme, in float.
# =============================================================================
def _fq(x: torch.Tensor, s: float, lo: int = INT8_MIN,
        hi: int = INT8_MAX) -> torch.Tensor:
    """Fake-quantize: round onto the grid of scale `s`, stay in float.

    `lo`/`hi` match the device op that would narrow here - [-128, 127] for
    `requant`, [-127, 127] for the symmetric `dyt`, [-1, 1] for `tquant`.
    """
    return torch.round(x / s).clamp(lo, hi) * s


@torch.no_grad()
def fake_quant_forward(model: transformer.Model, qm: QuantModel,
                       tokens: torch.Tensor) -> torch.Tensor:
    """The same quantization *scheme* as :func:`int_forward`, in float32.

    This is the check that separates "the scheme loses the model" from "the
    integer arithmetic has a bug". It clips at the same places and to the same
    scales, but it never leaves float: no requant word, no flooring shift, no
    int32 accumulator. If this scores like :func:`int_forward`, the fixed-point
    implementation is faithful and the loss is the quantization itself; if the
    two diverge, the integer path is the suspect.
    """
    B, T = tokens.shape
    kv, g, dh = qm.kv, qm.g, qm.dh
    dev = next(model.parameters()).device
    X = _fq(model.embedding(tokens), qm.s_x0)
    causal = torch.triu(torch.ones(T, T, device=dev) * -1e9, diagonal=1)

    for li, layer in enumerate(model.layers):
        s = qm.layers[li]["scales"]
        at = layer.attention
        Q = _fq(at.Wq(X), s["q"])
        # ...then the ternary narrow on top, exactly as int_forward does it.
        K = _fq(_fq(at.Wk(X), s["k"]), s["kt"], -1, 1)
        V = _fq(_fq(at.Wv(X), s["v"]), s["vt"], -1, 1)
        Qh = Q.reshape(B, T, kv, g, dh)
        Kh = K.reshape(B, T, kv, dh)
        Vh = V.reshape(B, T, kv, dh)
        S = _fq(torch.einsum("btkgh,bskh->btskg", Qh, Kh) / math.sqrt(dh), s["s"])
        P = _fq(torch.relu(S + causal[None, :, :, None, None]), s["p"])
        A = _fq(torch.einsum("btskg,bskh->btkgh", P, Vh).reshape(B, T, kv * g * dh), s["a"])
        a1, a2 = qm.layers[li]["dyt"]
        lo1 = -INT8_MAX if a1 is not None else INT8_MIN
        lo2 = -INT8_MAX if a2 is not None else INT8_MIN
        O = _fq(at.Wo(A), s["x"])            # pinned to the residual's scale
        XO = _fq(X + O, s["x"])              # pinned again by the second add
        X1 = _fq(layer.norm1(X + XO), s["x1"], lo1)
        H = _fq(layer.ff[0](X1), s["h"])
        HR = _fq(torch.relu(H), s["hr"])
        F = _fq(layer.ff[2](HR), s["x1"])    # pinned
        X = _fq(layer.norm2(X1 + F), s["x2"], lo2)
    return model.fc(X)


# =============================================================================
# Benchmark.
# =============================================================================
ANS = numbers_data.EQUALS_POS       # 15: '=' sits at 14, the answer starts here


def make_batch(n: int, seed: int, max_tokens: int, max_digits: int = 5):
    import random
    random.seed(seed)
    exprs, toks, masks = numbers_data.create_addition_batch(n, max_tokens,
                                                            max_digits=max_digits)
    return exprs, torch.tensor(toks), torch.stack(masks)


def predictions(logits: torch.Tensor) -> torch.Tensor:
    """The answer region, scored the way train.py trains it.

    The logit at position `t` predicts the token at `t+1`, and the answer
    occupies `[ANS, T)`, so the slice is `[ANS-1, T-1)`. (Note this is one
    position wider than `accel/tpulang/adder_export.py` scores — that script
    compares `logits[ANS:-1]` against `tokens[ANS+1:]`, which is internally
    consistent but drops the answer's leading digit from the metric.)
    """
    return logits.argmax(-1)[:, ANS - 1:-1]


def score(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[float, float]:
    """(exact-sequence, per-token) accuracy over the answer region."""
    hit = pred == tgt
    return float(hit.all(dim=1).double().mean()), float(hit.double().mean())


@dataclass
class Report:
    float_seq: float
    float_tok: float
    int_seq: float
    int_tok: float
    agreement: float
    clips: dict
    qm: QuantModel
    n_problems: int
    fq_seq: float | None = None      # the float cross-check, if it was run
    fq_tok: float | None = None


@torch.no_grad()
def benchmark(model: transformer.Model, n_problems: int = 256, n_calib: int = 128,
              max_tokens: int = 32, max_digits: int = 5, dyt_fixed: bool = True,
              eval_seed: int = 1234, calib_seed: int = 99,
              batch: int = 64, fake_quant: bool = False,
              use_qat: bool | None = None) -> Report:
    """Float baseline vs the integer pipeline, on the same problems.

    The calibration set is drawn from a different seed than the evaluation set,
    so the scales are never fitted to the problems they are scored on.
    """
    model.eval()
    _, tok_e, msk_e = make_batch(n_problems, eval_seed, max_tokens, max_digits)
    # No mask needed: MultiHeadAttention builds its own causal mask and ignores
    # the attn_mask it is handed, so calibration only needs the tokens.
    _, tok_c, _ = make_batch(n_calib, calib_seed, max_tokens, max_digits)
    tgt = tok_e[:, ANS:]

    # The float baseline has to be the *unquantized* network. On a QAT
    # checkpoint the ActQuant sites are live by default, so calling the model
    # directly would score the fake-quantized model and call it "float" —
    # flattering the comparison by moving the baseline, not the result.
    transformer.set_quant_enabled(model, False)
    try:
        pred_f = predictions(model(tok_e, msk_e))
        seq_f, tok_f = score(pred_f, tgt)
        qm = prepare(model, tok_c, dyt_fixed=dyt_fixed, use_qat=use_qat)
    finally:
        transformer.set_quant_enabled(model, True)

    stats: dict = {}
    preds = []
    for i in range(0, n_problems, batch):
        X0 = quantize_input(model, tok_e[i:i + batch], qm.s_x0)
        preds.append(predictions(int_forward(qm, X0, stats)))
    pred_i = torch.cat(preds)
    seq_i, tok_i = score(pred_i, tgt)

    fq_seq = fq_tok = None
    if fake_quant:
        fq_seq, fq_tok = score(predictions(fake_quant_forward(model, qm, tok_e)), tgt)

    return Report(
        float_seq=seq_f, float_tok=tok_f, int_seq=seq_i, int_tok=tok_i,
        agreement=float((pred_i == pred_f).all(dim=1).double().mean()),
        clips=stats, qm=qm, n_problems=n_problems, fq_seq=fq_seq, fq_tok=fq_tok,
    )


# =============================================================================
# Reporting.
# =============================================================================
def report_words(qm: QuantModel) -> None:
    L = len(qm.layers)
    print("\nrequant words (m0, n) per layer, and the multiplier each approximates:")
    print(f"  {'slot':4s} " + " ".join(f"{'L' + str(i):>18s}" for i in range(L)))
    for k in SLOTS:
        cells = [f"{lay['words'][k][0]:5d}/2^{lay['words'][k][1]:<2d}="
                 f"{lay['words'][k][0] / (1 << lay['words'][k][1]):7.4f}"
                 for lay in qm.layers]
        print(f"  {k:4s} " + " ".join(f"{c:>18s}" for c in cells))

    worst = []
    for k in SLOTS:
        errs = []
        for lay in qm.layers:
            m0, n = lay["words"][k]
            want = lay["mult"][k]
            errs.append(abs(m0 / (1 << n) - want) / want if want else 0.0)
        worst.append((max(errs), k))
    worst.sort(reverse=True)
    print(f"  worst word vs its exact multiplier: {worst[0][1]} at {worst[0][0] * 100:.3f}%"
          "  (a large value here means the word saturated, not that it rounded)")


def report_clips(rep: Report) -> None:
    """Saturation per requant site — the diagnostic that localizes a bad score.

    A high rate at O / XO / X1 / F means the *residual* path is losing data,
    which is the failure mode a per-tensor scale is prone to; a high rate at
    Q / K / V / H means the projection scale is too tight.
    """
    L = len(rep.qm.layers)
    print("\nsaturation rate at each requant (fraction of values clipped to +-127):")
    print(f"  {'site':5s} " + " ".join(f"{'L' + str(i):>9s}" for i in range(L)) + f" {'mean':>9s}")
    for k in ("Q", "K", "KT", "V", "VT", "A", "O", "XO", "X1", "H", "F", "X2"):
        v = rep.clips.get(k)
        if not v:
            continue
        per = torch.tensor(v, dtype=torch.float64).reshape(-1, L).mean(dim=0)
        print(f"  {k:5s} " + " ".join(f"{float(x) * 100:8.3f}%" for x in per) +
              f" {float(per.mean()) * 100:8.3f}%")


@torch.no_grad()
def report_dynamic_range(model: transformer.Model, tokens: torch.Tensor) -> None:
    """Within-tensor dynamic range — the statistic that predicts int8 failure.

    A per-tensor scale is pinned by the tensor's *maximum*, so the resolution
    every other element gets is set by how far the bulk sits below it. The
    statistic that matters is therefore `median/max`, not the usual outlier
    check against a high percentile: a tensor can have a perfectly well-behaved
    top (max/p99.9 ~ 1) and still lose most of its mass to zero.
    """
    keep: dict = {}
    instrumented_forward(model, tokens, keep=keep)
    print("\nwithin-tensor dynamic range (per-tensor int8 resolution):")
    print(f"  {'site':10s} {'|max|':>12s} {'median':>10s} {'med/max':>9s} {'%->0':>7s}")
    for (li, site) in sorted(keep):
        if site not in ("x", "p", "a", "o", "x1", "x2"):
            continue
        a = torch.cat(keep[(li, site)]).abs()
        if a.numel() == 0:
            continue
        mx, med = float(a.max()), float(a.median())
        if mx == 0:
            continue
        z = float((a < (mx / 127.0) / 2).double().mean()) * 100
        print(f"  L{li} {site:7s} {mx:12.4f} {med:10.4f} {med / mx:9.5f} {z:6.1f}%")
    print("  -> a value below half a quantization step becomes exactly 0.")


ARCHS = {
    "vanilla": transformer.adder_vanilla,
    "gqa": transformer.adder_gqa,
    "ternary_vanilla": transformer.adder_ternary_vanilla,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="ternary_vanilla")
    ap.add_argument("--model-path", default="model/saved/colab_ternary_mha_small_dyt.pt")
    ap.add_argument("-n", "--problems", type=int, default=256)
    ap.add_argument("-c", "--calib", type=int, default=128,
                    help="problems in the calibration set (disjoint from the eval set)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-digits", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dyt-scale", choices=["fixed", "calibrated"], default="fixed",
                    help="fixed pins a DyT output to 1/127, so the requant's clip IS "
                         "the hardtanh; calibrated uses absmax/127 instead, which gives "
                         "finer resolution when DyT never saturates but can clip early")
    ap.add_argument("--scales", choices=["auto", "qat", "calib"], default="auto",
                    help="where activation scales come from: the checkpoint's learned "
                         "LSQ scales, absmax over the calibration set, or auto (qat "
                         "whenever the checkpoint has them)")
    ap.add_argument("--words", action="store_true", help="print the requant words")
    ap.add_argument("--dynamic-range", action="store_true",
                    help="print the median/max diagnostic per site")
    ap.add_argument("--check", action="store_true",
                    help="verify the instrumented forward matches Model.forward")
    ap.add_argument("--fake-quant", action="store_true",
                    help="also run the same scheme in float, as a cross-check that "
                         "the integer arithmetic is faithful")
    args = ap.parse_args(argv)

    model = ARCHS[args.arch]()
    state = torch.load(args.model_path, map_location="cpu")
    # strict=False: checkpoints predating the act_scale buffers list them as
    # missing; they are unused here (this module derives every scale itself).
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    model.eval()

    print(f"arch {args.arch}  checkpoint {args.model_path}")
    print(f"  d={model.d} f={model.f} layers={len(model.layers)} "
          f"q_heads={model.q_heads} kv_heads={model.kv_heads} head_dim={model.head_dim}")
    if missing:
        print(f"  {len(missing)} buffers missing from the checkpoint (act_scale; unused here)")

    if args.check:
        _, tok, msk = make_batch(8, 7, args.max_tokens, args.max_digits)
        # instrumented_forward is the *float* forward written inline, so the
        # comparison is only meaningful with the QAT sites off.
        transformer.set_quant_enabled(model, False)
        try:
            d = check_instrumented(model, tok, msk)
        finally:
            transformer.set_quant_enabled(model, True)
        print(f"  instrumented forward vs Model.forward: max|diff| = {d:.3e}")
        if d > 1e-3:
            print("  !! they disagree; calibration would describe a different network")
            return 1

    rep = benchmark(model, n_problems=args.problems, n_calib=args.calib,
                    max_tokens=args.max_tokens, max_digits=args.max_digits,
                    dyt_fixed=(args.dyt_scale == "fixed"), batch=args.batch,
                    fake_quant=args.fake_quant,
                    use_qat={"auto": None, "qat": True, "calib": False}[args.scales])

    if args.words:
        report_words(rep.qm)
    if args.dynamic_range:
        _, tok_c, _ = make_batch(32, 99, args.max_tokens, args.max_digits)
        report_dynamic_range(model, tok_c)
    report_clips(rep)

    print(f"\naccuracy over {rep.n_problems} problems (T={args.max_tokens}, "
          f"answer region [{ANS}, {args.max_tokens})):")
    print(f"  {'':22s} {'exact-sequence':>15s} {'token':>9s}")
    print(f"  {'float (quant off)':22s} {rep.float_seq * 100:14.2f}% {rep.float_tok * 100:8.2f}%")
    print(f"  {'int8 activations':22s} {rep.int_seq * 100:14.2f}% {rep.int_tok * 100:8.2f}%")
    if has_qat_scales(model):
        # On a QAT checkpoint the float row is not an upper bound and a drop
        # from it is not a loss: the quantizers were present during training, so
        # the clipping is part of the function that was fitted. Removing them
        # gives a network the weights were never trained for.
        print("  (QAT checkpoint: the quantizers are part of the trained model, so the "
              "float\n   row is a different network, not a ceiling — compare against "
              "the cross-check.)")
    if rep.fq_seq is not None:
        print(f"  {'  same scheme, float':22s} {rep.fq_seq * 100:14.2f}% "
              f"{rep.fq_tok * 100:8.2f}%   <- cross-check")
    print(f"  the float and integer models agree on {rep.agreement * 100:.2f}% of sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
