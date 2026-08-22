import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.cpp_extension import load

import model.numbers_data as numbers_data

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")



# Symmetric int4 grid. -8 is representable, but no scale is ever derived from
# it: every scale here is absmax/qmax with qmax = 7, which is what ActQuant's
# absmax init already does and what keeps the grid symmetric about zero.
INT4_QMIN, INT4_QMAX = -8, 7


class RoundClip(torch.autograd.Function):
    """``round(x).clip(qmin, qmax)`` with a straight-through-estimator backward.

    The STE passes gradient only where the input was inside the clip, so a
    weight driven off the grid stops receiving it. ``(-1, 1)`` is
    :class:`TernaryLinear`'s grid, ``(-8, 7)`` is :class:`Int4Linear`'s.
    """

    @staticmethod
    def forward(input, qmin, qmax):
        return input.round().clip(qmin, qmax)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, qmin, qmax = inputs
        ctx.save_for_backward(input)
        ctx.qmin, ctx.qmax = qmin, qmax

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        inside = (input >= ctx.qmin) & (input <= ctx.qmax)
        return grad_output * inside, None, None


class FakeQuant(torch.autograd.Function):
    """Symmetric per-tensor fake quantization: ``round(x/s).clip(qmin,qmax) * s``.

    Returns the *dequantized* value so the surrounding math stays in real units;
    only the representable grid changes. ``qmin`` is -8 for a ``requant`` site
    and -7 for a ``dyt`` site (the hardtanh clip is symmetric because hardtanh
    is odd).

    Backward is LSQ (arxiv 1902.08153): straight-through for the input, and the
    analytic derivative w.r.t. the scalar step size, summed over the tensor
    because one ``s`` broadcasts over every element. ``grad_scale`` is the LSQ
    gradient-magnitude correction ``1/sqrt(numel * qmax)``.
    """

    @staticmethod
    def forward(input, scale, qmin, qmax, grad_scale):
        return (input / scale).round().clamp(qmin, qmax) * scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale, qmin, qmax, grad_scale = inputs
        ctx.save_for_backward(input, scale)
        ctx.qmin, ctx.qmax, ctx.grad_scale = qmin, qmax, grad_scale

    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        q = input / scale
        inside = (q >= ctx.qmin) & (q <= ctx.qmax)
        grad_input = grad_output * inside

        q_clipped = q.clamp(ctx.qmin, ctx.qmax)
        # d(round(q)*s)/ds = round(q) - q inside the range, = qmin/qmax outside.
        local = torch.where(inside, q_clipped.round() - q, q_clipped)
        grad_scale = (grad_output * local).sum() * ctx.grad_scale
        return grad_input, grad_scale.reshape(scale.shape), None, None, None


def dynamic_fake_quant(x, qmin=-128, qmax=127):
    """Per-tensor int8 fake-quant with a scale taken from the current absmax.

    For weights the hardware stores as plain int8, where the host derives the
    scale at export time rather than learning it. `adder_int4_vanilla` has none:
    the output head is an :class:`Int4Linear` too, so every weight in it is int4
    and this path is reached only by the float `adder_vanilla` / `adder_gqa`
    configs (see `Model.forward`).
    """
    scale = (x.detach().abs().max() / qmax).clamp_min(1e-8)
    return FakeQuant.apply(x, scale, qmin, qmax, 0.0)


class ActQuant(nn.Module):
    """One activation quantization site — one entry in the requant table.

    A site owns a per-tensor scale. Sites that the hardware *pins* to another
    tensor's scale (the residual adds, the identity requants) are expressed by
    sharing the same ``ActQuant`` instance, or by pinning ``fixed_scale``.

    ``fixed_scale=None`` learns the scale (LSQ), initialized from the absmax of
    the first tensor that flows through. DyT outputs are pinned to ``1/7``
    instead: ``hardtanh`` bounds them to ``[-1, 1]`` analytically, so no
    calibration set is involved (``adder_kernel.md`` §4).

    The default grid is symmetric **int4** (``[-8, 7]``). ``init="absmean"``
    seeds from the absmean instead of the absmax, which is what a grid with
    very few levels wants — an absmax seed rounds most of the tensor to zero
    and leaves LSQ climbing out of a dead gradient.
    """

    def __init__(self, qmin=INT4_QMIN, qmax=INT4_QMAX, fixed_scale=None,
                 init="absmax"):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.init = init
        self.enabled = True

        if fixed_scale is None:
            self.scale = nn.Parameter(torch.ones(()))
            self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        else:
            self.register_buffer("scale", torch.tensor(float(fixed_scale)))
            self.register_buffer("initialized", torch.ones((), dtype=torch.bool))

    def init_scale_from(self, x):
        """Seed the scale from ``x``'s absmax, once. No-op if already seeded.

        Exposed so a site can be seeded from a tensor other than the one it
        quantizes — ``RQ_S`` is seeded from the *post-mask, post-ReLU* scores,
        because a large negative score clips to -8 and ReLU takes it to
        exactly zero, so spending range on it is pure waste.
        """
        if bool(self.initialized):
            return
        with torch.no_grad():
            a = x.detach().abs()
            seed = a.mean() if self.init == "absmean" else a.max() / self.qmax
            self.scale.copy_(seed.clamp_min(1e-8))
            self.initialized.fill_(True)

    def forward(self, x):
        if not self.enabled:
            return x

        self.init_scale_from(x)

        scale = self.scale.abs().clamp_min(1e-8)
        grad_scale = 1.0 / math.sqrt(x.numel() * self.qmax)
        return FakeQuant.apply(x, scale, self.qmin, self.qmax, grad_scale)

    def extra_repr(self):
        kind = "learned" if isinstance(self.scale, nn.Parameter) else "pinned"
        return f"qmin={self.qmin}, qmax={self.qmax}, {kind}"


def set_quant_enabled(module, enabled):
    """Turn every quantization site on/off (float baseline vs. QAT)."""
    for m in module.modules():
        if isinstance(m, ActQuant):
            m.enabled = enabled
        elif isinstance(m, Model):
            m.quantize_head = enabled

class TernaryLinear(nn.Module):
    """Linear layer with 1.58-bit (ternary {-1, 0, 1}) weights, BitNet-style.

    **No config builds one any more** — `adder_int4_vanilla` uses
    :class:`Int4Linear`. It is kept because `model/quant.py` and the `accel/`
    export path still model a ternary weight, and both import it.

    Weights are scaled by their absmean and rounded/clipped to {-1, 0, 1}.
    RoundClip provides a straight-through estimator for the backward pass.

    Activations feeding the matmul can optionally be quantized to symmetric
    int8 with a single per-tensor scalar. The scalar is populated offline by
    ``calibrate_activations`` (absmax over a calibration set, ``scale =
    absmax / 127``) and stored in ``act_scale``. An uncalibrated layer keeps
    ``act_scale`` at NaN and passes activations through untouched, so existing
    checkpoints load and run exactly as before.
    """

    def __init__(self, in_dim, out_dim, bias=True, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

        # Per-tensor symmetric int8 activation scale. NaN => not calibrated.
        self.register_buffer("act_scale", torch.tensor(float("nan")))
        # Running absmax accumulated while calibrating (not persisted).
        self.register_buffer("_act_absmax", torch.zeros(()), persistent=False)
        self.calibrating = False

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Activations are quantized by the ActQuant site that *produced* x, not
        # here: on the TPU every input to a matmul is already an int8 tensor
        # left behind by the previous op's requant.
        scale = self.w.abs().mean() + self.eps
        w_quant = RoundClip.apply(self.w / scale, -1, 1) * self.w.abs().mean()
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


class Int4Linear(nn.Module):
    """Linear layer with symmetric per-tensor int4 weights.

    The int4 analogue of :class:`TernaryLinear`: one scale derived from the
    weight itself, :class:`RoundClip` giving the straight-through backward, no
    extra parameter and no calibration. Two differences from the ternary case:

    - the scale is ``absmax / 7``, not the absmean. Absmean is BitNet's rule for
      a grid with one nonzero level per sign; on 15 levels it puts most of the
      grid past the end of the weight distribution.
    - the scale is **detached**. ``absmax`` has a gradient that lands entirely
      on one element, which is a spike, not a signal; ``TernaryLinear`` gets
      away with a live scale because an absmean averages over the whole tensor.

    The weight is ``[in_dim, out_dim]`` with ``x @ w`` — the *transpose* of
    ``nn.Linear``, same as ``TernaryLinear``, so checkpoints are not
    interchangeable with a float config.
    """

    def __init__(self, in_dim, out_dim, bias=True, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Activations are quantized by the ActQuant site that *produced* x, not
        # here — every input to a matmul is already an int4 tensor left behind
        # by the previous op's requant.
        scale = (self.w.detach().abs().max() / INT4_QMAX).clamp_min(self.eps)
        w_quant = RoundClip.apply(self.w / scale, INT4_QMIN, INT4_QMAX) * scale
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(in_dim, out_dim, use_int4=False, bias=True):
    if use_int4:
        return Int4Linear(in_dim, out_dim, bias=bias)
    return nn.Linear(in_dim, out_dim, bias=bias)


# https://arxiv.org/pdf/2503.10622
# try using no gamma & beta at first
class DyT(nn.Module):
    def __init__(self, C, init_a=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_a)

    def forward(self, x):
        return F.hardtanh(x * self.alpha, min_val=-1, max_val=1)

class MultiHeadAttention(nn.Module):
    """Quantization sites, in ``adder_kernel.md`` §4 block order:

    ``RQ_Q``/``RQ_K``/``RQ_V`` (separate — each projection has its own scale),
    ``RQ_S`` (with ``1/sqrt(head_dim)`` folded in, so quantize *after* the
    divide), ``RQ_P`` and ``RQ_XO`` are the ``{1,0}`` identities and so share
    their producer's site, and ``RQ_A``. ``RQ_O`` is pinned to ``s_x`` because
    ``vecadd`` takes both residual operands at one scale — that is the site the
    clipping shows up at, so it deliberately keeps the *tight* ``s_x`` rather
    than widening to cover ``O``.

    Every tensor here is int4, K and V included. The old ``RQ_KT``/``RQ_VT``
    pair — a second narrow rounding K and V onto ``{-1, 0, 1}`` — is gone: it
    existed only because the MXU could not multiply an int8 activation by an
    int8 weight, so the operand landing on the weight side had to be a trit. At
    equal width there is nothing left for it to do but round twice.
    """

    def __init__(self, d, q_heads, kv_heads, head_dim, x_quant,
                 use_int4=False, use_bias=True):
        super().__init__()

        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d = d
        self.heads_per_q = q_heads // kv_heads
        self.head_dim = head_dim

        self.Wq = make_linear(d, q_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wk = make_linear(d, kv_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wv = make_linear(d, kv_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wo = make_linear(q_heads * self.head_dim, d, use_int4, bias=use_bias)

        self.q_q = ActQuant()   # RQ_Q
        self.q_k = ActQuant()   # RQ_K
        self.q_v = ActQuant()   # RQ_V
        # RQ_S: calibrated on the scores that *survive* the mask+ReLU, which is
        # what makes RQ_P the {1,0} identity. Learning the scale finds that on
        # its own: clipped-away negatives contribute nothing to the loss.
        self.q_s = ActQuant()
        self.q_p = self.q_s     # RQ_P  {1,0}: s_p = s_s
        self.q_a = ActQuant()   # RQ_A
        self.q_o = x_quant      # RQ_O  pinned to s_x
        self.q_xo = x_quant     # RQ_XO {1,0}: stays on s_x for the second add

    def forward(self, X, attn_mask, use_custom_attention=False):
        batch_size = X.shape[0]
        n_tokens = X.shape[1]

        Q = self.q_q(self.Wq(X))
        K = self.q_k(self.Wk(X))
        V = self.q_v(self.Wv(X))

        Q = torch.reshape(
            Q, [batch_size, n_tokens, self.kv_heads, self.heads_per_q, self.head_dim]
        )
        K = torch.reshape(K, [batch_size, n_tokens, self.kv_heads, self.head_dim])
        V = torch.reshape(V, [batch_size, n_tokens, self.kv_heads, self.head_dim])


        mask = torch.triu(torch.ones([n_tokens, n_tokens]) * -1e9, diagonal=1).reshape(
            [1, n_tokens, n_tokens, 1, 1]
        )
        mask = mask.to(device)

        attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(self.head_dim)
        self.q_s.init_scale_from(F.relu(attention_scores + mask))
        attention_scores = self.q_s(attention_scores)

        # The mask needs no scale: a masked entry is negative whatever s_s is,
        # so ReLU takes it to exactly zero. Exact, not a tolerance.
        attention_scores = F.relu(attention_scores + mask)  # btskg revert to softmax if training bad
        attention_scores = self.q_p(attention_scores)

        A = torch.einsum("btskg,bskh->btkgh", attention_scores, V).reshape(
            [batch_size, n_tokens, self.q_heads * self.head_dim]
        )
        A = self.q_a(A)

        O = self.q_o(self.Wo(A))
        return self.q_xo(O + X)

class Transformer(nn.Module):
    def __init__(
        self,
        d,
        f,
        q_heads,
        kv_heads,
        head_dim,
        x_quant,
        use_int4=False,
        use_bias=True,
    ):
        super().__init__()

        self.use_int4 = use_int4

        self.attention = MultiHeadAttention(
            d, q_heads, kv_heads, head_dim, x_quant,
            use_int4=use_int4, use_bias=use_bias,
        )
        self.norm1 = DyT(d)

        self.ff = nn.Sequential(
            make_linear(d, f, use_int4, bias=use_bias),
            nn.ReLU(),
            make_linear(f, d, use_int4, bias=use_bias),
        )

        self.norm2 = DyT(d)
        self.dropout = nn.Dropout(p=0.1)

        # RQ_X1 / RQ_X2 are `dyt` ops: hardtanh bounds them to [-1, 1], so their
        # scale is analytic (1/7) rather than calibrated, and the clip is
        # symmetric at -7 because hardtanh is odd.
        self.q_x1 = ActQuant(qmin=-INT4_QMAX, fixed_scale=1.0 / INT4_QMAX)
        self.q_h = ActQuant()                       # RQ_H
        self.q_hr = self.q_h                        # RQ_HR {1,0}: s_hr = s_h
        # RQ_F is pinned to s_x1 by the second residual add, but it is a
        # `requant`, so it clips at -8 rather than -7.
        self.q_f = ActQuant(fixed_scale=1.0 / INT4_QMAX)
        self.q_x2 = ActQuant(qmin=-INT4_QMAX, fixed_scale=1.0 / INT4_QMAX)

    def forward(self, X, attn_mask, use_custom_attention=False):
        # The attention residual is 2X + O: MultiHeadAttention already returned
        # O + X (at s_x), and X is added again here before dyt.
        X = self.q_x1(
            self.norm1(X + self.dropout(self.attention(X, attn_mask, use_custom_attention)))
        )
        # X = self.dropout(self.attention(X, attn_mask, use_custom_attention))

        H = self.q_h(self.ff[0](X))
        HR = self.q_hr(self.ff[1](H))
        Fout = self.q_f(self.ff[2](HR))
        return self.q_x2(self.norm2(X + self.dropout(Fout)))


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d=128,
        f=256,
        layers=8,
        q_heads=8,
        kv_heads=8,
        head_dim=None,
        use_int4=False,
        use_bias=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = d

        if head_dim == None:
            head_dim = d // q_heads

        self.d = d
        self.f = f
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, d)

        # The s_x chain. Layer 0 enters from the embedding and needs a
        # calibrated scale; every later layer enters on the previous
        # layer's DyT output, which is analytically 1/7, so its s_x *is* that
        # site. Sharing the module is what makes RQ_O / RQ_XO land on s_x.
        self.q_embed = ActQuant()

        blocks = []
        x_quant = self.q_embed
        for i in range(layers):
            block = Transformer(
                self.d,
                self.f,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                x_quant,
                use_int4=use_int4,
                use_bias=use_bias,
            )
            blocks.append(block)
            x_quant = block.q_x2
        self.layers = nn.ModuleList(blocks)
        # The output head is int4 like every other projection.
        self.fc = make_linear(self.d, self.vocab_size, use_int4, bias=use_bias)
        self.quantize_head = True
        self.dropout = nn.Dropout(p=0.1)

    # There is no positional encoding. The model's only position signal is the
    # causal mask: token t attends over t+1 keys, and attention here is ReLU
    # with no normalization over the source axis, so the *magnitude* of the
    # attention output carries the count. Removing the sinusoidal PE also
    # removes the one unbounded contribution to layer 0's input, which is why
    # `s_x0` is now the embedding's scale alone.

    def forward(self, inputs, attn_mask, use_custom_attention=False):
        X = self.q_embed(self.dropout(self.embedding(inputs)))
        for layer in self.layers:
            X = layer(X, attn_mask, use_custom_attention=use_custom_attention)

        # The head's output scale is irrelevant to an argmax, so the logits are
        # never requantized: the device spills the raw int32 accumulator.
        #
        # An int4 head carries its own weight quantization (Int4Linear's
        # absmax + RoundClip), exactly as the projections do, so there is
        # nothing here for `quantize_head` to switch — it and
        # `dynamic_fake_quant` now only reach the int8 `nn.Linear` head that a
        # float config still gets.
        if isinstance(self.fc, (TernaryLinear, Int4Linear)):
            return self.fc(X)
        w = self.fc.weight
        if self.quantize_head:
            w = dynamic_fake_quant(w)
        return F.linear(X, w, self.fc.bias)

    def sample_pred(self, logits):
        dist = Categorical(logits=logits)
        return dist.sample()

    def sample_pred_best(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)


def adder_vanilla():
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=4,
        q_heads=4,
        kv_heads=4,
    )
    return model


def adder_gqa():
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=6,
        q_heads=8,
        kv_heads=2,
    )
    return model


def adder_int4_vanilla():
    # int4 weights and int4 activations throughout, same QAT scheme as the
    # ternary config it replaces: every requant site is an LSQ-learned
    # ActQuant, the residual/identity sites share instances, the DyT sites are
    # pinned to 1/7. head_dim = d // q_heads = 16.
    #
    # use_bias=False: the TPU's VPU has no row-broadcast operand, so a [N] bias
    # over [T, N] costs a T-iteration vecadd/requant loop per linear (~5 per
    # layer). Dropping the biases keeps a hand-written tpulang layer
    # straight-line. See accel/tpulang/tpunn.md.
    model = Model(
        len(numbers_data.VOCAB),
        d=64,
        f=256,
        layers=4,
        q_heads=4,
        kv_heads=4,
        use_int4=True,
        use_bias=False,
    )
    return model
