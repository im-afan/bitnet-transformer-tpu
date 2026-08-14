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



def mha_torch(Q, K, V):
    batch_size = Q.shape[0]
    n_tokens = Q.shape[1]
    q_heads = Q.shape[2] * Q.shape[3]
    head_dim = Q.shape[4]

    mask = torch.triu(torch.ones([n_tokens, n_tokens]) * -1e9, diagonal=1).reshape(
        [1, n_tokens, n_tokens, 1, 1]
    )
    mask = mask.to(device)

    attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(head_dim)

    attention_scores = F.relu(attention_scores + mask)  # btskg revert to softmax if training bad
    A = torch.einsum("btskg,bskh->btkgh", attention_scores, V).reshape(
        [batch_size, n_tokens, q_heads * head_dim]
    )

    return A


class RoundClip(torch.autograd.Function):
    @staticmethod
    def forward(input):
        return input.round().clip(-1, 1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (input,) = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * (input.abs() <= 1)


class TernaryLinear(nn.Module):
    """Linear layer with 1.58-bit (ternary {-1, 0, 1}) weights, BitNet-style.

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

    def quantize_activations(self, x):
        """Fake-quantize activations to symmetric per-tensor int8.

        During calibration, tracks the running absmax instead of quantizing so a
        single scalar can be derived once the pass finishes.
        """
        if self.calibrating:
            self._act_absmax = torch.maximum(
                self._act_absmax, x.detach().abs().max()
            )
            return x
        if torch.isnan(self.act_scale):
            return x
        x_int = torch.round(x / self.act_scale).clamp_(-127, 127)
        return x_int * self.act_scale

    def forward(self, x):
        x = self.quantize_activations(x)
        scale = self.w.abs().mean() + self.eps
        w_quant = RoundClip.apply(self.w / scale) * self.w.abs().mean()
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(in_dim, out_dim, use_ternary=False, bias=True):
    if use_ternary:
        return TernaryLinear(in_dim, out_dim, bias=bias)
    return nn.Linear(in_dim, out_dim, bias=bias)


# https://arxiv.org/pdf/2503.10622
# try using no gamma & beta at first
class DyT(nn.Module):
    def __init__(self, C, init_a=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_a)

    def forward(self, x):
        return F.hardtanh(x * self.alpha, min_val=-1, max_val=1)


@torch.no_grad()
def calibrate_activations(model, run_forward):
    """Calibrate per-tensor symmetric int8 activation scales for a ternary model.

    ``run_forward(model)`` should push the whole calibration set through the model
    (looping over batches internally, with whatever masks/args the model needs)
    while every ``TernaryLinear`` records the absmax of its input activations across
    all those forward passes. Afterwards each
    layer's ``act_scale`` is set to ``absmax / 127`` — one scalar per layer, so the
    activation entering that matmul quantizes to symmetric int8 [-127, 127].

    Returns a dict mapping module name -> calibrated scale.
    """
    layers = [m for m in model.modules() if isinstance(m, TernaryLinear)]
    if not layers:
        raise ValueError("model has no TernaryLinear layers to calibrate")

    was_training = model.training
    model.eval()  # disable dropout so calibration matches inference
    for layer in layers:
        layer.calibrating = True
        layer._act_absmax.zero_()
        # Ignore any previously calibrated scale while recording absmax.
        layer.act_scale = torch.tensor(float("nan"), device=layer.act_scale.device)

    try:
        run_forward(model)
    finally:
        for layer in layers:
            layer.calibrating = False
        if was_training:
            model.train()

    scales = {}
    name_of = {m: n for n, m in model.named_modules()}
    for layer in layers:
        # A zero absmax (activations never nonzero) would make an unusable 0
        # scale; leave such a layer uncalibrated (NaN => passthrough) instead.
        if layer._act_absmax > 0:
            layer.act_scale = layer._act_absmax / 127.0
        scales[name_of[layer]] = layer.act_scale.item()
    return scales


class MultiHeadAttention(nn.Module):
    def __init__(self, d, q_heads, kv_heads, head_dim, use_ternary=False, use_bias=True):
        super().__init__()

        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d = d
        self.heads_per_q = q_heads // kv_heads
        self.head_dim = head_dim

        self.Wq = make_linear(d, q_heads * self.head_dim, use_ternary, bias=use_bias)
        self.Wk = make_linear(d, kv_heads * self.head_dim, use_ternary, bias=use_bias)
        self.Wv = make_linear(d, kv_heads * self.head_dim, use_ternary, bias=use_bias)
        self.Wo = make_linear(q_heads * self.head_dim, d, use_ternary, bias=use_bias)

    def forward(self, X, attn_mask, use_custom_attention=False):
        batch_size = X.shape[0]
        n_tokens = X.shape[1]

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        Q = torch.reshape(
            Q, [batch_size, n_tokens, self.kv_heads, self.heads_per_q, self.head_dim]
        )
        K = torch.reshape(K, [batch_size, n_tokens, self.kv_heads, self.head_dim])
        V = torch.reshape(V, [batch_size, n_tokens, self.kv_heads, self.head_dim])

        A = mha_torch(Q, K, V)

        O = self.Wo(A)
        return O + X

class Transformer(nn.Module):
    def __init__(
        self,
        d,
        f,
        q_heads,
        kv_heads,
        head_dim,
        use_ternary=False,
        use_bias=True,
    ):
        super().__init__()

        self.use_ternary = use_ternary

        self.attention = MultiHeadAttention(
            d, q_heads, kv_heads, head_dim, use_ternary=use_ternary, use_bias=use_bias
        )
        self.norm1 = DyT(d)


        self.ff = nn.Sequential(
            make_linear(d, f, use_ternary, bias=use_bias),
            nn.ReLU(),
            make_linear(f, d, use_ternary, bias=use_bias),
        )

        self.norm2 = DyT(d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, attn_mask, use_custom_attention=False):
        X = self.norm1(
            X + self.dropout(self.attention(X, attn_mask, use_custom_attention))
        )
        # X = self.dropout(self.attention(X, attn_mask, use_custom_attention))
        return self.norm2(X + self.dropout(self.ff(X)))


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
        use_ternary=False,
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
        self.layers = nn.ModuleList(
            [
                Transformer(
                    self.d,
                    self.f,
                    self.q_heads,
                    self.kv_heads,
                    self.head_dim,
                    use_ternary=use_ternary,
                    use_bias=use_bias,
                )
                for i in range(layers)
            ]
        )
        self.fc = make_linear(self.d, self.vocab_size, bias=use_bias)
        self.dropout = nn.Dropout(p=0.1)

    def positional_encoding(self, n_tokens):
        device = next(self.parameters()).device
        div = torch.exp(torch.arange(0, self.d, 2) * math.log(1 / 10000) * 1 / self.d)
        pos_tensor = torch.arange(n_tokens).unsqueeze(1)

        pe = torch.zeros((1, n_tokens, self.d))
        pe[:, :, 0::2] = torch.sin(pos_tensor * div)
        pe[:, :, 1::2] = torch.cos(pos_tensor * div)

        return pe.to(device)

    def forward(self, inputs, attn_mask, use_custom_attention=False):
        pe = self.positional_encoding(inputs.shape[1])
        X = self.dropout(self.embedding(inputs) + pe)
        for layer in self.layers:
            X = layer(X, attn_mask, use_custom_attention=use_custom_attention)

        return self.fc(X)

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


def adder_ternary_vanilla():
    # use_bias=False: the TPU's VPU has no row-broadcast operand, so a [N] bias
    # over [T, N] costs a T-iteration vecadd/requant loop per linear (~5 per
    # layer). Dropping the biases keeps a hand-written tpulang layer
    # straight-line. See accel/tpulang/tpunn.md.
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=128,
        layers=4,
        q_heads=4,
        kv_heads=4,
        use_ternary=True,
        use_bias=False,
    )
    return model
