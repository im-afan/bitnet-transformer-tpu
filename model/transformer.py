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

# CUDA reference accelerator now lives under accel/cuda/. Resolve its sources
# relative to the repo root so the extension loads regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUDA_DIR = os.path.join(_REPO_ROOT, "accel", "cuda")
mha_cuda = load(
    name="mha_cuda",
    sources=[
        os.path.join(_CUDA_DIR, "kernels.cu"),
        os.path.join(_CUDA_DIR, "bindings.cpp"),
    ],
    verbose=True,
)


def slow_mha_cuda(Q, K, V):
    batch_size = Q.shape[0]
    n_tokens = Q.shape[1]
    q_heads = Q.shape[2] * Q.shape[3]
    head_dim = Q.shape[4]

    out, _ = mha_cuda.mha_custom(Q, K, V)
    out = out.reshape([batch_size, n_tokens, q_heads * head_dim])
    return out


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

    attention_scores = F.softmax(attention_scores + mask, dim=2)  # btskg
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
        return grad_output * (1 - torch.tanh(input) ** 2)


class TernaryLinear(nn.Module):
    """Linear layer with 1.58-bit (ternary {-1, 0, 1}) weights, BitNet-style.

    Weights are scaled by their absmean and rounded/clipped to {-1, 0, 1}.
    RoundClip provides a straight-through estimator for the backward pass.
    """

    def __init__(self, in_dim, out_dim, bias=True, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.normal_(self.w, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        scale = self.w.abs().mean() + self.eps
        w_quant = RoundClip.apply(self.w / scale)
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(in_dim, out_dim, use_ternary=False, bias=True):
    if use_ternary:
        return TernaryLinear(in_dim, out_dim, bias=bias)
    return nn.Linear(in_dim, out_dim, bias=bias)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, q_heads, kv_heads, head_dim, use_ternary=False):
        super().__init__()

        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d = d
        self.heads_per_q = q_heads // kv_heads
        self.head_dim = head_dim

        self.Wq = make_linear(d, q_heads * self.head_dim, use_ternary)
        self.Wk = make_linear(d, kv_heads * self.head_dim, use_ternary)
        self.Wv = make_linear(d, kv_heads * self.head_dim, use_ternary)
        self.Wo = make_linear(q_heads * self.head_dim, d, use_ternary)

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

        A = None
        if use_custom_attention:
            A = slow_mha_cuda(Q, K, V)
        else:
            A = mha_torch(Q, K, V)

        O = self.Wo(A)
        return O + X


class Expert(nn.Module):
    def __init__(self, d, f, use_ternary=False):
        super().__init__()
        self.fc1 = make_linear(d, f, use_ternary)
        self.fc2 = make_linear(f, d, use_ternary)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, d, f, n_experts=8, top_k=2, use_ternary=False):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Gate stays full-precision; quantizing the router hurts routing quality.
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d, f, use_ternary) for _ in range(n_experts)]
        )

    def forward(self, X):
        # X: (B, T, d)
        B, T, d = X.shape
        X_flat = X.view(B * T, d)  # (B*T, d)

        logits = self.gate(X_flat)  # (B*T, E)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # (B*T, k)
        weights = F.softmax(weights, dim=-1)  # (B*T, k)

        out = torch.zeros_like(X_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (B*T,)
            w = weights[:, k].unsqueeze(1)  # (B*T, 1)
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](X_flat[mask])

        return out.view(B, T, d)


class Transformer(nn.Module):
    def __init__(
        self,
        d,
        f,
        q_heads,
        kv_heads,
        head_dim,
        n_experts=8,
        top_k=2,
        use_moe=False,
        use_ternary=False,
    ):
        super().__init__()

        self.use_moe = use_moe
        self.use_ternary = use_ternary

        self.attention = MultiHeadAttention(
            d, q_heads, kv_heads, head_dim, use_ternary=use_ternary
        )
        self.norm1 = nn.LayerNorm(d)

        if use_moe:
            self.ff = MoE(
                d, f, n_experts=n_experts, top_k=top_k, use_ternary=use_ternary
            )
        else:
            self.ff = nn.Sequential(
                make_linear(d, f, use_ternary),
                nn.GELU(),
                make_linear(f, d, use_ternary),
            )

        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, attn_mask, use_custom_attention=False):
        X = self.norm1(
            X + self.dropout(self.attention(X, attn_mask, use_custom_attention))
        )
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
        n_experts=8,
        top_k=2,
        use_moe=False,
        use_ternary=False,
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
                    n_experts=n_experts,
                    top_k=top_k,
                    use_moe=use_moe,
                    use_ternary=use_ternary,
                )
                for i in range(layers)
            ]
        )
        self.fc = nn.Linear(self.d, self.vocab_size)
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
        layers=6,
        q_heads=8,
        kv_heads=8,
        use_moe=False,
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
        use_moe=False,
    )
    return model


def adder_ternary_vanilla():
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=6,
        q_heads=8,
        kv_heads=8,
        use_moe=False,
        use_ternary=True,
    )
    return model
