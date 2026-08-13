#!/usr/bin/env python3
"""adder_export.py — take the real ternary adder checkpoint all the way to integers.

The last leg of [`adder_kernel.md`](adder_kernel.md) §7: everything else proves
`examples/adder_model.tpu` computes what the model *specifies* (the ISS matches an
independent PyTorch implementation, and the RTL matches the ISS, byte for byte).
None of it says the fully-integer model still does addition. That is what this
measures.

Four stages, each usable on its own:

  1. **load**       — `model/saved/colab_ternary_mha_small_nobias.pt` into
                      `adder_ternary_vanilla()`, and a float accuracy baseline.
  2. **calibrate**  — one instrumented float forward over a calibration set,
                      recording the absmax at every point the integer pipeline
                      needs a scale. The shipped checkpoint's `act_scale` buffers
                      are all NaN, so *every* scale is derived here, not just the
                      four attention-internal ones.
  3. **derive**     — the 13 requant `{m0,n}` words per layer (§4 of the doc),
                      including the two that are pinned by the residual adds.
  4. **evaluate**   — the integer pipeline over N problems, reporting
                      exact-sequence and token accuracy against the float model
                      and against ground truth, plus the clip rates that §8 says
                      are the diagnostic.

The integer forward here is deliberately the *same* computation as
`torch_ref.adder_model`, which is verified bit-exact against both the ISS and the
RTL. `--iss-check` closes that loop explicitly: it packs a real problem into the
kernel's DRAM layout, runs `examples/adder_model.tpu` in the ISS, and compares the
device's logits against this module's. When that passes, the accuracy numbers
below describe the actual kernel rather than a model of it.

    python accel/tpulang/adder_export.py                    # 256 problems
    python accel/tpulang/adder_export.py -n 512 --iss-check
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model.numbers_data as numbers_data                      # noqa: E402
import model.transformer as transformer                        # noqa: E402

M0_W, N_W = 12, 4
M0_MAX = (1 << M0_W) - 1          # 4095
N_MAX = (1 << N_W) - 1            # 15
T = 32                            # train.py --max_tokens
D = 128
HEADS = 4
DH = D // HEADS
ANS = numbers_data.EQUALS_POS     # '=' lands at index 14; answer starts at 15

CKPT = os.path.join(ROOT, "model", "saved", "colab_ternary_mha_small_nobias.pt")

# The 13 requant slots, in the order adder_model.tpu reads them out of DRAM.
SLOTS = ["Q", "K", "V", "S", "ID", "P", "A", "O", "X1", "H", "HR", "F", "X2"]


# =============================================================================
# Fixed-point helpers — identical arithmetic to requant.sv / iss.requant8.
# =============================================================================
def choose_word(m: float) -> tuple[int, int]:
    """Best `{m0, n}` with `m0/2**n ~= m`, `1 <= m0 <= 4095`, `0 <= n <= 15`.

    Larger `n` gives finer relative resolution, so the largest feasible shift
    wins. A multiplier outside the representable band saturates rather than
    silently wrapping; `report_words` flags it.
    """
    best = None
    for n in range(N_MAX, -1, -1):
        m0 = int(round(m * (1 << n)))
        if 1 <= m0 <= M0_MAX:
            best = (m0, n)
            break
    if best is None:
        best = (M0_MAX, 0) if m > 1 else (1, N_MAX)
    return best


def requant8(acc: torch.Tensor, m0: int, n: int) -> torch.Tensor:
    """`clip_int8((acc*m0 + 2**(n-1)) >> n)` — arithmetic (flooring) shift."""
    rnd = 0 if n == 0 else (1 << (n - 1))
    return torch.div(acc * m0 + rnd, 1 << n, rounding_mode="floor").clamp(-128, 127)


def clip_rate(acc: torch.Tensor, m0: int, n: int) -> float:
    """Fraction of a requant's inputs that saturate at +-127."""
    rnd = 0 if n == 0 else (1 << (n - 1))
    sh = torch.div(acc * m0 + rnd, 1 << n, rounding_mode="floor")
    return float(((sh > 127) | (sh < -128)).float().mean())


def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact `[M,K] @ [K,N]` over integers (torch has no integer BLAS kernel)."""
    return (a[:, :, None] * b[None, :, :]).sum(dim=1)


# =============================================================================
# 1. The model, and the float baseline.
# =============================================================================
def load_model(path: str = CKPT):
    m = transformer.adder_ternary_vanilla()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def make_batch(n: int, seed: int):
    """`n` addition problems at the kernel's T=32 context."""
    import random
    random.seed(seed)
    exprs, toks, masks = numbers_data.create_addition_batch(n, T)
    return exprs, torch.tensor(toks), torch.stack(masks)


def answer_slice(tokens: torch.Tensor) -> torch.Tensor:
    """The tokens a prediction is scored against: everything after `=`."""
    return tokens[:, ANS + 1:]


def score(pred_tok: torch.Tensor, tgt_tok: torch.Tensor) -> tuple[float, float]:
    """(exact-sequence, per-token) accuracy over the answer region."""
    hit = pred_tok == tgt_tok
    return float(hit.all(dim=1).float().mean()), float(hit.float().mean())


@torch.no_grad()
def float_predict(m, tokens, masks) -> torch.Tensor:
    logits = m(tokens, masks)
    return logits.argmax(-1)[:, ANS:-1]


# =============================================================================
# 2. Calibration — one instrumented float forward.
# =============================================================================
class Scales:
    """Running absmax at every site the integer pipeline needs a scale."""

    # "o" and "f" are the two residual *addends*. They are recorded because a
    # residual add works on int8 operands, so the addend shares the scale of the
    # thing it is added to — which means that scale has to cover the addend's
    # range too, not just the accumulator's. See derive_layer.
    SITES = ["x", "q", "k", "v", "s", "p", "a", "o", "x1", "h", "hr", "f", "x2"]

    def __init__(self, layers: int):
        self.m = [{s: 0.0 for s in self.SITES} for _ in range(layers)]

    def see(self, layer: int, site: str, t: torch.Tensor):
        v = float(t.detach().abs().max())
        if v > self.m[layer][site]:
            self.m[layer][site] = v

    def scale(self, layer: int, site: str) -> float:
        """absmax/127, the symmetric int8 convention `calibrate_activations` uses."""
        a = self.m[layer][site]
        return (a / 127.0) if a > 0 else 1.0


def trits_and_absmean(lin) -> tuple[torch.Tensor, float]:
    """`TernaryLinear` -> ({-1,0,1} matrix, absmean). Mirrors its forward()."""
    absmean = float(lin.w.abs().mean())
    scale = absmean + lin.eps
    t = (lin.w / scale).round().clamp(-1, 1)
    return t.to(torch.int64), absmean


@torch.no_grad()
def float_forward_instrumented(m, tokens, masks, sc: Scales | None = None):
    """`Model.forward` rewritten inline so every intermediate can be recorded.

    Mirrors model/transformer.py exactly: no LayerNorm, no bias, ReLU attention,
    ReLU feed-forward, dropout off. Verified against `m(tokens, masks)` in
    :func:`check_float_forward`, which is what makes it safe to calibrate from.
    """
    pe = m.positional_encoding(tokens.shape[1])
    X = m.embedding(tokens) + pe
    B, N = tokens.shape[0], tokens.shape[1]
    causal = torch.triu(torch.ones(N, N) * -1e9, diagonal=1)

    for li, layer in enumerate(m.layers):
        if sc: sc.see(li, "x", X)
        at = layer.attention
        Q, K, V = at.Wq(X), at.Wk(X), at.Wv(X)
        if sc:
            sc.see(li, "q", Q); sc.see(li, "k", K); sc.see(li, "v", V)

        Qh = Q.reshape(B, N, HEADS, DH)
        Kh = K.reshape(B, N, HEADS, DH)
        Vh = V.reshape(B, N, HEADS, DH)
        # [B, t, s, h] scores, matching mha_torch's einsum pair.
        S = torch.einsum("btkh,bskh->btsk", Qh, Kh) / math.sqrt(DH)
        if sc: sc.see(li, "s", S)
        P = torch.relu(S + causal[None, :, :, None])
        if sc: sc.see(li, "p", P)
        A = torch.einsum("btsk,bskh->btkh", P, Vh).reshape(B, N, D)
        if sc: sc.see(li, "a", A)

        O = at.Wo(A)
        if sc: sc.see(li, "o", O)
        X1 = X + O
        if sc: sc.see(li, "x1", X1)
        H = layer.ff[0](X1)
        if sc: sc.see(li, "h", H)
        HR = torch.relu(H)
        if sc: sc.see(li, "hr", HR)
        F = layer.ff[2](HR)
        if sc: sc.see(li, "f", F)
        X = X1 + F
        if sc: sc.see(li, "x2", X)

    return m.fc(X)


@torch.no_grad()
def check_float_forward(m, tokens, masks) -> float:
    """Max abs difference between the inline forward and `Model.forward`."""
    return float((float_forward_instrumented(m, tokens, masks)
                  - m(tokens, masks)).abs().max())


# =============================================================================
# 3. Derive the 13 words per layer.
# =============================================================================
def layer_input_scales(sc: Scales, n_layers: int) -> list[float]:
    """The residual-stream scale at each layer's input — the single source of
    truth, because it is used three times and must agree everywhere:

      * to quantize `X0` before layer 0,
      * as `s_x` inside layer `li`'s words,
      * as the *target* of layer `li-1`'s `RQ_X2`.

    Kept at `|X|/127`, deliberately. Widening it to `max(|X|,|O|)/127` removes
    the `RQ_O` clipping that §8.3 of the doc predicted — but it is a net *loss*,
    because X is the same tensor the Q/K/V/W1 projections read, and |O| runs up
    to 55x |X| in this checkpoint. Measured over 64 problems:

        s_x = |X|/127            token 63.18%    (clips O)
        s_x = max(|X|,|O|)/127   token 44.63%    (collapses X to ~2 levels)

    A rescaled second copy of X for the add gets 69.73%, at the cost of one more
    requant slot and a VPU pass per residual; it was not adopted because the
    model does not survive int8 activations at any of these settings (§7.6).
    """
    return [sc.scale(li, "x") for li in range(n_layers)]


def derive_layer(m, li: int, sc: Scales, s_x: float, s_x_next: float) -> tuple[dict, dict]:
    """Requant words + the scales they were built from, for one layer.

    Two of the thirteen are **pinned**, not chosen: `vecadd` adds int8 operands,
    so a residual is only meaningful if both sides share a scale. `RQ_O` must
    therefore land `Wo`'s output on `s_x`, and `RQ_F` must land `W2`'s on `s_x1`
    (adder_kernel.md §4). Those two are where clipping is expected.
    """
    layer = m.layers[li]
    at = layer.attention
    _, a_q = trits_and_absmean(at.Wq)
    _, a_k = trits_and_absmean(at.Wk)
    _, a_v = trits_and_absmean(at.Wv)
    _, a_o = trits_and_absmean(at.Wo)
    _, a_1 = trits_and_absmean(layer.ff[0])
    _, a_2 = trits_and_absmean(layer.ff[2])

    s = {k: sc.scale(li, k) for k in Scales.SITES}
    s["x2"] = s_x_next                      # must equal the next layer's s_x

    # --- two corrections that naive absmax calibration gets wrong ---------
    #
    # (a) The score scale is set by what SURVIVES, not by the widest score.
    #     S8 feeds `relu(S8 + mask)`, so a large *negative* score contributes
    #     nothing: it clips to -128 and ReLU takes it to exactly 0, which is the
    #     right answer. Calibrating s_s on |S| therefore spends the int8 range on
    #     values that are discarded. Measured on this checkpoint, layer 3's
    #     widest score is 1.32e6 while the widest *surviving* one is 5.87e3 —
    #     225x, i.e. ~7.8 of the 8 bits thrown away. Pinning s_s to the post-ReLU
    #     range is what makes RQ_P the {1,0} identity the design expected.
    s["s"] = s["p"]
    #
    # (b) The residual pair still shares one scale, and it is NOT widened to
    #     cover the addend — see :func:`layer_input_scales` for the measurement
    #     that settled it. `RQ_O`/`RQ_F` therefore clip, which is the cost side
    #     of the tradeoff and is reported as a clip rate rather than hidden.
    s["x"] = s_x

    mult = {
        "Q":  s["x"] * a_q / s["q"],
        "K":  s["x"] * a_k / s["k"],
        "V":  s["x"] * a_v / s["v"],
        "S":  s["q"] * s["k"] / (math.sqrt(DH) * s["s"]),
        "ID": 1.0,                          # narrow + clip only
        "P":  s["s"] / s["p"],
        "A":  s["p"] * s["v"] / s["a"],
        "O":  s["a"] * a_o / s["x"],        # pinned to the residual's scale
        "X1": s["x"] / s["x1"],
        "H":  s["x1"] * a_1 / s["h"],
        "HR": s["h"] / s["hr"],
        "F":  s["hr"] * a_2 / s["x1"],      # pinned to the residual's scale
        "X2": s["x1"] / s["x2"],
    }
    words = {k: (1, 0) if k == "ID" else choose_word(v) for k, v in mult.items()}
    return words, {"scales": s, "mult": mult}


def derive_all(m, sc: Scales):
    L = len(m.layers)
    s_x_all = layer_input_scales(sc, L)
    out = []
    for li in range(L):
        # Layer li's output scale IS layer li+1's input scale; the last layer
        # just feeds the head, which an argmax makes scale-invariant.
        s_next = s_x_all[li + 1] if li + 1 < L else sc.scale(li, "x2")
        w, meta = derive_layer(m, li, sc, s_x_all[li], s_next)
        out.append((w, meta))
    return out


# =============================================================================
# 4. The integer pipeline — the same computation adder_model.tpu performs.
# =============================================================================
@torch.no_grad()
def int_forward(m, X0q: torch.Tensor, words, mask_i8: torch.Tensor,
                stats: dict | None = None) -> torch.Tensor:
    """One sequence, int8 in -> int32 logits out. Structured exactly like
    `torch_ref.adder_model`, which is verified bit-exact against the ISS and RTL.

    `X0q` is `[T, D]` int8; the return is `[T, VOCAB]` int32.
    """
    X = X0q.to(torch.int64)
    for li, layer in enumerate(m.layers):
        at = layer.attention
        w = words[li][0]
        tq, _ = trits_and_absmean(at.Wq)
        tk, _ = trits_and_absmean(at.Wk)
        tv, _ = trits_and_absmean(at.Wv)
        to, _ = trits_and_absmean(at.Wo)
        t1, _ = trits_and_absmean(layer.ff[0])
        t2, _ = trits_and_absmean(layer.ff[2])

        Q = requant8(int_matmul(X, tq), *w["Q"])
        K = requant8(int_matmul(X, tk), *w["K"])
        V = requant8(int_matmul(X, tv), *w["V"])

        A = torch.zeros((T, D), dtype=torch.int64)
        for h in range(HEADS):
            sl = slice(h * DH, (h + 1) * DH)
            S8 = requant8(int_matmul(Q[:, sl], K[:, sl].t()), *w["S"])
            S8 = requant8(S8 + mask_i8, *w["ID"])
            P8 = requant8(S8.clamp(min=0), *w["P"])
            AhT = requant8(int_matmul(V[:, sl].t(), P8.t()), *w["A"])
            A[:, sl] = AhT.t()

        acc_o = int_matmul(A, to)
        if stats is not None:
            stats.setdefault("O", []).append(clip_rate(acc_o, *w["O"]))
        O = requant8(acc_o, *w["O"])
        X1 = requant8(X + O, *w["X1"])

        H = requant8(int_matmul(X1, t1), *w["H"])
        HR = requant8(H.clamp(min=0), *w["HR"])
        acc_f = int_matmul(HR, t2)
        if stats is not None:
            stats.setdefault("F", []).append(clip_rate(acc_f, *w["F"]))
        F = requant8(acc_f, *w["F"])
        X = requant8(X1 + F, *w["X2"])

    wfc, s_w = quantize_i8(m.fc.weight)
    return int_matmul(X, wfc.t())


def quantize_i8(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric per-tensor int8. The scale is irrelevant to an argmax."""
    a = float(t.detach().abs().max())
    s = a / 127.0 if a > 0 else 1.0
    return torch.round(t.detach() / s).clamp(-127, 127).to(torch.int64), s


def causal_mask_i8() -> torch.Tensor:
    """0 where key <= query, -128 above the diagonal (adder_kernel.md §4)."""
    t = torch.arange(T)
    return torch.where(t[None, :] <= t[:, None], 0, -128).to(torch.int64)


@torch.no_grad()
def quantize_input(m, tokens_1: torch.Tensor, s_x0: float) -> torch.Tensor:
    """embedding + positional encoding -> int8 at the layer-0 input scale."""
    pe = m.positional_encoding(tokens_1.shape[1])
    X = m.embedding(tokens_1) + pe
    return torch.round(X[0] / s_x0).clamp(-128, 127).to(torch.int64)


# =============================================================================
# Reporting.
# =============================================================================
@torch.no_grad()
def report_dynamic_range(m, tokens, masks) -> None:
    """Within-tensor dynamic range — why per-tensor int8 fails on this model.

    A per-tensor scale is pinned by the tensor's *maximum*, so the resolution
    every other element gets is set by how far the bulk sits below that maximum.
    The statistic that matters is therefore `median/max`, not the usual
    outlier check against a high percentile: here the top of the distribution is
    perfectly well behaved (max/p99.9 is only 1.1-2.5) while the *median* falls
    to ~0.1% of max by layer 3, so most of the tensor rounds to zero.
    """
    B, N = tokens.shape
    rec: dict = {}
    pe = m.positional_encoding(N)
    X = m.embedding(tokens) + pe
    causal = torch.triu(torch.ones(N, N) * -1e9, diagonal=1)
    for li, layer in enumerate(m.layers):
        at = layer.attention
        rec[(li, "x")] = X.flatten()
        Q, K, V = at.Wq(X), at.Wk(X), at.Wv(X)
        Qh = Q.reshape(B, N, HEADS, DH); Kh = K.reshape(B, N, HEADS, DH)
        Vh = V.reshape(B, N, HEADS, DH)
        S = torch.einsum("btkh,bskh->btsk", Qh, Kh) / math.sqrt(DH)
        P = torch.relu(S + causal[None, :, :, None])
        rec[(li, "P>0")] = P[P > 0].flatten()
        A = torch.einsum("btsk,bskh->btkh", P, Vh).reshape(B, N, D)
        rec[(li, "a")] = A.flatten()
        X1 = X + at.Wo(A)
        H = layer.ff[0](X1)
        X = X1 + layer.ff[2](torch.relu(H))

    print("\nwithin-tensor dynamic range (per-tensor int8 resolution):")
    print(f"  {'site':10s} {'|max|':>12s} {'median':>10s} {'med/max':>9s} {'%->0':>7s}")
    for (li, site), t in rec.items():
        a = t.abs().float()
        if a.numel() == 0:
            continue
        mx, med = float(a.max()), float(a.median())
        z = float((a < (mx / 127.0) / 2).float().mean()) * 100
        print(f"  L{li} {site:7s} {mx:12.2f} {med:10.4f} {med/mx:9.5f} {z:6.1f}%")
    print("  -> a value below half a quantization step becomes exactly 0.")


def report_words(derived) -> None:
    print("\nrequant words (m0, n) per layer, and the multiplier each approximates:")
    print(f"  {'slot':4s} " + " ".join(f"{'L'+str(i):>18s}" for i in range(len(derived))))
    for k in SLOTS:
        cells = []
        for w, meta in derived:
            m0, n = w[k]
            cells.append(f"{m0:5d}/2^{n:<2d}={m0/(1<<n):7.4f}")
        print(f"  {k:4s} " + " ".join(f"{c:>18s}" for c in cells))
    print("\n  relative error of each word vs the exact multiplier:")
    worst = []
    for k in SLOTS:
        errs = []
        for w, meta in derived:
            m0, n = w[k]
            want = meta["mult"][k]
            errs.append(abs(m0 / (1 << n) - want) / want if want else 0.0)
        worst.append((max(errs), k))
        print(f"    {k:4s} max {max(errs)*100:6.3f}%")
    worst.sort(reverse=True)
    print(f"    -> worst slot: {worst[0][1]} at {worst[0][0]*100:.3f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--problems", type=int, default=256)
    ap.add_argument("-c", "--calib", type=int, default=128,
                    help="problems in the calibration set (disjoint from the eval set)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--iss-check", action="store_true",
                    help="also run one problem through the ISS on the real kernel")
    args = ap.parse_args(argv)

    print(f"checkpoint: {os.path.relpath(args.ckpt, ROOT)}")
    m = load_model(args.ckpt)
    nan = sum(1 for _, b in m.named_buffers() if b.numel() == 1 and torch.isnan(b).all())
    print(f"  uncalibrated act_scale buffers: {nan} (every scale is derived here)")

    # ---- float baseline --------------------------------------------------
    _, tok_e, msk_e = make_batch(args.problems, seed=1234)
    tgt = answer_slice(tok_e)
    pred_f = float_predict(m, tok_e, msk_e)
    seq_f, tokacc_f = score(pred_f, tgt)
    print(f"\nfloat model ({args.problems} problems, T={T}):")
    print(f"  exact-sequence {seq_f*100:6.2f}%    token {tokacc_f*100:6.2f}%")

    d = check_float_forward(m, tok_e[:8], msk_e[:8])
    print(f"  inline forward vs Model.forward: max|diff| = {d:.3e}")
    if d > 1e-3:
        print("  !! inline forward disagrees with the model; calibration would be wrong")
        return 1

    # ---- calibrate on a disjoint set -------------------------------------
    _, tok_c, msk_c = make_batch(args.calib, seed=99)
    sc = Scales(len(m.layers))
    float_forward_instrumented(m, tok_c, msk_c, sc)
    print(f"\ncalibrated on {args.calib} disjoint problems; per-layer absmax:")
    for li in range(len(m.layers)):
        row = "  ".join(f"{k}={sc.m[li][k]:8.2f}" for k in ("x", "q", "s", "a", "x1", "h"))
        print(f"  L{li}  {row}")

    derived = derive_all(m, sc)
    report_words(derived)
    report_dynamic_range(m, tok_c[:32], msk_c[:32])

    # ---- integer pipeline ------------------------------------------------
    mask_i8 = causal_mask_i8()
    s_x0 = layer_input_scales(sc, len(m.layers))[0]
    stats: dict = {}
    preds = []
    for i in range(args.problems):
        X0q = quantize_input(m, tok_e[i:i + 1], s_x0)
        lg = int_forward(m, X0q, derived, mask_i8, stats)
        preds.append(lg.argmax(-1)[ANS:-1])
    pred_i = torch.stack(preds)
    seq_i, tokacc_i = score(pred_i, tgt)
    agree = float((pred_i == pred_f).all(dim=1).float().mean())

    print(f"\ninteger pipeline ({args.problems} problems):")
    print(f"  exact-sequence {seq_i*100:6.2f}%    token {tokacc_i*100:6.2f}%")
    print(f"  agrees with the float model on {agree*100:6.2f}% of sequences")
    print(f"\nclip rates at the two pinned requants (adder_kernel.md §8.3):")
    for k in ("O", "F"):
        v = torch.tensor(stats.get(k, [0.0]))
        per_layer = v.reshape(-1, len(m.layers)).mean(dim=0)
        print(f"  RQ_{k}: mean {float(v.mean())*100:6.3f}%   per layer " +
              " ".join(f"L{i}={float(x)*100:.2f}%" for i, x in enumerate(per_layer)))

    if args.iss_check:
        rc = iss_check(m, derived, mask_i8, s_x0, tok_e[0:1])
        if rc:
            return rc
    return 0


# =============================================================================
# Cross-check: the real kernel, on real weights, in the ISS.
# =============================================================================
def iss_check(m, derived, mask_i8, s_x0, tokens_1) -> int:
    """Pack one real problem into the kernel's DRAM layout, run
    examples/adder_model.tpu in the ISS, and compare its logits to int_forward.

    This is what makes the accuracy numbers above statements about the *kernel*
    rather than about this module.
    """
    import gen_vectors as gv
    from iss import TPU

    print("\nISS cross-check on the real checkpoint:")
    words, consts = gv.assemble_program(os.path.join(HERE, "examples", "adder_model.tpu"))
    tpu = TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W,
              mem_addr_w=gv.MEM_ADDR_W, m0_w=gv.M0_W, n_w=gv.N_W)

    def put(addr, byte):
        tpu.dram[tpu._d(addr)] = byte & 0xFF

    X0q = quantize_input(m, tokens_1, s_x0)
    for t in range(T):
        for d_ in range(D):
            put(consts["D_XIN"] + t * D + d_, int(X0q[t, d_]))
    for t in range(T):
        for s in range(T):
            put(consts["D_MASK"] + t * T + s, int(mask_i8[t, s]))

    wfc, _ = quantize_i8(m.fc.weight)
    V = m.vocab_size
    for v in range(V):
        for d_ in range(D):
            put(consts["D_WFC"] + v * D + d_, int(wfc[v, d_]))

    WCOL = consts["WCOL"]

    def pack(base, tmat):
        """[K][N] trits -> column-major 2-bit packed at `base`."""
        K, N = tmat.shape
        for n in range(N):
            colint = 0
            for k in range(K):
                t = int(tmat[k, n])
                colint |= (0 if t == 0 else (1 if t > 0 else 0b11)) << (2 * k)
            for b in range(WCOL):
                put(base + n * WCOL + b, (colint >> (8 * b)) & 0xFF)

    for li, layer in enumerate(m.layers):
        base = consts["D_L0"] + li * consts["LSTRIDE"]
        at = layer.attention
        tq, _ = trits_and_absmean(at.Wq)
        tk, _ = trits_and_absmean(at.Wk)
        tv, _ = trits_and_absmean(at.Wv)
        fused = torch.cat([tq, tk, tv], dim=1)
        pack(base + consts["O_WQKV"], fused)
        pack(base + consts["O_WO"], trits_and_absmean(at.Wo)[0])
        pack(base + consts["O_W1"], trits_and_absmean(layer.ff[0])[0])
        pack(base + consts["O_W2"], trits_and_absmean(layer.ff[2])[0])
        for i, k in enumerate(SLOTS):
            m0, n = derived[li][0][k]
            word = (n << M0_W) | m0
            for b in range(4):
                put(base + consts["O_RQW"] + 4 * i + b, (word >> (8 * b)) & 0xFF)

    tpu.run(words, max_steps=2_000_000)

    got = []
    for t in range(T):
        row = []
        for v in range(V):
            a = consts["D_LOG"] + t * (V * 4) + v * 4
            u = sum(tpu.dram[tpu._d(a + b)] << (8 * b) for b in range(4))
            row.append(u - (1 << 32) if u >= (1 << 31) else u)
        got.append(row)
    got = torch.tensor(got, dtype=torch.int64)
    want = int_forward(m, X0q, derived, mask_i8)

    bad = int((got != want).sum())
    print(f"  logits [{T}x{V}] int32: {bad}/{got.numel()} differ")
    if bad:
        print("  !! the ISS and this module disagree — the accuracy numbers above "
              "do not describe the kernel")
        return 1
    print("  exact — the accuracy numbers above are the kernel's")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
