# `model/` — the golden reference

A character-level decoder-style transformer trained to do multi-digit **addition**. Everything
under [`accel/`](../accel) is validated numerically against this code, so it is the definition
of correctness for the CUDA kernel and the TPU RTL — not the other way round. When the numerics
here change, the accelerators are wrong until they change too (see
[Accelerator contract](#accelerator-contract)).

```
model/
  transformer.py    architecture: MHA/GQA, ternary linears, MoE, named configs
  numbers_data.py   synthetic addition dataset + tokenizer
  train.py          training loop (gradient accumulation, checkpoint rotation)
  calibrate.py      offline int8 activation calibration for ternary checkpoints
  tests/
    test_inference.py   load a checkpoint, decode a batch, eyeball the answers
  notebook.ipynb    scratch experiments (stale imports — see Caveats)
  saved/            checkpoints
```

## Running

Python is a package rooted at the **repo root**; run with `-m` from there so
`import model.transformer` resolves.

```bash
python -m model.train --arch vanilla            # vanilla | gqa | ternary_vanilla
python -m model.tests.test_inference --arch vanilla --model-path model/saved/colab_vanilla_mha.pt
python -m model.calibrate --model-path model/saved/colab_ternary_mha.pt
```

Deps are just `torch` (+ `jupyter` for the notebook). The venv is checked in at `.venv/`
(Python 3.12); there is no `requirements.txt`. `device` is picked per-module as CUDA if
available, else CPU.

---

## 1. The task and the data (`numbers_data.py`)

### Vocabulary

13 tokens, one id per character. No BOS/EOS.

| token   | `0`–`9` | `+` | `=` | `N` (pad) |
| ------- | ------- | --- | --- | --------- |
| **id**  | 0–9     | 10  | 11  | 12        |

`PAD_ID = 12`, `PAD_TOKEN = 'N'`. `VOCAB`/`INV_VOCAB` are the two directions;
`detokenize` drops pads.

### Fixed answer position — the invariant everything else leans on

`EQUALS_POS = 15` is **the index of the first answer digit**, not the index of `=`. The `=`
sits at `EQUALS_POS - 1 = 14`. `generate_addition_expression` gets this by padding *between*
the operands and the `=`, so the operand text is left-aligned and the answer always starts at
the same offset regardless of how many digits the operands have:

```
index   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 ... 31
token   1  2  3  +  4  5  N  N  N  N  N  N  N  N  =  1  6  8  N ...  N
        └── operands ──┘  └──── pad ─────────┘  ^  └ answer ┘ └ tail pad ┘
                                                │  ^
                                        '=' at 14  │
                                                   EQUALS_POS = 15
```

`train.py` and `test_inference.py` slice with this constant rather than searching for `=`:
logits at position `EQUALS_POS - 1` (the `=` token) predict the digit at `EQUALS_POS`. Move the
constant and both slicings move with it; break the padding rule and they silently mis-align.

Answer digits are emitted **most-significant first** — this model does not use the
reversed-digit trick that some addition transformers rely on.

### Implied bounds

- `max_digits ≤ 6`. The pad count is `EQUALS_POS - len("a+b") - 1`; at 7-digit operands
  `"9999999+9999999"` is already 15 chars, the count goes negative, and the `=` lands at 15
  instead of 14 — every downstream slice is off by one, with no error raised.
- `max_tokens` must fit `14 + 1 + len(answer)`; with `max_digits=5` that is 21. The default 32
  leaves room.

### Sampling

`_sample_number` picks a **digit length** uniformly in `1..max_digits`, then a value uniformly
inside that length — so single-digit operands are as common as five-digit ones, which is
deliberate (uniform-over-value would make short operands vanishingly rare). Length 1 draws from
`0..9` inclusive.

### The attention mask

`tokenize(expr, max_tokens)` returns `(token_ids, mask)`. The mask is a `[T, T]` **additive**
float tensor built from the pad positions:

```python
mask = [-1e9 if id == PAD_ID else 0 for id in token_ids]   # [T]
mask = mask.repeat(T, 1)
mask = mask + mask.T                                        # [T, T]
```

so any pair touching a pad gets `-1e9` (pad↔pad gets `-2e9`, harmless once softmaxed).
`create_addition_batch(batch_size, max_tokens, max_digits)` returns
`(expressions, token_batches, attention_masks)` — three parallel lists; the caller stacks the
masks and tensorises the tokens.

> **This mask is currently threaded through the model and never applied.** See
> [Caveats](#caveats).

---

## 2. Architecture (`transformer.py`)

```
tokens [B, T]
   │ nn.Embedding(vocab, d)  +  sinusoidal PE            ← PE recomputed per forward
   │ dropout(0.1)
   ▼
 ┌──────────────────────── Transformer × layers ─────────────────────────┐
 │  X ─► MultiHeadAttention ─► (returns O + X)                           │
 │  norm1( X + dropout(attn_out) )              ← post-LN, LayerNorm(d)  │
 │  norm2( X + dropout(ff(X)) )                                          │
 │    ff = Linear(d,f) ─ GELU ─ Linear(f,d)     (or MoE if use_moe)      │
 └───────────────────────────────────────────────────────────────────────┘
   ▼
 nn.Linear(d, vocab)  ─►  logits [B, T, vocab]
```

Post-LN (normalisation *after* the residual add), sinusoidal absolute positional encoding
computed on the fly in `Model.positional_encoding` (standard `10000^(-2i/d)`, sin on even
channels, cos on odd), dropout `p=0.1` on the embedding and on both sub-block outputs.
`Model.sample_pred` samples from a `Categorical`; `sample_pred_best` takes the argmax.

### The 5-D attention layout — the accelerator contract

Q/K/V are **not** in the usual flat `[B, heads, T, head_dim]` layout. Grouped-query attention
is expressed structurally, by giving Q an extra `heads_per_q` axis that K/V simply do not have
and therefore broadcast over:

```
Q : [batch, tokens, kv_heads, heads_per_q, head_dim]      "btkgh"
K : [batch, tokens, kv_heads,              head_dim]      "bskh"
V : [batch, tokens, kv_heads,              head_dim]      "bskh"
```

with `heads_per_q = q_heads // kv_heads` (asserted to divide evenly). `mha_torch` is then two
einsums around a softmax:

```python
scores = einsum("btkgh,bskh->btskg", Q, K) / sqrt(head_dim)   # [B,T,S,K,G]
scores = softmax(scores + causal_mask, dim=2)                 # normalise over S
A      = einsum("btskg,bskh->btkgh", scores, V).reshape(B, T, q_heads * head_dim)
```

Three things are load-bearing for anyone reimplementing this:

- **Softmax is over `dim=2`**, the *source* axis `s`, in the 5-D `btskg` layout.
- The causal mask is built inside `mha_torch` as
  `triu(ones([T,T]) * -1e9, diagonal=1).reshape([1,T,T,1,1])` — strictly upper-triangular, so
  position `t` attends to `s ≤ t`. It is the **only** mask applied.
- The output reshape flattens `(kv_heads, heads_per_q, head_dim)` into
  `q_heads * head_dim` in that order, which is the order `Wo` expects.

`slow_mha_cuda` is the same signature backed by `mha_cuda.mha_custom`. It is selected by the
`use_custom_attention` flag, which threads `Model.forward → Transformer.forward →
MultiHeadAttention.forward`. **The extension `load(...)` at the top of the file is commented
out**, so this path raises `NameError` until it is re-enabled on a machine with a CUDA
toolchain.

### The double residual

`MultiHeadAttention.forward` ends with `return O + X` — it adds the residual itself. `Transformer.forward`
then adds it *again*:

```python
X = self.norm1(X + self.dropout(self.attention(X, attn_mask, use_custom_attention)))
#                └── X + dropout(O + X);  at eval this is 2X + O, not X + O
```

Whether or not that was intended, it is what the trained checkpoints were fitted to, so any
reimplementation must reproduce it. Removing it invalidates every checkpoint in `saved/`.

### Named configs

The factories at the bottom of the file are the source of truth for hyperparameters and are
wired to `train.py --arch`. `head_dim` defaults to `d // q_heads` when not given.

| factory                  | `--arch`          | `d` | `f` | layers | `q_heads` | `kv_heads` | `head_dim` | ternary | params |
| ------------------------ | ----------------- | --- | --- | ------ | --------- | ---------- | ---------- | ------- | ------ |
| `adder_vanilla`          | `vanilla`         | 128 | 512 | 4      | 4         | 4          | 32         | no      | ≈0.80 M |
| `adder_gqa`              | `gqa`             | 128 | 512 | 6      | 8         | 2          | 16         | no      | ≈1.04 M |
| `adder_ternary_vanilla`  | `ternary_vanilla` | 128 | 512 | 4      | 4         | 4          | 32         | yes     | ≈0.80 M |

`use_moe=False` in all three — the MoE path exists but nothing shipped uses it.

---

## 3. The ternary (BitNet-style) path

### `TernaryLinear`

Weights quantized to `{-1, 0, 1}` scaled by their absmean, with a straight-through estimator
for the backward pass:

```python
scale   = w.abs().mean() + eps                       # eps = 1e-5
w_quant = RoundClip.apply(w / scale) * w.abs().mean()
out     = quantize_activations(x) @ w_quant  (+ bias)
```

`RoundClip` is a `torch.autograd.Function`: forward `round().clip(-1, 1)`, backward
`grad * (|input| <= 1)` — gradient passes straight through inside the clip range and is killed
outside it. Bias (when present) stays full precision, as does the LayerNorm, the embedding, and
the final `fc`.

> **Weight orientation differs from `nn.Linear`.** `TernaryLinear.w` is `[in_dim, out_dim]` and
> the forward is `x @ w`, whereas `nn.Linear.weight` is `[out_dim, in_dim]` with `x @ Wᵀ`.
> Ternary and non-ternary checkpoints are therefore **not** interchangeable, and a reimplementation
> must know which convention it is reading. `make_linear(in_dim, out_dim, use_ternary, bias)` is
> the switch used everywhere.

### int8 activation quantization

Optional, off by default, orthogonal to the weight quantization. Each `TernaryLinear` carries a
single per-tensor symmetric int8 scale:

- `act_scale` — a **persistent** buffer, so it round-trips through `state_dict`. Initialised to
  `NaN`, which means *uncalibrated*: `quantize_activations` returns `x` untouched. Old
  checkpoints therefore behave exactly as before.
- `_act_absmax` — non-persistent, only used while calibrating.
- `calibrating` — a plain bool flag; while set, the layer records
  `max(_act_absmax, |x|.max())` and passes `x` through unquantized.

Once calibrated, the forward fake-quantizes: `round(x / scale).clamp(-127, 127) * scale`.

`calibrate_activations(model, run_forward)` orchestrates it: flips every `TernaryLinear` into
calibrating mode, resets the absmax, wipes any prior scale, puts the model in `eval()` (dropout
off, so calibration matches inference), calls `run_forward(model)` — which is responsible for
looping over the whole calibration set with whatever masks the model needs — restores the
training flag in a `finally`, then sets `act_scale = absmax / 127`. A layer whose absmax stayed
0 is left at `NaN` (passthrough) rather than given an unusable zero scale. Returns
`{module_name: scale}`.

### `calibrate.py`

The driver: builds `adder_ternary_vanilla()`, loads a checkpoint with `strict=False` (older
checkpoints predate the `act_scale` buffers and show up as *missing*; any **unexpected** key is
still a hard error), pre-generates `--batches × --batch-size` synthetic batches, runs
`calibrate_activations`, prints the per-layer scales, and saves to `--out`
(default: `<model-path stem>_int8act.pt`). Seeded with `--seed` (default 0) so the calibration
set is reproducible.

---

## 4. MoE (`use_moe=True`)

Top-k routing over `n_experts` two-layer GELU `Expert`s. The `gate` is deliberately kept
full-precision `nn.Linear` even in ternary models — quantizing the router degrades routing
quality far more than it saves. Tokens are flattened to `[B*T, d]`, `topk` logits are softmaxed
*over the selected k only*, and each expert is applied to its masked subset:

```python
for k in range(top_k):
    for e in range(n_experts):
        mask = indices[:, k] == e
        out[mask] += weights[:, k:k+1][mask] * experts[e](X_flat[mask])
```

That is `top_k × n_experts` masked gathers per forward — correct and simple, not fast. There is
no load-balancing auxiliary loss. Unused by the shipped configs.

---

## 5. Training (`train.py`)

```bash
python -m model.train --arch vanilla --mini_batch_size 256 --batch_size 512 \
                      --max_tokens 32 --max_digits 5 --save_freq 50
```

Adam, `lr=1e-3`, fixed. Data is generated fresh every step — there is no held-out split and no
epoch in the usual sense; `epochs=100 × batches=10000` is effectively "run until you stop it".

**Gradient accumulation.** `steps_per_batch = batch_size // mini_batch_size`; each iteration
does a forward/backward on `mini_batch_size` examples, and `optim.step()` + `zero_grad()` only
fire every `steps_per_batch` micro-steps. So `--batch_size` is the *effective* batch and
`--mini_batch_size` is what has to fit in memory.

**Loss.** Cross-entropy over the answer span, aligned by the fixed position:

```python
pred = pred[:, EQUALS_POS - 1 : -1].flatten(0, 1)   # logits that predict positions 15..T-1
y    = tokens[:, EQUALS_POS :].flatten(0, 1)        # targets at positions 15..T-1
```

Note the target span runs to the end of the sequence, so the **trailing pad tokens are part of
the loss** — the model is explicitly trained to emit `N` after the last answer digit. That is a
real (if easy) part of the objective, not an oversight to "fix" without retraining.

**Checkpointing.** Every `save_freq` *micro*-steps, `model/saved/test_model_<timestamp>.pt`,
keeping only the last 3 (older files are deleted). These are gitignored; the committed
`colab_*.pt` checkpoints are not produced by this path. The printed line is labelled
`Epoch {i}` but reports the mean loss over the last `save_freq` micro-steps.

---

## 6. Inference (`tests/test_inference.py`)

Not a `pytest` test — a script. It builds the architecture, loads `--model-path`, runs one batch
of 32 expressions at `max_tokens=64`, and prints the expression, the expected digits, and
`sample_pred_best(pred)[:, EQUALS_POS-1:-1]` side by side for a human to compare. There is no
assertion and no accuracy number.

## 7. Checkpoints (`saved/`)

| file                              | build with                | notes                                              |
| --------------------------------- | ------------------------- | -------------------------------------------------- |
| `colab_vanilla_mha.pt`            | `adder_vanilla()`         | the default reference                               |
| `colab_gqa.pt`                    | `adder_gqa()`             | grouped-query, 8 q-heads over 2 kv-heads            |
| `colab_ternary_mha.pt`            | `adder_ternary_vanilla()` | ternary weights, activations **not** calibrated     |
| `colab_ternary_mha_int8act.pt`    | `adder_ternary_vanilla()` | the above after `python -m model.calibrate`         |

Loading a checkpoint into the wrong factory fails on shape mismatch — except that
`colab_ternary_mha.pt` needs `strict=False` against the current `TernaryLinear` (missing
`act_scale`), which is what `calibrate.py` does.

---

## Accelerator contract

What `accel/` reimplements, and therefore what cannot change here unilaterally:

| Contract | Where it is defined | Who depends on it |
| --- | --- | --- |
| 5-D `btkgh`/`bskh` layout, the einsum pair, `1/sqrt(head_dim)`, softmax over `dim=2`, strict-upper-triangular `-1e9` causal mask | `mha_torch` | `accel/cuda/kernels.cu` (asserts max abs diff < 1e-2 in `accel/cuda/tests/test_cuda_mha.py`); any future TPU attention block |
| Output reshape order `(kv_heads, heads_per_q, head_dim) → q_heads*head_dim` | `mha_torch` | same |
| The double residual (`O + X` inside attention, `X + dropout(...)` outside) | `MultiHeadAttention` / `Transformer` | any full-model reimplementation |
| GELU in the FFN, post-LN placement | `Transformer` | `accel/tpu/docs/vpu.md` (activation LUTs, LayerNorm reductions) |
| Ternary weight scheme (absmean scale, `{-1,0,1}`) and per-tensor int8 activation scale | `TernaryLinear` | TPU MXU datapath and requant (`accel/tpu/docs/mxu.md`) |
| `EQUALS_POS`, vocab ids, pad convention | `numbers_data.py` | anything decoding golden vectors |

Golden vectors exported for the TPU flow live under `accel/tpulang/` (`gen_vectors.py`,
`torch_ref.py`); `torch_ref.py` is deliberately an *independent* PyTorch leg that shares no code
with the ISS.

---

## Caveats

Known rough edges — documented so they are not rediscovered as bugs.

- **The padding mask is never applied.** `attn_mask` is built by `tokenize`, stacked by the
  callers, and passed all the way down to `MultiHeadAttention.forward` — which ignores it.
  `mha_torch(Q, K, V)` takes no mask argument and applies only its own causal mask. Pad
  positions are therefore attended to normally. The model works anyway because the pad run is at
  a fixed place in a fixed-format sequence, but wiring the mask in would change the numerics and
  invalidate the checkpoints.
- **`mha_torch` builds its mask on the module-level `device`,** not the model's. On a machine
  with CUDA available but the model kept on CPU, the mask lands on the GPU and the add raises a
  device mismatch.
- **`test_inference.py` defaults `--custom-attention` to `True`,** which routes to
  `slow_mha_cuda` → the commented-out extension → `NameError`. Pass `--custom-attention ""` or
  edit the default. Note also that `type=bool` makes any non-empty string truthy, so
  `--custom-attention False` does *not* disable it — the usual argparse trap.
- **`test_inference.py`'s default `--model-path` is `saved/colab_vanilla_mha.pt`,** relative to
  the repo root where it does not exist. Pass `model/saved/colab_vanilla_mha.pt`.
- **`train.py`'s `--arch` dispatch falls through to ternary** for anything that is not `gqa` or
  `vanilla`; `argparse` `choices` is what actually constrains it.
- **`notebook.ipynb` uses flat imports** (`import train`, `from transformer import Model`) from
  before the package layout, and one cell unpacks `create_addition_batch` into two values when it
  now returns three. It needs updating before it will run from the repo root.
