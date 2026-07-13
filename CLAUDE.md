# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small decoder-style transformer trained to do multi-digit **addition** (a character-level
sequence model), plus custom hardware/backend accelerators for its core attention/matmul ops.
The PyTorch model in `model/` is the **golden reference**; every accelerator in `accel/` is
validated numerically against it.

## Commands

Python is a package rooted at the repo, so **run from the repo root** with `-m` so imports
resolve (`import model.transformer`, etc.):

```bash
python -m model.train --arch vanilla        # train (arch: vanilla | gqa | ternary_vanilla)
python -m model.tests.test_inference --arch vanilla --model-path model/saved/colab_vanilla_mha.pt
python accel/cuda/tests/test_cuda_mha.py    # CUDA kernel vs. torch reference — requires a GPU
```

`test_cuda_mha.py` JIT-compiles the CUDA kernel via `torch.utils.cpp_extension.load`, so it
needs a CUDA toolchain + GPU. It asserts max abs diff < 1e-2 against the einsum reference.

The venv is checked in at `.venv/` (Python 3.12). There is no requirements.txt; deps are just
`torch` (+ jupyter for `model/notebook.ipynb`).

## Architecture

**`model/transformer.py`** — the whole model. Key things to know before editing:

- **5-D attention tensor layout.** Q/K/V are shaped `[batch, tokens, kv_heads, heads_per_q,
  head_dim]` (K/V drop the `heads_per_q` axis) rather than the usual flat head layout. GQA is
  expressed by broadcasting `kv_heads` over `heads_per_q`. The attention math is the einsum
  pair `"btkgh,bskh->btskg"` (scores) and `"btskg,bskh->btkgh"` (weighted values) in
  `mha_torch`. **This exact layout and einsum is the contract the CUDA kernel implements** — if
  you change it, `accel/cuda/kernels.cu` must change to match.
- **`use_custom_attention` flag** threads from `Model.forward` → `Transformer` →
  `MultiHeadAttention.forward` and switches between `mha_torch` (default) and `slow_mha_cuda`
  (the compiled kernel). The kernel `load(...)` at the top of the file is currently commented
  out, so the custom path only works once that's re-enabled with a GPU present.
- **Ternary (BitNet-style) weights.** `TernaryLinear` quantizes weights to {-1,0,1} scaled by
  absmean, with `RoundClip` providing a straight-through-estimator backward. `make_linear(...,
  use_ternary)` selects it everywhere. The MoE router `gate` stays full-precision on purpose.
- **MoE** (`use_moe`) is a top-k expert FFN; it exists but the shipped `adder_*` configs don't
  use it.
- **Named configs** at the bottom (`adder_vanilla`, `adder_gqa`, `adder_ternary_vanilla`) are
  the source of truth for hyperparameters and are wired to the `--arch` choices in `train.py`.

**`model/numbers_data.py`** — synthetic dataset. Non-obvious invariants:

- **Fixed answer position.** `EQUALS_POS = 15` is hard-coded: operands are padded so `=` always
  lands at index 15, so the answer tokens always start at a fixed offset. `train.py` and
  `test_inference.py` slice predictions/targets using `EQUALS_POS` — they rely on this.
- Padding token is `'N'` (`PAD_ID = 12`); `tokenize` builds an additive `-1e9` attention mask
  from the pad positions (`mask + mask.T`).
- Operand lengths are sampled uniformly over digit-count (`_sample_number`), not uniformly over
  value.

**`model/train.py`** — `batch_size` vs `mini_batch_size` implements **gradient accumulation**
(`steps_per_batch = batch_size // mini_batch_size`); `optim.step()` only fires every
`steps_per_batch` micro-steps. Checkpoints go to `model/saved/test_model_*.pt`, keeping only
the last 3 (these are gitignored; committed checkpoints like `colab_vanilla_mha.pt` are not).

## Accelerators (`accel/`)

- **`cuda/`** (`kernels.cu` + `bindings.cpp`) — hand-written CUDA MHA/GQA kernel, the GPU
  reference. Loaded into PyTorch as an extension. Correctness is defined by matching
  `mha_torch`.
- **`tpu/`** — a SystemVerilog TPU targeting an FPGA, **early / mostly scaffolding** (see
  `accel/tpu/README.md` and `accel/tpu/docs/`). No synthesizable RTL of the array yet; blocks
  are to be validated against golden vectors exported from `model/`.

When touching attention numerics, keep the three implementations in sync: `mha_torch`
(reference), the CUDA kernel, and any future TPU block — they all implement the same 5-D layout
and masking convention.

## Coding Practices

- **Do**: read files to gain a good understanding of code, ask questions when design choices are unclear,
and write code. Plan out anything in markdown files when it is a large task.
- **Don't**: run code or any other commands (such as checking for packages, etc) unless otherwise told to do so.
If extra context is needed about my development environment or feedback is needed for your code, please ask.
