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

## In progress: intermittent UART corruption (as of 2026-08-05)

Live bug hunt on branch `uart-debug`. `host/test_uart_link.py` passes most of the time and
fails after tens of seconds of traffic. **The visible symptom — the device sending more bytes
than the host asked for — is the aftermath, not the fault.** A single lost/extra byte desyncs
the command FSM permanently, it lands back in `IDLE` mid-payload, and every payload byte that
isn't `R`/`W`/`I`/`G` then produces a NAK byte (`uart_interface.sv:201`). Don't chase the byte
count; chase the one divergence that precedes it.

**Ruled out — do not re-litigate these.** Each cost a real experiment:

| Hypothesis | Killed by |
|---|---|
| Off-centre sample point (`START` runs `CLK_PER_BIT+1`; CPB 104 vs 104.1667) | Echo image clean across ±3% baud. Bias is ~4 clocks of the 52 available |
| Metastability / no `ASYNC_REG` on `uart_receiver.sv:16` | 83 ns period gives astronomical MTBF; echo clean for 5 min (~3.3 MiB each way) |
| `uart_receiver` / `uart_transmitter` themselves | Same 5-min echo soak — it instantiates both **unmodified** |
| Switching noise / SSO from the 30 bank-14 SRAM pins | No failure clustering across `sram_roundtrip`'s six patterns |
| TX→RX crosstalk (J17/J18 are the `IO_L7P`/`L7N` pair, physically adjacent) | Echo image has the identical pinout and runs full duplex |
| Timing closure | `build/cmod_a7/reports/`: WNS 34.129 ns of 83.333 ns, TNS 0, 0 failing endpoints. DRC is advisory-only |

**Leading hypothesis.** The fault is logical and lives in `uart_interface.sv`, not in the UART
primitives or the physical layer. `uart_echo.sv` pushes on `rx_byte` unconditionally every
cycle; `uart_interface` consumes it **only** in `IDLE`, `RX_ADDR`, `RX_LEN`, `WR_RX`, `IMEM_RX`
and silently drops it everywhere else. `SEND_STATUS` + `SEND_STATUS_WAIT` is a ~1050-clock
blind window — a full byte time — sitting exactly where the host sends its next command. There
is no recovery because `UART_RX_TIMEOUT = 0` (`tpu_top.sv:68`, `cmod_a7_top.sv:56`) compiles
the mid-frame abort at `uart_interface.sv:346` out entirely. **This is why the echo image
cannot reproduce it** — the echo has no blind states and no protocol to desync.

**Next experiments**, in value order: (1) `write_mem` with `len=1` in a loop vs `len=4096` —
same byte count, wildly different turnaround-to-payload ratio, so failure rate scaling with
*commands* vs *bytes* localises it; (2) set `UART_RX_TIMEOUT` non-zero (~`20 * UART_CPB`) —
masks rather than fixes, but turns a NAK flood into one legible timeout; (3) add RX→TX
turnaround to the echo image, which is the one traffic shape it never exercises.

See `docs/uart_selftest.md` for the echo self-test (`make echo`, `board=cmod_a7_echo`,
`host/uart_echo.py`). **Reflash `board=cmod_a7` before running `test_uart_link.py` or
`run_program.py`** — they time out against the echo bitstream.

## Coding Practices

- **Do**: read files to gain a good understanding of code, ask questions when design choices are unclear,
and write code. Plan out anything in markdown files when it is a large task.
- **Don't**: run code or any other commands (such as checking for packages, etc) unless otherwise told to do so.
If extra context is needed about my development environment or feedback is needed for your code, please ask.
