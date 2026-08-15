# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small decoder-style transformer trained to do multi-digit **addition** (a character-level
sequence model), plus custom hardware/backend accelerators for its core attention/matmul ops.
The PyTorch model in `model/` is the **golden reference**; every accelerator in `accel/` is
validated numerically against it.

The current front line is the TPU: `accel/tpulang/examples/adder_model.tpu` runs the whole
ternary adder model as one program, verified against PyTorch and the RTL. What that exposed —
the model is not int8-quantizable without normalization — is what the `no-layernorm` branch is
now reacting to. See [The int8 finding](#the-int8-finding-and-why-the-model-keeps-changing).

## Commands

Python is a package rooted at the repo, so **run from the repo root** with `-m` so imports
resolve (`import model.transformer`, etc.):

```bash
python -m model.train --arch ternary_vanilla    # train (arch: vanilla | gqa | ternary_vanilla)
python -m model.tests.test_inference --arch ternary --model-path model/saved/colab_ternary_mha_small.pt
python -m model.calibrate --model-path model/saved/colab_ternary_mha_small.pt   # int8 act scales
python -m model.quant --model-path model/saved/colab_ternary_mha_small_dyt.pt  # int8 accuracy bench
python accel/cuda/tests/test_cuda_mha.py        # CUDA kernel vs. its own reference — needs a GPU
```

`model/quant.py` is the **hardware-exact** int8 benchmark: integers end to end, one
`clip_int8((acc*m0 + 2**(n-1)) >> n)` requant per site, so its accuracy number is what the TPU
would score rather than an estimate. `model/calibrate.py` is the *fake-quant* path
(`TernaryLinear.quantize_activations`, float32 throughout) and answers a different, weaker
question. Useful flags: `--check` (instrumented forward vs `Model.forward`), `--fake-quant`
(the same scheme in float — if it disagrees with the integer run, the fixed-point code is the
suspect, not the model), `--words`, `--dynamic-range`, `--dyt-scale calibrated`.

**Measured on `colab_ternary_mha_small_dyt.pt` (256 problems): float 93.36% exact-sequence /
99.59% token, int8 activations 0.00% / 51.56%.** The fake-quant cross-check lands on
0.00% / 51.52%, so this is the scheme, not the arithmetic. Mechanism: DyT saturates
*completely* — `x`/`x1`/`x2` all have median = max = 1.0, i.e. the residual stream is a sign
vector — while `Wo(A)` grows unbounded with depth, so the two addends of the attention
residual differ by 9.6x at L0 and **3630x at L3**. `vecadd` takes two int8 operands at one
scale, so pinning to `s_x` clips 71% of `O` (92% at L3). This is `adder_kernel.md` §8.1's
finding with a sharper mechanism: DyT bounds the residual stream but nothing bounds what is
added to it.

Only ternary checkpoints are committed now (`model/saved/colab_ternary_mha_small*.pt`); the old
`colab_vanilla_mha.pt` / `colab_gqa.pt` / `colab_ternary_mha.pt` were deleted. `test_inference.py`
defaults to `saved/colab_ternary_mha_small.pt`, which is wrong relative to the repo root — pass
`model/saved/...`.

`test_cuda_mha.py` JIT-compiles the CUDA kernel via `torch.utils.cpp_extension.load`, so it
needs a CUDA toolchain + GPU. It asserts max abs diff < 1e-2 against a **softmax** einsum
reference it defines inline — see the CUDA note below; it no longer matches the model.

The venv is checked in at `.venv/` (Python 3.12). There is no requirements.txt; deps are just
`torch` (+ jupyter for `model/notebook.ipynb`), plus `pyserial` for the FPGA host driver.

TPU stack (these are plain scripts, not `-m` modules — each adds its own directory to
`sys.path`, so run them by path from the repo root):

```bash
python accel/tpulang/assembler.py accel/tpulang/examples/adder_model.tpu   # 196 words
python accel/tpulang/gen_vectors.py -p accel/tpulang/examples/adder_model.tpu -o vectors_model
python accel/tpulang/torch_ref.py                         # every example kernel: ISS vs PyTorch
python accel/tpulang/adder_export.py -n 256 --iss-check   # real checkpoint → integers, accuracy
python accel/tpu/host/run_program.py --dry-run            # toolchain only, no board
cd accel/tpu/tb && make list                              # RTL testbenches (Icarus)
```

`make` targets in `tb/`: `sim` (default TB), `uart` (`tpu_top_uart_tb`), `cosim` (host driver
vs RTL over a simulated UART), `echo`/`mem`/`bram` (bring-up images), `wave`, `list`, `all`.

## Architecture

**`model/transformer.py`** — the whole model. It has changed a lot recently; the shape below is
what is in the tree, not what `model/README.md` describes (that file is stale — see
[Stale docs](#stale-docs)).

- **5-D attention tensor layout.** Q/K/V are shaped `[batch, tokens, kv_heads, heads_per_q,
  head_dim]` (K/V drop the `heads_per_q` axis) rather than the usual flat head layout. GQA is
  expressed by broadcasting `kv_heads` over `heads_per_q`. The attention math is the einsum
  pair `"btkgh,bskh->btskg"` (scores) and `"btskg,bskh->btkgh"` (weighted values). This layout
  is the contract every accelerator implements.
- **ReLU attention, not softmax.** `P = relu(S + causal_mask)`, with the mask built as
  `triu(ones([T,T]) * -1e9, diagonal=1)`. There is no normalization over the source axis at
  all. (The comment in the code says "revert to softmax if training bad".)
- **Attention is inlined.** The old `mha_torch` / `slow_mha_cuda` helpers are **gone**; the math
  now lives directly in `MultiHeadAttention.forward`. `use_custom_attention` still threads
  `Model.forward → Transformer → MultiHeadAttention.forward` but is **accepted and ignored** —
  there is no CUDA path from the model any more.
- **No LayerNorm.** Removed in `2f6e32f`; `norm1`/`norm2` are now `DyT` (dynamic tanh,
  arxiv 2503.10622) — `hardtanh(x * alpha)` with a single learned scalar and no gamma/beta.
  The committed `colab_ternary_mha_small_nobias.pt` predates DyT (it has *no* normalization at
  all); `colab_ternary_mha_small_dyt.pt` is the DyT run.
- **Double residual.** `MultiHeadAttention.forward` ends in `return O + X`, and
  `Transformer.forward` adds `X` again. Deliberate or not, the checkpoints were fitted to it,
  so a reimplementation must reproduce it.
- **The padding mask is threaded through and never applied.** `attn_mask` reaches
  `MultiHeadAttention.forward` and is unused; only the causal mask is applied.
- **Ternary (BitNet-style) weights.** `TernaryLinear` quantizes weights to {-1,0,1} scaled by
  absmean, with `RoundClip` providing a straight-through-estimator backward. Note the weight is
  `[in_dim, out_dim]` with `x @ w` — the *transpose* of `nn.Linear`, so ternary and non-ternary
  checkpoints are not interchangeable. `make_linear(..., use_ternary)` selects it. **`Model.fc`
  never passes `use_ternary`**, so the output head is always a full-precision `nn.Linear`.
- **int8 activation quantization** (optional, orthogonal to the weights). Each `TernaryLinear`
  carries a persistent `act_scale` buffer, `NaN` = uncalibrated = passthrough.
  `calibrate_activations(model, run_forward)` records a per-tensor absmax and sets
  `act_scale = absmax / 127`; `model/calibrate.py` is the driver (loads with `strict=False`
  because older checkpoints predate the buffer).
- **MoE is gone** — the `use_moe` expert FFN was removed, along with the vanilla/GQA checkpoints.
- **Named configs** at the bottom (`adder_vanilla`, `adder_gqa`, `adder_ternary_vanilla`) are
  the source of truth for hyperparameters and are wired to the `--arch` choices in `train.py`.
  `adder_ternary_vanilla` is now `d=128, f=128, layers=4, q_heads=kv_heads=4, use_bias=False`
  — `f` dropped from 512, and the biases are off because the TPU's VPU has no row-broadcast
  operand (see the comment on that factory and `accel/tpulang/tpunn.md`).

**`model/numbers_data.py`** — synthetic dataset. Non-obvious invariants:

- **Fixed answer position.** `EQUALS_POS = 15` is the index of the first **answer digit** (`=`
  is at 14). Operands are padded so this holds regardless of operand length; `train.py` and
  `test_inference.py` slice with the constant rather than searching for `=`. Implies
  `max_digits ≤ 6` — at 7 digits the pad count goes negative and every slice silently shifts.
- Padding token is `'N'` (`PAD_ID = 12`); `tokenize` builds an additive `-1e9` attention mask
  from the pad positions (`mask + mask.T`) — which the model then ignores.
- Operand lengths are sampled uniformly over digit-count (`_sample_number`), not uniformly over
  value. Answer digits are most-significant-first (no reversed-digit trick).

**`model/train.py`** — `batch_size` vs `mini_batch_size` implements **gradient accumulation**
(`steps_per_batch = batch_size // mini_batch_size`); `optim.step()` only fires every
`steps_per_batch` micro-steps. **Gradient clipping** (`--grad_clip`, default 1.0) is applied to
the *accumulated* gradient immediately before the step — clipping each micro-batch instead
would bound the partial sums separately; the reported `grad_norm` mean/max/clipped counts are
the pre-clip norms. The loss spans `EQUALS_POS-1 : -1` against `EQUALS_POS:`, so the trailing
pad tokens are part of the objective on purpose. Checkpoints go to `model/saved/test_model_*.pt`,
keeping only the last 3 (gitignored; the committed `colab_*.pt` are not produced by this path).

## Accelerators (`accel/`)

- **`cuda/`** (`kernels.cu` + `bindings.cpp`) — hand-written CUDA MHA/GQA kernel. **Now out of
  sync with the model**: it implements *softmax* attention and the model does ReLU attention,
  and nothing in `model/` loads it any more. It still passes its own self-contained test. Treat
  it as legacy unless someone re-syncs it.
- **`tpu/`** — a SystemVerilog TPU running on a **Digilent Cmod A7-35T** (see
  `accel/tpu/README.md` and `accel/tpu/docs/`). The array, VPU, banked scratchpad, DMA +
  external SRAM, scalar unit and UART link are all synthesizable and fit; `host/run_program.py`
  loads a program over UART, runs it, and checks the readback against both the ISS and PyTorch.
  Recent RTL work: `perf_counters.sv` (replaces `cycle_timer.sv`) and the **macro-op ISA**
  (`docs/macro_ops.md`) — phases 0–5 are *built and passing*: `setcfgr`, MXU config strides,
  `matmul_t` (hardware tile loop), `vecmatmul`, and a hardware `softmax`. `layernorm` + the
  `rsqrt` LUT are not built. Still stubbed: the inter-TPU LINK (`wrneigh` is a completing no-op).
- **19-bit DRAM addressing from programs.** `scalar_unit.sv` used to truncate a `rdmem`/`wrmem`
  DRAM address to 16 bits, so a *program* could only reach the low 64 KB of the 512 KB SRAM even
  though the host always could. Widening it to `MEM_ADDR_W = 19` is what let the 96 KB of model
  weights stay resident. It broke three things that were invisible while both widths agreed
  below 64 KB, all documented in `accel/tpulang/adder_kernel.md` §2: `li` had to stop
  sign-extending, the testbenches' DRAM byte maps were sized off `ADDR_W` (so high expectations
  were *silently dropped* by `$readmemh`), and `tpu_top_tb`'s backdoor pokes truncated too.
  **When something addresses DRAM, check it is not using `ADDR_W`.**
- **`tpulang/`** — the TPU's software stack: the `.tpu` assembly language + `assembler.py`,
  `iss.py` (bit-exact with the RTL), `luts.py` (the VPU's gelu/exp ROMs), `torch_ref.py`
  (independent PyTorch checks), `gen_vectors.py` (golden vectors for `tpu/tb/`),
  `adder_export.py` (real checkpoint → integers → accuracy), and `examples/`.
- **`examples/adder_model.tpu` is not an example — it is the whole shipped model** (four layers
  + the output head) in **196 words of 1024**, one program, one run. Its byte-level contract —
  DRAM/scratchpad maps, the 13 requant `{m0,n}` words per layer and where their scales come
  from, the ternary packing, the verification order — is `adder_kernel.md`. Read that before
  touching the kernel or the host staging.
- `pytpu.py` + `pytpu.md` replaced the old `pytpu/` package directory. `core.py` is an empty
  placeholder.

When touching attention numerics, keep the implementations in sync: `model/transformer.py`
(reference), `torch_ref.adder_model`, `examples/adder_model.tpu` + `iss.py`, and the RTL.

## The int8 finding, and why the model keeps changing

**The kernel is correct and the checkpoint is not int8-quantizable.** On
`colab_ternary_mha_small_nobias.pt`, via `adder_export.py -n 256 -c 128 --iss-check`:

| | exact-sequence | token |
| --- | --- | --- |
| float model | **96.09%** | 99.76% |
| int8 activations, this kernel | **0.00%** | 63.23% |

The ISS cross-check passed (0 of 416 logits differ against the independent reference on those
very weights), so this measures the kernel, not a model of it. **Do not go looking for a bug in
the kernel, the calibration, or the requant words** (worst relative error 0.87%). The cause was
established four ways in `adder_kernel.md` §7.6:

1. Fake-quantizing in pure float reproduces it — nothing integer-specific is involved.
2. Any single quantization site is fatal alone (`X1` → 0.00%, Q/K/V → 3.12%, `A` → 1.56%).
3. The model tolerates ±10% *multiplicative* jitter everywhere at 100% exact-sequence, i.e.
   100× more error than int8 injects. It is not fragile to perturbation in general.
4. Quantization error is *absolute*, and a per-tensor scale is pinned by the maximum — so what
   matters is `median/max`, and with no normalization that collapses with depth: 1.3% of layer
   0's residual stream rounds to exactly zero, 91.6% of layer 3's `A` does. The usual outlier
   check misses it because the *top* of the distribution is well behaved (`max/p99.9` ≈ 1.1–2.5).

The remedy is per-channel/per-token activation scales, **which this ISA cannot express**
(`requant` takes one `{m0,n}` per dispatch; `matmul_t.rq` reads one `cfg scalar`). So either
retrain with normalization — `tpunn.md` §1 measured 98.4% exact-sequence with ternary weights
*and* int8 activations back when the model had LayerNorm — or add a per-channel requant path to
the VPU, a much larger change. The `no-layernorm` branch's DyT experiment is an attempt at
option (a) with something cheaper than LayerNorm; DyT is one multiply + a clamp, which the VPU
can already do.

Related, from tuning the synthetic requant words: moving `RQ_A` by two shifts (`{1,6}`→`{1,8}`)
took `A` from 45% saturated at ±127 to 99.3% *exactly zero*. Each layer's requant multiplies the
residual stream by a constant, so an error compounds geometrically with nothing renormalizing it.

## Stale docs

Several docs describe an earlier model or ISA. When they disagree with the code, the code wins;
fix the doc if you are in there anyway.

- **`model/README.md`** — the most stale. Describes LayerNorm/post-LN, softmax attention, GELU
  in the FFN, `mha_torch`/`slow_mha_cuda`, MoE, `f=512` for the ternary config, and the deleted
  vanilla/GQA/`colab_ternary_mha` checkpoints. Its *invariants* sections (`EQUALS_POS`, the
  sampling, the `TernaryLinear` weight orientation, the double residual) are still correct.
- **`accel/tpulang/tpunn.md`** — superseded by `adder_kernel.md`; written against the older
  model (LayerNorm, GELU, softmax, biases, `f=512`) and the pre-macro-op ISA. Still the only
  record of the 98.4% ternary+int8 number quoted above.
- **`accel/tpu/docs/README.md`** — calls `macro_ops.md` a "proposal, not built"; phases 0–5 are
  built. Its own §1/§4 already carry a "the component doc wins" caveat.
- **`accel/tpu/docs/isa.md` §5 and `accel/tpulang/README.md` §1.4** — the grouped instruction
  summaries omit `matmul_t`, `vecmatmul` and `softmax` (the opcode map in isa.md Appendix A.3
  has them).

## Solved: intermittent UART corruption (root-caused 2026-08-07)

**The fault was on the host, not the FPGA. Reading the serial port while the USB-serial
bridge is still transmitting corrupts the byte in flight.** The host's IN requests disturb
the FT2232H's transmit bit timing by roughly half a bit, so the device samples that byte on
its bit boundaries and decodes `sent[k]` **or** `sent[k-1]` for every bit `k` — usually
`value << 1`, sometimes `value | (value << 1)`. Fixed in `host/tpu_uart.py`: `TPUUart._send`
now waits for a frame to clear the wire before anything reads (see that docstring and
`TX_SETTLE_S` / `TX_RATE_SLACK`). All four commands and `test_uart_link.py`'s `raw_exchange`
route through it.

`tpu_uart.py` had always issued `ser.write(frame)` then read the reply immediately, so the
read landed on the *first few bytes of every command* — i.e. the header, where a corrupted
length or address does the most damage. A 16-bit length field arriving one bit left-shifted
is the whole of the original symptom: the device really did read 0x0180 bytes when asked for
0x0140, so "the device sent more bytes than the host asked for" was literal, not a desync
artefact.

**How it was pinned down**, on the echo image (`cmod_a7_echo` reports every byte it received,
so no protocol can hide anything), 64-byte bursts, 6400 bytes per arm:

| host behaviour | corrupted | positions in the burst |
|---|---|---|
| read immediately after write | 72/6400 | 1–6 |
| sleep past the whole burst, then read | **0/6400** | — |
| sleep over only the *first half*, then read | 77/6400 | **33–46** |

The third row is the proof: delaying the read moves the damage to exactly where the read
begins. It is not a probability spread over the burst — it is one mangled byte per burst,
located wherever the host started reading.

**Ruled out — do not re-litigate.** Each cost a real experiment:

| Hypothesis | Killed by |
|---|---|
| Anything in the RTL | `uart_receiver` alone, and the whole `uart_memory` path, decode a bit-exact back-to-back stream perfectly at the real 104.1667 clocks/bit — all 256 byte values, 40 frames. See "the simulation blind spot" below |
| `uart_interface`'s `SEND_STATUS` blind window | Co-simulation passes identically with and without the `rx_hold` fix; `rx_overrun`/`collision` never set on hardware either |
| External SRAM, `sram_controller`, the 30 bank-14 pins, SSO | `cmod_a7_bram` (on-chip BRAM, no SRAM chip, no SRAM pins) reproduces at 23.8% vs 27.5% — indistinguishable |
| Byte value, and the byte before it on the wire | 0x40 corrupts 15.8% in a header slot and 0/5184 in a payload slot; sweeping the predecessor over 0x00/0x01/0x03/0x0f/0x40/0xff changes nothing |
| Position in the burst per se | The damage tracks the *read*, not the offset — see the table above |
| Baud mismatch / sampling phase | Host baud swept 108k–122k: flat 12–30% across ±3%, so no phase minimum. Device sample points are within 1.5 clocks of centre by construction |
| Long low runs on the line, DC/slew recovery | Corruption rate is flat as the predecessor's low run goes 9 → 4 bit times |
| The `activity` LED's load step | Flat ~1.1% at every inter-burst idle from 0 ms to 200 ms, including idles shorter than the LED's 43.7 ms hold |
| `reset_input_buffer()` (PurgeComm) perturbing the bridge | With it, without it, and with it 20 ms early: all ~1.1% |
| Metastability, TX→RX crosstalk, timing closure | Previously ruled out; nothing since contradicts them |

**The simulation blind spot, now fixed in understanding.** `tb/uart_memory_cosim_tb.sv`'s
`drive_byte` clocks every bit for exactly `CPB` clocks, so its host is bit-exact with the
device and has *zero* baud error — a real 115200 host against a 12 MHz CPB=104 device runs at
104.1667. That is why `make cosim` could never see this, and it is worth remembering before
trusting a green co-simulation run about anything analogue or timing-related. Driving the same
RTL at a fractional bit period (accumulator carrying the remainder across bits and bytes) is
still clean, which is what exonerated the RTL.

**Still worth doing.** `UART_RX_TIMEOUT = 0` is what turns one corrupted byte into a
permanently wedged link: the FSM sits mid-frame forever and only a reflash clears it, which
is why a single failure used to cascade into every later test in the suite. Setting it to
`20 * UART_CPB` in `boards/*/board.tcl` makes a corrupted frame cost one legible timeout
instead. That is hardening, not the fix — the fix is in the host — but the link should not
depend on the host never glitching. Measured floor is one byte time; at `200` clocks (~2 bit
times) the abort fires mid-frame during ordinary streaming and every command NAKs.

See `docs/uart_selftest.md` for the echo self-test (`make echo`, `board=cmod_a7_echo`,
`host/uart_echo.py`). **Reflash `board=cmod_a7` before running `test_uart_link.py` or
`run_program.py`** — they time out against the echo bitstream. Note that bitstreams under
`synth/build/` are not rebuilt by `mode=program`; check their mtime against `rtl/` before
concluding anything from a board run.

## Long simulations

A full-model `tpu_top_tb` run is ~2.36 M clocks (23.58 ms simulated at 100 MHz) and dumps
400+ MB of waveform at the defaults, so it needs `-DWATCHDOG_NS=400000000 -DNO_VCD` (both are
options on `tpu_top_tb.sv`, defaults unchanged). The command line is in `adder_kernel.md` §7.7.
The ISS runs the same four-layer forward plus the PyTorch reference and the byte comparison in
~19 s, so leg-B verification is an edit-run loop, not a batch job.

## Coding Practices

- **Do**: read files to gain a good understanding of code, ask questions when design choices are unclear,
and write code. Plan out anything in markdown files when it is a large task.
- **Don't**: run code or any other commands (such as checking for packages, etc) unless otherwise told to do so.
If extra context is needed about my development environment or feedback is needed for your code, please ask.
