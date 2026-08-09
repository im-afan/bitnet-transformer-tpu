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
`torch` (+ jupyter for `model/notebook.ipynb`), plus `pyserial` for the FPGA host driver.

TPU stack (these are plain scripts, not `-m` modules — each adds its own directory to
`sys.path`, so run them by path from the repo root):

```bash
python accel/tpulang/gen_vectors.py -p accel/tpulang/examples/relu_layer.tpu
python accel/tpulang/torch_ref.py                         # example kernels: ISS vs PyTorch
python accel/tpulang/pytpu/examples/transformer_layer.py  # build + verify one layer
python accel/tpu/host/run_program.py --dry-run            # toolchain only, no board
cd accel/tpu/tb && make list                              # RTL testbenches (Icarus)
```

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
- **`tpu/`** — a SystemVerilog TPU running on a **Digilent Cmod A7-35T** (see
  `accel/tpu/README.md` and `accel/tpu/docs/`). The array, VPU, banked scratchpad, DMA +
  external SRAM, scalar unit and UART link are all synthesizable and fit; `host/run_program.py`
  loads a program over UART, runs it, and checks the readback against both the ISS and
  PyTorch. Stubbed: the inter-TPU LINK (`wrneigh` is a completing no-op).
- **`tpulang/`** — the TPU's software stack: the `.tpu` assembly language + `assembler.py`,
  `iss.py` (bit-exact with the RTL), `luts.py` (the VPU's gelu/exp ROMs), `torch_ref.py`
  (independent PyTorch checks), and `gen_vectors.py` (golden vectors for `tpu/tb/`).
  `tpulang/pytpu/` composes parameterized `.tpu` templates into whole layers — it builds one
  quantized transformer layer in 861 of 1024 instruction words.

When touching attention numerics, keep the three implementations in sync: `mha_torch`
(reference), the CUDA kernel, and any future TPU block — they all implement the same 5-D layout
and masking convention.

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

## Coding Practices

- **Do**: read files to gain a good understanding of code, ask questions when design choices are unclear,
and write code. Plan out anything in markdown files when it is a large task.
- **Don't**: run code or any other commands (such as checking for packages, etc) unless otherwise told to do so.
If extra context is needed about my development environment or feedback is needed for your code, please ask.
