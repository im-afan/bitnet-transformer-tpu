# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small decoder-style transformer trained to do multi-digit **addition** (a character-level
sequence model), plus custom hardware/backend accelerators for its core attention/matmul ops.
The PyTorch model in `model/` is the **golden reference**; every accelerator in `accel/` is
validated numerically against it.

The current front line is the TPU, and it is **caught up with the model**.
[`accel/tpu/fw/adder.c`](accel/tpu/fw/adder.c) runs the whole int4 adder model —
four transformer layers plus the output head — as one firmware kernel, 518
commands and 1544 bytes of RISC-V, verified against the ISS and the RTL and
scored on the addition task:

| | |
| --- | --- |
| `cd accel/tpu/tb && make fw FWPROG=adder` | 526 879 checks, **0 errors**, 439 917 clocks, 518/518 commands matched |
| `python accel/tpulang/adder_export.py -n 256` | **100.00% exact-sequence, 100.00% token** on `model/saved/int4_d64_f256_l4.pt`, identical to the PyTorch QAT model |

Everything is int4 now, weights *and* activations, on both sides: the RTL and
`iss.py` take **int4 weights in a row-major 4-bit packed layout** and every
narrow clips to `[-8, 7]`. Row-major inverted the attention transpose relative
to the retired ternary kernel — `P@V`'s weight operand is now the free one and
`Q@K^T`'s is the one needing the transposing DMA. `model/quant.py` and
`model/calibrate.py` are the last things still describing the ternary/int8
model and are stale against `model/transformer.py`.

How the model got here — the model is not int8-quantizable *post-hoc*, and QAT
is what fixed it — is [The int8
finding](#the-int8-finding-and-why-the-model-keeps-changing).

## Commands

Python is a package rooted at the repo, so **run from the repo root** with `-m` so imports
resolve (`import model.transformer`, etc.):

```bash
python -m model.train --arch int4_vanilla       # train (arch: vanilla | gqa | int4_vanilla)
python -m model.tests.test_inference --arch int4 --model-path model/saved/colab_ternary_mha_small.pt
python -m model.calibrate --model-path model/saved/colab_ternary_mha_small.pt   # int8 act scales
python -m model.quant --model-path model/saved/ternary_mha.pt  # int8 accuracy bench
python accel/cuda/tests/test_cuda_mha.py        # CUDA kernel vs. its own reference — needs a GPU
```

`model/quant.py` is the **hardware-exact** int8 benchmark: integers end to end, one
`clip_int8((acc*m0 + 2**(n-1)) >> n)` requant per site, so its accuracy number is what the TPU
would score rather than an estimate. `model/calibrate.py` is the *fake-quant* path
(`TernaryLinear.quantize_activations`, float32 throughout) and answers a different, weaker
question. Useful flags: `--check` (instrumented forward vs `Model.forward`), `--fake-quant`
(the same scheme in float — if it disagrees with the integer run, the fixed-point code is the
suspect, not the model), `--words`, `--dynamic-range`, `--dyt-scale calibrated`.

The numbers below are the **PTQ-era** ones and are still what that checkpoint scores; the
last `model/saved/ternary_mha.pt` (QAT, `layers=2`) scored 100.00% / 100.00% instead — see
[The int8 finding](#the-int8-finding-and-why-the-model-keeps-changing).

**Measured on `colab_ternary_mha_small_dyt.pt` (256 problems): float 93.36% exact-sequence /
99.59% token, int8 activations 0.00% / 51.56%.** The fake-quant cross-check lands on
0.00% / 51.52%, so this is the scheme, not the arithmetic. Mechanism: DyT saturates
*completely* — `x`/`x1`/`x2` all have median = max = 1.0, i.e. the residual stream is a sign
vector — while `Wo(A)` grows unbounded with depth, so the two addends of the attention
residual differ by 9.6x at L0 and **3630x at L3**. `vecadd` takes two int8 operands at one
scale, so pinning to `s_x` clips 71% of `O` (92% at L3). This is `adder_kernel.md` §8.1's
finding with a sharper mechanism: DyT bounds the residual stream but nothing bounds what is
added to it.

**`model/saved/int4_d64_f256_l4.pt` is the live checkpoint** — the `adder_int4_vanilla`
retrain, QAT throughout, **100.00% exact-sequence / 100.00% token** over 256 problems both
in PyTorch and through `accel/tpu/fw/adder.c` on the ISS. It is what
`accel/tpulang/adder_export.py` defaults to. Untracked, like every checkpoint here.

`model/saved/ternary_mha.pt` is the last *ternary* QAT checkpoint (2 layers) and
**predates ternary K/V, the removal of the positional encoding, the ternary output head
and `layers=4`**. It no longer loads at all: `fc.weight` is an unexpected key against a
`TernaryLinear` head. There is also **no ternary arch to retrain it with** —
`adder_ternary_vanilla` was replaced by `adder_int4_vanilla`, so restoring the
ternary/int8 path means restoring that factory (git history) first.
The three `colab_ternary_mha_small*.pt` are committed but **deleted in the working tree**, so
any command naming one fails with `FileNotFoundError` until it is restored from git; the old
`colab_vanilla_mha.pt` / `colab_gqa.pt` / `colab_ternary_mha.pt` were deleted. `test_inference.py`
defaults to `saved/colab_ternary_mha_small.pt`, which is wrong relative to the repo root — pass
`model/saved/...`.

`test_cuda_mha.py` JIT-compiles the CUDA kernel via `torch.utils.cpp_extension.load`, so it
needs a CUDA toolchain + GPU. It asserts max abs diff < 1e-2 against a **softmax** einsum
reference it defines inline — see the CUDA note below; it no longer matches the model.

The venv is checked in at `.venv/` (Python 3.12). There is no requirements.txt; deps are just
`torch` (+ jupyter for `model/notebook.ipynb`), plus `pyserial` for the FPGA host driver.

TPU stack. **There is no assembler and no `.tpu` language any more** — the TPU has
one command producer, PicoRV32 firmware in `accel/tpu/fw/`:

```bash
make -C accel/tpu/fw                        # C firmware -> matmul.hex (RISC-V gcc)
make -C accel/tpu/fw PROG=adder             # ...or mha / ffn / matmul_loop
make -C accel/tpu/fw trace PROG=adder       # the kernel's command trace, host cc only
python accel/tpulang/fw_vectors.py -t <trace> -o accel/tpu/tb/vectors_fw -k adder
python accel/tpu/host/run_fw_matmul.py --dry-run   # operands + reference, no board
cd accel/tpu/tb && make fw FWPROG=adder     # the kernel through the whole core (~3 min)
cd accel/tpu/tb && make list                # RTL testbenches (Icarus)

python accel/tpulang/adder_export.py -n 256          # accuracy on the addition task
python accel/tpulang/adder_export.py --dump-rq -n 0  # the 16 requant words per layer
```

`make` targets in `tb/`: `sim` (default TB), `cosim` (host driver vs RTL over a
simulated UART), `fw` / `fwsweep` (C firmware; `fw` needs a RISC-V gcc, and
regenerates golden vectors from the kernel's own native trace first),
`echo`/`mem`/`bram` (bring-up images), `wave`, `list`, `all`.

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
- **Every activation is int4, K and V included.** The old `q_kt`/`q_vt` — a second narrow
  rounding K and V onto `{-1,0,1}` on top of the int8 `q_k`/`q_v` — are **gone**, and so is
  the `TernaryQuant` helper. They existed only because the MXU multiplies an int8 activation
  by a ternary *weight* and cannot multiply int8 by int8, so the operand landing on the
  weight side had to be a trit (`adder_kernel.md` §2.5). With weights and activations at the
  same width the second rounding does nothing but lose precision.
- **There is no positional encoding.** `Model.positional_encoding` is gone and
  `Model.forward` embeds and nothing else. The only position signal left is the causal
  mask; ReLU attention has no source-axis normalization, so the magnitude of `P @ V`
  carries the count of visible keys. The same removal under softmax would leave the
  model position-blind.
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
- **int4 weights.** `Int4Linear` quantizes weights to `[-8, 7]` scaled by `absmax/7`, with
  `RoundClip` (now taking explicit `qmin`/`qmax`) providing the straight-through-estimator
  backward. The scale is **detached** — an absmax gradient lands entirely on one element —
  unlike `TernaryLinear`, whose absmean averages over the tensor and can stay live. The
  weight is `[in_dim, out_dim]` with `x @ w`, the *transpose* of `nn.Linear`, so int4 and
  float checkpoints are not interchangeable. `make_linear(..., use_int4)` selects it —
  **including `Model.fc`**. A float config (`adder_vanilla`, `adder_gqa`) still gets an
  `nn.Linear` head, which is the only thing `quantize_head` / `dynamic_fake_quant` still
  apply to. The head's output is never requantized — an argmax does not care about scale.
  `TernaryLinear` is retained but **no config builds one**; it is what `model/quant.py` and
  the `accel/` export path import.
- **int4 activation quantization is QAT, not post-hoc.** Every requant site is an
  `ActQuant` module holding one per-tensor scale, learned by LSQ (`FakeQuant` implements the
  straight-through input gradient and the analytic scale gradient). Sites the hardware *pins*
  to another tensor's scale share one `ActQuant` **instance** — `q_o`/`q_xo` are the residual
  stream's `x_quant`, `q_p is q_s`, `q_hr is q_h` — so the sharing in `__init__` is load-bearing,
  not a shorthand. The default grid is `INT4_QMIN, INT4_QMAX = -8, 7`. DyT outputs are pinned
  to `fixed_scale=1/7` (`hardtanh` bounds them analytically) with `qmin=-7`, symmetric because
  hardtanh is odd. `set_quant_enabled(model, False)` turns them all off — **necessary to get a
  real float baseline**, since they are live by default. `TernaryLinear.act_scale` is the *old*
  PTQ buffer and is vestigial; `model/calibrate.py` drives that dead path.
- **MoE is gone** — the `use_moe` expert FFN was removed, along with the vanilla/GQA checkpoints.
- **Named configs** at the bottom (`adder_vanilla`, `adder_gqa`, `adder_int4_vanilla`) are
  the source of truth for hyperparameters and are wired to the `--arch` choices in `train.py`.
  `adder_int4_vanilla` is `d=64, f=256, layers=4, q_heads=kv_heads=4` (so `head_dim=16`),
  `use_int4=True, use_bias=False`. The biases are off because the TPU's VPU has no
  row-broadcast operand (see the comment on that factory).
  It replaced `adder_ternary_vanilla` (`d=128, f=128`, ternary weights, int8 activations),
  and **the `accel/` stack has followed**: the MXU multiplies an int4 weight by an int4
  activation, `accel/tpu/fw/adder.c` is this exact shape, and
  `accel/tpulang/adder_export.py` scores it. `model/quant.py` and `model/calibrate.py`
  are the leftovers — both still describe the ternary/int8 model and both fail on the
  missing `adder_ternary_vanilla` rather than exporting something wrong.

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
- **The VPU implements seven ops and nothing else.** `VOP_DOT`, `ADD`, `RELU`,
  `REQUANT`, `DYT`, `TQUANT`, `VECMATMUL`. `TQUANT` (`0x22`, `VOP_TQUANT = 17`) is the
  newest: `requant`'s fixed point clipped to +-1, written **2 bits wide** in the MXU's
  weight encoding, four elements to a byte — int8 in, a packed ternary weight column
  out. It is what lets an activation be a weight operand, and therefore what put
  attention's `Q@K^T` and `P@V` on the array. With `Model.fc` ternary too, `VECMATMUL`
  and `VOP_DOT` now have **no caller in the shipped kernel at all** — both are kept
  for the model shape that needs an int8 x int8 matmul, and dropping them would mean
  dropping the whole reduction path plus its counter block, which is an area decision
  rather than a deletion. Its `vlen` must be a multiple of 4 (the write strobe is per
  byte) and its destination advances a quarter as fast as its source. `GELU`, `EXP`, `SQUARE`, `ELEMENT_MUL`, `SCALAR_MUL/ADD/DIV`,
  `REDUCEMAX`, `REDUCESUM` and the `SOFTMAX` macro op were **removed** — they served a
  softmax/LayerNorm/GELU model and the current one is ReLU attention + DyT + ReLU FFN.
  Gone with them: both 256-entry activation ROMs, `rtl/luts/`, `accel/tpulang/luts.py`,
  the `GELU_INIT`/`EXP_INIT` parameters, the restoring divider, the softmax sequencer, the
  reduction path's **max fold** (`DOT` is the only reduction, so `acc` always opens at 0),
  `cfg vscalar`, and `examples/softmax_row.tpu` / `softmax_rows.tpu`. Measured by OOC
  synthesis of `vpu` alone: **10012 → 5162 LUTs (−48%), 897 → 667 FFs (−26%), 90 → 32 DSPs
  (−64%)**. Full table and what restoring softmax would cost: `accel/tpu/docs/vpu.md`
  §Removed ops.
  - **Retired opcodes are holes, not free space.** `0x02`, `0x05`, `0x0A`–`0x0F`, `0x1B`,
    `0x20` are not reallocated, so a stale binary decodes to an unknown op rather than a
    different one. Likewise `cfg` index 9 (`vscalar`) is vacant — renumbering 10–17 would
    silently repoint every `setcfg` in every program. `0x22` is now `tquant`; new ops go
    at `0x23`+.
  - **`vecdot` survives although no kernel issues it**: it is `vecmatmul`'s inner
    primitive, so the datapath is mandatory and the opcode costs one decode arm.
- **`tpu/`** — a SystemVerilog TPU running on a **Digilent Cmod A7-35T** (see
  `accel/tpu/README.md` and `accel/tpu/docs/`). The array, VPU, banked scratchpad, DMA +
  external SRAM, scalar unit and UART link are all synthesizable, and the design **fits**:
  `make bit` completes at 13342 LUTs of 20800 (64%), 7543 FFs, WNS +26.2 ns, timing met
  (measured 2026-08-17). The long-standing "857 LUTs over" note stopped reproducing — the
  MXU is 5930 LUTs post-route against the 14476 post-synth that note was written about, and
  nothing in the memory-path work touched it. `accel/tpu/docs/synth.md` §5 carries the
  current numbers beside the historical ones.
  `host/run_program.py`
  loads a program over UART, runs it, and checks the readback against both the ISS and PyTorch.
  Recent RTL work: `perf_counters.sv` (replaces `cycle_timer.sv`) and the **macro-op ISA**
  (`docs/macro_ops.md`) — phases 0–5 are *built and passing*: `setcfgr`, MXU config strides,
  `matmul_t` (hardware tile loop) and `vecmatmul`. Phase 5's hardware `softmax` was built,
  validated, and then **removed** (see the VPU note above); `layernorm` + the `rsqrt` LUT
  are not built and are now dropped — DyT replaced LayerNorm and needs no macro op. Still stubbed: the inter-TPU LINK (`wrneigh` is a completing no-op).
- **Dispatch is a 128-bit macro-op pushed into a per-unit queue, not a wire bundle
  plus a global config file.** `cmd_queue.sv` + `cmd_{mxu,vpu,dma}.sv` sit in front
  of each unit; a command carries its own operands and geometry, so nothing an
  earlier dispatch (or an earlier *program*) left in a register can reach it. Two
  producers push: `scalar_unit.sv` still runs tpulang and still waits after every
  dispatch — so **every `.tpu` program, `iss.py` and every golden vector is
  unchanged** — and `cpu_subsys.sv` (PicoRV32 + AXI4-Lite, `rtl/vendor/picorv32.v`)
  pushes the same commands from firmware. Firmware is `accel/tpu/fw/`: `tpu.h` is
  the MMIO aperture plus one builder per command, and the Makefile wants a
  bare-metal RISC-V gcc at **`-march=rv32ic_zmmul`** — *not* `rv32imc`, the core
  is built with `ENABLE_DIV(0)` and a `div` traps. Two C kernels, same
  `[8x32] @ [32x16]` problem and same addresses: `matmul.c` (one `matmul_t`, the
  array walks the 4x2 tile grid) and `matmul_loop.c` (the grid walked in C as 8
  single-tile dispatches, `.acc` across the contraction — the firmware analogue
  of `tiled_matmul.tpu` vs `tiled_matmul_hw.tpu`). `host/run_fw_matmul.py` loads
  either over `'I'` (word address bit 12 picks the CPU's RAM over IMEM) and
  checks the result; `make -C accel/tpu/fw PROG=matmul_loop` builds the second. Two more kernels
  exercise the rest of the machine: **`ffn.c`** (X@W1 -> relu -> requant -> @W2)
  is the first firmware to issue a **VPU** command at all, and **`mha.c`** (one
  head of ReLU attention) adds the **transposing DMA** and the `quant4` pack.
  Both pass through `make fw FWPROG=ffn|mha` and are checked against an
  independent Python reference as well as the ISS.
  Both build with Homebrew `riscv64-elf-gcc` 16.2.0 and both **pass in
  simulation** — `cd accel/tpu/tb && make fw [FWPROG=matmul_loop]` runs the image
  out of `FW_INIT` through the whole core, 129 checks, 0 errors. **Not yet run on
  hardware.** `make fwsweep` walks the shape (both kernels take `M=`/`KTILES=`/
  `NTILES=` at build time) and measures the point of the exercise: the hardware
  tile walk costs the CPU **~380 clocks at every shape** (5 commands, flat over a
  205x range of array work), the C loop costs **~85 clocks per dispatch**, of
  which only `max(0, 85 − MXU-clocks-per-tile)` is exposed — zero once M > ~24.
  `docs/picorv32_migration.md` §9.6–9.7 carry the breakdown; §9.5 now also records
  that the `qfull` counter is **structurally 0** (both producers gate `cmd_we`
  with `!cmd_full`, so `p_cmd_we & p_cmd_full` never fires) and proves nothing.
  `docs/picorv32_migration.md` is the
  design record and carries the format table, the measured baseline and the phase
  plan; §0's table is what actually passes today. **Measured cost of the plane on
  the four-layer model: 690 705 → 693 107 clocks (+0.35%), byte-identical
  output.** Issue overhead went 6 914 → 9 229 clocks (1.00% → 1.33% of the run);
  the units gave back 1 536 of that (the requant literal), and the DMA's skid
  drain cost 1 622. None of the ~1.45x overlap is claimed yet — the scalar unit
  still waits after every dispatch, and `ovlap` reads 0.
  - **The requant `{m0,n}` is now a literal in the command**, not a scratchpad
    address. Same 16 bits either way, and it deleted a two-state fetch in both
    `mxu.sv` and `vpu.sv`. The scalar unit, whose ISA still names an *address*,
    reads the word over the S port before packing — a shim that exists only while
    tpulang is a producer.
  - **`scratchpad.sv`'s exclusivity invariant is gone.** It held only because
    issue-and-wait serialized the units; queues break it on purpose. Arbitration
    is now real, with grants (`V_rgnt`/`s_rgnt`/`dma_rgnt` and the write pair),
    ordered so the requester that cannot stall wins: reads `A>W>C>V>s>DMA`,
    writes `C>V>s>DMA`. The VPU freezes its FSM for a clock when denied; the DMA
    parks fill bytes in a skid buffer and, past that, stops the SRAM read stream
    through `sram.sv`'s new `dout_ready`. **If you make two units run at once,
    check every path into the scratchpad takes its grant back** — a denied
    requester that ignores it loses the access silently.
  - **`swait` stopped meaning "control overhead"** once dispatch went through
    queues. The counter block is 10 wide now: 0–6 unchanged (so the UART `'T'`
    reply stays prefix-compatible), plus `idlec` (clocks with *no* unit busy —
    this is what instruction overhead costs), `qfull` and `ovlap`. Under the
    scalar unit `ovlap` and `qfull` are identically 0, which is the check that
    the command plane is in the path and not yet changing behaviour.
- **`sram.sv` moves ranges, not bytes.** One request is a start address, a byte count and an
  address stride; `dma.sv` issues one range per source row (in linear mode, one for the whole
  transfer) and streams it. **1 clock/byte on fills, 2 on spills**, against ~8 before, and the
  full adder model through `tb/tpu_top_tb.sv` went **1 226 722 → 541 590 clocks (2.27x)** with
  byte-identical output — DMA fell from 66% of the run to 24%. Two things to know before
  touching it: writes are two clocks because **WE# is generated on the falling edge**, so its
  rising edge lands half a clock clear of the address/data change (the chip's address hold is
  0 ns, i.e. met by skew alone if you drive it from the rising edge — the failure mode is a
  byte written to its *neighbour*, and both `sram_tb` and `dma_tb` now check for it); and the
  `stride` exists for `wrmem.t`, whose destination is column-major and would otherwise be one
  range per byte. `CLOCKS_PER_ACCESS` is now *extra* clocks per beat and is 0 on every board.
  `accel/tpu/docs/dma.md` §7 and `rtl/sram.sv`'s header are the full story.
- **19-bit DRAM addressing from programs.** `scalar_unit.sv` used to truncate a `rdmem`/`wrmem`
  DRAM address to 16 bits, so a *program* could only reach the low 64 KB of the 512 KB SRAM even
  though the host always could. Widening it to `MEM_ADDR_W = 19` is what let the 96 KB of model
  weights stay resident. It broke three things that were invisible while both widths agreed
  below 64 KB, all documented in `accel/tpulang/adder_kernel.md` §2: `li` had to stop
  sign-extending, the testbenches' DRAM byte maps were sized off `ADDR_W` (so high expectations
  were *silently dropped* by `$readmemh`), and `tpu_top_tb`'s backdoor pokes truncated too.
  **When something addresses DRAM, check it is not using `ADDR_W`.**
- **`tpulang/`** — three files, all that is left of the software stack after the
  assembler and the `.tpu` language were deleted. `iss.py` (bit-exact with the RTL;
  the instruction decoder is gone, `exec_command`/`run_trace` are the way in),
  `fw_vectors.py` (a firmware kernel's command trace → golden DRAM images + the
  expected command stream, plus the **one** definition of each kernel's synthetic
  operands), and `adder_export.py` (real checkpoint → requant table → trace →
  accuracy). The directory name is now a fossil.
- **`fw/adder.c` is not an example — it is the whole shipped model** (four layers
  + the output head) in **518 commands and 1544 bytes**, one program, one run.
  `LAYERS` is the only thing that moves when the model's depth changes. Its
  byte-level contract — DRAM/scratchpad maps, the 16 requant `{m0,n}` words per
  layer, the row-major int4 packing, why K is transposed and V is not — is that
  file's header and `accel/tpu/fw/README.md`. Read those before touching the
  kernel or the host staging.
  - **The requant table is a compile-time input**, because the `{m0,n}` word is a
    literal in the macro-op and the CPU has no path to DRAM or the scratchpad.
    `fw/adder_rq.h` is the checked-in default and is tuned for `fw_vectors.py`'s
    synthetic operands, so the RTL regression needs no checkpoint;
    `adder_export.py` generates a real one and compiles the kernel against it
    with `-DADDER_RQ_H`.

When touching attention numerics, keep the implementations in sync:
`model/transformer.py` (reference), `fw/adder.c` + `iss.py`, and the RTL.

## The int8 finding, and why the model keeps changing

> **The numbers in this section describe the architecture *before* ternary K/V.**
> `model/saved/ternary_mha.pt` was fitted with int8 K and V, a positional encoding and
> `vecmatmul` attention; it has no `q_kt`/`q_vt` and its weights never saw a ternary K,
> so `adder_export.py` on it no longer measures the shipped kernel. **The model needs
> retraining** before rows 8-10 of `adder_kernel.md` §7 can be repeated, and the ternary
> config that would train it no longer exists — see the note at the top of this file.
> `QATCalibration` now falls back to the
> calibration set for any site whose `ActQuant.initialized` flag is clear, so a stale
> checkpoint degrades rather than silently quantizing with a scale of 1.0. Everything
> below about *why* PTQ fails here is unaffected by the change.

**Resolved on the `qat` branch: train through the quantizers and the same kernel scores
100.00% exact-sequence / 100.00% token** (`adder_export.py -n 256 -c 128 --iss-check` on
`model/saved/ternary_mha.pt`, ISS cross-check 0/416 logits differ; `adder_kernel.md` §7.6c).
Nothing in the kernel changed *at that point* — it was byte-identical to the one that scored
0.00% below; the ternary-K/V rewrite came after. What
changed is that every requant site is an `ActQuant` present during training with an LSQ-learned
scale, so the clipping is part of the fitted function rather than damage inflicted afterwards.

Three consequences worth internalizing before reading the PTQ history below:

- **The "float" row is no longer a ceiling.** Removing the quantizers from a QAT checkpoint
  gives 0.00% — a network the weights were never trained for, not the model's true accuracy.
  `quant.benchmark` disables the sites explicitly for that row; `--fake-quant` (same scheme in
  float32) is the honest cross-check, and it also scores 100.00%.
- **Saturation rates stopped being a health check.** L0's `RQ_O` word is `3838/2^3` = 479.75 and
  clips **99.88%** of `O` — the layer learned to use the requant as a `sign()`. `report_clips`
  localizes damage only for a PTQ checkpoint.
- **Scales come from the checkpoint, not from calibration.** `quant.QATCalibration` reads the
  learned `ActQuant.scale` (`--scales qat|calib|auto`). Re-deriving by absmax moves every
  rounding grid the weights were fitted against, so it is not a neutral substitution — the
  control is `--scales calib` on the *same* QAT checkpoint, which scores **0.00%**. It takes
  `.abs()`: LSQ can drive a scale negative and `ActQuant.forward` uses the magnitude.

The rest of this section is the **post-training** quantization history. It is still correct
about those checkpoints, and its account of the `vecadd`-single-scale constraint is still what
forces `RQ_O` to be pinned — but it is no longer the live problem.

**The kernel is correct and the PTQ checkpoints are not int8-quantizable.** On
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

The remedy looked like per-channel/per-token activation scales, **which this ISA cannot
express** (`requant` takes one `{m0,n}` per dispatch; `matmul_t.rq` reads one `cfg scalar`) — so
the options were: retrain with normalization (`tpunn.md` §1 measured 98.4% with LayerNorm), or
add a per-channel requant path to the VPU. The `no-layernorm` branch tried the first with DyT
(one multiply + a clamp, which the VPU already does); that fixed the diagnosed mechanism and
still scored 0.00%, because what it exposed was the residual *addend* — `vecadd` puts `X` and
`O = Wo(A)` on one scale while they differ by up to 7259× (`adder_kernel.md` §7.6b).

**QAT is the third option and it needs no VPU change at all**: rather than finding scales the
trained weights tolerate, it trains weights that tolerate the scales. That is what the `qat`
branch did, and it is why the header above supersedes this analysis.

Related, from tuning the synthetic requant words: moving `RQ_A` by two shifts (`{1,6}`→`{1,8}`)
took `A` from 45% saturated at ±127 to 99.3% *exactly zero*. Each layer's requant multiplies the
residual stream by a constant, so an error compounds geometrically with nothing renormalizing it.

## Stale docs

Several docs describe an earlier model or ISA. When they disagree with the code, the code wins;
fix the doc if you are in there anyway.

- **`model/README.md`** — the most stale. Describes LayerNorm/post-LN, softmax attention, GELU
  in the FFN, sinusoidal positional encoding, `mha_torch`/`slow_mha_cuda`, MoE, `f=512` for the
  ternary config, and the deleted vanilla/GQA/`colab_ternary_mha` checkpoints. Its *invariants* sections (`EQUALS_POS`, the
  sampling, the `TernaryLinear` weight orientation, the double residual) are still correct.
- **`accel/tpulang/tpunn.md`** — superseded by `adder_kernel.md`; written against the older
  model (LayerNorm, GELU, softmax, biases, `f=512`) and the pre-macro-op ISA. Still the only
  record of the 98.4% ternary+int8 number quoted above.
  It additionally describes the `gelu`/`exp` LUTs and `luts.py`, all now deleted.
- **`accel/tpu/docs/README.md`** — fixed for the VPU trim, but its own §1/§4 still carry a
  "the component doc wins" caveat; `docs/vpu.md` is authoritative on the op set.

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

A full-model run is `make fw FWPROG=adder` in `accel/tpu/tb`: **439 917 clocks**, about
**3 minutes** of Icarus, 526 879 checks. It regenerates the golden vectors from the
kernel's own native trace first, so a stale image cannot silently be compared against
the wrong expectations. `fw_matmul_tb.sv` does not dump a VCD at all, and the tb
Makefile already passes the kernel a 60 ms watchdog (the 2 ms default is sized for the
small kernels and would trip on this one while it worked perfectly).

The split is `mxu = 206 361`, `dma = 131 168`, `vpu = 84 352`, `idlec = 18 035` (4.1%,
which is what the CPU costs as a command producer). `qfull` and `ovlap` are both 0:
nothing overlaps yet, because the kernel fences after every cross-unit dependency.

For an edit-run loop, use the **ISS** rather than the RTL — one four-layer forward is
**1.9 s**, so `python accel/tpulang/adder_export.py -n 4` is a ~10 s check that the
kernel still computes the model. The RTL run is what proves the *hardware* agrees; it
is not where you iterate. `make fw FWPROG=ffn` (or `mha`) is the ~5 s smoke test that
the dispatch plane still works at all.

**A long Icarus run prints nothing until it halts, which makes "slow" and "deadlocked"
look identical from outside.** Redirect the log to a file rather than piping it through
`tail`/`head`, which buffer the whole stream.