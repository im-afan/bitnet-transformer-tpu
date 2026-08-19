# TPU Design Doc

## Component docs

Detailed per-component design notes (this file is the overview):

| Doc                              | Component                                              |
| -------------------------------- | ----------------------------------------------------- |
| [scratchpad.md](scratchpad.md)   | On-chip BRAM working memory, banking, sizing          |
| [mxu.md](mxu.md)                 | Weight-stationary ternary systolic matrix unit        |
| [vpu.md](vpu.md)                 | SIMD vector unit: activations, reductions, attention  |
| [dma.md](dma.md)                 | DMA engine and the external SRAM controller ("DRAM")  |
| [comms.md](comms.md)             | 2D inter-TPU link for scale-out (**stubbed in RTL**)  |
| [scalar_unit.md](scalar_unit.md) | Control processor microarchitecture                   |
| [scalar_unit_pipeline.md](scalar_unit_pipeline.md) | Plan to pipeline it — **proposal, not built**; the RTL is still the multi-cycle FSM |
| [isa.md](isa.md)                 | Guide to writing TPU programs in tpulang (+ encoding/opcode appendix) |
| [macro_ops.md](macro_ops.md)     | Moving tiling/attention into hardware (CISC macro-ops). Phases 0–4 **are built** (`setcfgr`, MXU strides, `matmul_t`, `vecmatmul`); `softmax` was built then removed and `layernorm` is dropped — see its banner |
| [uart_host.md](uart_host.md)     | The host link: frame format, the five commands, arbitration |
| [uart_selftest.md](uart_selftest.md) | The `cmod_a7_echo` bring-up image and how to use it |
| [synth.md](synth.md)             | Vivado build flow, Cmod A7-35T deployment, sizing reality check |

The assembly language that lowers 1:1 onto the ISA is documented in
[`accel/tpulang/README.md`](../../tpulang/README.md), and the host driver that speaks the
protocol in [`accel/tpu/host/README.md`](../host/README.md).

> **This file is the original design sketch**, kept because §1 is still an accurate
> statement of intent. Where it disagrees with a component doc above, the component doc
> wins — §4's instruction list in particular predates the real ISA (see
> [isa.md](isa.md) for what the machine actually implements).

## 1. Overview

This is a design for a TPU that goes on low-end FPGA boards for inferencing transformers.
To minimize area usage, we target highly quantized neural networks: those with weights in ${1, 0, -1}$ 
and 8-bit activations. We target running a ternary version of the digit-predicting adder LLM. We also design for scale-out,
allowing multiple devices to communicate during inference.

We use the same general architecture as google TPU:

### 1. Scratchpad

We will use DMA to access RAM and load into a specific address in our scratchpad memory, which uses FPGA BRAM. 

### 2. MXU

The MXU is a weight-stationary systolic array. It will load the ternary weights from scratchpad memory into registers in each PE.
Then, it loads activations from the scratchpad into the activation buffer and feeds them into the systolic array in a staggered order. 
The scratchpad should consist of enough BRAM blocks to fully load 1 column of activations in 1 clock.
The MXU output immediately gets written to another address in scratchpad memory.

### 3. VPU

The VPU performs the remaining pointwise vector operations using SIMD: `relu`, vector add, the `requant`/`dyt`/`tquant` narrows, and the `vecmatmul` macro op (built on a dot-product reduction). `tquant` is the one that narrows to a *trit* rather than to int8, packed 2 bits wide in the MXU's weight layout — it is what makes K and V ternary and therefore what moved both attention matmuls onto the array. The output head went with them once `Model.fc` became ternary, so `vecmatmul` has no caller in the shipped kernel at all — it is kept for the model shape that needs an int8 × int8 matmul. It is deliberately no larger than that — the activation LUTs, broadcast/scalar ops, divider, reductions and the softmax macro op were removed once the model stopped needing them ([vpu.md §Removed ops](vpu.md#removed-ops)). 
It contains multiple ALUs that act on data from a single scratchpad memory access.

### 4. Communication interface

Furthermore, we implement an inter-TPU interface that allows a TPU to receive requests to write to the RAM of its neighbors. For now, we implement a 2d connection scheme (4 neighbors).

*Status: designed in [comms.md](comms.md), but not built. `tpu_top.sv` ties the `nb_*` port
off, so the `wrneigh` instruction completes as a no-op in both the RTL and the ISS.*

### 5. Scalar Unit

Handles all the control, dispatches instructions to the VPU and MXU. Reads from instruction memory.
Also handles scalar operations such as adding, multiplying numbers in memory.

The instruction set that grew out of this sketch is specified in [isa.md](isa.md) and
summarized, with assembler syntax, in
[`accel/tpulang/README.md`](../../tpulang/README.md#14-instruction-set). Two things about the
real ISA differ from the shape implied above and are worth stating here, because they change
how programs are written:

- **Operands are registers, not immediates.** An instruction names registers; the register
  *contents* are the byte addresses the unit operates on. So `matmul rout, ract, rweight`,
  after an `li` puts each address in a register.
- **Sizes are not in the instruction.** They come from config registers set beforehand —
  `setcfg tlen` (MXU token count), `setcfg vlen` (VPU vector length), `setcfg len` (DMA byte
  count). Stale config is the most common silent bug in a tpulang program.

The dispatched ops are `matmul` / `matmul_t` (with `.acc`/`.rq` flags), `vecmatmul`,
`vecdot`, `vecadd`, `relu`, `requant`, `dyt`, and `tquant`; memory movement is `rdmem` / `wrmem` (DMA between DRAM and scratchpad) and
`wrneigh`; control is `adds`, `subs`, `muls`, `cmps`, `li`, `loads`, `stores`, `setcfg`,
`branch` (and the `beq`/`bne`/`blt`/`bge` forms), `jmp`, `wait`, and `halt`.







