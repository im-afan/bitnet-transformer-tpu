# Custom TPU (FPGA)

A custom matrix/attention accelerator for the transformer in `../../model`, written in
**SystemVerilog** and targeting an **FPGA** (toolchain not yet fixed — the layout below is
kept vendor-agnostic; vendor-specific files get their own subfolders when a target board is
chosen).

The PyTorch model in `model/` is the **golden reference**. Every hardware block is validated
by simulating it against vectors produced by that reference (the same role the CUDA kernel in
`../cuda` already plays for the GPU path).

## Directory layout

| Path           | Contents                                                                        |
| -------------- | ------------------------------------------------------------------------------- |
| `rtl/`         | Synthesizable SystemVerilog: PE / systolic array, accumulators, control FSM, memory-mapped register file, host interface. |
| `tb/`          | Testbenches and simulation-only SystemVerilog (drivers, scoreboards, assertions). |
| `sim/`         | Simulator run scripts and generated artifacts (waveforms, logs — gitignored).   |
| `constraints/` | Pin assignment and timing constraints, one file per target board (`.xdc` / `.sdc` / `.pcf`). |
| `synth/`       | Synthesis / place-and-route / bitstream build scripts, one flow per toolchain. Vivado lives in `synth/vivado/`, with per-board definitions and top-level wrappers under `synth/vivado/boards/<board>/`. Build output goes to `synth/build/` (gitignored). |
| `host/`        | Host-side driver + Python runtime that streams weights/activations to the FPGA and exposes it as a model backend. |
| `docs/`        | Microarchitecture notes: ISA / command format, register + memory map, dataflow, numerics. |

## Roadmap

1. ~~`docs/` — pin down datatype, tile size, and the systolic-array dataflow.~~ **done**
2. ~~`rtl/` — the blocks, the array, control + memory map.~~ **done**
3. ~~`tb/` — unit-test each block, then end-to-end over the UART link against golden
   vectors from `tpulang/gen_vectors.py`.~~ **done**
4. `constraints/` + `synth/` — **in progress.** Target is the Digilent Cmod A7-35T; the
   Vivado flow and board wrapper are in place (see [`docs/synth.md`](docs/synth.md)).
   Remaining blocker: `scratchpad.sv` is a behavioural model with six read ports on a
   flat byte array and needs banking into true 2-port BRAM before the design fits.
5. `host/` — wire the FPGA in as a selectable backend, mirroring the `use_custom_attention` path.
