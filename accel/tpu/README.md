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
| `synth/`       | Synthesis / place-and-route / bitstream build scripts, one flow per toolchain.  |
| `host/`        | Host-side driver + Python runtime that streams weights/activations to the FPGA and exposes it as a model backend. |
| `docs/`        | Microarchitecture notes: ISA / command format, register + memory map, dataflow, numerics. |

## Roadmap (no RTL written yet)

1. `docs/` — pin down datatype (e.g. int8/bf16), tile size, and the systolic-array dataflow.
2. `rtl/` — single processing element, then the array, then the surrounding control + memory map.
3. `tb/` + `sim/` — unit-test each block against golden vectors exported from `model/`.
4. `constraints/` + `synth/` — target a concrete board, synthesize, flash.
5. `host/` — wire the FPGA in as a selectable backend, mirroring the `use_custom_attention` path.
