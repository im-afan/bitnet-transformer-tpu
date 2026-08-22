# Custom TPU (FPGA)

> **The scalar unit and tpulang are gone.** The TPU has **one** command
> producer: PicoRV32 firmware in `accel/tpu/fw/`, pushing 128-bit macro-ops
> through the MMIO aperture. `scalar_unit.sv`, `assembler.py`, `gen_vectors.py`,
> `torch_ref.py`, `pytpu.py`, `adder_export.py`, every `examples/*.tpu`, the
> `.tpu` testbenches and `isa.md` were deleted; `iss.py` survives as the golden
> numerics behind `fw_vectors.py`'s command front end. Anything below that
> describes a `.tpu` program, an assembler or the scalar unit is history.
> See `docs/picorv32_migration.md` §11.



A custom matrix/attention accelerator for the transformer in `../../model`, written in
**SystemVerilog** and running on a **Digilent Cmod A7-35T** (`xc7a35t-cpg236-1`) via a
non-project Vivado batch flow. The tree is still organised so a second board is a copy of
`synth/vivado/boards/cmod_a7/`; nothing outside those directories is vendor-specific.

The PyTorch model in `model/` is the **golden reference**. Every hardware block is validated
by simulating it against vectors produced by that reference (the same role the CUDA kernel in
`../cuda` already plays for the GPU path).

**Current state.** The design simulates and, at the last bitstream that was built, ran on
the board. The **whole model runs on it**: `fw/adder.c` is four transformer layers plus the
output head in 518 macro-ops, and `cd tb && make fw FWPROG=adder` puts the real PicoRV32
image through the real core and checks every DRAM byte *and* every command word against the
ISS — 526 879 checks, 0 errors, 439 917 clocks. Scored on the addition task through the ISS
(`python ../tpulang/adder_export.py -n 256`) it lands at **100.00% exact-sequence**, the same
as the PyTorch checkpoint it came from.

**It fits.** `make bit` completes with timing met: **13342 LUTs of 20800 (64%)**, 7543 FFs,
33 BRAM tiles, 31 DSPs, WNS +26.2 ns. An earlier note here said it was 857 LUTs over; that
stopped reproducing (the MXU is less than half the size it was measured at) and it is not
this section's doing. See [`docs/synth.md`](docs/synth.md) §5 for the current numbers next to
the historical ones, and [`host/README.md`](host/README.md) for the link.

The software side is now two directories: [`fw/`](fw) is the kernels (C, one file each), and
[`../tpulang`](../tpulang) is what survived the toolchain deletion — the bit-exact ISS, the
golden-vector generator that drives it from a kernel's own command trace, and the checkpoint
exporter.

## Directory layout

| Path           | Contents                                                                        |
| -------------- | ------------------------------------------------------------------------------- |
| `rtl/`         | Synthesizable SystemVerilog: systolic array (`mxu.sv`), vector unit (`vpu.sv`), banked scratchpad, DMA + external SRAM controller, scalar unit, UART interface, and `tpu_top.sv` tying them together. There is no `rtl/luts/` any more — the VPU's activation ROMs went with the `gelu`/`exp` instructions ([docs/vpu.md §Removed ops](docs/vpu.md#removed-ops)), so the design reads no `$readmemh` file by default. |
| `tb/`          | Icarus testbenches, one per block plus `fw_matmul_tb.sv`, which runs any C kernel out of `FW_INIT` through the whole core (`make fw FWPROG=<kernel>`). `make` targets in `tb/Makefile`; `vectors_fw/` holds golden vectors generated per run by `../tpulang/fw_vectors.py` from the kernel's own native command trace. |
| `sim/`         | Placeholder for simulator artifacts (waveforms, logs — gitignored). Testbenches currently build and run in place under `tb/`. |
| `constraints/` | Pin assignment and timing constraints, one `.xdc` per target board. |
| `synth/`       | Vivado non-project build (`synth/vivado/build.tcl`), with per-board definitions and top-level wrappers under `synth/vivado/boards/<board>/`. Build output goes to `synth/build/` (gitignored). |
| `fw/`          | C firmware for the PicoRV32 command producer (`rtl/cpu_subsys.sv`): the MMIO driver header, one `.c` per kernel, `start.S`, linker script and Makefile. Needs a RISC-V cross gcc — see [`fw/README.md`](fw/README.md). |
| `host/`        | Python host driver: the UART protocol (`tpu_uart.py`), the end-to-end runner (`run_fw_matmul.py`), and link self-tests. |
| `docs/`        | Microarchitecture notes: ISA / command format, register + memory map, dataflow, numerics. Start at [`docs/README.md`](docs/README.md). |

Four board targets are defined under `synth/vivado/boards/`: `cmod_a7` (the real design),
`cmod_a7_mem` and `cmod_a7_bram` (memory-path bring-up variants), and `cmod_a7_echo` (the
UART self-test image — see [`docs/uart_selftest.md`](docs/uart_selftest.md)). **Reflash
`board=cmod_a7` before running `host/test_uart_link.py` or `host/run_program.py`**; they
time out against the echo bitstream.

## Roadmap

1. ~~`docs/` — pin down datatype, tile size, and the systolic-array dataflow.~~ **done**
2. ~~`rtl/` — the blocks, the array, control + memory map.~~ **done**
3. ~~`tb/` — unit-test each block, then end-to-end over the UART link against golden
   vectors from `tpulang/gen_vectors.py`.~~ **done**
4. `constraints/` + `synth/` — Vivado flow and board wrapper: **done**; a design that
   fits: **regressed**, currently 4.1% over on LUTs (`docs/synth.md` §5).
   `scratchpad.sv` was the original blocker — a flat byte array with six read ports, which
   Vivado could only implement as ~524k FFs. It is now byte-lane banked into genuine
   dual-port BRAM. What pushed it back over is the macro-op work in the MXU.
5. ~~Run a real program on the board and verify it.~~ **done.** `host/run_program.py`
   assembles, loads, runs, and checks the result against the ISS *and* an independent
   PyTorch reference.
6. `host/` — **next.** Wire the FPGA in as a selectable model backend, mirroring the
   `use_custom_attention` path. Today the host runs standalone `.tpu` kernels, not
   `model/transformer.py` itself.
7. Fit a second transformer layer. One quantized layer already measures 861 of 1024
   instruction words, so the lever is hoisting the four inlined ternary matmuls into a
   single parameter-driven copy.

Known gaps on hardware: the inter-TPU LINK (`wrneigh`) is stubbed in `tpu_top.sv` and
completes as a no-op, and `UART_RX_TIMEOUT` is 0 in every board definition, so a corrupted
frame wedges the receive FSM until reflash (see `CLAUDE.md` for why that is a hardening item
rather than the fix).
