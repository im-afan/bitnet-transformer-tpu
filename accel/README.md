# Accelerators

> **The scalar unit and tpulang are gone.** The TPU has **one** command
> producer: PicoRV32 firmware in `accel/tpu/fw/`, pushing 128-bit macro-ops
> through the MMIO aperture. `scalar_unit.sv`, `assembler.py`, `gen_vectors.py`,
> `torch_ref.py`, `pytpu.py`, `adder_export.py`, every `examples/*.tpu`, the
> `.tpu` testbenches and `isa.md` were deleted; `iss.py` survives as the golden
> numerics behind `fw_vectors.py`'s command front end. Anything below that
> describes a `.tpu` program, an assembler or the scalar unit is history.
> See `docs/picorv32_migration.md` §11.



Hardware/backend implementations of the operations in the `../model` reference. Each backend
is validated against the PyTorch golden model.

- **`cuda/`** — custom CUDA MHA/GQA kernel loaded into PyTorch via `torch.utils.cpp_extension`.
  Serves as the GPU reference and correctness baseline.
- **`tpu/`** — custom SystemVerilog TPU targeting a Digilent Cmod A7-35T, plus its host
  driver and Vivado build. Runs real programs on real silicon over a UART link.
  See [`tpu/README.md`](tpu/README.md).
- **`tpulang/`** — the TPU's software stack: the `.tpu` assembly language and assembler, an
  instruction-set simulator that is bit-exact with the RTL, the activation-LUT generator,
  PyTorch references for the example kernels, and the golden-vector generator that feeds
  `tpu/tb/`. See [`tpulang/README.md`](tpulang/README.md).

There is no `compiler/`: the job that name implied — turning a model into a TPU instruction
stream — is done by hand in `tpulang/`, and nothing traces PyTorch with `torch.fx`.
`torch_ref.py` runs in the other direction, as an independent check on kernels the ISS and
the FPGA have already executed.
