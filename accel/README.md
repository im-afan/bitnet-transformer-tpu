# Accelerators

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
  - **`tpulang/pytpu/`** — a Python *composer* one level up: it instantiates hand-written
    parameterized `.tpu` templates and concatenates them into an ordinary `.tpu` file. It
    builds a whole quantized transformer layer this way. See
    [`tpulang/pytpu/README.md`](tpulang/pytpu/README.md).

There is no `compiler/`: the job that name implied — turning a model into a TPU instruction
stream — is split between `tpulang/` (language + tools) and `tpulang/pytpu/` (composition),
and neither traces PyTorch with `torch.fx`. `torch_ref.py` runs in the other direction, as
an independent check on kernels the ISS and the FPGA have already executed.
