# Accelerators

Hardware/backend implementations of the operations in the `../model` reference. Each backend
is validated against the PyTorch golden model.

- **`cuda/`** — custom CUDA MHA/GQA kernel loaded into PyTorch via `torch.utils.cpp_extension`.
  Serves as the GPU reference and correctness baseline.
- **`tpu/`** — custom SystemVerilog TPU targeting an FPGA. See `tpu/README.md`.
- **`compiler/`** — compiles torch expressions into TPU instructions: traces a PyTorch
  function with `torch.fx`, and emits an instruction stream for `tpu/`'s scalar unit plus
  the scratchpad image it runs against. See `compiler/README.md`.
