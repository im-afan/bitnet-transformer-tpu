# Accelerators

Hardware/backend implementations of the operations in the `../model` reference. Each backend
is validated against the PyTorch golden model.

- **`cuda/`** — custom CUDA MHA/GQA kernel loaded into PyTorch via `torch.utils.cpp_extension`.
  Serves as the GPU reference and correctness baseline.
- **`tpu/`** — custom SystemVerilog TPU targeting an FPGA. See `tpu/README.md`.
