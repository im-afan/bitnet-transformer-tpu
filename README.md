# Bitnet Transformer TPU 

A small transformer (a digit-addition sequence model) plus custom hardware accelerators for
its core attention/matmul ops.

## Layout

```
model/            PyTorch golden reference — architecture, training, data, checkpoints
  transformer.py    model definition (vanilla MHA, GQA, ternary/BitNet, MoE)
  numbers_data.py   synthetic addition-problem dataset
  train.py          training loop (with gradient accumulation)
  notebook.ipynb    experiments
  saved/            checkpoints
  tests/            inference / correctness checks

accel/            accelerator backends, each validated against the golden model
  cuda/             custom CUDA MHA/GQA kernel (GPU reference)
  tpu/              custom SystemVerilog TPU — runs on a Digilent Cmod A7-35T
    rtl/ tb/          design + testbenches
    synth/ constraints/  Vivado build, per-board definitions
    host/             Python UART driver and end-to-end runner
    docs/             microarchitecture notes
  tpulang/          the TPU's software stack: assembly language, assembler,
                    bit-exact ISS, activation LUTs, PyTorch references
```

## Running

All Python is packaged under `model/`; run from the repo root so the package resolves:

```bash
python -m model.train --arch vanilla          # vanilla | gqa | ternary_vanilla
python -m model.tests.test_inference --arch vanilla
python accel/cuda/tests/test_cuda_mha.py      # requires CUDA
```

The TPU stack runs from the repo root too:

```bash
python accel/tpulang/torch_ref.py                        # example kernels: ISS vs PyTorch
python accel/tpu/host/run_program.py --dry-run           # toolchain only, no board
cd accel/tpu/tb && make list                             # RTL testbenches (Icarus)
```

## Where to read next

| Doc | Covers |
| --- | --- |
| [`model/README.md`](model/README.md) | The reference model: data format, architecture, the ternary path, and the numerical contract the accelerators implement |
| [`accel/README.md`](accel/README.md) | How the three backends relate |
| [`accel/tpu/README.md`](accel/tpu/README.md) | The hardware: layout, current state, roadmap |
| [`accel/tpu/docs/README.md`](accel/tpu/docs/README.md) | Microarchitecture notes, one doc per block |
| [`accel/tpulang/README.md`](accel/tpulang/README.md) | The assembly language and its toolchain |
