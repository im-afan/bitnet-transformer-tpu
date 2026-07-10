# transformer_impl

A small transformer (a digit-addition sequence model) plus custom hardware accelerators for
its core attention/matmul ops.

## Layout

```
model/            PyTorch golden reference — architecture, training, data, checkpoints
  transformer.py    model definition (vanilla MHA + GQA)
  numbers_data.py   synthetic addition-problem dataset
  train.py          training loop
  notebook.ipynb    experiments
  saved/            checkpoints
  tests/            inference / correctness checks

accel/            accelerator backends, each validated against the golden model
  cuda/             custom CUDA MHA/GQA kernel (GPU reference)
  tpu/              custom SystemVerilog TPU for FPGA (in progress)
```

## Running

All Python is packaged under `model/`; run from the repo root so the package resolves:

```bash
python -m model.train --arch vanilla
python -m model.tests.test_inference --arch vanilla
python accel/cuda/tests/test_cuda_mha.py   # requires CUDA
```

See `accel/tpu/README.md` for the hardware effort.
