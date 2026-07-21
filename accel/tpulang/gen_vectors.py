#!/usr/bin/env python3
"""gen_vectors.py — golden test-vector generator for tpu_top_tb.sv.

Ties the whole toolchain together end to end:

  1. generate deterministic sample tensors (int8 activations, ternary weights,
     the requant {m0,n} word) and lay them out in a scratchpad image exactly
     where the program's address arithmetic expects them;
  2. assemble a ``.tpu`` program into machine words (assembler.py);
  3. run those words over the scratchpad in the ISS (iss.py) to get the golden
     final memory image;
  4. emit three ``$readmemh``-loadable files the SystemVerilog testbench loads:

       tpu_prog.hex      instruction words   (one 32-bit word per line, dense)
       tpu_spad_in.hex   input tensors       (one byte per line, sparse @addr)
       tpu_spad_exp.hex  expected outputs    (one byte per line, sparse @addr)

The expected file contains exactly the bytes the program *wrote* (tracked by the
ISS), so the testbench checks precisely the program's outputs, whatever program
it is. The default program is examples/relu_layer.tpu — one ternary NN layer plus
a ReLU, the shape the adder model runs — but any ``.tpu`` whose tensors match the
geometry below can be dropped in.

    python gen_vectors.py                       # default program + geometry
    python gen_vectors.py -p examples/foo.tpu   # a different program
"""

from __future__ import annotations

import argparse
import os

from assembler import assemble
from iss import TPU, CFG_TLEN, CFG_VLEN, CFG_SCALAR

# --- geometry — MUST match the DUT localparams in tpu_top_tb.sv ---------------
ROWS = 8          # MXU contraction dim d (activation vector length)
COLS = 8          # MXU output features
T = 4             # token rows
ADDR_W = 16
M0_W, N_W = 12, 4

# --- scratchpad address map (matches relu_layer.tpu / tpu_top_tb.sv) ----------
ACT = 0x0000      # A[t][i] int8, row-major
WGT = 0x1000      # W[i][j] ternary, column-major 2-bit packed
RQW = 0x7000      # requant {m0,n} word (int32)

# --- requant scale {m0=1, n=0}: identity + int8 clip -> plain integer matmul --
RQ_M0, RQ_N = 1, 0

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROG = os.path.join(HERE, "examples", "relu_layer.tpu")
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "tpu", "tb", "vectors"))


def gen_tensors() -> tuple[list, list]:
    """Bounded, deterministic operands (identical to the old in-TB generator)."""
    act = [[((t * 3 + i) % 9) - 4 for i in range(ROWS)] for t in range(T)]     # [-4,4]
    wgt = [[((i + 2 * j) % 3) - 1 for j in range(COLS)] for i in range(ROWS)]   # {-1,0,1}
    return act, wgt


def build_input_image(tpu: TPU, act: list, wgt: list) -> dict:
    """Write the tensors into the ISS scratchpad; return {addr: byte} for the file."""
    img: dict[int, int] = {}

    def put(addr: int, byte: int):
        tpu.mem[tpu._a(addr)] = byte & 0xFF
        img[addr] = byte & 0xFF

    # Activations: A[t][i] int8, row-major.
    for t in range(T):
        for i in range(ROWS):
            put(ACT + t * ROWS + i, act[t][i])

    # Weights: W[i][j] ternary, column-major 2-bit packed (00=0, 01=+1, 11=-1).
    wcol_bytes = (ROWS * 2) // 8
    for j in range(COLS):
        colint = 0
        for i in range(ROWS):
            code = 0 if wgt[i][j] == 0 else (1 if wgt[i][j] == 1 else 0b11)
            colint |= code << (2 * i)
        for b in range(wcol_bytes):
            put(WGT + j * wcol_bytes + b, (colint >> (8 * b)) & 0xFF)

    # Requant {m0,n} word: m0 in low M0_W bits, n above.
    word = (RQ_N << M0_W) | RQ_M0
    for b in range(4):
        put(RQW + b, (word >> (8 * b)) & 0xFF)

    return img


def preset_cfg(tpu: TPU) -> None:
    """Config the host presets before the run (the SETCFGs in the program also
    set these; presetting keeps the ISS correct even for programs that don't)."""
    tpu.cfg[CFG_TLEN] = T
    tpu.cfg[CFG_VLEN] = T * COLS
    tpu.cfg[CFG_SCALAR] = RQW


def emit_words(path: str, words: list, note: str) -> None:
    with open(path, "w") as f:
        f.write(f"// {note}\n")
        for w in words:
            f.write(f"{w & 0xFFFFFFFF:08x}\n")


def emit_bytes(path: str, img: dict, note: str) -> None:
    """Sparse $readmemh byte image: `@addr` markers before each contiguous run."""
    with open(path, "w") as f:
        f.write(f"// {note}\n")
        prev = None
        for addr in sorted(img):
            if addr != prev:
                f.write(f"@{addr:04x}\n")
            f.write(f"{img[addr] & 0xFF:02x}\n")
            prev = addr + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--program", default=DEFAULT_PROG, help="tpulang .tpu source")
    ap.add_argument("-o", "--out-dir", default=OUT_DIR, help="output directory")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.program) as f:
        words = assemble(f.read())

    tpu = TPU(rows=ROWS, cols=COLS, addr_w=ADDR_W, m0_w=M0_W, n_w=N_W)
    act, wgt = gen_tensors()
    in_img = build_input_image(tpu, act, wgt)
    preset_cfg(tpu)

    tpu.run(words)
    exp_img = {a: tpu.mem[a] for a in sorted(tpu.written)}

    prog_path = os.path.join(args.out_dir, "tpu_prog.hex")
    in_path = os.path.join(args.out_dir, "tpu_spad_in.hex")
    exp_path = os.path.join(args.out_dir, "tpu_spad_exp.hex")
    name = os.path.basename(args.program)
    emit_words(prog_path, words, f"instruction words assembled from {name}")
    emit_bytes(in_path, in_img, f"input tensors for {name} (ACT/WGT/RQW)")
    emit_bytes(exp_path, exp_img, f"golden outputs for {name} (ISS-computed)")

    print(f"program : {len(words):4d} words         -> {prog_path}")
    print(f"inputs  : {len(in_img):4d} bytes         -> {in_path}")
    print(f"expected: {len(exp_img):4d} bytes written -> {exp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
