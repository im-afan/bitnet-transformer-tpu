#!/usr/bin/env python3
"""run_fw_matmul.py — run a C firmware matmul (``../fw/*.c``) on the board.

The board-side counterpart of `make -C ../tb fw`: the PicoRV32 is the producer,
so there is no assembler and no ISS in the loop:

  1. read the built firmware image (``fw/matmul.hex``, one word per line);
  2. build the A and W operands and a plain-Python reference for ``A @ W``;
  3. ``I`` load the firmware — the ``'I'``/``'G'`` word address carries
     :data:`tpu_uart.FW_BASE` in its top bit to select the CPU's RAM over the
     scalar unit's IMEM;
  4. ``W`` write the operands into external SRAM ("DRAM");
  5. ``G`` release the core. The firmware always starts at 0 (``PROGADDR_RESET``
     is fixed in the RTL), so only the producer bit of the address matters;
  6. wait for the core to go idle, ``T`` read the counters, ``R`` read the int32
     result back and compare it to the reference.

The geometry below must match ``fw/matmul.c``; it is the same problem
the retired ``tiled_matmul_hw.tpu`` ran, so historical numbers can be
compared directly.

``fw/matmul_loop.c`` is the same problem with the tile grid walked in firmware
rather than by the array, and lands the same int32 C at the same address, so
this script checks it unchanged: pass ``--fw ../fw/matmul_loop.hex``.

    python accel/tpu/host/run_fw_matmul.py --dry-run   # operands + reference only
    python accel/tpu/host/run_fw_matmul.py -p COM5

Requires ``pyserial`` (except under ``--dry-run``).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPULANG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tpulang"))
for _p in (HERE, TPULANG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fw_vectors import a_val, operand_image, w_val            # noqa: E402
from tpu_uart import (                                        # noqa: E402
    FW_AW, FW_BASE, ProtocolError, TPUUart, autodetect_port, contiguous_runs,
    parse_hex_program, probe_idle, report_run_time, wait_until_idle,
)

DEFAULT_FW = os.path.normpath(os.path.join(HERE, "..", "fw", "matmul.hex"))

# ---- geometry (mirrors fw/matmul.c and fw/matmul_loop.c) --------------------
# The kernels' default shape. --m/--ktiles/--ntiles follow a firmware rebuilt at
# another one (`make -C accel/tpu/fw M=... KTILES=... NTILES=...`).
ROWS, COLS = 8, 8
M, KTILES, NTILES = 8, 4, 2
K = N = AROW = WROW = CROW = C_BYTES = 0        # filled in by set_shape()

A_ADDR, W_ADDR, C_ADDR = 0x0000, 0x2000, 0x4000


def set_shape(m: int, ktiles: int, ntiles: int) -> None:
    global M, KTILES, NTILES, K, N, AROW, WROW, CROW, C_BYTES
    M, KTILES, NTILES = m, ktiles, ntiles
    K, N = KTILES * ROWS, NTILES * COLS
    AROW = K              # A row stride, bytes
    WROW = (N * 4) // 8   # W row stride, bytes (row-major int4)
    CROW = N * 4          # C row stride, bytes (int32)
    C_BYTES = M * N * 4


set_shape(M, KTILES, NTILES)


def build_operands() -> tuple[dict, list, list]:
    """DRAM image ``{addr: byte}`` plus the two matrices, as plain lists.

    The image comes from `fw_vectors.operand_image`, which is also what the
    simulation vectors are built from — one definition of the layout, so a board
    run and `make fw` cannot be staging different bytes. ``w_full[n][k]`` is
    W[k][n]; the *wire* layout is row-major int4 and lives in that function.
    """
    a_full = [[a_val(m, k) for k in range(K)] for m in range(M)]
    w_full = [[w_val(k, n) for k in range(K)] for n in range(N)]
    return operand_image(M, K, N), a_full, w_full


def reference(a_full: list, w_full: list) -> list:
    return [[sum(a_full[m][k] * w_full[n][k] for k in range(K)) for n in range(N)]
            for m in range(M)]


def decode_c(data: bytes) -> list:
    """The read-back bytes as an [M][N] int32 matrix."""
    return [[int.from_bytes(data[m * CROW + n * 4:m * CROW + n * 4 + 4],
                            "little", signed=True)
             for n in range(N)] for m in range(M)]


def print_matrix(mat: list, title: str) -> None:
    print(f"\n{title}  [{len(mat)}x{len(mat[0])}]")
    for row in mat:
        print("  " + " ".join(f"{v:5d}" for v in row))


def compare(got: list, want: list, limit: int = 8) -> int:
    bad = 0
    for m in range(M):
        for n in range(N):
            if got[m][n] != want[m][n]:
                bad += 1
                if bad <= limit:
                    print(f"  MISMATCH C[{m}][{n}]: device {got[m][n]}  "
                          f"reference {want[m][n]}")
    if bad > limit:
        print(f"  ... and {bad - limit} more")
    return bad


def load_and_run(tpu: TPUUart, words: list, in_img: dict, args) -> bytes:
    in_runs = contiguous_runs(in_img)

    if not probe_idle(tpu):
        print("core is busy; waiting for it to halt before loading...")
        wait_until_idle(tpu, args.run_timeout, args.poll_interval)

    tpu.load_program(FW_BASE, words)
    print(f"loaded  : {len(words)} words -> firmware RAM[0]")

    for addr, blob in in_runs:
        tpu.write_mem(addr, blob)
    print(f"wrote   : {len(in_img)} operand bytes in {len(in_runs)} frame(s)")

    tpu.go(FW_BASE)
    print("started : CPU released from reset (ACK)")

    waited = wait_until_idle(tpu, args.run_timeout, args.poll_interval)
    print(f"halted  : core idle after {waited:.2f}s")
    report_run_time(tpu, args.clk_mhz)

    data = tpu.read_mem(C_ADDR, C_BYTES)
    print(f"read    : {len(data)} result bytes from {C_ADDR:#06x}")
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("-t", "--timeout", type=float, default=2.0,
                    help="per-reply timeout in seconds (default: %(default)s)")
    ap.add_argument("--fw", default=DEFAULT_FW,
                    help="firmware image, one hex word per line (default: fw/matmul.hex)")
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--run-timeout", type=float, default=10.0,
                    help="give up if the core has not halted by then (default: %(default)s)")
    ap.add_argument("--clk-mhz", type=float, default=12.0,
                    help="device clock, for the 'T' counters (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the operands and the reference only; no board")
    ap.add_argument("--m", type=int, default=M, help="token rows (default %(default)s)")
    ap.add_argument("--ktiles", type=int, default=KTILES,
                    help="contraction tiles, K = 8*this (default %(default)s)")
    ap.add_argument("--ntiles", type=int, default=NTILES,
                    help="output tiles, N = 8*this (default %(default)s)")
    args = ap.parse_args(argv)

    set_shape(args.m, args.ktiles, args.ntiles)
    in_img, a_full, w_full = build_operands()
    want = reference(a_full, w_full)

    print(f"problem : [{M}x{K}] @ [{K}x{N}]  "
          f"({KTILES}x{NTILES} tiles of {ROWS}x{COLS})")
    print(f"inputs  : {len(in_img)} bytes of DRAM image "
          f"(A@{A_ADDR:#06x}, W@{W_ADDR:#06x})")
    print(f"expected: {C_BYTES} bytes of int32 C@{C_ADDR:#06x}")

    if args.dry_run:
        print("\n--dry-run: nothing sent to the board")
        print_matrix(want, "C = A @ W (reference)")
        return 0

    try:
        with open(args.fw) as fh:
            words = parse_hex_program(fh.read())
    except OSError as exc:
        print(f"error: {exc} — build it first with `make -C accel/tpu/fw`",
              file=sys.stderr)
        return 1
    if len(words) > (1 << FW_AW):
        print(f"error: {len(words)} words does not fit the {1 << FW_AW}-word "
              f"firmware RAM", file=sys.stderr)
        return 1
    print(f"firmware: {os.path.basename(args.fw)}, {len(words)} words "
          f"({4 * len(words)} bytes)")

    port = args.port or autodetect_port()
    try:
        with TPUUart(port, args.baud, args.timeout) as tpu:
            data = load_and_run(tpu, words, in_img, args)
    except (ProtocolError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    got = decode_c(data)
    print_matrix(got, "C = A @ W (from the FPGA)")
    print()
    bad = compare(got, want)
    if bad:
        print(f"FAILED: {bad}/{M * N} elements differ from the reference")
        return 1
    print(f"PASSED: all {M * N} elements match the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
