#!/usr/bin/env python3
"""run_program.py — assemble a tpulang program, run it on the FPGA, read it back.

The end-to-end host flow, over the UART link only — the same sequence
``tb/tpu_top_uart_tb.sv`` drives in simulation, but against real hardware:

  1. assemble the ``.tpu`` source into machine words (``tpulang/assembler.py``);
  2. build the DRAM input image and the golden output image with the *same*
     builders ``tpulang/gen_vectors.py`` uses for the testbench vectors, so the
     host and the simulation agree on layout by construction;
  3. ``I`` load the program into instruction memory;
  4. ``W`` write the input tensors into external SRAM ("DRAM");
  5. ``G`` start the scalar unit at PC 0;
  6. wait for the core to go idle, then ``R`` read the output bytes back and
     compare them to the golden image.

Default program is ``tpulang/examples/tiled_matmul.tpu``: a [8x32] @ [32x16]
int8-by-ternary matmul streamed tile-by-tile through fixed 180-byte scratchpad
buffers. Any other example works too — the image builder is picked from the
program's own ``.equ`` constants exactly as ``gen_vectors.py`` picks it.

**Waiting for the run.** The link has no status-read command, so completion
cannot be polled directly (``docs/uart_host.md`` §6). What *can* be observed is
the arbitration rule: while the core is running, every command is NAK'd and
touches nothing. So :func:`wait_until_idle` issues a harmless 2-byte read as a
probe — a NAK means "still running", data back means "idle, results are
readable". On the Cmod A7 the same state is on ``led[1]`` (``done``).

Geometry must match the bitstream: ``boards/cmod_a7/board.tcl`` builds the 8x8
array these programs and vectors are written against.

    python accel/tpu/host/run_program.py -p COM5
    python accel/tpu/host/run_program.py -p /dev/ttyUSB1 --verify-inputs
    python accel/tpu/host/run_program.py --program ../../tpulang/examples/relu_layer.tpu -p COM5
    python accel/tpu/host/run_program.py --dry-run          # toolchain only, no board

Requires ``pyserial`` (except under ``--dry-run``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# The toolchain modules use flat imports among themselves (assembler/iss/
# gen_vectors), and tpu_uart.py is deliberately standalone — so put both
# directories on the path rather than turning either into a package.
HERE = os.path.dirname(os.path.abspath(__file__))
TPULANG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tpulang"))
for _p in (HERE, TPULANG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_vectors as gv                                    # noqa: E402
from iss import TPU, s8                                     # noqa: E402
from tpu_uart import NakError, ProtocolError, TPUUart       # noqa: E402

DEFAULT_PROGRAM = os.path.join(TPULANG_DIR, "examples", "tiled_matmul.tpu")

BOOT_PC = 0

# Idle probe: any valid read works, so use a small one at address 0. Two bytes,
# not one — a 1-byte read cannot tell a NAK (0x15) from a data byte of 0x15.
PROBE_ADDR, PROBE_LEN = 0x0, 2

# A NAK'd read answers with one byte and then silence, so the driver only learns
# it was rejected once the reply times out — i.e. every busy probe costs a full
# timeout. Keep the probe's own timeout short (the real reply takes ~0.2 ms of
# wire time plus USB latency) so the wait loop stays responsive; --timeout still
# governs the actual data transfers.
PROBE_TIMEOUT = 0.3


# ---- Image helpers ----------------------------------------------------------

def contiguous_runs(img: dict) -> list:
    """Collapse a sparse ``{addr: byte}`` image into ``[(addr, bytes), ...]``.

    One UART command per run keeps the frame count (and the 6-byte header tax)
    down; the tensors are laid out densely, so this is a handful of frames.
    """
    runs: list = []
    start: int | None = None
    buf = bytearray()
    for addr in sorted(img):
        if start is not None and addr != start + len(buf):
            runs.append((start, bytes(buf)))
            start, buf = None, bytearray()
        if start is None:
            start = addr
        buf.append(img[addr] & 0xFF)
    if start is not None:
        runs.append((start, bytes(buf)))
    return runs


def decode_c_matrix(consts: dict, byte_at) -> list:
    """Reassemble the tiled-matmul C tiles into one [M][N] int8 matrix."""
    Tt, MT, NT = consts["T"], consts["MTILES"], consts["NTILES"]
    cols, ctile, cbase = consts["COLS"], consts["CTILE"], consts["C"]
    out = [[0] * (NT * cols) for _ in range(MT * Tt)]
    for mt in range(MT):
        for nt in range(NT):
            base = cbase + (mt * NT + nt) * ctile
            for t in range(Tt):
                for j in range(cols):
                    out[mt * Tt + t][nt * cols + j] = s8(byte_at(base + t * cols + j))
    return out


def print_matrix(mat: list, title: str) -> None:
    print(f"\n{title}  [{len(mat)}x{len(mat[0])}]")
    for row in mat:
        print("  " + " ".join(f"{v:4d}" for v in row))


# ---- Stage 1: toolchain (assemble + simulate) -------------------------------

class Plan:
    """Everything the device needs, plus what it should produce."""

    def __init__(self, words, consts, in_img, exp_img, ref, shape):
        self.words = words          # instruction words for 'I'
        self.consts = consts        # .equ constants + labels from the source
        self.in_img = in_img        # {addr: byte} DRAM inputs for 'W'
        self.exp_img = exp_img      # {addr: byte} golden DRAM outputs for 'R'
        self.ref = ref              # independent Python reference, matmul only
        self.shape = shape          # human-readable problem shape, or None


def build_plan(path: str) -> Plan:
    """Assemble the program and run it in the ISS to get the golden images."""
    words, consts = gv.assemble_program(path)
    tpu = TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W,
              mem_addr_w=gv.MEM_ADDR_W, m0_w=gv.M0_W, n_w=gv.N_W)

    ref, shape = None, None
    if "MTILES" in consts:                       # the streaming tiled matmul
        in_img, ref, dims = gv.build_matmul_image(tpu, consts)
        shape = f"[{dims['M']}x{dims['K']}] @ [{dims['K']}x{dims['N']}]"
    else:                                        # fixed-geometry programs
        in_img = gv.build_default_image(tpu)

    tpu.run(words)
    exp_img = {a: tpu.dram[a] for a in sorted(tpu.dram_written)}

    if ref is not None:
        bad = gv.check_reference(tpu, ref)
        if bad:
            raise SystemExit(
                f"ISS disagrees with the Python reference in {bad} place(s) — "
                "the toolchain is broken, not the board; fix that before loading"
            )
    if not exp_img:
        raise SystemExit(
            f"{os.path.basename(path)} spills nothing to DRAM with wrmem, so there "
            "is nothing to read back over UART"
        )
    return Plan(words, consts, in_img, exp_img, ref, shape)


# ---- Stage 2: device --------------------------------------------------------

def autodetect_port() -> str:
    """Find the board's USB-serial port (the Cmod A7's FT2232 UART channel)."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyserial is required: pip install pyserial") from exc

    cands = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x0403, 0x6010)]
    if not cands:  # any FTDI part, then
        cands = [p for p in list_ports.comports() if p.vid == 0x0403]
    if not cands:
        raise SystemExit("no FTDI serial port found — pass --port explicitly")
    if len(cands) > 1:
        names = ", ".join(f"{p.device} ({p.description})" for p in cands)
        raise SystemExit(f"several candidate ports, pass --port: {names}")
    print(f"port    : {cands[0].device} ({cands[0].description}) [autodetected]")
    return cands[0].device


def probe_idle(tpu: TPUUart) -> bool:
    """True if the core is idle — i.e. it answers a read instead of NAK'ing it."""
    saved, tpu.timeout = tpu.timeout, PROBE_TIMEOUT
    try:
        tpu.read_mem(PROBE_ADDR, PROBE_LEN)
        return True
    except NakError:
        return False
    finally:
        tpu.timeout = saved


def wait_until_idle(tpu: TPUUart, limit: float, interval: float) -> float:
    """Block until the core stops NAK'ing commands. Returns seconds waited."""
    t0 = time.monotonic()
    while True:
        if probe_idle(tpu):
            return time.monotonic() - t0
        waited = time.monotonic() - t0
        if waited >= limit:
            raise SystemExit(
                f"core still busy {waited:.1f}s after 'G' — it never halted "
                f"(check led[1]/done), or the link is out of sync"
            )
        time.sleep(interval)


def load_and_run(tpu: TPUUart, plan: Plan, args) -> dict:
    """Load, run, and read back. Returns the hardware bytes as ``{addr: byte}``."""
    in_runs = contiguous_runs(plan.in_img)
    exp_runs = contiguous_runs(plan.exp_img)

    # The core has priority over the link, so a run left over from a previous
    # invocation would NAK the preload (and a NAK mid-write is a desync, not a
    # routine error). Confirm idle before sending anything.
    if not probe_idle(tpu):
        print("core is busy; waiting for it to halt before loading...")
        wait_until_idle(tpu, args.run_timeout, args.poll_interval)

    # 1) program -> IMEM
    tpu.load_program(0, plan.words)
    print(f"loaded  : {len(plan.words)} instruction words -> IMEM[0]")

    # 2) tensors -> DRAM
    for addr, blob in in_runs:
        tpu.write_mem(addr, blob)
    print(f"wrote   : {len(plan.in_img)} input bytes in {len(in_runs)} frame(s)")

    if args.verify_inputs:
        bad = 0
        for addr, blob in in_runs:
            got = tpu.read_mem(addr, len(blob))
            bad += sum(1 for i, b in enumerate(got) if b != blob[i])
        if bad:
            raise SystemExit(f"input readback FAILED: {bad} byte(s) differ — "
                             "bad SRAM, bad wiring, or a marginal link")
        print(f"verify  : all {len(plan.in_img)} input bytes read back identical")

    # 3) run
    tpu.go(BOOT_PC)
    print(f"started : pc={BOOT_PC} (ACK)")

    # 4) wait for the halt, out-of-band (no status command on this link)
    if args.wait:
        time.sleep(args.wait)
    waited = wait_until_idle(tpu, args.run_timeout, args.poll_interval)
    print(f"halted  : core idle after {args.wait + waited:.2f}s")

    # 5) results -> host
    hw: dict = {}
    for addr, blob in exp_runs:
        for i, b in enumerate(tpu.read_mem(addr, len(blob))):
            hw[addr + i] = b
    print(f"read    : {len(hw)} output bytes in {len(exp_runs)} frame(s)")
    return hw


# ---- Stage 3: report --------------------------------------------------------

def compare(hw: dict, exp_img: dict, limit: int = 8) -> int:
    bad = 0
    for addr in sorted(exp_img):
        got, want = hw[addr], exp_img[addr]
        if got != want:
            bad += 1
            if bad <= limit:
                print(f"  MISMATCH @{addr:05x}: device {s8(got):4d}  golden {s8(want):4d}")
    if bad > limit:
        print(f"  ... and {bad - limit} more")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=115200,
                    help="must match UART_CPB on the device (default: %(default)s)")
    ap.add_argument("-t", "--timeout", type=float, default=2.0,
                    help="per-reply timeout in seconds (default: %(default)s)")
    ap.add_argument("--program", default=DEFAULT_PROGRAM,
                    help="tpulang .tpu source (default: examples/tiled_matmul.tpu)")
    ap.add_argument("--wait", type=float, default=0.1,
                    help="fixed settle time after 'G' before probing (default: %(default)s)")
    ap.add_argument("--poll-interval", type=float, default=0.05,
                    help="gap between idle probes (default: %(default)s)")
    ap.add_argument("--run-timeout", type=float, default=10.0,
                    help="give up if the core has not halted by then (default: %(default)s)")
    ap.add_argument("--verify-inputs", action="store_true",
                    help="read the tensors back after writing them (bring-up check)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble + simulate only; never open the serial port")
    args = ap.parse_args(argv)

    plan = build_plan(args.program)
    name = os.path.basename(args.program)
    print(f"program : {name}  ({len(plan.words)} words, "
          f"{gv.ROWS}x{gv.COLS} array)")
    if plan.shape:
        print(f"problem : {plan.shape}  (ISS matches the Python reference)")
    print(f"inputs  : {len(plan.in_img)} bytes of DRAM image")
    print(f"expected: {len(plan.exp_img)} bytes spilled by wrmem")

    if args.dry_run:
        print("\n--dry-run: toolchain only, nothing sent to the board")
        if plan.ref is not None:
            print_matrix(decode_c_matrix(plan.consts, lambda a: plan.exp_img[a]),
                         "C = A @ W (golden)")
        return 0

    port = args.port or autodetect_port()
    try:
        with TPUUart(port, args.baud, args.timeout) as tpu:
            hw = load_and_run(tpu, plan, args)
    except (ProtocolError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if plan.ref is not None:
        print_matrix(decode_c_matrix(plan.consts, lambda a: hw[a]),
                     "C = A @ W (from the FPGA)")

    print()
    bad = compare(hw, plan.exp_img)
    if bad:
        print(f"FAILED: {bad}/{len(plan.exp_img)} output bytes differ from the golden image")
        return 1
    print(f"PASSED: all {len(plan.exp_img)} output bytes match the golden image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
