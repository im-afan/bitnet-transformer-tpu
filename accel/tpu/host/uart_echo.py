#!/usr/bin/env python3
"""uart_echo.py — host side of the UART echo self-test (``docs/uart_selftest.md``).

Sends fixed-size blocks of random bytes at the `cmod_a7_echo` bitstream and
checks that exactly those bytes come back. The device runs `rtl/uart_echo.sv`,
which is store-and-forward: it buffers ``BLOCK_LEN`` bytes in registers, sends
those ``BLOCK_LEN`` bytes back, and repeats. No commands, no addresses, no modes.

    python accel/tpu/host/uart_echo.py -p COM5                # 30-second run
    python accel/tpu/host/uart_echo.py -p COM5 --minutes 30   # soak until it breaks
    python accel/tpu/host/uart_echo.py -p COM5 --baud 117000  # margin check, below
    python accel/tpu/host/uart_echo.py --offline              # check the forensics, no board

**``--block`` must equal the device's ``BLOCK_LEN``** (``BLOCK_LEN`` in
`boards/cmod_a7_echo/board.tcl`, 64 by default). Nothing on the wire negotiates
it: the block length *is* the protocol. Get it wrong and the run fails in the
first block — see :func:`resync`, which says so explicitly rather than leaving
you to read it out of a byte diff.

**Half duplex, by construction.** The device is never transmitting and receiving
at once, so this no longer exercises simultaneous traffic the way a streaming
echo did. What it does exercise instead is the *turnaround*: a burst in, a gap,
a burst out, then the next block's first byte arriving right behind the reply —
the shape of the real protocol's traffic.

**Fresh random data every block**, so a readback of stale buffered bytes can never
be mistaken for a pass.

**The `--baud` flag is the cheap experiment worth running.** 8N1 tolerates roughly
±5% of bit-period error. Sweeping the host a few percent either side of 115200
measures where the receiver actually falls over. A margin that is *asymmetric*
(clean at −3%, broken at +2%) means the sample point sits off-centre and the fix
is arithmetic in `uart_receiver.sv`. A margin that is symmetric and wide, with
failures still happening at 0%, means noise or metastability instead.

On a mismatch the script prints what it can work out about *how* the stream went
wrong — in particular whether the received stream is the sent stream shifted by a
whole bit, which is the skipped/extra-bit theory, confirmed or refuted. See
:func:`report_mismatch`.

Requires ``pyserial`` (except under ``--offline``).
"""

from __future__ import annotations

import argparse
import random
import sys
import time

DEFAULT_BAUD = 115200
DEFAULT_BLOCK = 64            # = BLOCK_LEN in boards/cmod_a7_echo/board.tcl
DEFAULT_SECONDS = 30.0

# Bytes per block are the device's register file, not a buffer the host can pick
# freely: this has to match the synthesised BLOCK_LEN. The cap is only here to
# keep a fat-fingered --block from producing a run that appears to hang while it
# waits for a reply that will never be BLOCK_LEN bytes long.
MAX_BLOCK = 4096


# ---- Bit-level forensics ----------------------------------------------------
#
# These are the reason this script exists rather than a five-line loop: they turn
# "a byte was wrong" into a statement about which layer was wrong. All pure
# functions over bytes, and all covered by --offline.

def to_bits(data: bytes) -> list[int]:
    """Serialise ``data`` as it appears on the wire in 8N1.

    One frame per byte: a low start bit, eight data bits least-significant
    first, then a high stop bit. Ten bits per byte, so byte ``k`` owns
    ``bits[10*k : 10*k+10]`` and its data bits are at ``10*k+1 .. 10*k+8``.
    """
    bits: list[int] = []
    for b in data:
        bits.append(0)
        bits.extend((b >> i) & 1 for i in range(8))
        bits.append(1)
    return bits


def explain_bit_shift(sent: bytes, got: bytes, k: int, span: int = 3) -> int | None:
    """Is ``got[k]`` what you get by sampling the sent bitstream ``s`` bits off?

    The direct test of the skipped/extra-bit theory. A receiver that frames a
    byte early — because a glitch in the previous stop bit looked like a start
    bit — samples the eight data bits one place to the left, pulling in the
    neighbouring frame's stop bit at one end and dropping a real bit at the
    other. That is not a random corruption: it produces one specific value, and
    this recomputes it.

    Returns the offset in bits (negative = sampled early, positive = late), or
    ``None`` if no offset in ``±span`` explains the byte — in which case the
    theory is wrong for this failure and it is isolated bit-sampling noise.

    The classic case, and the one `tb/uart_receiver_tb.sv` already reproduces: a
    0x40 preceded by a zero byte arrives as 0x80 at an offset of −1.
    """
    bits = to_bits(sent)
    base = 10 * k + 1
    for s in range(-span, span + 1):
        if s == 0:
            continue
        lo = base + s
        if lo < 0 or lo + 8 > len(bits):
            continue
        val = 0
        for i in range(8):
            val |= bits[lo + i] << i
        if val == got[k]:
            return s
    return None


def find_byte_shift(sent: bytes, got: bytes, k: int,
                    window: int = 64, span: int = 4) -> int | None:
    """Is ``got`` from around ``k`` onward just ``sent`` slid by whole bytes?

    A different failure from the one above: not a mis-sampled byte but a lost or
    invented *frame*. If ``got[j:] == sent[j+1:]`` the device dropped a byte; if
    ``got[j:] == sent[j-1:]`` it produced one that was never sent. Both leave
    every following byte "wrong" while the payload is actually intact, which a
    plain byte-by-byte diff reports as total corruption.

    The realignment does not necessarily start at ``k`` itself. A dropped frame
    realigns right there, but an *invented* one puts a junk byte at ``k`` and
    only lines up again from ``k+1``, so a couple of start positions are tried.
    The comparison window is clipped to what both streams actually have, since a
    failure near the end of a block has little left to match against.

    Returns the offset in bytes (positive = the device dropped that many,
    negative = it produced that many extra), or ``None``.
    """
    for j in (k, k + 1, k + 2):
        for d in range(-span, span + 1):
            if d == 0 or j + d < 0:
                continue
            n = min(window, len(got) - j, len(sent) - (j + d))
            if n < 8:                     # too little left to be convincing
                continue
            if got[j:j + n] == sent[j + d:j + d + n]:
                return d
    return None


def first_difference(sent: bytes, got: bytes) -> int | None:
    """Index of the first differing byte over the common prefix, or ``None``."""
    for i, (g, w) in enumerate(zip(got, sent)):
        if g != w:
            return i
    return None


def report_mismatch(sent: bytes, got: bytes, block_no: int,
                    bytes_before: int, elapsed: float) -> None:
    """Print everything derivable about a failed block."""
    print()
    print(f"  MISMATCH in block {block_no}")
    print(f"    {bytes_before + len(sent)} bytes into the run, {elapsed:.1f}s in")
    print(f"    sent {len(sent)} bytes, {len(got)} came back")

    k = first_difference(sent, got)

    if k is None:
        # Every byte that arrived was right; the stream is simply the wrong
        # length. Short means a frame was lost, long means one was invented.
        if not got:
            print( "    nothing came back at all — check the bitstream is the "
                   "echo image, the port, and that --baud matches UART_CPB")
        elif len(got) < len(sent):
            print(f"    every byte received was correct — {len(sent) - len(got)} "
                  f"never arrived. A frame was lost, not corrupted.")
            print(f"    (if this is the first block: --block {len(sent)} may not "
                  f"be the device's BLOCK_LEN — {len(got)} came back)")
        else:
            print(f"    every expected byte was correct, plus "
                  f"{len(got) - len(sent)} extra — the device invented a frame.")
        return

    print(f"    first bad byte at index {k}: "
          f"got {got[k]:#04x}, want {sent[k]:#04x}, xor {got[k] ^ sent[k]:08b}")

    bits_wrong = 0
    ndiff = 0
    for g, w in zip(got, sent):
        if g != w:
            ndiff += 1
            bits_wrong |= g ^ w
    print(f"    {ndiff}/{min(len(got), len(sent))} bytes differ; "
          f"bits ever wrong: {bits_wrong:08b}")

    lo = max(0, k - 4)
    hi = min(len(sent), k + 5)
    print(f"    sent[{lo}:{hi}]  {sent[lo:hi].hex(' ')}")
    print(f"    got [{lo}:{hi}]  {bytes(got[lo:hi]).hex(' ')}")

    # --- the two structural explanations ---
    d = find_byte_shift(sent, got, k)
    if d is not None:
        if d > 0:
            print(f"    >> the stream realigns at a {d}-byte offset: the device "
                  f"DROPPED {d} frame(s) at index {k}, everything after is intact")
        else:
            print(f"    >> the stream realigns at a {d}-byte offset: the device "
                  f"produced {-d} EXTRA frame(s) at index {k}")

    s = explain_bit_shift(sent, got, k)
    if s is not None:
        when = "early" if s < 0 else "late"
        print(f"    >> byte {k} is exactly what you get sampling the sent "
              f"bitstream {abs(s)} bit {when} — a shifted frame, not noise")
    elif d is None:
        print( "    >> no whole-bit or whole-byte offset explains this: the "
               "framing held and an individual bit was mis-sampled")


# ---- Link -------------------------------------------------------------------

def echo_block(ser, data: bytes, quiet: float) -> bytes:
    """Write one whole block, then read the reply back.

    Sequential because the device is: it says nothing until the last byte of the
    block has arrived. There is no concurrency to arrange here, and a reader
    thread would only obscure that.

    ``quiet`` is an inter-chunk timeout, not a total one: the clock restarts
    every time bytes arrive, so a block is not penalised for taking as long as
    the baud rate says it must, while a device that has stopped talking is
    noticed within ``quiet`` seconds. It also bounds the failure case that
    matters — a device stuck mid-block, waiting for bytes the host has already
    sent — instead of hanging the run.

    **The wait before reading is not politeness, it is the measurement.**
    Reading the port while the USB-serial bridge is still shifting the block out
    corrupts the byte in flight — see ``tpu_uart.TPUUart._send``. Without it this
    function reports one mangled byte per block and blames the FPGA for it,
    which is exactly the trap this script exists to avoid: on a 64-byte block,
    reading immediately corrupted 72/6400 bytes and waiting first corrupted 0.
    """
    ser.write(data)
    ser.flush()
    time.sleep(10.0 * len(data) / ser.baudrate * 1.02 + 0.005)

    got = bytearray()
    deadline = time.monotonic() + quiet
    while len(got) < len(data) and time.monotonic() < deadline:
        chunk = ser.read(len(data) - len(got))
        if chunk:
            got.extend(chunk)
            deadline = time.monotonic() + quiet
    return bytes(got)


def resync(ser, block: int, quiet: float = 0.3) -> int | None:
    """Bring the device's block counter to a known position, and say what it was.

    A store-and-forward device has state the streaming echo did not: a partial
    block. Interrupt a previous run (or feed the board stray bytes) and it sits
    holding ``k`` bytes, expecting ``block - k`` more — after which every
    exchange is short by ``k`` and a byte diff blames the link for what is really
    a bookkeeping problem.

    One block of zero padding fixes it whatever ``k`` was. The device completes
    its block after ``block - k`` of them and replies; the remaining ``k`` land
    during that reply and are dropped, since the reply occupies a full ``block``
    byte times and ``k < block``. Either way it ends up back at zero.

    The reply is also the diagnostic. It is the ``k`` stale bytes the device was
    holding followed by our zeros, so the count of leading non-zero bytes *is*
    ``k`` — reported, not just corrected. (A stale byte that was itself zero
    hides in the padding and undercounts; that costs nothing, since the
    alignment is fixed regardless.)

    Returns ``k``, or ``None`` if the device did not return a full block: wrong
    bitstream, wrong port, wrong baud, or a ``BLOCK_LEN`` that is not ``block``.

    One deliberate side effect: if ``k`` was non-zero, the padding that gets
    dropped latches the board's sticky ``overrun`` flag (led[0]). That is why the
    caller prints the recovery — a fast-blinking led[0] afterwards is this, not a
    fault found during the run.
    """
    drain(ser, quiet)
    got = echo_block(ser, b"\x00" * block, quiet)
    drain(ser, quiet)

    if len(got) != block:
        return None
    return next((i for i, b in enumerate(got) if b == 0), len(got))


def drain(ser, quiet: float = 0.3) -> bytes:
    """Read until the device has said nothing for ``quiet`` seconds."""
    out = bytearray()
    saved, ser.timeout = ser.timeout, quiet
    try:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                return bytes(out)
            out += chunk
    finally:
        ser.timeout = saved


def autodetect_port() -> str:
    """Find the board's USB-serial port (the Cmod A7's FT2232 UART channel).

    Same rule as `test_uart_link.py`, repeated rather than imported so this file
    stays standalone.
    """
    try:
        from serial.tools import list_ports
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyserial is required: pip install pyserial") from exc

    cands = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x0403, 0x6010)]
    if not cands:                      # any FTDI part, then
        cands = [p for p in list_ports.comports() if p.vid == 0x0403]
    if not cands:
        raise SystemExit("no FTDI serial port found — pass --port explicitly")
    if len(cands) > 1:
        names = ", ".join(f"{p.device} ({p.description})" for p in cands)
        raise SystemExit(f"several candidate ports, pass --port: {names}")
    return cands[0].device


def run(cfg) -> int:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: pip install pyserial") from exc

    port = cfg.port or autodetect_port()
    duration = cfg.minutes * 60.0 if cfg.minutes else cfg.seconds

    print(f"port    : {port} @ {cfg.baud} baud 8N1"
          f"{'' if cfg.port else ' [autodetected]'}")
    if cfg.baud != DEFAULT_BAUD:
        err = 100.0 * (cfg.baud - DEFAULT_BAUD) / DEFAULT_BAUD
        print(f"          {err:+.2f}% off the device's {DEFAULT_BAUD} "
              f"— margin check, mismatches here are the point")
    print(f"block   : {cfg.block} bytes in then {cfg.block} back, fresh random "
          f"data each time (must match the device's BLOCK_LEN)")
    print(f"run for : {duration:.0f}s")
    print()

    rng = random.Random(cfg.seed)
    ser = serial.Serial(port, cfg.baud, timeout=0.2, write_timeout=10.0)
    try:
        stale = drain(ser)
        if stale:
            print(f"  note: dropped {len(stale)} stale byte(s) left on the link")

        held = resync(ser, cfg.block)
        if held is None:
            print(f"  {cfg.block} bytes of padding did not produce a "
                  f"{cfg.block}-byte reply — the device never completed a block.")
            print( "  check: the echo bitstream is loaded, --port, --baud, and "
                   "that --block matches BLOCK_LEN in board.tcl (a BLOCK_LEN "
                   "larger than --block looks exactly like this).")
            return 1
        if held:
            print(f"  note: device was holding {held} byte(s) of a previous "
                  f"block; realigned. led[0] will be flagging the padding this "
                  f"discarded — that is the recovery, not a fault")
        else:
            print( "  aligned: device was empty and returned the padding block")
        print()

        t0 = time.monotonic()
        blocks = 0
        total = 0
        last_print = t0

        while time.monotonic() - t0 < duration:
            data = bytes(rng.randbytes(cfg.block))
            got = echo_block(ser, data, cfg.timeout)
            elapsed = time.monotonic() - t0

            if got != data:
                report_mismatch(data, got, blocks, total, elapsed)
                extra = drain(ser)
                if extra:
                    print(f"    {len(extra)} further byte(s) arrived after the "
                          f"block: {extra[:16].hex(' ')}"
                          f"{' ...' if len(extra) > 16 else ''}")
                print()
                print(f"FAILED after {total + len(data)} bytes / "
                      f"{blocks + 1} blocks / {elapsed:.1f}s")
                return 1

            # A correct block that is nonetheless followed by extra bytes is the
            # same class of bug seen on the real link (the device over-running a
            # reply), so it fails the run rather than being silently drained.
            if ser.in_waiting:
                extra = drain(ser)
                print(f"\n  {len(extra)} extra byte(s) after a correct "
                      f"{len(data)}-byte block: {extra[:16].hex(' ')}")
                print(f"\nFAILED after {total + len(data)} bytes / "
                      f"{blocks + 1} blocks / {elapsed:.1f}s")
                return 1

            blocks += 1
            total += len(data)

            if time.monotonic() - last_print >= 5.0:
                last_print = time.monotonic()
                print(f"  {elapsed:6.1f}s  {blocks:5d} blocks  "
                      f"{total / 1024:8.1f} KiB  ok  "
                      f"({total / elapsed / 1024:.1f} KiB/s)")

        elapsed = time.monotonic() - t0
        print()
        print(f"PASSED: {total} bytes / {blocks} blocks over {elapsed:.1f}s "
              f"with no mismatch ({total / elapsed / 1024:.1f} KiB/s)")
        return 0
    finally:
        ser.close()


# ---- Offline checks ---------------------------------------------------------

class FakeDevice:
    """`rtl/uart_echo.sv`'s protocol in twenty lines, enough for :func:`resync`.

    Buffers ``block`` bytes, then returns them; a byte written while a reply is
    still outstanding is dropped and sets ``overrun``, as on the board. ``held``
    seeds it mid-block, which is the state `resync` exists to clear.

    Not a serial port simulator — no timing, no framing, no corruption. It models
    exactly the one thing under test: the block counter.
    """

    def __init__(self, block: int, held: bytes = b""):
        self.block = block
        self.buf = bytearray(held)
        self.out = bytearray()
        self.sending = False
        self.overrun = False
        self.timeout = 0.2
        self.baudrate = DEFAULT_BAUD

    def write(self, data: bytes) -> int:
        for b in data:
            if self.sending:            # nothing is listening during the reply
                self.overrun = True
                continue
            self.buf.append(b)
            if len(self.buf) == self.block:
                self.out += self.buf
                self.buf.clear()
                self.sending = True
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, n: int = 1) -> bytes:
        chunk = bytes(self.out[:n])
        del self.out[:n]
        if not self.out:
            self.sending = False
        return chunk

    @property
    def in_waiting(self) -> int:
        return len(self.out)


def offline() -> int:
    """Exercise the forensics and the alignment logic on known cases.

    Worth having: this code only ever runs on the day something breaks, which is
    the worst moment to discover it was wrong.
    """
    failed = 0

    def check(cond, what):
        nonlocal failed
        if cond:
            print(f"  ok    {what}")
        else:
            failed += 1
            print(f"  FAIL  {what}")

    check(to_bits(b"\x00") == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          "to_bits: 0x00 is a start bit, eight lows and a stop bit")
    check(to_bits(b"\xff") == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "to_bits: 0xff is a start bit then all highs")
    check(to_bits(b"\x01") == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
          "to_bits: data bits go out least-significant first")

    # The failure tb/uart_receiver_tb.sv reproduces: a false start bit in the
    # stop bit before a 0x40 frames it a bit early and it arrives as 0x80.
    sent = bytes([0x00, 0x40])
    check(explain_bit_shift(sent, bytes([0x00, 0x80]), 1) == -1,
          "explain_bit_shift: 0x40 -> 0x80 is a one-bit-early frame")
    check(explain_bit_shift(sent, bytes([0x00, 0x41]), 1) is None,
          "explain_bit_shift: a plain flipped bit is not explained as a shift")

    ramp = bytes(range(40))
    dropped = ramp[:5] + ramp[6:] + b"\xaa"        # frame 5 lost, length made up
    check(find_byte_shift(ramp, dropped, 5) == 1,
          "find_byte_shift: a dropped frame realigns at +1")
    inserted = ramp[:5] + b"\xaa" + ramp[5:-1]     # a frame that was never sent
    check(find_byte_shift(ramp, inserted, 5) == -1,
          "find_byte_shift: an invented frame realigns at -1")
    noisy = bytearray(ramp)
    noisy[5] ^= 0x08
    check(find_byte_shift(ramp, bytes(noisy), 5) is None,
          "find_byte_shift: one bad byte is not a realignment")

    check(first_difference(ramp, ramp) is None,
          "first_difference: identical streams have no first difference")
    check(first_difference(ramp, bytes(noisy)) == 5,
          "first_difference: finds the index")

    # --- the block protocol, against FakeDevice ---
    payload = bytes(range(64))
    check(echo_block(FakeDevice(64), payload, 0.05) == payload,
          "echo_block: a full block goes out and comes back")

    check(resync(FakeDevice(64), 64, 0.05) == 0,
          "resync: an empty device reports 0 stale bytes")

    dev = FakeDevice(64, held=b"\xaa\xbb\xcc")
    check(resync(dev, 64, 0.05) == 3,
          "resync: a device 3 bytes into a block reports 3")
    check(dev.overrun,
          "resync: realigning discards padding, which the board flags")
    check(echo_block(dev, payload, 0.05) == payload,
          "resync: the exchange after it is aligned")

    check(resync(FakeDevice(32), 64, 0.05) is None,
          "resync: a device whose BLOCK_LEN is smaller returns None")
    check(resync(FakeDevice(128), 64, 0.05) is None,
          "resync: a device whose BLOCK_LEN is larger returns None")

    print()
    if failed:
        print(f"FAILED: {failed} offline check(s)")
        return 1
    print("PASSED: all offline checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD,
                    help="host baud. The device is fixed at %(default)s; setting "
                         "this a few %% either side measures the sampling margin")
    ap.add_argument("--block", type=lambda s: int(s, 0), default=DEFAULT_BLOCK,
                    help="bytes per exchange. Must equal the device's BLOCK_LEN "
                         "(board.tcl, default %(default)s) — this is not a host "
                         "buffer size, it is the protocol")
    ap.add_argument("-s", "--seconds", type=float, default=DEFAULT_SECONDS,
                    help="how long to run (default: %(default)s)")
    ap.add_argument("-m", "--minutes", type=float,
                    help="how long to run, in minutes (overrides --seconds)")
    ap.add_argument("-t", "--timeout", type=float, default=2.0,
                    help="seconds of silence that counts as the device having "
                         "stopped talking (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the payload RNG for a reproducible run")
    ap.add_argument("--offline", action="store_true",
                    help="check the failure-analysis code, no board needed")
    cfg = ap.parse_args(argv)

    if cfg.offline:
        return offline()
    if not 1 <= cfg.block <= MAX_BLOCK:
        raise SystemExit(f"--block must be 1..{MAX_BLOCK}")
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
