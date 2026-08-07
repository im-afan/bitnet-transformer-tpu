#!/usr/bin/env python3
"""test_uart_link.py — self-checking bring-up tests for the TPU's UART link.

Exercises `rtl/uart_interface.sv` through `tpu_uart.py` against a real board, with
no toolchain and no program involved: every test writes its own expected data and
checks what comes back, so a run is pass/fail on its own with nothing to eyeball.
This is the thing to run *before* `run_program.py` — it separates "the link and
the SRAM work" from "the core computes the right answer".

    python accel/tpu/host/test_uart_link.py -p COM5
    python accel/tpu/host/test_uart_link.py -p COM5 --length 16384 --slow
    python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
    python accel/tpu/host/test_uart_link.py --offline      # driver checks, no board
    python accel/tpu/host/test_uart_link.py --sim          # against RTL, no board

`--sim` puts an Icarus Verilog simulation of `rtl/uart_memory.sv` — the
cmod_a7_mem bitstream's core — where the serial port would be, so every board
test below runs against the real RTL with no hardware (see `sim_link.py`). It is
roughly 10 000x slower than the wire, so it defaults to a shorter `--length` and
only a few `--iters` of the soak tests; it also has no analogue signalling, no
FTDI and no clock skew, so it can only find bugs that are in the RTL or in the
host's idea of the protocol. That is most of the search space, and unlike the
board it says exactly which clock cycle went wrong.

The core one is `sram_roundtrip`: `W` a block of external SRAM, `R` it back, and
compare byte for byte, once per data pattern. The rest widen that in the
directions a fresh board actually fails in — address lines (`sram_address_bus`),
frame boundaries (`sram_short_frames`), one command corrupting another's data
(`sram_isolation`), and the reject paths (`link_rejects_*`).

**These tests overwrite SRAM**, including the region `run_program.py` preloads
tensors into — that's fine (it rewrites them every run), but nothing on the board
survives a run of this file.

**The core must be idle.** While a program runs, every command is NAK'd and
touches nothing (`uart_interface.sv`'s arbitration rule), so the suite checks for
idle up front and stops with a message rather than reporting 20 bogus failures.

Requires `pyserial` (except under `--offline`).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time

# tpu_uart.py is deliberately standalone (no package, no repo imports), so add
# its directory rather than importing it as `host.tpu_uart`.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from tpu_uart import (                                        # noqa: E402
    CMD_READ,
    IMEM_LIMIT,
    MAX_LEN,
    MEM_ADDR_W,
    MEM_LIMIT,
    STAT_NAK,
    NakError,
    ProtocolError,
    ReplyTimeout,
    TPUUart,
    _check_imem,
    _check_mem,
    _check_pc,
    _header,
    pack_words,
    parse_hex_program,
)

# The `_`-prefixed imports above are the driver's frame-validation and framing
# internals. `host_validation` / `frame_encoding` test them directly on purpose:
# they mirror the RTL's VALIDATE state, and a drift between them is exactly the
# bug that desyncs the link (see the driver's module docstring).

DEFAULT_BASE = 0x0
DEFAULT_LENGTH = 4096      # ~0.36 s each way at 115200 — long enough to matter
DEFAULT_SEED = 20260802

SIM_LENGTH = 128           # --sim: a frame or two of turnaround is the whole bug
SIM_ITERS = 3              # --sim: the 1000-iteration soaks are a board thing


# ---- Harness ----------------------------------------------------------------

class Failure(AssertionError):
    """A test's own check failed — as opposed to the link falling over."""


class Test:
    def __init__(self, name, fn, needs_board, slow):
        self.name = name
        self.fn = fn
        self.needs_board = needs_board
        self.slow = slow


TESTS: list[Test] = []


def board_test(name, slow=False):
    """Register a test that needs a board on the other end of the port."""
    def deco(fn):
        TESTS.append(Test(name, fn, True, slow))
        return fn
    return deco


def offline_test(name):
    """Register a test of the host driver alone (runs with no hardware)."""
    def deco(fn):
        TESTS.append(Test(name, fn, False, False))
        return fn
    return deco


def check_bytes(addr: int, got: bytes, want: bytes, what: str, limit: int = 8) -> None:
    """Compare a readback against what was written; raise :class:`Failure` if not.

    The report is aimed at hardware bring-up, so it prints the XOR of each bad
    byte and the union of every bit that was ever wrong — a single stuck data
    line shows up as the same bit in every mismatch, which a plain byte diff
    buries.
    """
    if got == want:
        return
    if len(got) != len(want):
        raise Failure(f"{what}: got {len(got)} bytes back, expected {len(want)}")

    diffs = [(addr + i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
    bits = 0
    for _, g, w in diffs:
        bits |= g ^ w
    for a, g, w in diffs[:limit]:
        print(f"       @{a:05x}  got {g:#04x}  want {w:#04x}  xor {g ^ w:08b}")
    if len(diffs) > limit:
        print(f"       ... and {len(diffs) - limit} more")
    raise Failure(
        f"{what}: {len(diffs)}/{len(want)} bytes differ "
        f"(bits ever wrong: {bits:08b})"
    )


def roundtrip(tpu: TPUUart, addr: int, data: bytes, what: str) -> None:
    """Write, read back, compare — the whole point of the file, in one helper."""
    tpu.write_mem(addr, data)
    check_bytes(addr, tpu.read_mem(addr, len(data)), data, what)


def raw_exchange(tpu: TPUUart, frame: bytes, nbytes: int, timeout: float) -> bytes:
    """Send a hand-built frame and collect up to ``nbytes`` of reply.

    The driver validates every frame before sending, which is exactly what the
    reject tests need to bypass — they are checking the *device's* VALIDATE
    state. Only ever used with header-only commands, so a rejection cannot leave
    payload bytes behind to be re-decoded as commands.
    """
    tpu.ser.reset_input_buffer()
    # Via the driver's _send, not a bare write: reading while the frame is
    # still going out corrupts the byte in flight (see TPUUart._send).
    tpu._send(frame)
    saved, tpu.ser.timeout = tpu.ser.timeout, timeout
    try:
        reply = tpu.ser.read(nbytes)
    finally:
        tpu.ser.timeout = saved
    tpu.ser.reset_input_buffer()   # drop anything a mis-decode put on the wire
    return reply


def link_alive(tpu: TPUUart, addr: int) -> None:
    """Assert the link still carries a normal transaction (used after a reject)."""
    probe = bytes([0xC3, 0x3C, 0xA5, 0x5A])
    try:
        roundtrip(tpu, addr, probe, "link check after rejection")
    except ProtocolError as exc:
        raise Failure(
            f"link did not survive the rejected frame ({type(exc).__name__}: {exc}) "
            "— the FSM did not return to IDLE"
        ) from exc


# ---- Data patterns ----------------------------------------------------------
#
# Each targets a different failure: a constant catches stuck data bits, walking
# ones catches shorted neighbours, the address signature catches a byte landing
# at the wrong address, and random catches whatever the structured ones miss.

def addr_signature(a: int) -> int:
    """A byte that depends on the *address*, so aliasing can't go unnoticed."""
    return ((a * 0x1B) ^ (a >> 5) ^ 0x5A) & 0xFF


def patterns(base: int, n: int, seed: int) -> list[tuple[str, bytes]]:
    rng = random.Random(seed)
    return [
        ("ramp",          bytes(i & 0xFF for i in range(n))),
        ("zeros",         bytes(n)),
        ("ones",          b"\xff" * n),
        ("walking-ones",  bytes(1 << (i % 8) for i in range(n))),
        ("addr-sig",      bytes(addr_signature(base + i) for i in range(n))),
        ("random",        rng.randbytes(n)),
    ]


# ---- Failure forensics ------------------------------------------------------
#
# The intermittent failure this file is currently chasing is a *length* problem,
# not a data problem: the device transmits more bytes than the host asked for.
# uart_interface can only ever transmit SRAM read data or a one-byte status, so
# the total size of a reply recovers the length field the FSM actually ran with
# -- and that number is what separates the candidate causes from each other.

def drain_until_quiet(tpu: TPUUart, quiet_s: float = 0.5) -> bytes:
    """Read until the device has said nothing for ``quiet_s``.

    Closing the port at the first error truncates the evidence: the device may
    still be streaming, and how much more it had to say is the whole diagnosis.
    """
    out = bytearray()
    saved, tpu.ser.timeout = tpu.ser.timeout, quiet_s
    try:
        while True:
            chunk = tpu.ser.read(4096)
            if not chunk:
                return bytes(out)
            out += chunk
    finally:
        tpu.ser.timeout = saved


def report_reply(asked: int, total: int, tail: bytes) -> None:
    """Say what a reply of ``total`` bytes implies about the header the FSM saw."""
    print(f"       host asked for {asked} ({asked:#06x}) bytes, "
          f"device sent {total} ({total:#06x})")
    if total > asked:
        if total == (asked << 1) & 0xFFFF:
            print("       total == asked << 1: the length byte was received one "
                  "bit shifted -- RX-side corruption of a byte that did arrive")
        elif (total & 0xFF) in (0x00, 0xFF) and total > 0xFF:
            print("       low byte of the length looks like payload data: a header "
                  "byte was DROPPED and the frame shifted, not corrupted")
        else:
            print("       neither a one-bit shift nor an obvious frame shift -- "
                  "capture this number, it does not match either theory")
    # Run-length summary: `a5 x191  15 x27` says more than 200 lines of hexdump.
    runs: list[list[int]] = []
    for b in tail:
        if runs and runs[-1][0] == b:
            runs[-1][1] += 1
        else:
            runs.append([b, 1])
    if runs:
        shown = "  ".join(f"{b:02x} x{n}" for b, n in runs[:12])
        more = "  ..." if len(runs) > 12 else ""
        print(f"       tail: {shown}{more}")


# ---- Board tests ------------------------------------------------------------

@board_test("sram_write")
def sram_write(tpu, cfg):
    """Hammer one W/R pair until the link breaks, then measure *how* it broke.

    Built to capture the state of the wire at the moment it goes wrong rather
    than to pass or fail, because the failure is intermittent and only appears
    after tens of seconds of traffic:

      * every iteration writes different data, so a readback of stale buffered
        bytes cannot be mistaken for a correct one;
      * the input buffer is asserted empty before each command, which catches an
        over-long reply on the iteration that caused it rather than two commands
        later, when the link has already desynced beyond reading;
      * on any failure the link is drained to silence and the *total* byte count
        is reported, which recovers the length field the device actually used.
    """
    n = cfg.length

    for i in range(cfg.iters):
        # Distinct every iteration: stale data cannot look like a pass.
        data = bytes((addr_signature(cfg.base + j) ^ (i * 7 + 1)) & 0xFF
                     for j in range(n))

        if tpu.ser.in_waiting:
            tail = drain_until_quiet(tpu)
            print(f"     iteration {i}: {len(tail)} unread bytes before the command")
            report_reply(n, n + len(tail), tail)
            raise Failure(f"iteration {i}: {len(tail)} stale bytes on the link")

        t0 = time.monotonic()
        try:
            tpu.write_mem(cfg.base, data)
            got = tpu.read_mem(cfg.base, n)
        except (ProtocolError, NakError, ReplyTimeout) as exc:
            partial = getattr(exc, "partial", b"")
            tail = drain_until_quiet(tpu)
            print(f"     iteration {i}: {type(exc).__name__}: {exc}")
            print(f"       {len(partial)} bytes arrived before the error, "
                  f"{len(tail)} more after draining")
            report_reply(n, len(partial) + len(tail), partial + tail)
            raise Failure(f"iteration {i}: {type(exc).__name__}") from exc

        dt = max(time.monotonic() - t0, 1e-6)

        if tpu.ser.in_waiting:
            tail = drain_until_quiet(tpu)
            print(f"     iteration {i}: {len(tail)} extra bytes after a {n}-byte read")
            report_reply(n, n + len(tail), tail)
            raise Failure(f"iteration {i}: device over-ran the read "
                          f"by {len(tail)} bytes")

        check_bytes(cfg.base, got, data, f"iteration {i}")

        if i % 25 == 0:
            print(f"     iteration {i:5d}  {n} B @{cfg.base:#07x}  ok  ({dt:.2f}s)")


@board_test("sram_write_simple")
def sram_write_simple(tpu, cfg):
    """The same W/R hammer as `sram_write`, with none of the forensics.

    `sram_write` wraps every command in input-buffer assertions, a drain to
    silence and a reply-length analysis. That is all machinery standing between
    the link and the verdict, and machinery can be wrong: an `in_waiting` check
    that trips on its own timing, or a drain that swallows a byte the next
    command needed, would look exactly like the bug it is hunting.

    So this is the control. Write, read back, compare, repeat — nothing else
    touches the port. Whatever the driver raises propagates to the harness,
    which reports it as an ERROR and moves on. Run both over the same traffic:
    if this one passes where `sram_write` fails, the difference is in the
    instrumentation and not in the link, and that is worth knowing before
    another day goes into chasing the device.

        python accel/tpu/host/test_uart_link.py -p COM5 --only simple

    `--only sram_write` matches both tests (it is a substring filter); `--only
    simple` selects this one alone.
    """
    n = cfg.length
    rng = random.Random(cfg.seed)

    for i in range(cfg.iters):
        # Fresh data every iteration, so a readback of stale buffered bytes
        # cannot be mistaken for a pass. The one guard worth keeping, because it
        # costs nothing and its absence makes a pass meaningless.
        data = rng.randbytes(n)

        t0 = time.monotonic()
        roundtrip(tpu, cfg.base, data, f"iteration {i}")
        dt = max(time.monotonic() - t0, 1e-6)

        if i % 1 == 0:
            print(f"     iteration {i:5d}  {n} B @{cfg.base:#07x}  ok  ({dt:.2f}s)")


@board_test("sram_roundtrip")
def sram_roundtrip(tpu, cfg):
    """`W` a block of SRAM, `R` it back, compare — once per data pattern.

    The self-check the whole file is built around: the host is the only source of
    truth for what should be in memory, so a pass means the command framing, the
    SRAM controller and the physical link all agree.
    """
    for name, data in patterns(cfg.base, cfg.length, cfg.seed):
        t0 = time.monotonic()
        tpu.write_mem(cfg.base, data)
        got = tpu.read_mem(cfg.base, len(data))
        dt = max(time.monotonic() - t0, 1e-6)
        check_bytes(cfg.base, got, data, f"pattern {name}")
        print(f"     {name:<13} {len(data)} B @{cfg.base:#07x}  ok  "
              f"({dt:.2f}s, {2 * len(data) / dt / 1024:.1f} KiB/s both ways)")


@board_test("sram_isolation")
def sram_isolation(tpu, cfg):
    """Two far-apart regions must not disturb each other.

    Writes region A, then region B, then reads *both* back. A roundtrip that
    checks each region right after writing it would pass even if every write
    landed at the same place; this ordering does not. Region B sits at the very
    top of the 19-bit space, so this doubles as the end-of-SRAM check.
    """
    n = min(256, cfg.length)
    a_addr = cfg.base
    b_addr = MEM_LIMIT - n
    if a_addr + n > b_addr:                     # overlapping regions prove nothing
        a_addr = 0
    data_a = bytes(addr_signature(a_addr + i) for i in range(n))
    data_b = bytes(addr_signature(b_addr + i) for i in range(n))

    tpu.write_mem(a_addr, data_a)
    tpu.write_mem(b_addr, data_b)
    check_bytes(a_addr, tpu.read_mem(a_addr, n), data_a, "region A after writing B")
    check_bytes(b_addr, tpu.read_mem(b_addr, n), data_b, "region B")
    print(f"     {n} B @{a_addr:#07x} and @{b_addr:#07x} (top of SRAM) both intact")


@board_test("sram_address_bus")
def sram_address_bus(tpu, cfg):
    """One byte at each one-hot address — catches shorted or open address lines.

    Writes a distinct signature at 0, at every 2**k inside the 19-bit space, and
    at the last byte, then reads them all back afterwards. If two address lines
    are swapped or one is stuck, a pair of these alias and each read comes back
    holding the other's signature — which the values being distinct makes obvious.

    The readbacks are one byte each, which the protocol cannot tell from a lone
    NAK (both are a single byte, 0x15). The signatures start at 0x40 so no
    expected value is 0x15 and a rejected read still fails the comparison.
    """
    addrs = [0] + [1 << k for k in range(MEM_ADDR_W)] + [MEM_LIMIT - 1]
    values = {a: 0x40 + i for i, a in enumerate(addrs)}
    assert STAT_NAK not in values.values()

    for a, v in values.items():
        tpu.write_mem(a, bytes([v]))
    bad = []
    for a, v in values.items():
        got = tpu.read_mem(a, 1)[0]
        if got != v:
            alias = next((x for x, y in values.items() if y == got), None)
            where = f" (that is the byte written at {alias:#07x})" if alias is not None else ""
            bad.append(f"@{a:#07x}: got {got:#04x}, want {v:#04x}{where}")
    for line in bad:
        print(f"       {line}")
    if bad:
        raise Failure(f"{len(bad)}/{len(addrs)} one-hot addresses read back wrong")
    print(f"     {len(addrs)} one-hot addresses across the {MEM_ADDR_W}-bit space ok")


@board_test("sram_short_frames")
def sram_short_frames(tpu, cfg):
    """Tiny transfers at odd offsets — the FSM's `idx + 1 == len` termination.

    Off-by-one in that comparison shows up on short frames and nowhere else: a
    4096-byte block that stops one byte early still looks fine unless the last
    byte happens to differ. Odd base offsets go with it because the SRAM
    controller works in bytes and nothing here should care about alignment.

    Payload bytes avoid 0x15 for the same reason as `sram_address_bus`: a
    one-byte read is indistinguishable from a NAK.
    """
    base = cfg.base if cfg.base + 16 <= MEM_LIMIT else 0
    for length in (1, 2, 3, 4, 5, 7, 8, 9):
        for off in (0, 1, 3):
            addr = base + off
            data = bytes(
                (0x30 + i * 13 + length * 7 + off) & 0xFF for i in range(length)
            )
            data = bytes(b ^ 0x01 if b == STAT_NAK else b for b in data)
            roundtrip(tpu, addr, data, f"len={length} at +{off}")
    print("     lengths 1..9 at offsets +0/+1/+3 all round-trip")


@board_test("link_rejects_bad_command")
def link_rejects_bad_command(tpu, cfg):
    """An unknown command byte gets a NAK, and the FSM goes back to IDLE.

    IDLE's `default` branch. Worth its own test because the recovery is what
    matters: after the NAK the very next byte must be decoded as a fresh command,
    or one line-noise byte wedges the link for good.
    """
    for cmd in (b"X", b"\x00", b"\xff"):
        reply = raw_exchange(tpu, cmd, 1, 0.5)
        if reply != bytes([STAT_NAK]):
            raise Failure(
                f"command {cmd.hex()} answered {reply.hex(' ') or '<nothing>'}, "
                f"expected NAK ({STAT_NAK:#04x})"
            )
    link_alive(tpu, cfg.base)
    print("     3 unknown command bytes NAK'd; link still usable afterwards")


@board_test("link_rejects_out_of_range")
def link_rejects_out_of_range(tpu, cfg):
    """The device's own VALIDATE rules, checked by bypassing the host's.

    `tpu_uart.py` refuses these frames client-side, so they normally never reach
    the board — but that host-side check is only safe if it is a superset of what
    the RTL rejects. These are all reads (header only, no data phase), so a
    rejection cannot leave payload bytes to be mis-decoded.
    """
    cases = [
        ("address above the 19-bit space", _header(CMD_READ, MEM_LIMIT, 4)),
        ("range running past the end",     _header(CMD_READ, MEM_LIMIT - 2, 8)),
        ("zero length",                    _header(CMD_READ, cfg.base, 0)),
    ]
    for why, frame in cases:
        reply = raw_exchange(tpu, frame, 2, 0.5)
        if reply != bytes([STAT_NAK]):
            raise Failure(
                f"{why}: answered {reply.hex(' ') or '<nothing>'}, "
                f"expected a lone NAK ({STAT_NAK:#04x})"
            )
    link_alive(tpu, cfg.base)
    print("     3 invalid frames NAK'd with no data phase; link still usable")


@board_test("sram_long_transfer", slow=True)
def sram_long_transfer(tpu, cfg):
    """A transfer past the 16-bit length field, so the driver has to split it.

    `write_mem` / `read_mem` chop anything over 65535 bytes into back-to-back
    commands; this is the only test that reaches that path, and it also leaves
    the link running flat out for ~12 s, which is where a marginal cable or a
    baud-rate mismatch finally shows itself. Opt-in via `--slow`.
    """
    n = MAX_LEN + 4096
    base = cfg.base if cfg.base + n <= MEM_LIMIT else 0
    data = random.Random(cfg.seed).randbytes(n)
    t0 = time.monotonic()
    roundtrip(tpu, base, data, f"{n}-byte split transfer")
    dt = max(time.monotonic() - t0, 1e-6)
    print(f"     {n} B @{base:#07x} across 2 frames each way  ok  "
          f"({dt:.1f}s, {2 * n / dt / 1024:.1f} KiB/s both ways)")


# ---- Offline tests (no board) -----------------------------------------------

@offline_test("host_validation")
def host_validation(tpu, cfg):
    """The host must reject exactly the frames the RTL's VALIDATE state rejects.

    Not a nicety: the FSM NAKs *before* the data phase and returns to IDLE, so a
    write the device rejects leaves the host's payload to be re-decoded as
    commands. The driver avoids that by never sending an invalid frame — which
    only holds while these rules match `uart_interface.sv`.
    """
    bad = [
        ("SRAM addr above 2^19",     _check_mem,  (MEM_LIMIT, 1)),
        ("SRAM addr above 2^24",     _check_mem,  (1 << 24, 1)),
        ("SRAM range past the end",  _check_mem,  (MEM_LIMIT - 4, 8)),
        ("SRAM zero length",         _check_mem,  (0, 0)),
        ("SRAM length past 16 bits", _check_mem,  (0, MAX_LEN + 1)),
        ("IMEM addr above 2^10",     _check_imem, (IMEM_LIMIT, 4)),
        ("IMEM length not a word",   _check_imem, (0, 6)),
        ("IMEM zero length",         _check_imem, (0, 0)),
        ("IMEM range past the end",  _check_imem, (IMEM_LIMIT - 1, 8)),
        ("boot PC above 2^10",       _check_pc,   (IMEM_LIMIT,)),
    ]
    good = [
        ("last SRAM byte",           _check_mem,  (MEM_LIMIT - 1, 1)),
        ("SRAM range up to the end", _check_mem,  (MEM_LIMIT - 8, 8)),
        ("max SRAM length",          _check_mem,  (0, MAX_LEN)),
        ("last IMEM word",           _check_imem, (IMEM_LIMIT - 1, 4)),
        ("last boot PC",             _check_pc,   (IMEM_LIMIT - 1,)),
    ]
    for why, fn, args in bad:
        try:
            fn(*args)
        except ValueError:
            continue
        raise Failure(f"{why}: accepted, but the device would NAK it")
    for why, fn, args in good:
        try:
            fn(*args)
        except ValueError as exc:
            raise Failure(f"{why}: rejected by the host, but the device accepts it "
                          f"({exc})") from exc
    print(f"     {len(bad)} invalid frames rejected, {len(good)} legal ones accepted")


@offline_test("frame_encoding")
def frame_encoding(tpu, cfg):
    """Header layout, word packing and program parsing, byte for byte.

    Endianness here is the one thing no readback can catch: a host that packs the
    address the wrong way round still round-trips its own writes perfectly, and
    only the *addresses* are wrong.
    """
    cases = [
        (_header(0x52, 0x001000, 64), "520010000040", "R 0x1000 len 64"),
        (_header(0x57, 0x07FFFF, 1),  "5707ffff0001", "W 0x7ffff len 1"),
        (_header(0x47, 0x000000),     "47000000",     "G pc 0 (no length)"),
        (_header(0x49, 0x000002, 8),  "490000020008", "I word 2 len 8"),
    ]
    for got, want, why in cases:
        if got.hex() != want:
            raise Failure(f"{why}: encoded {got.hex(' ')}, expected {want}")

    packed = pack_words([0x54000010, 0x000000FF])
    if packed.hex() != "54000010000000ff":
        raise Failure(f"pack_words is not big-endian: {packed.hex(' ')}")
    try:
        pack_words([1 << 32])
        raise Failure("pack_words accepted a value wider than 32 bits")
    except ValueError:
        pass

    words = parse_hex_program("54000010  // load\n\n  ff  \n// comment only\n")
    if words != [0x54000010, 0xFF]:
        raise Failure(f"parse_hex_program returned {[hex(w) for w in words]}")
    try:
        parse_hex_program("123456789")
        raise Failure("parse_hex_program accepted a 9-digit word")
    except ValueError:
        pass
    print(f"     {len(cases)} headers, word packing and program parsing all exact")


# ---- Runner -----------------------------------------------------------------

def autodetect_port() -> str:
    """Find the board's USB-serial port (the Cmod A7's FT2232 UART channel).

    Same rule as `run_program.py`, repeated rather than imported: that module
    pulls in the whole tpulang toolchain at import time, and this file has no
    business needing it.
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


def require_idle(tpu: TPUUart) -> None:
    """Stop unless the core is idle — it NAKs every command while running.

    There is no status command, so "idle" is inferred the way `run_program.py`
    infers it: a harmless 2-byte read either answers with data or is rejected.
    """
    try:
        tpu.read_mem(0, 2)
    except NakError:
        raise SystemExit(
            "the core is running — every command would be NAK'd. Reset the board "
            "(or wait for the program to halt; led[1] is `done`) and re-run."
        ) from None
    except ReplyTimeout as exc:
        raise SystemExit(
            f"no reply from the board ({exc}). Check the port, that the bitstream "
            "is loaded, and that --baud matches UART_CPB in board.tcl."
        ) from None


def select(only: str | None, slow: bool, offline: bool) -> list[Test]:
    tests = [t for t in TESTS if not (offline and t.needs_board)]
    if only:
        tests = [t for t in tests if only in t.name]
        if not tests:
            raise SystemExit(f"no test matches {only!r}: "
                             f"{', '.join(t.name for t in TESTS)}")
        return tests                    # an explicit filter overrides --slow
    return [t for t in tests if slow or not t.slow]


def run(tests: list[Test], tpu: TPUUart | None, cfg) -> int:
    """Run every test, keep going after a failure, return the failure count."""
    failed = 0
    for t in tests:
        print(f"\n-- {t.name}")
        t0 = time.monotonic()
        try:
            t.fn(tpu, cfg)
        except Failure as exc:
            failed += 1
            print(f"   FAIL   {exc}")
        except (ProtocolError, ValueError, OSError) as exc:
            failed += 1
            print(f"   ERROR  {type(exc).__name__}: {exc}")
        else:
            print(f"   ok     ({time.monotonic() - t0:.2f}s)")
            continue
        if tpu is not None:
            # Under --sim the exact clock cycle is recoverable, which is the
            # whole reason for running there — print it while the failure is
            # still on screen rather than making anyone dig through a VCD.
            sim = getattr(tpu, "sim", None)
            if sim is not None:
                print(f"       {sim.summary()}")
                sim.dump_fsm()
            # A failure can leave the device mid-frame or with reply bytes still
            # queued; drain before the next test so one failure doesn't cascade.
            tpu.resync()
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=115200,
                    help="must match UART_CPB on the device (default: %(default)s)")
    ap.add_argument("-t", "--timeout", type=float, default=5.0,
                    help="per-reply timeout in seconds (default: %(default)s)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=DEFAULT_BASE,
                    help="SRAM byte address the block tests use (default: 0x0)")
    ap.add_argument("--length", type=lambda s: int(s, 0), default=None,
                    help=f"bytes per pattern (default: {DEFAULT_LENGTH}, "
                         f"or {SIM_LENGTH} under --sim)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="seed for the random pattern (default: %(default)s)")
    ap.add_argument("--only", metavar="SUBSTR",
                    help="run only tests whose name contains this")
    ap.add_argument("--slow", action="store_true",
                    help="also run the >64 KiB transfer test (~12 s)")
    ap.add_argument("--offline", action="store_true",
                    help="run only the tests that need no board")
    ap.add_argument("--iters", type=int, default=None, metavar="N",
                    help="iterations of the sram_write soak tests "
                         "(default: 1000 on a board, 3 under --sim)")
    ap.add_argument("--trace", action="store_true",
                    help="use tracer")
    sim = ap.add_argument_group("simulation (--sim)")
    sim.add_argument("--sim", action="store_true",
                     help="run against an Icarus simulation of rtl/uart_memory.sv "
                          "instead of a board (see sim_link.py)")
    sim.add_argument("--sim-fsm", action="store_true",
                     help="log uart_interface's state machine and print the tail "
                          "of it whenever a test fails")
    sim.add_argument("--sim-vcd", metavar="FILE",
                     help="dump waves to FILE (roughly triples the run time)")
    sim.add_argument("--sim-keep", action="store_true",
                     help="keep the bridge's stream files for inspection")
    cfg = ap.parse_args(argv)

    if cfg.sim and cfg.offline:
        raise SystemExit("--sim and --offline are mutually exclusive")
    if cfg.iters is None:
        cfg.iters = SIM_ITERS if cfg.sim else 1000
    # The simulator moves about 10 000x slower than the wire, so the board
    # default of 4096 B per pattern would take hours. Anything the link gets
    # wrong, it gets wrong within a frame or two of a turnaround.
    if cfg.length is None:
        cfg.length = SIM_LENGTH if cfg.sim else DEFAULT_LENGTH

    if not 1 <= cfg.length <= MAX_LEN:
        raise SystemExit(f"--length must be 1..{MAX_LEN} (one frame per pattern)")
    if cfg.base + cfg.length > MEM_LIMIT:
        raise SystemExit(f"--base {cfg.base:#x} + --length {cfg.length} runs past "
                         f"the end of SRAM ({MEM_LIMIT:#x})")

    tests = select(cfg.only, cfg.slow, cfg.offline)
    names = ", ".join(t.name for t in tests)

    if cfg.offline:
        print(f"offline : {len(tests)} test(s) — {names}")
        failed = run(tests, None, cfg)
    elif cfg.sim:
        import sim_link                                          # noqa: PLC0415

        work = os.path.join(HERE, "..", "tb", "cosim")
        print(f"device  : Icarus simulation of rtl/uart_memory.sv "
              f"(cmod_a7_mem, CPB 104, RX_TIMEOUT 0)")
        print(f"region  : {cfg.length} B at {cfg.base:#07x} "
              f"(SRAM is {MEM_LIMIT // 1024} KiB, behavioral chip model)")
        print(f"tests   : {len(tests)} — {names}")
        try:
            vvp = sim_link.build(work)
            ser = sim_link.SimSerial(
                vvp, work, fsm_trace=cfg.sim_fsm, vcd=cfg.sim_vcd, verbose=True
            )
        except sim_link.SimLinkError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            tpu = sim_link.SimTPUUart(ser, cfg.timeout, trace=cfg.trace)
            require_idle(tpu)
            failed = run(tests, tpu, cfg)
            print(f"\n{ser.summary()}")
        finally:
            ser.close()
            if not cfg.sim_keep:
                for name in ("cosim_h2d.txt", "cosim_d2h.txt"):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(work, name))
    else:
        port = cfg.port or autodetect_port()
        print(f"port    : {port} @ {cfg.baud} baud 8N1"
              f"{'' if cfg.port else ' [autodetected]'}")
        print(f"region  : {cfg.length} B at {cfg.base:#07x} "
              f"(SRAM is {MEM_LIMIT // 1024} KiB — these tests overwrite it)")
        print(f"tests   : {len(tests)} — {names}")
        try:
            with TPUUart(port, cfg.baud, cfg.timeout, trace=cfg.trace) as tpu:
                require_idle(tpu)
                failed = run(tests, tpu, cfg)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print()
    if failed:
        print(f"FAILED: {failed}/{len(tests)} test(s)")
        return 1
    print(f"PASSED: all {len(tests)} test(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
