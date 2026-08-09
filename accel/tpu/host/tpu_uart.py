"""Host-side driver for the TPU's UART command link (``rtl/uart_interface.sv``).

The FPGA is a pure slave: the host sends a fixed-header command frame and the
device answers. Five commands, four of them with a 3-byte big-endian address
field:

===== ====== ============= ========================== =====================
CMD   byte   frame                                    reply
===== ====== ============= ========================== =====================
``R`` 0x52   read  SRAM    ``CMD A2 A1 A0 L1 L0``      ``len`` data bytes
``W`` 0x57   write SRAM    ``CMD A2 A1 A0 L1 L0`` +    ACK / NAK
                           ``data[len]``
``I`` 0x49   write IMEM    ``CMD A2 A1 A0 L1 L0`` +    ACK / NAK
                           ``data[len]``
``G`` 0x47   go / run      ``CMD A2 A1 A0``            ACK / NAK
``T`` 0x54   read timer    ``CMD``                     4 counter bytes
===== ====== ============= ========================== =====================

* ``R``/``W`` addresses are 19-bit external-SRAM byte addresses; ``len`` is a
  byte count in ``1..65535``.
* ``I`` addresses are *instruction word indices* (10-bit); ``len`` is still in
  **bytes** and must be a multiple of 4. Each word is sent MSB first.
* ``G`` carries no length and no payload — it pulses the scalar unit's run
  trigger with ``run_pc = addr``.
* ``T`` is a single byte and cannot fail: it returns the scalar unit's
  run-length counter (``rtl/cycle_timer.sv``) as 4 bytes MSB first — the core
  clocks the last run took, or how far the current one has got.
* Read replies are header-less: the host already knows ``len`` and simply reads
  that many bytes. A *rejected* read answers with a lone NAK instead.

Two device behaviours drive the design here:

**Rejected commands do not consume the data phase.** The FSM validates right
after the header and, on failure, sends NAK and returns to IDLE — so payload
bytes the host had already queued would be re-decoded as fresh command bytes.
Every command is therefore validated host-side against the exact same rules the
RTL applies (:func:`_check_mem`, :func:`_check_imem`) before a single byte goes
out, and a NAK on a write is treated as a desync (the link is drained and the
error raised) rather than a routine error.

**The core has priority.** Any command arriving while the scalar unit is running
is NAK'd and touches nothing, so preload/readback only works while the core is
idle. ``T`` is the exception — it reads a counter and contends over nothing, so
it is answered at any time. There is still no ``busy``/``done`` status command:
:meth:`TPUUart.read_timer` tells you how long the core has been running, not
whether it has stopped (the count also sits still if the run never started), so
completion is inferred as before, from a command that stops being NAK'd.

Both failure modes are byte-level, so the driver can log every byte it moves
(see :class:`UartTrace`)::

    with TPUUart("COM3", trace=True) as tpu:    # live hexdump to stderr
        tpu.go(0)

    tpu = TPUUart("COM3", trace=UartTrace(sink=None))   # record silently
    ...
    tpu.dump_trace()                            # print it after the fact

From the CLI, ``--trace`` / ``--trace-file FILE``.

Requires ``pyserial``.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

# ---- Protocol constants (mirror rtl/uart_interface.sv) -----------------------

CMD_READ = 0x52   # 'R'
CMD_WRITE = 0x57  # 'W'
CMD_IMEM = 0x49   # 'I'
CMD_GO = 0x47     # 'G'
CMD_TIMER = 0x54  # 'T'

TIMER_BYTES = 4   # width of the 'T' reply (device counter is 32-bit)

STAT_ACK = 0x06
STAT_NAK = 0x15

DEFAULT_BAUD = 115200      # CLK_PER_BIT = 868 @ 100 MHz, 8N1

MEM_ADDR_W = 19            # external SRAM byte address
IMEM_AW = 10               # instruction memory word address
MEM_LIMIT = 1 << MEM_ADDR_W
IMEM_LIMIT = 1 << IMEM_AW

MAX_LEN = 0xFFFF           # 16-bit length field
MAX_IMEM_LEN = MAX_LEN & ~0x3   # 'I' payloads must be a multiple of 4 bytes
ADDR_LIMIT = 1 << 24       # 3-byte address field

# ---- Transmit settling (see TPUUart._send) ----------------------------------
#
# Reading the port while the USB-serial bridge is still shifting a frame out
# corrupts the byte in flight, so _send waits for the frame to clear the wire
# before anything reads. Both guards sit on top of the computed wire time
# because that time cannot be measured, only predicted:
#
#   TX_SETTLE_S    write() returns when the bytes reach the driver, not when
#                  the bridge starts transmitting; the USB OUT is scheduled a
#                  frame or two later. A fixed pad covers that start-up skew,
#                  which does not scale with the transfer.
#   TX_RATE_SLACK  the real bit rate is only nominally the baud we asked for
#                  (the FT2232H divides 120 MHz, the device divides its own
#                  clock), so the true wire time drifts from the computed one
#                  in proportion to the transfer. 2% covers a 4 KiB frame with
#                  room to spare.
#
# Costs one settling time per command. That is invisible next to the wire time
# of a block transfer and a few ms on a short one.
TX_SETTLE_S = 0.005
TX_RATE_SLACK = 1.02


class ProtocolError(Exception):
    """The device answered, but not the way the protocol says it should."""


class NakError(ProtocolError):
    """The device rejected the command (bad frame, or the core was busy)."""


class ReplyTimeout(ProtocolError):
    """The device did not send the expected number of bytes in time.

    ``partial`` holds whatever did arrive — the read path uses it to tell a
    rejected command (a lone NAK) from a link that simply went quiet.
    """

    def __init__(self, message: str, partial: bytes = b""):
        super().__init__(message)
        self.partial = partial


# ---- Frame validation -------------------------------------------------------
#
# These mirror the RTL's VALIDATE state exactly. Catching a bad frame here (as
# a ValueError, before any byte is sent) keeps the host and device in lockstep;
# letting the device reject it instead risks a desync on the write path.

def _check_mem(addr: int, length: int) -> None:
    if not 0 <= addr < ADDR_LIMIT:
        raise ValueError(f"address {addr:#x} does not fit the 24-bit field")
    if addr >= MEM_LIMIT:
        raise ValueError(f"address {addr:#x} outside the {MEM_ADDR_W}-bit SRAM space")
    if not 1 <= length <= MAX_LEN:
        raise ValueError(f"length {length} outside 1..{MAX_LEN}")
    if addr + length > MEM_LIMIT:
        raise ValueError(
            f"range {addr:#x}+{length} runs past the end of SRAM ({MEM_LIMIT:#x})"
        )


def _check_imem(word_addr: int, length: int) -> None:
    if not 0 <= word_addr < ADDR_LIMIT:
        raise ValueError(f"address {word_addr:#x} does not fit the 24-bit field")
    if word_addr >= IMEM_LIMIT:
        raise ValueError(
            f"word address {word_addr:#x} outside the {IMEM_AW}-bit IMEM space"
        )
    if not 1 <= length <= MAX_LEN:
        raise ValueError(f"length {length} outside 1..{MAX_LEN}")
    if length % 4:
        raise ValueError(f"IMEM payload {length} bytes is not a multiple of 4")
    if word_addr + length // 4 > IMEM_LIMIT:
        raise ValueError(
            f"{length // 4} words at {word_addr:#x} run past the end of IMEM "
            f"({IMEM_LIMIT} words)"
        )


def _check_pc(pc: int) -> None:
    if not 0 <= pc < IMEM_LIMIT:
        raise ValueError(f"boot PC {pc:#x} outside the {IMEM_AW}-bit IMEM space")


def _header(cmd: int, addr: int, length: int | None = None) -> bytes:
    """CMD + 24-bit address (MSB first) [+ 16-bit length (MSB first)]."""
    frame = bytes([cmd]) + addr.to_bytes(3, "big")
    if length is not None:
        frame += length.to_bytes(2, "big")
    return frame


def pack_words(words: Iterable[int]) -> bytes:
    """Pack 32-bit instruction words into the big-endian byte stream ``I`` wants."""
    out = bytearray()
    for i, w in enumerate(words):
        if not 0 <= w <= 0xFFFFFFFF:
            raise ValueError(f"instruction word {i} ({w:#x}) is not a 32-bit value")
        out += w.to_bytes(4, "big")
    return bytes(out)


# ---- Byte-level tracing -----------------------------------------------------
#
# Debugging this link means answering "which byte went where, and when" — a
# desync shows up as the device replying to bytes the host thought were payload,
# which is invisible at the level of "write_mem raised NakError".
#
# Every transfer goes through _TracedSerial, a transparent proxy around the
# pyserial port, so the trace covers the driver's own traffic *and* anything a
# caller does with ``tpu.ser`` directly (the raw frame pokes in
# test_uart_link.py, say). Bytes thrown away by reset_input_buffer() are
# recorded too: stale reply bytes sitting in the input buffer are exactly the
# evidence that the previous command desynced.

TRACE_TX = "TX"      # host -> device
TRACE_RX = "RX"      # device -> host
TRACE_DROP = "DROP"  # device -> host, discarded unread by the driver
TRACE_NOTE = "--"    # no bytes; a decode ("ACK") or a driver-side remark


@dataclass
class TraceEvent:
    """One serial operation: a run of bytes moving in one direction."""

    t: float            # seconds since the trace started
    kind: str           # TRACE_TX | TRACE_RX | TRACE_DROP | TRACE_NOTE
    data: bytes = b""
    context: str = ""   # driver call in flight, e.g. "write_mem[0x1000+8]"
    note: str = ""      # decode ("ACK") or anomaly ("short read: ...")


class UartTrace:
    """A log of every byte crossing the link.

    Args:
        sink: where to emit events as they happen — a text file object, a
            ``callable(line)``, or ``None`` to only accumulate them in
            :attr:`events` (dump later with :meth:`dump`).
        width: bytes per hexdump row.
        max_events: keep only the most recent N events (``None`` = keep all).
            Byte counters stay exact regardless.
        enabled: recording can be flipped at runtime via this attribute, so a
            long session can trace just the interesting command.
    """

    def __init__(
        self,
        sink: Any = sys.stderr,
        width: int = 16,
        max_events: int | None = None,
        enabled: bool = True,
    ):
        self.sink = sink
        self.width = width
        self.enabled = enabled
        self.events: deque[TraceEvent] = deque(maxlen=max_events)
        self.n_tx = 0
        self.n_rx = 0
        self.n_dropped = 0
        self._t0 = time.monotonic()
        self._ctx: list[str] = []

    # -- recording ------------------------------------------------------------

    def record(self, kind: str, data: bytes = b"", note: str = "") -> None:
        if not self.enabled:
            return
        data = bytes(data)
        ev = TraceEvent(
            t=time.monotonic() - self._t0,
            kind=kind,
            data=data,
            context=self._ctx[-1] if self._ctx else "",
            note=note,
        )
        if kind == TRACE_TX:
            self.n_tx += len(data)
        elif kind == TRACE_RX:
            self.n_rx += len(data)
        elif kind == TRACE_DROP:
            self.n_dropped += len(data)
        self.events.append(ev)
        if self.sink is not None:
            self._emit(ev)

    def note(self, text: str) -> None:
        """Record a zero-byte annotation at the current point in the stream."""
        self.record(TRACE_NOTE, b"", text)

    @contextlib.contextmanager
    def context(self, label: str) -> Iterator["UartTrace"]:
        """Tag every event recorded inside the block with ``label``."""
        self._ctx.append(label)
        try:
            yield self
        finally:
            self._ctx.pop()

    def reset(self) -> None:
        """Drop recorded events and restart the clock."""
        self.events.clear()
        self.n_tx = self.n_rx = self.n_dropped = 0
        self._t0 = time.monotonic()

    # -- rendering ------------------------------------------------------------

    def format_event(self, ev: TraceEvent) -> list[str]:
        size = "      " if ev.kind == TRACE_NOTE else f"{len(ev.data):5d}B"
        tail = "  ".join(s for s in (ev.context, ev.note) if s)
        lines = [f"{ev.t:9.4f}  {ev.kind:<4} {size}  {tail}".rstrip()]
        lines += _hexdump(ev.data, self.width, indent=" " * 11)
        return lines

    def summary(self) -> str:
        elapsed = self.events[-1].t if self.events else 0.0
        drop = f", {self.n_dropped} B dropped" if self.n_dropped else ""
        return (
            f"uart trace: {len(self.events)} events, TX {self.n_tx} B, "
            f"RX {self.n_rx} B{drop}, {elapsed:.3f} s"
        )

    def format(self) -> str:
        """The whole trace as text, one block per event plus a summary line."""
        lines: list[str] = []
        for ev in self.events:
            lines += self.format_event(ev)
        lines.append(self.summary())
        return "\n".join(lines)

    def dump(self, sink: Any = None) -> None:
        """Write the whole trace to ``sink`` (default: this trace's own sink,
        else stderr). Useful when constructed with ``sink=None``."""
        _write_lines(sink if sink is not None else (self.sink or sys.stderr),
                     self.format().splitlines())

    def _emit(self, ev: TraceEvent) -> None:
        _write_lines(self.sink, self.format_event(ev))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)


def _hexdump(data: bytes, width: int = 16, indent: str = "") -> list[str]:
    """``offset  hex bytes  |printable|`` rows — every byte, none elided."""
    rows = []
    for off in range(0, len(data), width):
        chunk = data[off : off + width]
        text = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
        rows.append(
            f"{indent}{off:04x}  {chunk.hex(' '):<{width * 3 - 1}}  |{text}|"
        )
    return rows


def _write_lines(sink: Any, lines: Sequence[str]) -> None:
    """Emit to a file object or a ``callable(line)``."""
    if hasattr(sink, "write"):
        sink.write("\n".join(lines) + "\n")
        flush = getattr(sink, "flush", None)
        if flush:
            flush()
    else:
        for line in lines:
            sink(line)


def _as_trace(spec: Any) -> UartTrace:
    """Normalise the ``trace=`` argument into a (possibly disabled) UartTrace.

    ``None``/``False`` -> disabled, ``True`` -> live to stderr, a file object or
    ``callable(line)`` -> live to that, a UartTrace -> itself.
    """
    if isinstance(spec, UartTrace):
        return spec
    if spec is None or spec is False:
        return UartTrace(sink=None, enabled=False)
    if spec is True:
        return UartTrace(sink=sys.stderr)
    return UartTrace(sink=spec)


class _TracedSerial:
    """Transparent proxy around a ``serial.Serial`` that records every byte.

    Everything not overridden here (``in_waiting``, ``close``, assignments to
    ``timeout``, ...) passes straight through to the real port, so this is a
    drop-in for code that reaches into ``tpu.ser``.
    """

    def __init__(self, ser: Any, trace: UartTrace):
        object.__setattr__(self, "_ser", ser)
        object.__setattr__(self, "_trace", trace)

    # -- the traced surface ---------------------------------------------------

    def write(self, data: bytes) -> int:
        n = self._ser.write(data)
        if self._trace.enabled:
            short = "" if n is None or n == len(data) else f"short write: {n}/{len(data)}"
            self._trace.record(TRACE_TX, data, short)
        return n

    def read(self, size: int = 1) -> bytes:
        data = self._ser.read(size)
        self._record_rx(data, size)
        return data

    def read_until(self, *args, **kwargs) -> bytes:
        data = self._ser.read_until(*args, **kwargs)
        self._record_rx(data)
        return data

    def read_all(self) -> bytes:
        data = self._ser.read_all()
        self._record_rx(data)
        return data

    def reset_input_buffer(self) -> None:
        # Consume-then-discard rather than discard, so the trace shows what was
        # thrown away — that is the tell-tale of a previous desync.
        if self._trace.enabled:
            pending = self._ser.in_waiting
            if pending:
                self._trace.record(
                    TRACE_DROP, self._ser.read(pending), "discarded (input buffer reset)"
                )
        self._ser.reset_input_buffer()

    def _record_rx(self, data: bytes, wanted: int | None = None) -> None:
        if not self._trace.enabled:
            return
        note = ""
        if wanted is not None and len(data) != wanted:
            note = f"short read: wanted {wanted}, got {len(data)} (timeout)"
        elif not data:
            note = "nothing received (timeout)"
        self._trace.record(TRACE_RX, data, note)

    # -- pass-through ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ser, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._ser, name, value)

    def __repr__(self) -> str:
        return f"<traced {self._ser!r}>"


# ---- Driver -----------------------------------------------------------------

class TPUUart:
    """A connection to the TPU's UART command interface.

    Usable as a context manager::

        with TPUUart("/dev/ttyUSB0") as tpu:
            tpu.write_mem(0x1000, tensor_bytes)
            tpu.load_program(0, words)
            tpu.go(0)
            result = tpu.read_mem(0x2000, 256)

    Pass ``trace=True`` (or a :class:`UartTrace`) to log every byte in both
    directions; see :attr:`trace` and :meth:`dump_trace`.
    """

    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_BAUD,
        timeout: float = 2.0,
        rx_timeout_s: float = 0.01,
        trace: Any = None,
    ):
        """
        Args:
            port: serial device (``/dev/ttyUSB0``, ``COM3``, ...).
            baud: must match ``CLK_PER_BIT`` on the device.
            timeout: seconds to wait for a reply before :class:`ReplyTimeout`.
                Reads of large blocks need this to cover the wire time, so
                :meth:`read_mem` extends it by the transfer duration.
            rx_timeout_s: the device's ``RX_TIMEOUT`` expressed in seconds —
                how long :meth:`resync` idles the line to make a mid-frame FSM
                abort. Only meaningful if the device was built with a non-zero
                ``RX_TIMEOUT``.
            trace: byte-level logging. ``None``/``False`` off, ``True`` for a
                live hexdump on stderr, a file object or ``callable(line)`` to
                send it elsewhere, or a :class:`UartTrace` you configured
                yourself (e.g. ``UartTrace(sink=None)`` to record quietly and
                :meth:`dump_trace` later).
        """
        try:
            import serial  # noqa: PLC0415  (optional dep, imported lazily)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyserial is required for the TPU UART host: pip install pyserial"
            ) from exc

        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.rx_timeout_s = rx_timeout_s
        self.trace = _as_trace(trace)
        # 8N1 — the device's uart_receiver/uart_transmitter take no other format.
        # Wrapped so `self.ser` records everything, including callers' raw pokes.
        self.ser = _TracedSerial(
            serial.Serial(
                port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout
            ),
            self.trace,
        )
        self.trace.note(f"open {port} @ {baud} 8N1, reply timeout {timeout}s")

    # ---- plumbing -----------------------------------------------------------

    def close(self) -> None:
        self.trace.note(f"close {self.port}")
        self.ser.close()

    def dump_trace(self, sink: Any = None) -> None:
        """Print the recorded byte trace (see :meth:`UartTrace.dump`)."""
        self.trace.dump(sink)

    def __enter__(self) -> "TPUUart":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _byte_time(self, n: int) -> float:
        """Wire time for ``n`` bytes at 10 bits/byte (8N1)."""
        return 10.0 * n / self.baud

    def _send(self, frame: bytes) -> None:
        """Put ``frame`` on the wire and wait until it has actually left.

        **Do not read from the port before this returns.** Reading while the
        USB-serial bridge is still shifting a frame out corrupts the byte in
        flight: the host's IN requests disturb the transmitter's bit timing by
        about half a bit, so the device samples that byte on its bit boundaries
        and decodes ``sent[k]`` or ``sent[k-1]`` for every bit k. In practice
        that is one mangled byte per command, and because the read is issued
        the moment the write returns, it lands in the first few bytes — i.e.
        the header, where a wrong length or address does the most damage. A
        16-bit length field that comes out one bit left-shifted is why the
        device used to "send more bytes than the host asked for".

        Measured on a Cmod A7 (FT2232H, 115200 8N1) with the echo image, which
        reports every received byte: reading immediately corrupted 72/6400
        bytes, all in burst positions 1..6; sleeping past the burst first
        corrupted 0/6400; sleeping over only the first half moved the damage to
        positions 33..46, i.e. exactly where the read began. The FPGA is not
        involved — the same RTL decodes a bit-identical stream perfectly in
        simulation, and the fault reproduces on all three board images.

        ``flush()`` is not enough on its own: on Windows it returns once the
        bytes reach the driver, not once they reach the wire. There is no API
        that reports the UART shift register, so the wire time is computed and
        waited out, with two guards on top of it (:data:`TX_SETTLE_S`,
        :data:`TX_RATE_SLACK`) — see those constants for why a bare wire time
        is not enough.
        """
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(self._byte_time(len(frame)) * TX_RATE_SLACK + TX_SETTLE_S)

    def _recv(self, n: int, timeout: float | None = None) -> bytes:
        """Read exactly ``n`` bytes or raise, leaving whatever arrived in the message."""
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            data = self.ser.read(n)
        finally:
            self.ser.timeout = self.timeout
        if len(data) != n:
            raise ReplyTimeout(
                f"expected {n} byte(s), got {len(data)}: {data.hex(' ') or '<nothing>'}",
                partial=data,
            )
        return data

    def _status(self) -> None:
        """Consume the one-byte reply of a W / I / G command."""
        st = self._recv(1)[0]
        if st == STAT_ACK:
            self.trace.note("ACK")
            return
        if st == STAT_NAK:
            self.trace.note("NAK")
            raise NakError(
                "device sent NAK — the core was busy, or host and device are "
                "out of sync (the frame itself was validated before sending)"
            )
        self.trace.note(f"bogus status byte {st:#04x} (expected ACK/NAK)")

        raise ProtocolError(f"expected ACK/NAK, got {st:#04x}")

    def resync(self) -> None:
        """Recover from a suspected desync: drain the link and idle it.

        Idling longer than the device's ``RX_TIMEOUT`` makes a mid-frame FSM
        abort back to IDLE. If the device was built with ``RX_TIMEOUT = 0``
        (the RTL default) that abort does not exist and a device stuck mid-frame
        can only be recovered by resetting it — this call still drains stale
        reply bytes, which is what a NAK-after-write leaves behind.
        """
        with self.trace.context("resync"):
            self.ser.reset_input_buffer()
            time.sleep(max(self.rx_timeout_s, self._byte_time(4)))
            self.ser.reset_input_buffer()

    # ---- commands -----------------------------------------------------------

    def read_mem(self, addr: int, length: int) -> bytes:
        """``R`` — read ``length`` bytes of external SRAM from ``addr``.

        Transfers longer than the 16-bit length field are split into
        back-to-back commands.
        """
        if length <= 0:
            raise ValueError(f"length {length} must be >= 1")
        out = bytearray()
        for off in range(0, length, MAX_LEN):
            n = min(MAX_LEN, length - off)
            out += self._read_chunk(addr + off, n)
        return bytes(out)

    def _read_chunk(self, addr: int, length: int) -> bytes:
        _check_mem(addr, length)
        with self.trace.context(f"read_mem[{addr:#08x}+{length}]"):
            self.ser.reset_input_buffer()
            self._send(_header(CMD_READ, addr, length))
            # The reply is header-less, so allow for the whole block on the wire.
            try:
                return self._recv(
                    length, timeout=self.timeout + self._byte_time(length)
                )
            except ReplyTimeout as exc:
                # A rejected read answers with a lone NAK and no data. len == 1
                # is inherently ambiguous (a data byte of 0x15 looks identical),
                # but a short reply of exactly that byte is a rejection for
                # every len > 1.
                if length > 1 and exc.partial == bytes([STAT_NAK]):
                    self.trace.note("NAK (read rejected)")
                    raise NakError("device rejected the read command") from exc
                raise

    def write_mem(self, addr: int, data: bytes) -> None:
        """``W`` — write ``data`` to external SRAM at ``addr``, waiting for the ACK."""
        if not data:
            raise ValueError("nothing to write (len == 0 is rejected by the device)")
        for off in range(0, len(data), MAX_LEN):
            chunk = data[off : off + MAX_LEN]
            self._payload_cmd(CMD_WRITE, addr + off, chunk, _check_mem)

    def load_program(self, word_addr: int, words: Sequence[int] | bytes) -> None:
        """``I`` — load instruction words into the scalar unit's IMEM.

        ``words`` is a sequence of 32-bit ints (packed MSB first) or an already
        packed big-endian byte string. ``word_addr`` is a *word* index.
        """
        payload = words if isinstance(words, (bytes, bytearray)) else pack_words(words)
        if not payload:
            raise ValueError("nothing to load (len == 0 is rejected by the device)")
        for off in range(0, len(payload), MAX_IMEM_LEN):
            chunk = bytes(payload[off : off + MAX_IMEM_LEN])
            self._payload_cmd(
                CMD_IMEM, word_addr + off // 4, chunk, _check_imem
            )

    def _payload_cmd(self, cmd: int, addr: int, payload: bytes, check) -> None:
        """Header + data phase + status, for the two commands that carry data."""
        check(addr, len(payload))
        name = "write_mem" if cmd == CMD_WRITE else "load_program"
        with self.trace.context(f"{name}[{addr:#08x}+{len(payload)}]"):
            self.ser.reset_input_buffer()
            # One write() so the trace shows the header and data phase as the
            # single burst the device actually sees.
            self._send(_header(cmd, addr, len(payload)) + payload)
            try:
                self._status()
            except ProtocolError as exc:
                # The device NAKs before the data phase, so it will have decoded
                # our payload as command bytes. Clear the wreckage before
                # propagating.
                self.trace.note(f"{type(exc).__name__}: {exc} — resyncing")
                self.resync()
                raise

    def go(self, pc: int = 0) -> None:
        """``G`` — start the scalar unit at ``pc`` (a 4-byte frame, no length).

        Returns as soon as the device ACKs the launch; the program itself runs
        asynchronously and this link has no way to poll for completion.
        """
        _check_pc(pc)
        with self.trace.context(f"go[pc={pc}]"):
            self.ser.reset_input_buffer()
            self._send(_header(CMD_GO, pc))
            self._status()

    def read_timer(self) -> int:
        """``T`` — the scalar unit's run-length counter, in core clocks.

        Counts the interval the core was ``busy``: reset at the start of each
        run ('G' → the first instruction fetch) and frozen when the program
        halts, so after a run this is exactly how long that run took, and during
        one it is how far it has got. Answered whether or not the core is
        running — the one command that is.

        The frame is the single command byte; the reply is 4 bytes MSB first and
        no status, so unlike every other command there is nothing to validate,
        nothing to reject, and no NAK to disambiguate from data.

        Divide by the device's clock frequency for seconds (12 MHz on the Cmod
        A7). ``0xFFFFFFFF`` is the counter saturating rather than wrapping —
        358 s at 12 MHz, so treat it as "no plausible run is this long".
        """
        with self.trace.context("read_timer"):
            self.ser.reset_input_buffer()
            self._send(bytes([CMD_TIMER]))
            return int.from_bytes(self._recv(TIMER_BYTES), "big")


# ---- Program file parsing ---------------------------------------------------

_HEX_WORD = re.compile(r"^[0-9a-fA-F]{1,8}$")


def parse_hex_program(text: str) -> list[int]:
    """Parse a ``$readmemh``-style program (one 32-bit hex word per line).

    Accepts the ``//`` comments and whitespace used by ``tb/vectors/tpu_prog.hex``.
    """
    words: list[int] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//")[0].strip()
        if not line:
            continue
        for tok in line.split():
            if not _HEX_WORD.match(tok):
                raise ValueError(f"line {lineno}: {tok!r} is not a 32-bit hex word")
            words.append(int(tok, 16))
    return words


# ---- CLI --------------------------------------------------------------------

def _auto_int(s: str) -> int:
    return int(s, 0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tpu_uart",
        description="Host driver for the TPU UART link (read/write SRAM, load IMEM, run).",
    )
    p.add_argument("-p", "--port", required=True, help="serial port, e.g. /dev/ttyUSB0")
    p.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("-t", "--timeout", type=float, default=2.0, help="reply timeout (s)")
    p.add_argument(
        "-T", "--trace", action="store_true",
        help="hexdump every byte sent and received to stderr",
    )
    p.add_argument(
        "--trace-file", metavar="FILE",
        help="write that trace to FILE instead of stderr (implies --trace)",
    )
    p.add_argument(
        "--trace-width", type=int, default=16, metavar="N",
        help="bytes per hexdump row (default 16)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="read SRAM ('R')")
    r.add_argument("addr", type=_auto_int)
    r.add_argument("length", type=_auto_int)
    r.add_argument("-o", "--out", help="write to this file instead of stdout hex")

    w = sub.add_parser("write", help="write SRAM ('W')")
    w.add_argument("addr", type=_auto_int)
    src = w.add_mutually_exclusive_group(required=True)
    src.add_argument("--hex", help="payload as a hex string, e.g. deadbeef")
    src.add_argument("--file", help="payload from a binary file")

    ld = sub.add_parser("load", help="load instruction memory ('I')")
    ld.add_argument("file", help="$readmemh-style program (one hex word per line)")
    ld.add_argument("--addr", type=_auto_int, default=0, help="IMEM word index")
    ld.add_argument("--go", action="store_true", help="run from --addr after loading")

    g = sub.add_parser("go", help="start the scalar unit ('G')")
    g.add_argument("pc", type=_auto_int, nargs="?", default=0)

    tm = sub.add_parser("timer", help="read the run-length counter ('T')")
    tm.add_argument("--clk-mhz", type=float, default=12.0,
                    help="device clock, to print the count in seconds too "
                         "(default: %(default)s, the Cmod A7)")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Trace lives outside the `with` so a failure to even open the port, or an
    # exception mid-command, still leaves the summary printed.
    trace: UartTrace | None = None
    trace_fh = None
    if args.trace or args.trace_file:
        if args.trace_file:
            trace_fh = open(args.trace_file, "w", encoding="utf-8")
        trace = UartTrace(sink=trace_fh or sys.stderr, width=args.trace_width)

    try:
        with TPUUart(args.port, args.baud, args.timeout, trace=trace) as tpu:
            if args.cmd == "read":
                data = tpu.read_mem(args.addr, args.length)
                if args.out:
                    with open(args.out, "wb") as fh:
                        fh.write(data)
                    print(f"read {len(data)} bytes from {args.addr:#x} -> {args.out}")
                else:
                    for off in range(0, len(data), 16):
                        row = data[off : off + 16]
                        print(f"{args.addr + off:06x}  {row.hex(' ')}")

            elif args.cmd == "write":
                if args.hex:
                    payload = bytes.fromhex(args.hex)
                else:
                    with open(args.file, "rb") as fh:
                        payload = fh.read()
                tpu.write_mem(args.addr, payload)
                print(f"wrote {len(payload)} bytes to {args.addr:#x} (ACK)")

            elif args.cmd == "load":
                with open(args.file) as fh:
                    words = parse_hex_program(fh.read())
                tpu.load_program(args.addr, words)
                print(f"loaded {len(words)} words at IMEM[{args.addr}] (ACK)")
                if args.go:
                    tpu.go(args.addr)
                    print(f"started at pc={args.addr} (ACK)")

            elif args.cmd == "go":
                tpu.go(args.pc)
                print(f"started at pc={args.pc} (ACK)")

            elif args.cmd == "timer":
                cycles = tpu.read_timer()
                us = cycles / args.clk_mhz
                print(f"{cycles} clocks ({us:.1f} us @ {args.clk_mhz:g} MHz)"
                      + ("  [SATURATED]" if cycles == 0xFFFFFFFF else ""))

    except (ProtocolError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if trace is not None:
            print(trace.summary(), file=sys.stderr)
            if trace_fh is not None:
                trace_fh.write(trace.summary() + "\n")
                trace_fh.close()
                print(f"trace written to {args.trace_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
