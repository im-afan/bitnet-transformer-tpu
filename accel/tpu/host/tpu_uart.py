"""Host-side driver for the TPU's UART command link (``rtl/uart_interface.sv``).

The FPGA is a pure slave: the host sends a fixed-header command frame and the
device answers. Four commands, all with a 3-byte big-endian address field:

===== ====== ============= ========================== =====================
CMD   byte   frame                                    reply
===== ====== ============= ========================== =====================
``R`` 0x52   read  SRAM    ``CMD A2 A1 A0 L1 L0``      ``len`` data bytes
``W`` 0x57   write SRAM    ``CMD A2 A1 A0 L1 L0`` +    ACK / NAK
                           ``data[len]``
``I`` 0x49   write IMEM    ``CMD A2 A1 A0 L1 L0`` +    ACK / NAK
                           ``data[len]``
``G`` 0x47   go / run      ``CMD A2 A1 A0``            ACK / NAK
===== ====== ============= ========================== =====================

* ``R``/``W`` addresses are 19-bit external-SRAM byte addresses; ``len`` is a
  byte count in ``1..65535``.
* ``I`` addresses are *instruction word indices* (10-bit); ``len`` is still in
  **bytes** and must be a multiple of 4. Each word is sent MSB first.
* ``G`` carries no length and no payload — it pulses the scalar unit's run
  trigger with ``run_pc = addr``.
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
idle. There is no status-read command, so the host cannot poll ``busy``/``done``
over this link — after :meth:`TPUUart.go` you wait out-of-band (see README).

Requires ``pyserial``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Iterable, Sequence

# ---- Protocol constants (mirror rtl/uart_interface.sv) -----------------------

CMD_READ = 0x52   # 'R'
CMD_WRITE = 0x57  # 'W'
CMD_IMEM = 0x49   # 'I'
CMD_GO = 0x47     # 'G'

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


# ---- Driver -----------------------------------------------------------------

class TPUUart:
    """A connection to the TPU's UART command interface.

    Usable as a context manager::

        with TPUUart("/dev/ttyUSB0") as tpu:
            tpu.write_mem(0x1000, tensor_bytes)
            tpu.load_program(0, words)
            tpu.go(0)
            result = tpu.read_mem(0x2000, 256)
    """

    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_BAUD,
        timeout: float = 2.0,
        rx_timeout_s: float = 0.01,
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
        # 8N1 — the device's uart_receiver/uart_transmitter take no other format.
        self.ser = serial.Serial(
            port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout
        )

    # ---- plumbing -----------------------------------------------------------

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> "TPUUart":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _byte_time(self, n: int) -> float:
        """Wire time for ``n`` bytes at 10 bits/byte (8N1)."""
        return 10.0 * n / self.baud

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
            return
        if st == STAT_NAK:
            raise NakError(
                "device sent NAK — the core was busy, or host and device are "
                "out of sync (the frame itself was validated before sending)"
            )
        raise ProtocolError(f"expected ACK/NAK, got {st:#04x}")

    def resync(self) -> None:
        """Recover from a suspected desync: drain the link and idle it.

        Idling longer than the device's ``RX_TIMEOUT`` makes a mid-frame FSM
        abort back to IDLE. If the device was built with ``RX_TIMEOUT = 0``
        (the RTL default) that abort does not exist and a device stuck mid-frame
        can only be recovered by resetting it — this call still drains stale
        reply bytes, which is what a NAK-after-write leaves behind.
        """
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
        self.ser.reset_input_buffer()
        self.ser.write(_header(CMD_READ, addr, length))
        self.ser.flush()
        # The reply is header-less, so allow for the whole block on the wire.
        try:
            return self._recv(length, timeout=self.timeout + self._byte_time(length))
        except ReplyTimeout as exc:
            # A rejected read answers with a lone NAK and no data. len == 1 is
            # inherently ambiguous (a data byte of 0x15 looks identical), but a
            # short reply of exactly that byte is a rejection for every len > 1.
            if length > 1 and exc.partial == bytes([STAT_NAK]):
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
        self.ser.reset_input_buffer()
        self.ser.write(_header(cmd, addr, len(payload)) + payload)
        self.ser.flush()
        try:
            self._status()
        except ProtocolError:
            # The device NAKs before the data phase, so it will have decoded our
            # payload as command bytes. Clear the wreckage before propagating.
            self.resync()
            raise

    def go(self, pc: int = 0) -> None:
        """``G`` — start the scalar unit at ``pc`` (a 4-byte frame, no length).

        Returns as soon as the device ACKs the launch; the program itself runs
        asynchronously and this link has no way to poll for completion.
        """
        _check_pc(pc)
        self.ser.reset_input_buffer()
        self.ser.write(_header(CMD_GO, pc))
        self.ser.flush()
        self._status()


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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        with TPUUart(args.port, args.baud, args.timeout) as tpu:
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

    except (ProtocolError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
