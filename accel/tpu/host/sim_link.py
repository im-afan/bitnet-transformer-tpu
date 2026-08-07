"""sim_link.py — talk to a *simulated* board instead of a real one.

Presents an Icarus Verilog simulation of ``rtl/uart_memory.sv`` (the cmod_a7_mem
bitstream's core: uart_receiver -> uart_interface -> sram_controller -> a
behavioral SRAM chip) as something ``tpu_uart.TPUUart`` cannot tell from a
``serial.Serial``. So ``test_uart_link.py --sim`` runs its real frame encoder,
its real reply parser and its real reject cases against real RTL, with no board::

    python accel/tpu/host/test_uart_link.py --sim --only sram_roundtrip

``tb/uart_memory_cosim_tb.sv`` is the other half; its header documents the
line-oriented file protocol the two halves speak. Only two things about it
matter here.

**There is no wall clock on the other side.** The simulator advances at whatever
rate the host CPU manages — a couple of milliseconds of simulated time per real
second — so a reply timeout of "5 seconds" is meaningless: waiting that long in
simulated time takes an hour, and waiting that long in real time never fires
because the device is merely slow, not silent. Every timeout here is therefore
measured in **bit times of simulated line silence** (:data:`SLACK_BITS`), which
is what the host is really asking about. A separate real-time guard
(:data:`WALL_STALL_S`) exists only to notice that the simulator has died.

**The host can outrun the wire.** ``write()`` returns as soon as the bytes are
appended to the file; the simulator then clocks them in at 10 bit times each. A
naive timeout would fire while the device was still being handed a 4 KiB write
it had not finished hearing. The testbench therefore reports how many host bytes
it has fully clocked in, and a reply is only overdue once that counter has
stopped moving too — see :meth:`SimSerial._wait`.
"""

from __future__ import annotations

import collections
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TPU_DIR = os.path.dirname(HERE)
TB_DIR = os.path.join(TPU_DIR, "tb")
RTL_DIR = os.path.join(TPU_DIR, "rtl")

TB_TOP = "uart_memory_cosim_tb"

# Exactly the sources boards/cmod_a7_mem builds, plus the testbench.
RTL_SOURCES = [
    "uart_memory.sv",
    "uart_interface.sv",
    "uart_receiver.sv",
    "uart_transmitter.sv",
    "sram.sv",
]

# How much simulated line silence counts as "the device has stopped talking",
# in bit times. The device's real gaps are a handful of clocks (an SRAM access,
# a state transition), so ~40 byte times is enormously generous while still
# being reachable: 400 bits is 416 us of simulated time, a fraction of a second
# of real time.
SLACK_BITS = 400

# Simulated silence to insert before answering `in_waiting`, so that a byte the
# device sends just after its reply — the hardware symptom this whole exercise
# is about — has had time to appear rather than being missed by a host that
# asked too early.
QUIET_BITS = 40

# Real seconds with no output at all from the simulator before we conclude it
# has died or wedged. Not a protocol timeout: this one means the *simulation* is
# broken, not the link.
WALL_STALL_S = 120.0


class SimLinkError(RuntimeError):
    """The simulator itself failed — as opposed to the link misbehaving."""


def build(
    build_dir: str,
    quiet: bool = False,
    params: dict[str, int] | None = None,
    sources: list[str] | None = None,
) -> str:
    """Compile the co-simulation testbench; return the path to the vvp image.

    Args:
        params: testbench parameter overrides, e.g. ``{"RX_TIMEOUT": 2080}``.
        sources: absolute paths replacing :data:`RTL_SOURCES`. The point of this
            is fault injection — pointing the build at a deliberately broken
            copy of ``uart_interface.sv`` is how you check that this harness can
            still fail, which a suite that has only ever printed PASS has not
            demonstrated.
    """
    if shutil.which("iverilog") is None:
        raise SimLinkError("iverilog not found on PATH — install Icarus Verilog")
    os.makedirs(build_dir, exist_ok=True)
    out = os.path.join(build_dir, TB_TOP + ".vvp")
    srcs = sources or [os.path.join(RTL_DIR, s) for s in RTL_SOURCES]
    srcs = list(srcs) + [os.path.join(TB_DIR, TB_TOP + ".sv")]
    cmd = ["iverilog", "-g2012", "-o", out]
    for k, v in (params or {}).items():
        cmd += [f"-P{TB_TOP}.{k}={v}"]
    cmd += srcs
    if not quiet:
        print(f"build   : {' '.join(os.path.basename(s) for s in srcs)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SimLinkError(f"iverilog failed:\n{proc.stdout}{proc.stderr}")
    return out


class SimSerial:
    """A ``serial.Serial`` work-alike whose far end is a running simulation.

    Implements the surface ``tpu_uart.py`` and ``test_uart_link.py`` actually
    touch: :meth:`read`, :meth:`write`, :meth:`flush`, :meth:`reset_input_buffer`,
    :attr:`in_waiting`, :attr:`timeout` and :meth:`close`.
    """

    def __init__(
        self,
        vvp: str,
        work_dir: str,
        slack_bits: int = SLACK_BITS,
        fsm_trace: bool = False,
        vcd: str | None = None,
        verbose: bool = False,
    ):
        if shutil.which("vvp") is None:
            raise SimLinkError("vvp not found on PATH — install Icarus Verilog")
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.slack_bits = slack_bits
        self.verbose = verbose
        self.timeout: float | None = None      # accepted and ignored; see module doc

        self._h2d_path = os.path.join(work_dir, "cosim_h2d.txt")
        self._d2h_path = os.path.join(work_dir, "cosim_d2h.txt")

        # Both files must exist before vvp starts: the testbench opens the h2d
        # stream for reading in its first delta cycle and gives up if it is not
        # there.
        open(self._h2d_path, "wb").close()
        open(self._d2h_path, "wb").close()

        argv = [
            "vvp", vvp,
            f"+h2d={self._h2d_path}",
            f"+d2h={self._d2h_path}",
        ]
        if fsm_trace:
            argv.append("+fsm")
        if vcd:
            argv.append(f"+vcd={vcd}")

        self._proc = subprocess.Popen(
            argv, cwd=work_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        self._h2d = open(self._h2d_path, "ab", buffering=0)
        self._d2h = open(self._d2h_path, "rb")
        self._tail = b""

        # ---- link state, all fed by _pump() ---------------------------------
        self.bit_ns = 0             # simulated ns per UART bit
        self.sim_ns = 0             # simulator's current time
        self.consumed = 0           # host bytes the device has fully clocked in
        self.written = 0            # host bytes handed to the bridge
        self.dropped = 0            # bytes discarded by reset_input_buffer()
        self.finished = False
        self.events: list[tuple[str, int]] = []      # rx_overrun / collision
        self.fsm: collections.deque = collections.deque(maxlen=400)
        self._rx = bytearray()
        self._bad_stop = 0
        self._last_wall = time.monotonic()

        self._await_preamble()

    # ---- lifecycle ----------------------------------------------------------

    def _await_preamble(self) -> None:
        """Block until the testbench has released reset and told us its timing."""
        deadline = time.monotonic() + 30.0
        while self.bit_ns == 0 or self.sim_ns == 0:
            self._pump()
            if self.finished or self._proc.poll() is not None:
                raise SimLinkError(f"simulator exited during start-up:\n{self._drain_stdout()}")
            if time.monotonic() > deadline:
                raise SimLinkError("simulator never came up (no preamble in 30 s)")
            time.sleep(0.002)
        if self.verbose:
            print(f"sim     : link up at {self.sim_ns} ns, {self.bit_ns} ns/bit")

    def _drain_stdout(self) -> str:
        try:
            return self._proc.communicate(timeout=5)[0] or ""
        except Exception:                                   # noqa: BLE001
            return ""

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._h2d.write(b"100\n")                   # shut down cleanly
                self._h2d.flush()
                self._proc.wait(timeout=15)
            except Exception:                               # noqa: BLE001
                self._proc.kill()
        for fh in (self._h2d, self._d2h):
            try:
                fh.close()
            except Exception:                               # noqa: BLE001
                pass

    # ---- the bridge ---------------------------------------------------------

    def _pump(self) -> bool:
        """Absorb whatever the simulator has written. True if anything arrived."""
        chunk = self._d2h.read()
        if not chunk:
            if time.monotonic() - self._last_wall > WALL_STALL_S:
                alive = self._proc.poll() is None
                raise SimLinkError(
                    f"simulator produced nothing for {WALL_STALL_S:.0f} s "
                    f"({'still running' if alive else 'exited'})"
                )
            return False
        self._last_wall = time.monotonic()
        self._tail += chunk
        lines = self._tail.split(b"\n")
        self._tail = lines.pop()                            # partial last line
        for raw in lines:
            self._handle(raw.decode("ascii", "replace").strip())
        return True

    def _handle(self, line: str) -> None:
        if not line:
            return
        f = line.split()
        tag = f[0]
        if tag == "B":                                      # byte off uart_tx
            self._rx.append(int(f[1], 16))
            self.sim_ns = int(f[2])
            if f[3] == "0":
                self._bad_stop += 1
        elif tag == "T":                                    # heartbeat
            self.sim_ns, self.consumed = int(f[1]), int(f[2])
        elif tag == "P":                                    # preamble
            self.bit_ns = int(f[3])
        elif tag == "R":
            self.sim_ns = int(f[1])
        elif tag == "E":                                    # sticky flag went high
            self.events.append((f[1], int(f[2])))
            if self.verbose:
                print(f"sim     : {f[1]} asserted at {int(f[2])} ns")
        elif tag == "F":                                    # FSM step (+fsm)
            self.fsm.append((int(f[1]), int(f[2]), int(f[3]), int(f[4], 16)))
        elif tag == "X":
            self.sim_ns, self.consumed = int(f[1]), int(f[2])
            self.finished = True

    def _advance(self, bits: int) -> None:
        """Let the simulation run on by ``bits`` bit times of simulated time."""
        target = self.sim_ns + bits * self.bit_ns
        while self.sim_ns < target and not self.finished:
            if not self._pump():
                time.sleep(0.001)

    def _wait(self, n: int) -> None:
        """Wait until ``n`` bytes are buffered, or the device falls silent.

        Silence is judged from the last thing that moved — a byte arriving *or*
        the device finishing another host byte. The second half is what stops a
        4 KiB write from timing out while it is still being clocked in: the host
        is thousands of bit times ahead of the wire at that point, and none of
        that time is the device being unresponsive.
        """
        limit = self.slack_bits * self.bit_ns
        progress = self.sim_ns
        seen_rx, seen_consumed = len(self._rx), self.consumed

        while len(self._rx) < n:
            if not self._pump():
                time.sleep(0.001)
            if len(self._rx) != seen_rx or self.consumed != seen_consumed:
                seen_rx, seen_consumed = len(self._rx), self.consumed
                progress = self.sim_ns
            if self.finished:
                return
            if self.sim_ns - progress > limit:
                return

    # ---- the serial.Serial surface ------------------------------------------

    def write(self, data: bytes) -> int:
        if self.finished:
            raise SimLinkError("simulation has finished — the link is closed")
        self._h2d.write(b"".join(b"%03x\n" % b for b in data))
        self._h2d.flush()
        self.written += len(data)
        return len(data)

    def flush(self) -> None:
        pass                                                # write() is unbuffered

    def read(self, size: int = 1) -> bytes:
        self._wait(size)
        n = min(size, len(self._rx))
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def read_all(self) -> bytes:
        self._pump()
        out = bytes(self._rx)
        self._rx.clear()
        return out

    def reset_input_buffer(self) -> None:
        self._pump()
        self.dropped += len(self._rx)
        self._rx.clear()

    @property
    def in_waiting(self) -> int:
        # Give the device a few byte times to say something before reporting
        # silence, or "no extra bytes after the reply" would be true simply
        # because we asked before the extra byte could exist.
        self._advance(QUIET_BITS)
        return len(self._rx)

    # ---- reporting ----------------------------------------------------------

    def summary(self) -> str:
        ms = self.sim_ns / 1e6
        flags = ", ".join(f"{w}@{t} ns" for w, t in self.events) or "none"
        extra = f", {self._bad_stop} framing errors" if self._bad_stop else ""
        return (
            f"sim: {ms:.2f} ms simulated, {self.written} B to device "
            f"({self.consumed} clocked in), {self.dropped} B discarded"
            f"{extra}; sticky flags: {flags}"
        )

    STATES = [
        "IDLE", "RX_ADDR", "RX_LEN", "VALIDATE", "RD_ISSUE", "RD_WAIT", "RD_TX",
        "RD_TX_WAIT", "WR_RX", "WR_ISSUE", "WR_WAIT", "IMEM_RX", "IMEM_WR",
        "SEND_STATUS", "SEND_STATUS_WAIT",
    ]

    def dump_fsm(self, sink=sys.stdout, limit: int = 60) -> None:
        """Print the tail of the uart_interface state trace (needs ``+fsm``)."""
        if not self.fsm:
            print("       (no FSM trace — re-run with --sim-fsm)", file=sink)
            return
        print(f"       last {min(limit, len(self.fsm))} uart_interface states:", file=sink)
        for t, st, pend, hold in list(self.fsm)[-limit:]:
            name = self.STATES[st] if st < len(self.STATES) else f"?{st}"
            print(f"       {t:>12} ns  {name:<17} rx_pending={pend} rx_hold={hold:#04x}",
                  file=sink)


class SimTPUUart:
    """:class:`tpu_uart.TPUUart` with :class:`SimSerial` underneath.

    Built by hand rather than subclassed because ``TPUUart.__init__`` opens a
    real port; everything after that is reused unchanged, which is the point of
    the exercise — the driver under test must be the same object the board sees.
    """

    def __new__(cls, sim: SimSerial, timeout: float = 5.0, trace=None):
        from tpu_uart import TPUUart, _TracedSerial, _as_trace     # noqa: PLC0415

        tpu = TPUUart.__new__(TPUUart)
        tpu.port = "sim"
        tpu.baud = 115200
        tpu.timeout = timeout
        tpu.rx_timeout_s = 0.0
        tpu.trace = _as_trace(trace)
        tpu.ser = _TracedSerial(sim, tpu.trace)
        tpu.sim = sim

        # resync() sleeps in *real* seconds to idle the line past the device's
        # RX_TIMEOUT. Real sleep buys no simulated time at all, so idle the
        # simulated line instead — same intent, the only unit that exists here.
        def resync() -> None:
            with tpu.trace.context("resync"):
                tpu.ser.reset_input_buffer()
                sim._advance(4 * 10 + SLACK_BITS)
                tpu.ser.reset_input_buffer()

        tpu.resync = resync
        return tpu
