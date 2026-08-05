# UART echo self-test

A standalone FPGA image that does one thing: every byte arriving on `uart_rx` goes
straight back out on `uart_tx`. No commands, no protocol, no modes — it echoes
from reset until power-off. The host streams random bytes at it, reads
concurrently, and compares.

The point is to shrink the search space behind the intermittent corruption on the
real link. A bad byte in `host/test_uart_link.py` could come from
`uart_receiver`, `uart_transmitter`, `uart_interface`, the SRAM controller, the
arbitration mux, the cable or the host. This image deletes four of those.

It instantiates the **same** `uart_receiver` and `uart_transmitter` the production
image uses, unmodified. An instrument that alters the thing it measures is
worthless, so neither file was touched.

---

## Running it

```bash
# 1. simulate (no hardware)
cd accel/tpu/tb && make echo

# 2. build (Vivado). Output: synth/build/cmod_a7_echo/cmod_a7_echo_top.bit
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_echo mode=rtl
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_echo mode=bit
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_echo mode=program

# 3. drive it
python accel/tpu/host/uart_echo.py --offline               # check the forensics first
python accel/tpu/host/uart_echo.py -p COM5                 # 30-second run
python accel/tpu/host/uart_echo.py -p COM5 --minutes 30    # soak until it breaks
python accel/tpu/host/uart_echo.py -p COM5 --baud 117000   # sampling-margin check
```

`mode=bit` writes to `synth/build/cmod_a7_echo/`, so the self-test and production
bitstreams can never be confused. **Reflash `board=cmod_a7` before running
`test_uart_link.py` or `run_program.py`** — they will time out against this image.

**LEDs.** `led[0]` is a ~1.4 Hz heartbeat, so a dead clock or an unconfigured FPGA
is distinguishable from a dead link without opening a terminal; it switches to a
fast blink if the echo FIFO ever overflowed (sticky, cleared by `btn[0]`).
`led[1]` lights for ~44 ms per received byte — solid while a block streams.

---

## Files

**New**

| Path | What |
|---|---|
| `rtl/uart_echo.sv` | The core: `uart_receiver` → 256-deep FIFO → `uart_transmitter`, plus the status LEDs |
| `synth/vivado/boards/cmod_a7_echo/cmod_a7_echo_top.sv` | Pin wrapper. Clock and reset copied verbatim from `cmod_a7_top` |
| `synth/vivado/boards/cmod_a7_echo/board.tcl` | New board target — `build.tcl` already dispatches on `board=`, so it needed no changes |
| `constraints/cmod_a7_echo.xdc` | Six pins. `cmod_a7.xdc` is not reusable: `set_property` against an empty `get_ports` is an error, and this top declares none of the 31 SRAM ports |
| `tb/uart_echo_tb.sv` | Full-duplex testbench: drives `uart_rx` while decoding `uart_tx` |
| `host/uart_echo.py` | Host driver, soak loop and failure analysis |

**Modified:** `synth/vivado/sources.tcl` (reads `uart_echo.sv`), `tb/Makefile`
(`make echo`), `host/README.md`.

`rtl/uart_receiver.sv` and `rtl/uart_transmitter.sv` are **not touched**.

---

## Design notes

**Why a FIFO rather than a holding register.** The transmitter can only start once
a byte has fully arrived, so at the same baud it permanently lags by about one
byte — and it never quite catches up. A transmitted byte costs `10*CLK_PER_BIT`
clocks on the wire plus 2 for the start handshake, against a received byte
arriving every 10 host bit periods. At 12 MHz the FPGA's bit period is 0.16% short
of the host's, which claws back most of it, but the net is still ~0.03 µs of lag
per byte, so occupancy creeps up over a long unbroken stream.

256 entries is about 1.3 bytes of that headroom per 4096-byte block, and the host
waits for each block to return before sending the next, which resets the drift.
So `overflow` should never set; if it does, that is a finding, and the board says
so on `led[0]` with no host involved.

**Why the pop logic looks the way it does.** `if (!tx_busy) { strobe start; hand
over the byte }` is the same shape as `uart_interface`'s `RD_TX` state, so the
transmitter is driven exactly as it is in the production design. Likewise the
push edge-detects `rx_valid` (a level, held from `STOP` until half way through the
next start bit) precisely as `uart_interface.sv:116` derives `rx_byte` — this core
should see what the real consumer sees, including any byte the receiver merges or
invents.

**The testbench runs at `CLK_PER_BIT = 104`**, matching the hardware, not the 868
the other UART testbenches use for a 100 MHz core. Sampling margin is what is
under test and it is proportionally tighter at 104. The headline case is
`back-to-back`: 256 bytes with zero inter-byte gap, decoded concurrently. Frame
counts are taken from inside the DUT (`dut.rx_byte`, `dut.tx_start`) rather than
inferred from the pins — counting frames off the wire would mean telling a start
bit from a data bit that happens to be low, which is the very thing under test.

---

## Reading a failure

`host/uart_echo.py` does not just print a diff. Three structurally different bugs
produce an identical-looking byte-by-byte mismatch, so on failure it works out
which one it is:

| Report | Meaning |
|---|---|
| *"byte k is exactly what you get sampling the sent bitstream 1 bit early"* | The frame was **mis-framed** — a glitch in the previous stop bit looked like a start bit, so the eight data bits were sampled one place across. This is the skipped/extra-bit theory, confirmed, with the byte it happened at. The classic instance is `0x40` arriving as `0x80`, which `tb/uart_receiver_tb.sv` already reproduces |
| *"the stream realigns at a +1 byte offset"* | A whole **frame was lost**; everything after it is intact. A start bit was missed, or the byte was overrun |
| *"the stream realigns at a −1 byte offset"* | A frame that was never sent was **invented** — a spurious start bit got through |
| *"no whole-bit or whole-byte offset explains this"* | Framing held and an individual **bit was mis-sampled**. Noise or metastability, not timing |
| *"every byte received was correct — N never arrived"* | Length-only failure; nothing was corrupted |

Plus the usual: first bad index, `got`/`want`/`xor`, the union of every bit ever
wrong across the run (a stuck line is the same bit every time, noise is not), a
context dump either side, and how far into the run it failed — an error *rate* is
a number you can compare across fixes.

A correct block followed by extra bytes also fails the run rather than being
quietly drained: that is the same class of bug as the device over-running a reply
on the real link.

The failure-analysis helpers are themselves covered by `--offline`, which runs
them against known corruptions. That code only executes on the day something
breaks, which is the worst possible moment to find out it was wrong.

### `--baud` is the cheap experiment

8N1 tolerates roughly ±5% of bit-period error. Sweep the host a few percent either
side of 115200 and see where it actually falls over:

* **Asymmetric** (clean at −3%, broken at +2%) ⇒ the receiver's sample point sits
  off-centre. That is arithmetic in `uart_receiver.sv`, and there are two known
  contributors: `START` occupies `CLK_PER_BIT + 1` cycles rather than
  `CLK_PER_BIT`, putting every sample ~2 clocks late in its bit, and `CLK_PER_BIT`
  is 104 against a true 104.1667.
* **Symmetric and wide, but still failing at 0%** ⇒ not timing. Noise or
  metastability — the synchroniser flops in `uart_receiver.sv:16` carry no
  `ASYNC_REG` attribute, so the pair can be split across slices.

---

## What this will not tell you

An echo cannot separate "the receiver read the byte wrong" from "the transmitter
sent it back wrong". A mismatch implicates both blocks.

If a long soak **reproduces** the failure, that is still a large narrowing and the
next step is obvious. If it stays **clean**, that is also useful: it clears the
receiver and the transmitter, and the hunt moves up to `uart_interface.sv` — a
much better place to be than where this started.

Adding direction isolation (the FPGA generating a known stream the host verifies,
and vice versa) is the next step if the result comes back ambiguous. It is
deliberately not built yet.
