# UART echo self-test

A standalone FPGA image that does one thing: it receives **64 bytes** into a
register file, sends those 64 bytes back, and repeats, from reset until
power-off. No commands, no addresses, no modes — the block length is the whole
protocol, fixed at synthesis (`BLOCK_LEN` in `boards/cmod_a7_echo/board.tcl`).
The host sends a 64-byte block of random data, reads the reply, and compares.

The point is to shrink the search space behind the intermittent corruption on the
real link. A bad byte in `host/test_uart_link.py` could come from
`uart_receiver`, `uart_transmitter`, `uart_interface`, the SRAM controller, the
arbitration mux, the cable or the host. This image deletes four of those.

It instantiates the **same** `uart_receiver` and `uart_transmitter` the production
image uses, unmodified. An instrument that alters the thing it measures is
worthless, so neither file was touched.

**Store-and-forward, not streaming** — and that changes what is measured, so it
is worth being explicit about the trade:

* The link is now **half duplex by construction**. The device never transmits
  while receiving, so TX→RX crosstalk and receiver behaviour under simultaneous
  transmit are no longer exercised. The earlier streaming echo covered those.
* In exchange it reproduces the **turnaround** the production protocol has: a
  burst in, a gap, a burst out, then the host's next byte arriving right behind
  the reply. That gap is exactly where `uart_interface`'s blind `SEND_STATUS`
  window sat, so this is the shape of the real traffic.
* A byte arriving while the reply is going out has nowhere to go. It is dropped
  and latched on `overrun` → `led[0]`, so the host outrunning the turnaround is a
  visible event rather than a silent corruption.
* The device now has **state** the streaming echo did not: a partial block. The
  host walks it to a known position before starting (`resync` in
  `host/uart_echo.py`), or an interrupted previous run leaves every exchange
  short by the same few bytes and the byte diff blames the link.

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
fast blink if a byte ever arrived while the device was sending a block back
(sticky, cleared by `btn[0]`). `led[1]` lights for ~44 ms per received byte —
solid while a block streams in, dark during the reply.

**`--block` must equal `BLOCK_LEN`.** Nothing on the wire negotiates it. A host
block smaller than the device's leaves the device waiting forever for the rest;
larger, and the tail is dropped as an overrun. Either way `resync` reports it at
startup rather than letting it surface as a byte diff.

---

## Files

**New**

| Path | What |
|---|---|
| `rtl/uart_echo.sv` | The core: `uart_receiver` → 64-byte register file → `uart_transmitter`, sequenced RECV/SEND, plus the status LEDs |
| `synth/vivado/boards/cmod_a7_echo/cmod_a7_echo_top.sv` | Pin wrapper. Clock and reset copied verbatim from `cmod_a7_top` |
| `synth/vivado/boards/cmod_a7_echo/board.tcl` | New board target — `build.tcl` already dispatches on `board=`, so it needed no changes |
| `constraints/cmod_a7_echo.xdc` | Six pins. `cmod_a7.xdc` is not reusable: `set_property` against an empty `get_ports` is an error, and this top declares none of the 31 SRAM ports |
| `tb/uart_echo_tb.sv` | Testbench: drives a block into `uart_rx` and decodes the reply off `uart_tx` |
| `host/uart_echo.py` | Host driver, soak loop and failure analysis |

**Modified:** `synth/vivado/sources.tcl` (reads `uart_echo.sv`), `tb/Makefile`
(`make echo`), `host/README.md`.

`rtl/uart_receiver.sv` and `rtl/uart_transmitter.sv` are **not touched**.

---

## Design notes

**Why no FIFO any more.** A streaming echo needs one: the transmitter can only
start once a byte has fully arrived, so it permanently lags by about a byte and
never quite catches up, and occupancy creeps up over a long unbroken stream. A
block echo has no such race — the two phases do not overlap, so one counter and a
flat `logic [7:0] block_mem [0:63]` are the entire buffer, and the only way to
lose a byte is to send one during the reply, which is what `overrun` reports.

**The sequencer is two states and one counter.** `RECV` fills `0..BLOCK_LEN-1`
and hands over to `SEND`, which drains the same indices and hands back. `SEND`
returns to `RECV` when the *last* byte is handed to the transmitter, not when it
has finished leaving the pin — so the final frame is still on the wire for ~10 bit
periods afterwards. Deliberate: a host that waits for the whole reply is
unaffected, and one that starts early gets its byte accepted rather than counted
as an overrun.

**Why the pop logic looks the way it does.** `if (!tx_busy) { strobe start; hand
over the byte }` is the same shape as `uart_interface`'s `RD_TX` state, so the
transmitter is driven exactly as it is in the production design. Likewise the
capture edge-detects `rx_valid` (a level, held from `STOP` until half way through
the next start bit) precisely as `uart_interface.sv:116` derives `rx_byte` — this
core should see what the real consumer sees, including any byte the receiver
merges or invents.

**The testbench runs at `CLK_PER_BIT = 104`**, matching the hardware, not the 868
the other UART testbenches use for a 100 MHz core. Sampling margin is what is
under test and it is proportionally tighter at 104. It checks the four things the
block protocol adds on top of byte fidelity: that nothing comes back before the
block is complete, that a second block immediately after the first is not off by
a byte (a sequencer returning to `RECV` with a stale count would shift the payload
rather than corrupt it), that a byte sent during the reply is dropped and flagged,
and that the block after that one still lines up. Frame counts are taken from
inside the DUT (`dut.rx_byte`, `dut.tx_start`) rather than inferred from the pins —
counting frames off the wire would mean telling a start bit from a data bit that
happens to be low, which is the very thing under test.

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

## Sibling image: UART + block RAM (`cmod_a7_bram`)

The echo image removes the protocol along with the memory, so a clean echo run
says nothing about `uart_interface`. `boards/cmod_a7_bram` (`rtl/uart_bram.sv`)
is the rung that removes only the memory:

| Image | Core | Protocol | Memory |
|---|---|---|---|
| `cmod_a7` | yes | yes | external SRAM |
| `cmod_a7_mem` | no | yes | external SRAM |
| `cmod_a7_bram` | no | yes | on-chip block RAM |
| `cmod_a7_echo` | no | no | 64-byte register file |

`cmod_a7_mem` narrowed the fault to `uart_interface` or `sram_controller`. This
image is the cut between them: `uart_interface` is instantiated unmodified, the
wire protocol is byte-for-byte identical (same commands, 3-byte addresses, 19-bit
range checks, same turnaround), and `bram_controller` (`rtl/bram.sv`) reproduces
`sram_controller`'s handshake **cycle for cycle**, beat length and
`CLOCKS_PER_ACCESS` included —
a faster controller would move every turnaround and the comparison would be
worthless. What leaves with the SRAM is the bidirectional bus, the OE#/CE#
timing, the chip, and the 30 switching bank-14 pins.

*Still corrupts* ⇒ the fault is in `uart_interface` (or the host).
*Runs clean* ⇒ the fault needs the external memory present.

```bash
cd accel/tpu/tb && make bram                       # simulate
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_bram mode=deploy
python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
```

**Only 2\*\*`BRAM_AW` bytes exist** (64 KiB by default; 2\*\*19 will not fit in an
Artix-7 35T). The protocol space stays the full 19 bits because the range checks
are part of what is under test, so addresses above the window fold down onto it —
deterministically, and flagged on the sticky `aliased` output. Consequences for
`test_uart_link.py`, which otherwise runs unchanged:

* `sram_isolation` and `sram_address_bus` **fail, expectedly**. Both probe the top
  of the 19-bit space to check the address lines of a chip this image does not
  have.
* `sram_long_transfer` (`--slow`) writes 69631 bytes from the base — build with
  `bram_aw=17` (128 KiB, 32 of the 35T's 50 BRAM tiles) or skip it.
* Memory comes up **zeroed** by configuration rather than holding what the SRAM
  was last left with.

`led[0]` is the heartbeat / collision flag as elsewhere. `led[1]` is the activity
LED, plus a fast blink when the link is idle if any access ever aliased.

---

## What this will not tell you

An echo cannot separate "the receiver read the byte wrong" from "the transmitter
sent it back wrong". A mismatch implicates both blocks.

Nor, in this form, anything about **simultaneous** transmit and receive: the
device is strictly half duplex now, so a clean run no longer clears TX→RX
crosstalk or the receiver's behaviour while the transmitter is driving. If that
is the hypothesis under test, the streaming echo in this file's history is the
image to build.

If a long soak **reproduces** the failure, that is still a large narrowing and the
next step is obvious. If it stays **clean**, that is also useful: it clears the
receiver and the transmitter, and the hunt moves up to `uart_interface.sv` — a
much better place to be than where this started.

Adding direction isolation (the FPGA generating a known stream the host verifies,
and vice versa) is the next step if the result comes back ambiguous. It is
deliberately not built yet.
