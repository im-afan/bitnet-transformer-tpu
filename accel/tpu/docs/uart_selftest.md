# UART self-test image — plan

A standalone FPGA image whose **only** job is to qualify the serial link. No TPU
core, no SRAM, no command FSM, no external memory pins — just
`uart_receiver` → (small FIFO / checker) → `uart_transmitter` plus counters. The
host streams random bytes at it and checks what comes back.

The point is not to reproduce the bug (`host/test_uart_link.py` already does that).
The point is to **localise** it: today a corrupted byte could come from the
receiver, the transmitter, the `uart_interface` FSM, the SRAM controller, the
arbitration mux, the cable, or the host. This image deletes six of those seven so
that a failure has one place left to be.

Status: **plan only.** Nothing below is implemented yet.

---

## 1. What we already know

From the current design and from the forensics already built into
`host/test_uart_link.py` (`report_reply`, which was written to chase exactly this
bug):

* The failure is intermittent, appears only after tens of seconds of traffic, and
  presents as a **length/alignment** problem — the device sends more bytes than
  the host asked for — which is what a one-bit shift in the `L1 L0` length field
  looks like once `uart_interface` acts on it.
* A one-bit shift is consistent with either a **dropped byte** (frame shift) or a
  **mis-sampled bit** inside a byte that did arrive. `report_reply` already
  distinguishes those two by arithmetic on the reply size, but it only ever sees
  the aftermath, several layers downstream.

Things in the current RTL that are *candidates*, none of them yet the confirmed
cause. Listed here because the instrument is designed to discriminate between
them, not because I intend to change them pre-emptively:

| # | Observation | Where |
|---|---|---|
| H1 | `rx_reg1/rx_reg2` have no `ASYNC_REG` attribute, so the synchroniser pair can be split across slices — degraded MTBF, rare single-bit flips | `rtl/uart_receiver.sv:16` |
| H2 | Each data bit is decided by **one** sample at `clk_cnt == CLK_PER_BIT/2`. A single noise spike at that instant flips a bit with nothing to outvote it | `rtl/uart_receiver.sv:102` |
| H3 | The stop bit is never validated. `STOP` returns to `IDLE` on the first high sample, i.e. at the *start* of the stop bit, so the receiver is re-armed and watching for a start bit for the whole stop-bit period | `rtl/uart_receiver.sv:66` |
| H4 | `START` occupies `CLK_PER_BIT + 1` cycles (`clk_cnt` wraps, then the FSM spends one more cycle in `START` before `DATA`), so every sample point sits ~2 clocks late in its bit. Small at CPB=104, but it is a systematic bias, not noise | `rtl/uart_receiver.sv:61-64, 84-90` |
| H5 | `uart_tx` is driven combinationally from `state` and `shift[bit_cnt]`, both of which change on the same edge — the output pin can glitch at every bit boundary | `rtl/uart_transmitter.sv:36-42` |
| H6 | 12 MHz / 115200 = 104.1667 rounded to 104: −0.16% bit period. Negligible alone, but it eats margin that H4 also eats, in the same direction | `boards/cmod_a7/board.tcl:43` |
| H7 | Nothing measures overrun. `uart_interface` edge-detects a *level* `valid` (`rx_byte`), so a byte arriving while the FSM is off in `WR_WAIT`/`RD_TX_WAIT` is silently lost rather than flagged | `rtl/uart_interface.sv:116` |

The self-test image is built to produce a measurement that picks between these.
See §7 for the mapping from observation to hypothesis.

---

## 2. Deliverables

**New files**

| Path | What |
|---|---|
| `rtl/uart_selftest.sv` | Board-independent self-test core: protocol FSM, echo FIFO, LFSR generator/checker, counters |
| `synth/vivado/boards/cmod_a7_selftest/board.tcl` | New *board* target — `build.tcl` needs no changes, it already dispatches on `board=` |
| `synth/vivado/boards/cmod_a7_selftest/cmod_a7_selftest_top.sv` | Pin wrapper: clock, POR, buttons, LEDs, UART. **No SRAM pins** |
| `constraints/cmod_a7_selftest.xdc` | Only the pins this image has. `cmod_a7.xdc` cannot be reused — it constrains 31 SRAM ports that no longer exist |
| `tb/uart_selftest_tb.sv` | Icarus testbench for the core, including deliberate fault injection (§5) |
| `host/uart_selftest.py` | Host driver + test suite. Standalone, `pyserial` only, same conventions as `tpu_uart.py` |

**Modified files**

| Path | Change | Risk |
|---|---|---|
| `rtl/uart_receiver.sv` | Add `frame_error` / `overrun` / `busy` outputs. Add `(* ASYNC_REG *)`. Add a `SAMPLE_VOTES` parameter (1 or 3) for majority-vote sampling, **defaulting to 1 so behaviour is unchanged** | Additive; `tb/uart_receiver_tb.sv` and `tb/tpu_top_uart_tb.sv` must still pass unmodified |
| `rtl/uart_transmitter.sv` | Register `uart_tx` (H5). One flop, uniform one-clock delay | `tb/uart_transmitter_tb.sv` must still pass |
| `synth/vivado/sources.tcl` | Add `uart_selftest.sv` to `RTL_SRCS` | None — Vivado prunes unreferenced modules after `-top` elaboration |
| `tb/Makefile` | `make selftest` convenience target (the existing `RTL=` override already works) | None |
| `host/README.md` | Document the bring-up order: selftest image → `test_uart_link.py` → `run_program.py` | None |

**Untouched:** `tpu_top.sv`, `uart_interface.sv`, `cmod_a7_top.sv`, `cmod_a7.xdc`.
The production image is not disturbed while we measure.

Housekeeping: `rtl/uart_receiver__1.sv` is an untracked stray copy (a different
file entirely, not a variant) — delete it, since `find_module` in `build.tcl`
scans every source for `module <name>` and a second file declaring
`uart_receiver` is a trap waiting to happen.

**Design rule for all of it: the instrument must be simpler than the thing it
measures.** If the self-test image can fail in an ambiguous way, it has failed at
its job. Concretely: the echo path is a single-clock-domain ring buffer with a
write pointer and a read pointer and nothing else; there is no arbitration, no
external memory, no multi-cycle handshake.

---

## 3. Wire protocol

Deliberately not the `R`/`W`/`I`/`G` protocol — a different command set means
flashing the wrong bitstream is instantly obvious instead of subtly wrong.

**Every command is a fixed 4-byte header.** Uniform parsing: one counter, always
four bytes, no per-command header length. Unused fields are ignored.

```
CMD  LEN_HI  LEN_LO  SEED
```

| CMD | ASCII | Name | Host sends after header | Device replies |
|---|---|---|---|---|
| `0x21` | `!` | ECHO | `LEN` arbitrary bytes | those `LEN` bytes, unmodified |
| `0x22` | `"` | SINK | `LEN` bytes of PRBS(`SEED`) | nothing during the phase; STATS block ×3 after |
| `0x24` | `$` | SRC | — | `LEN` bytes of PRBS(`SEED`), back to back |
| `0x28` | `(` | STATS | — | STATS block ×3 |
| `0x30` | `0` | CLEAR | — | STATS block ×3, then counters zeroed |

Any other first byte → one `0x15` (NAK) and back to IDLE, matching
`uart_interface`'s recovery contract.

Command bytes are pairwise ≥2 bits apart and no single-bit error or one-place
shift of one produces another, so a corrupted command is rejected rather than
silently executed as a different one. They are all printable, so a plain terminal
session works for poking at it by hand.

### PRBS

16-bit Fibonacci LFSR, taps x¹⁶+x¹⁴+x¹³+x¹¹ (period 65535), advanced 8 steps per
byte; the byte is the low 8 bits. Seeded from the header's `SEED` byte
(`seed == 0 → 0x00FF`). Mirrored exactly in `host/uart_selftest.py` and in the
testbench, and unit-tested against a hard-coded first-32-bytes vector in all
three, so a drift between them can't be mistaken for a link fault.

### STATS block — 24 bytes, big-endian

```
off  len  field
 0    2   0xA5 0x5A        magic (lets the host resynchronise in a garbled stream)
 2    1   version = 0x01   identifies the image over the wire
 3    1   last command byte accepted
 4    1   flags: b0 frame_error seen, b1 overrun seen, b2 mismatch seen,
                 b3 fifo_full seen, b4 break seen
 5    1   0x00 reserved
 6    4   rx_bytes        bytes the receiver completed since CLEAR
10    4   tx_bytes        bytes the transmitter completed since CLEAR
14    4   mismatches      SINK-mode compare failures
18    4   first_bad_idx   payload index of the first SINK mismatch
22    1   first_bad_expected
23    1   first_bad_got
```

Sent **three times back to back**. The block is the one reply we cannot afford to
have corrupted by the very fault we are measuring; three copies let the host take
a majority and, separately, *notice* that a copy was corrupted — which is itself
a TX-side datapoint.

`rx_bytes` vs. what the host actually sent is the single most valuable number in
this whole exercise: it separates "bytes were lost" from "bytes arrived wrong".

---

## 4. RTL structure — `rtl/uart_selftest.sv`

```
        uart_rx ──► uart_receiver ──► data/valid ──┬──► header FSM (5 states)
                                                   ├──► echo FIFO (256 × 8, ring)
                                                   └──► LFSR checker  (SINK)
                                                                │
        uart_tx ◄── uart_transmitter ◄── tx mux ◄───────────────┤
                                                   ├──► LFSR generator (SRC)
                                                   └──► STATS shifter
```

Parameters: `CLK_PER_BIT`, `FIFO_AW` (default 8 → 256 deep), `VERSION`.

Outputs for the board wrapper: `activity` (one-shot stretched per RX byte),
`sticky_error` (OR of the flag bits, latched until CLEAR), `heartbeat`.

**FSM:** `IDLE → HDR (×4) → {ECHO | SINK | SRC | REPORT} → IDLE`. Five states plus
the header counter. Nothing else.

**Echo FIFO sizing.** RX and TX run at the same baud, and TX can only start after
a byte has fully arrived, so the transmitter permanently lags by ~1 byte and never
catches up — occupancy sits at 1–2. Depth 256 is absurd overkill on purpose: if
`fifo_full` ever sets, that is a genuine finding, not a capacity problem. The FIFO
is also why the image can accept a fully back-to-back stream with zero inter-byte
gap, which is the case the production link actually runs and the case most likely
to expose a re-arm timing bug (H3).

**Counting `rx_bytes`.** Counted inside `uart_selftest` off the rising edge of
`valid`, the same way `uart_interface` does it (`rx_byte`), so the count reflects
what a consumer of this receiver would actually see — including any byte the
receiver merges or drops.

**Overrun.** Set when `valid` rises while the previous byte has not been consumed.
In this image the consumer is always ready, so an overrun means the *receiver*
produced two `valid` rises closer together than a byte period — i.e. it decoded a
frame it should not have.

**`frame_error`.** Requires the receiver to actually look at the stop bit, which
today it does not (H3). The change: hold `STOP` until `clk_cnt == CLK_PER_BIT/2`,
sample there, `frame_error <= !rx_reg2`, then return to `IDLE`. This also moves
the re-arm point from the *start* of the stop bit to its centre, which is strictly
more correct — but it is a behavioural change to a module the production image
shares, so it lands **only after** `uart_receiver_tb` and `tpu_top_uart_tb` both
pass with it (§5), and it is the one receiver change that is not
parameter-gated-off by default.

---

## 5. Simulation — `tb/uart_selftest_tb.sv`

Run before anything is flashed. Two jobs:

**Protocol coverage.** All five commands. ECHO of 512 pseudo-random bytes sent
back-to-back with *zero* inter-byte gap. SRC/SINK against a Python-cross-checked
PRBS vector. STATS field-by-field.

**Fault injection — the part that validates the instrument.** A tester task drives
`uart_rx` with deliberate defects and the testbench asserts that the counters
report exactly what was injected:

| Injected | Expected in STATS |
|---|---|
| 1-clock low glitch during a stop bit | no extra byte; `rx_bytes` unchanged (the existing start-bit re-validation should absorb this) |
| a low glitch of ¾ bit during idle | ditto, or `frame_error` — either is fine, silent corruption is not |
| stop bit driven low | `frame_error` set, `rx_bytes` still increments |
| host baud +3% / −3% | still clean (8N1 tolerance is ~±5%) |
| host baud +6% / −6% | fails — and this is how we calibrate what the margin actually is |
| one data bit inverted for 2 clocks at the sample instant | mismatch counted, `first_bad_*` correct |
| two frames with zero gap | both received, no overrun |

The ±baud rows are the important ones: they turn "does it work" into "how much
margin is there, and is the margin symmetric". An asymmetric margin is a sample
point that is off-centre, which is H4/H6 and is fixable arithmetic rather than
noise.

Wiring: `make TEST=uart_selftest RTL="../rtl/uart_selftest.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv"`,
plus a `make selftest` shorthand.

---

## 6. Host — `host/uart_selftest.py`

Standalone, no repo imports, `pyserial` only — same rule `tpu_uart.py` follows.
Reuses the harness shape of `test_uart_link.py` (registered tests, keep going
after a failure, forensics on failure) so it reads familiarly.

```
python accel/tpu/host/uart_selftest.py -p COM5                  # full suite, ~1 min
python accel/tpu/host/uart_selftest.py -p COM5 --soak 30        # 30-minute soak
python accel/tpu/host/uart_selftest.py -p COM5 --only baud_sweep
python accel/tpu/host/uart_selftest.py --offline                # PRBS/parser checks, no board
```

| Test | What it proves |
|---|---|
| `identify` | STATS magic + version → the right bitstream is flashed and answering |
| `echo_short` | Lengths 1..16, every one a fresh random block. Off-by-one in the length handling |
| `echo_block` | 4096 random bytes, one shot |
| `echo_soak` | Loops with fresh random data until failure or `--soak N` minutes. **The headline test — this is the user-visible bug** |
| `tx_only` | SRC of 64 KiB, host verifies. Clean here ⇒ the transmitter is not the problem |
| `rx_only` | SINK of 64 KiB, device verifies, STATS reports. Isolates the receiver |
| `gap_sweep` | ECHO with inter-byte gaps of 0 / 1 / 4 / 16 bit times. A re-arm bug (H3) is gap-dependent; noise is not |
| `baud_sweep` | ECHO at host baud ×(1 ± 1%, 2%, 3%, 4%). pyserial sets arbitrary baud on FTDI parts |
| `duplex` | SRC and a host write overlapping, so both directions are active at once |

**Forensics on failure** — the reason this is worth writing rather than just
eyeballing a hexdump:

1. First mismatching index, `got`, `want`, `got ^ want`, and the union of every
   bit that was ever wrong across the run (a stuck line is the same bit every
   time; noise is not).
2. **Bit-level realignment check.** Re-serialise both the sent and the received
   streams as 8N1 bit sequences (start, 8 data LSB-first, stop) and slide one
   against the other. If the received stream matches the sent stream shifted by
   ±1 bit from some byte onwards, that *is* the user's skipped/extra-bit
   hypothesis, confirmed, with the exact byte it started at. If no shift aligns,
   the hypothesis is wrong and it's isolated sampling noise.
3. Total byte count received vs. expected — a short reply and a long reply mean
   opposite things.
4. STATS pulled immediately afterwards, printed field by field, with
   `rx_bytes` compared against the host's own sent count.
5. Where the failure sat in time (seconds into the run, bytes into the run) —
   an error rate is a number you can compare across fixes.

The suite exits non-zero on any failure and prints a one-line summary suitable
for pasting into a bug note.

**One false-pass to be aware of:** a pure byte-for-byte echo would also pass if
the RX and TX pins were shorted or the FTDI were in loopback mode. `identify`
(STATS has content no loopback could invent) and `tx_only` (the FPGA generates
data the host never sent) both rule that out, which is a second reason those two
tests exist.

---

## 7. Reading the result

The whole design above exists to fill in this table:

| Observation | Conclusion |
|---|---|
| `tx_only` clean over many MB, `echo_soak` fails | Fault is on the **RX** side. H5 (TX glitching) is out |
| `rx_only`: `mismatches > 0` but `rx_bytes == sent` | Bytes all framed correctly, individual **bits** mis-sampled → H1 / H2 |
| `rx_only`: `rx_bytes < sent` | Bytes **dropped** — a start bit missed, or overrun (H7). Check the `overrun` flag |
| `rx_only`: `rx_bytes > sent` | **Spurious** start bits — a glitch got past the re-validation, or the stop bit is being re-decoded → H3 |
| `frame_error > 0` | Sample point drifting off the bit, or the stop bit is being eaten → H3 / H4 / H6 |
| Fails at gap = 0, clean at gap ≥ 1 bit | Re-arm timing relative to the stop bit → **H3** |
| Fails at +2% baud, clean at −2% (or vice versa) | Sample point biased off-centre → **H4 / H6**, and the fix is arithmetic |
| Margin symmetric to ±4%, still fails randomly at 0% | Not timing. Metastability or line noise → **H1 / H2** |
| `tx_only` itself fails | H5, or the host/cable — swap the cable before touching RTL |
| Everything here is clean but `test_uart_link.py` still fails | The fault is *above* the receiver: `uart_interface`, the SRAM mux, or arbitration. Different investigation, and a valuable answer |

That last row matters as much as the others. If the self-test image is clean under
a long soak, we have eliminated the UART blocks entirely and the search moves to
`uart_interface.sv` — which is a much better place to be than where we are now.

---

## 8. Order of work

| # | Step | Gate |
|---|---|---|
| 1 | `uart_selftest.sv` + `uart_selftest_tb.sv`, ECHO path only | `make selftest` green |
| 2 | Receiver `frame_error`/`overrun` outputs; re-run `uart_receiver_tb`, `uart_interface_tb`, `tpu_top_uart_tb` | all three still pass |
| 3 | SRC / SINK / STATS + fault injection in the tb | injected faults are counted exactly |
| 4 | Board wrapper, `.xdc`, `board.tcl`; `build.tcl board=cmod_a7_selftest mode=rtl` then `mode=bit` | elaborates, then routes with timing met |
| 5 | `host/uart_selftest.py`; `--offline` first | PRBS matches the tb vector |
| 6 | Flash, `identify`, `echo_block` | link answers |
| 7 | `echo_soak`, then the isolation + sweep tests | a filled-in §7 table |
| 8 | Fix per §7, re-run, then re-run `test_uart_link.py` on the production image | the original bug is gone |

Steps 1–5 need no hardware. Step 4 needs Vivado. Steps 6–8 need the board.

Nothing before step 8 changes the behaviour of the production bitstream: the
receiver gains outputs and a defaulted-off parameter, the transmitter gains one
flop, and the self-test lives in its own board target with its own output
directory (`synth/build/cmod_a7_selftest/`), so the two `.bit` files can never be
confused.

---

## 9. Open questions

1. **Is `--soak` reproducing the same bug?** If the self-test image runs clean for
   an hour while `test_uart_link.py` fails in 30 s, that is a result (§7, last
   row) — but it also means the echo path is not stressing whatever matters.
   Backup plan: add a mode that mimics the production traffic *shape* — a burst
   in, a turnaround, a burst out — since the RX→TX turnaround is the one thing
   pure echo does not reproduce faithfully.
2. **Do you want `SAMPLE_VOTES=3` (majority-vote sampling) built in from the
   start**, so step 7 can A/B it in one session rather than needing a rebuild?
   It costs 2 flops and a 3-input majority gate. I'd default it off but wire the
   parameter through both board targets — say the word if you'd rather not carry
   it until it's justified.
3. **LED assignment.** Proposed: `led[0]` = ~1 Hz heartbeat (proves the clock and
   the bitstream are alive independently of the UART), `led[1]` = sticky error,
   cleared by `btn[0]` reset or the CLEAR command. `btn[1]` is free — it could
   trigger a standalone SRC burst so the TX path can be checked with nothing but
   a terminal open.
