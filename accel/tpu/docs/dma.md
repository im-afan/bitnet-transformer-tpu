# DMA Engine

The block that moves bytes between **DRAM** (the CMOD A7's external SRAM, driven by
[`sram_controller`](../rtl/sram.sv)) and the on-chip [scratchpad](scratchpad.md), without the
scalar unit copying each byte itself. It is the hardware behind the `ReadMemory` /
`WriteMemory` ISA ops ([scalar_unit.md](scalar_unit.md) §Comms/Memory). No RTL exists yet —
`rtl/dma.sv` is an empty stub; this doc is the plan.

---

## 0. What a DMA is (and why we need one)

**DMA = Direct Memory Access.** It's a small dedicated state machine whose only job is to copy
a block of memory from one place to another *on its own*, so the main processor (here, the
[scalar unit](scalar_unit.md)) doesn't have to spend an instruction per byte.

The pattern everywhere in this TPU is **issue-and-wait**: the scalar unit hands a unit a
command (`start` + operands), then blocks until that unit raises `done`. The MXU and VPU
already work this way. The DMA is the same idea applied to a *memory copy*:

> "Copy `len` bytes from DRAM address X into scratchpad address Y, tell me when you're done."

The scalar unit issues that one command and goes to sleep; the DMA grinds through the bytes and
pulses `done`. That's the whole contract. Everything below is *how* it grinds through the bytes.

### Why it's not just a `for` loop

The two memories the DMA bridges have **completely different shapes**:

| Memory                | Width per access | Timing                          | Size (this board)   |
| --------------------- | ---------------- | ------------------------------- | ------------------- |
| Scratchpad (BRAM)     | **64 bytes**     | 1 cycle, synchronous read       | 64 KB (`ADDR_W=16`) |
| DRAM (external SRAM)  | **1 byte**       | ~3 cycles/byte, `start`/`done`  | 512 KB (`ADDR_W=19`)|

So a DMA transfer is fundamentally a **width + rate adapter**: it takes wide, fast scratchpad
words and serializes them into narrow, slow byte transactions to the SRAM (and vice-versa).
That mismatch is the entire reason this is a real block and not a wire.

---

## 1. Where it sits

```
   scalar_unit  ── dispatch (dma_start, dma_write, addrs, len / busy, done) ──►  ┌─────────┐
                                                                                 │   DMA   │
   scratchpad  ◄── dma_* port (64B synchronous read/write, byte-strobed) ──────► │  engine │
                                                                                 │  (FSM)  │
   sram_ctrl   ◄── start/we/addr/din / dout/busy/done (1 byte per txn) ────────► └─────────┘
       │
       └── FPGA pins ──► external SRAM chip (constraints/constraings-cmod.xdc)
```

The DMA is a **master on two buses at once**. It is a *slave* to the scalar unit (it accepts a
command) and a *master* to both the scratchpad and the SRAM controller (it drives their
requests). All three interfaces already exist — the DMA is the missing middle:

- **Scalar side** — [`scalar_unit.sv`](../rtl/scalar_unit.sv) already exposes
  `dma_start / dma_write / dma_scratch_addr / dma_dram_addr / dma_len / dma_busy / dma_done`.
  In [`tpu_top.sv`](../rtl/tpu_top.sv) these dispatch outputs are currently dangling and
  `dma_busy=0 / dma_done=1` are tied off so a `ReadMemory`/`WriteMemory` is a completing no-op.
- **Scratchpad side** — [`scratchpad.sv`](../rtl/scratchpad.sv) has a dedicated `dma_*` port
  (`DMA_BYTES=64` wide, synchronous read valid next cycle, byte-strobed write). Today
  `tpu_top` wires that port out to the host as `host_mem_*` for preload/readback.
- **DRAM side** — [`sram_controller`](../rtl/sram.sv): one `start`/`busy`/`done` transaction
  per byte, `we` selects read vs. write, 19-bit address, 8-bit data.

Landing the DMA means: connect its dispatch inputs to the scalar unit, its scratchpad master
port to the scratchpad `dma_*` port (replacing the `host_mem_*` pass-through), and its SRAM
master port to a `sram_controller` instance whose chip pins go to the top-level.

---

## 2. The command (scalar-unit interface)

Latched on `dma_start` (one-cycle pulse), held until `dma_done`. This is fixed by the scalar
unit — the DMA consumes it as-is.

| Signal             | Dir | Width  | Meaning                                                    |
| ------------------ | --- | ------ | ---------------------------------------------------------- |
| `dma_start`        | in  | 1      | begin a transfer (pulse)                                   |
| `dma_write`        | in  | 1      | **1 = spill** scratch → DRAM · **0 = fill** DRAM → scratch |
| `dma_scratch_addr` | in  | 16     | scratchpad byte address (base)                             |
| `dma_dram_addr`    | in  | 16*    | DRAM byte address (base) — *see §7 note on width*          |
| `dma_len`          | in  | 16     | number of bytes to move                                    |
| `dma_busy`         | out | 1      | transfer in progress                                       |
| `dma_done`         | out | 1      | transfer complete (single-cycle pulse)                     |

`dma_write` mirrors the ISA: `WriteMemory` (spill) sets it, `ReadMemory` (fill) clears it. The
handshake is identical to MXU/VPU dispatch, so the scalar unit's `U_DMA` wait state
(`sel_done = dma_done`) already knows how to block on it.

---

## 3. Data movement: the two flows

The engine walks a byte counter `i` from `0` to `len-1`. Each byte touches both memories; the
order depends on direction.

### Fill — DRAM → scratchpad (`dma_write = 0`, `ReadMemory`)

For each byte `i`:
1. **SRAM read** — `sram.start=1, we=0, addr = dram_addr + i`; wait for `sram.done`; capture
   `sram.dout`.
2. **Scratchpad write** — one byte via a one-hot strobe: `dma_waddr = scratch_addr + i`,
   `dma_wdata[7:0] = byte`, `dma_wstrb = 1` (lane 0 only). Completes in one cycle.

### Spill — scratchpad → DRAM (`dma_write = 1`, `WriteMemory`)

For each byte `i`:
1. **Scratchpad read** — `dma_re=1, dma_raddr = scratch_addr + i`; `dma_rdata` is valid the
   *next* cycle (synchronous-read contract); take lane 0.
2. **SRAM write** — `sram.start=1, we=1, addr = dram_addr + i, din = byte`; wait `sram.done`.

The SRAM is the bottleneck (~3 cycles/byte vs. 1 for the scratchpad), so v1 deliberately does
**byte-at-a-time** transfers and doesn't try to batch. It's the simplest thing that's correct;
§6 covers the line-buffered optimization once v1 passes.

---

## 4. FSM

The engine's whole memory is small: one byte counter (`i`), one direction bit (`wr`), the
latched base addresses (`scratch_addr`, `dram_addr`) and `len`, and a one-byte holding register
(`byte`). Everything the DMA does is a loop that, once per byte, reads from one memory into
`byte` and writes `byte` into the other. The states exist because **each memory access takes
more than one cycle**, so a single "copy one byte" step has to be spread across several states
that each drive one phase of a handshake.

```
IDLE ──dma_start──► (latch wr, scratch_addr, dram_addr, len; i=0; busy=1)
  │
  ├─ if len == 0 ──────────────────────────────────► DONE      (empty transfer)
  │
  ├─ wr == 0  (FILL: DRAM → scratch) ───────────────────────────────┐
  │     SR_RD   : sram.start=1, we=0, addr=dram_addr+i              │
  │     SR_WAIT : wait sram.done → byte <= sram.dout                │
  │     SP_WR   : dma_we=1, waddr=scratch_addr+i, wdata=byte,       │
  │              wstrb=1                                            │
  │     STEP    : i++ ; (i < len) ? SR_RD : DONE                    │
  │                                                                 │
  ├─ wr == 1  (SPILL: scratch → DRAM) ──────────────────────────────┤
  │     SP_RD   : dma_re=1, raddr=scratch_addr+i                    │
  │     SP_WAIT : (rdata valid) byte <= dma_rdata[7:0]              │
  │     SW_WR   : sram.start=1, we=1, addr=dram_addr+i, din=byte    │
  │     SW_WAIT : wait sram.done                                    │
  │     STEP    : i++ ; (i < len) ? SP_RD : DONE                    │
  │                                                                 │
DONE : dma_done=1 (one cycle), busy=0 ──► IDLE  ◄──────────────────┘
```

### What each state means

**Shared**

- **`IDLE`** — the resting state. The DMA sits here doing nothing, `busy=0`, waiting for
  `dma_start`. When it arrives, this is the *only* moment the command inputs are sampled: it
  copies `dma_write → wr`, the two addresses and `len` into its own registers, zeroes the byte
  counter `i`, raises `busy`, and jumps into whichever loop `wr` selects. Latching here is what
  lets the scalar unit drop its inputs after the one-cycle `start` pulse — the DMA no longer
  depends on them.

- **`STEP`** — the loop's "bottom". One byte has fully landed in its destination, so this state
  advances the counter (`i++`) and makes the one branch decision in the whole machine: if there
  are more bytes (`i < len`) go back to the top of the loop for byte `i`; otherwise the transfer
  is finished, so go to `DONE`. (In RTL this can be folded into the last write state to save a
  cycle; it's drawn separately here so the loop is obvious.)

- **`DONE`** — the finish line. Pulses `dma_done` high for exactly one cycle and drops `busy`,
  which is the signal the scalar unit's `U_DMA` wait state is blocking on. Then it falls back to
  `IDLE`, ready for the next command.

**Fill loop (`wr = 0`, DRAM → scratchpad)** — read a byte *out of* the slow SRAM, then drop it
into the fast scratchpad:

- **`SR_RD`** ("SRAM read, issue") — kick off one SRAM read: drive `sram.start=1`, `we=0`
  (read), and `addr = dram_addr + i`. This asserts `start` for a *single* cycle; the actual data
  won't be ready this cycle, so we immediately move on to wait for it.
- **`SR_WAIT`** ("SRAM read, wait") — the SRAM controller is off doing its own multi-cycle
  access. The DMA parks here until `sram.done` goes high, then captures the returned byte into
  its holding register (`byte <= sram.dout`). This state is why the counter isn't just a `for`
  loop — most of the transfer time is spent sitting in these wait states.
- **`SP_WR`** ("scratchpad write") — the byte is in hand, so write it into the scratchpad in one
  cycle: `dma_we=1`, `dma_waddr = scratch_addr + i`, the byte on lane 0 of `dma_wdata`, and a
  one-hot `dma_wstrb = 1` so *only* that one byte is written and the other 63 lanes of the wide
  word are left untouched. Then fall through to `STEP`.

**Spill loop (`wr = 1`, scratchpad → DRAM)** — read a byte *out of* the fast scratchpad, then
push it into the slow SRAM. Note the read/write order is reversed vs. fill, because now the
scratchpad is the source:

- **`SP_RD`** ("scratchpad read, issue") — request the byte: `dma_re=1`,
  `dma_raddr = scratch_addr + i`. The scratchpad's read is **synchronous** — the data is *not*
  on `dma_rdata` this cycle, it appears next cycle — so we can't use it yet.
- **`SP_WAIT`** ("scratchpad read, capture") — exactly one cycle later the data is valid;
  latch lane 0 into the holding register (`byte <= dma_rdata[7:0]`). This state exists purely to
  honor that one-cycle read latency; skipping it would capture stale data from the *previous*
  read.
- **`SW_WR`** ("SRAM write, issue") — start one SRAM write: `sram.start=1`, `we=1` (write),
  `addr = dram_addr + i`, `din = byte`. Like `SR_RD`, this only pulses `start` for one cycle.
- **`SW_WAIT`** ("SRAM write, wait") — park until `sram.done` confirms the byte was committed to
  the chip, then fall through to `STEP`. The SRAM controller itself ends the write pulse cleanly
  while the address/data are still held (see `sram.sv`), so the DMA just has to wait for its
  `done`.

### Why the wait states are unavoidable

Two of the DMA's neighbors answer *later* than the cycle you ask them:

1. **The SRAM controller** takes several cycles per access and reports completion with `done`.
   The DMA must pulse `start` once and then wait — re-driving `start` while `sram.busy` is high
   would restart or corrupt the in-flight access.
2. **The scratchpad read** returns data one cycle after `dma_re` (the synchronous-read contract
   every other unit relies on). Hence the dedicated `SP_WAIT` capture cycle in the spill loop.

The write side of the scratchpad, by contrast, is single-cycle, which is why fill's `SP_WR` and
the counter bump can be done immediately with no wait.

Two more small points from the diagram:

- **`len == 0`** short-circuits `IDLE → DONE` so an empty transfer still produces a proper
  `done` handshake instead of hanging or underflowing the counter.
- **Symmetry.** Fill is *(SRAM read → scratch write)* and spill is *(scratch read → SRAM
  write)* — the same two-access shape with source and destination swapped. Only the counter,
  the `done` handshake, and `IDLE` are shared between the two loops.

---

## 5. Address & timing facts to respect

- **Single clock domain.** The SRAM is asynchronous but its controller is clocked on the same
  `clk` as the DMA and scratchpad, so there is **no clock-domain crossing** here — unlike the
  comms link ([comms.md](comms.md) §4). That keeps v1 simple: no async FIFOs.
- **No arbitration needed in v1.** The scalar unit blocks on `dma_done` (issue-and-wait), so
  while the DMA runs the MXU/VPU are idle and nobody else touches the scratchpad. The
  scratchpad's "DMA fills idle cycles" note ([scratchpad.md](scratchpad.md) §4) is about a
  future overlapped mode; v1 has the memory to itself.
- **Byte-serial handles any alignment / length.** Because every access is one byte at an
  absolute address, unaligned bases and a `len` that isn't a multiple of 64 just work — no
  special first/last-word masking (that complexity only appears in the §6 optimization).

---

## 6. Later: line-buffered (burst) mode

v1 wastes the scratchpad's 64-byte width — it reads/writes one lane at a time. Once correct,
the throughput win is to **buffer a full 64-byte scratchpad line** and stream it to/from the
SRAM as 64 back-to-back byte transactions:

- **Fill:** collect 64 SRAM bytes into a line register, then do **one** 64-byte scratchpad
  write with a full strobe.
- **Spill:** do **one** 64-byte scratchpad read, then drain the 64 bytes to the SRAM.

This amortizes the scratchpad access and is the natural place to later overlap SRAM traffic
with compute. It adds alignment bookkeeping (partial head/tail lines when `scratch_addr` or
`len` isn't 64-aligned), which is exactly why it's **not** in v1. A further step is a
prefetch/double-buffer so the next line's SRAM reads overlap the current line's scratchpad
write.

---

## 7. Open questions / decisions

- **DRAM address width.** `dma_dram_addr` is currently **16 bits** (scalar unit + `tpu_top`),
  but the SRAM is **19 bits / 512 KB**. As-is the DMA can only reach the low 64 KB. Options:
  widen `dma_dram_addr` to ≥19 bits, or add a DRAM **base/segment config register** (the scalar
  unit already has a `cfg[]` file — `dma_len` comes from `CFG_LEN`) and form the full address as
  `{cfg_base, dma_dram_addr}`. **Recommend** widening the DRAM address, since the point of this
  work is to use the SRAM as main memory *beyond* the scratchpad.
- **`tpu_top` rewire.** Landing the DMA means the scratchpad `dma_*` port stops being the
  `host_mem_*` pass-through. Decide whether the host still needs a direct BRAM path (e.g. a mux:
  host owns the port while `!busy`, DMA owns it during a program) or whether host preload now
  goes DRAM→DMA→scratchpad like everything else.
- **Error signalling.** Should an out-of-range DRAM address (≥ SRAM depth) fault back to the
  scalar unit, or wrap silently as the controller does today? No fault path exists yet.
- **`dma_len` units / max.** 16 bits = up to 64 KB per transfer, which exceeds the 64 KB
  scratchpad — fine, but confirm callers never expect a length in *words*.

---

## 8. Validation plan

Mirror the existing block tests (`make TEST=dma sim`, iverilog `-g2012`), reusing the
behavioral async-SRAM model from [`sram_tb.sv`](../tb/sram_tb.sv):

1. **Bench topology.** Instantiate the real `dma` + real `sram_controller` + real `scratchpad`
   (or a behavioral scratchpad model), plus the behavioral SRAM chip model on the controller's
   chip pins. Drive the scalar-side dispatch directly from the testbench.
2. **Fill then verify.** Preload the SRAM model with a known pattern, issue a `ReadMemory`
   (fill), wait `dma_done`, and read the scratchpad back through its port — every byte in
   `[scratch_addr, scratch_addr+len)` must equal the SRAM source. Reference = a plain memcpy.
3. **Spill then verify.** Preload the scratchpad, issue a `WriteMemory` (spill), and read the
   SRAM model back.
4. **Round-trip.** Spill a scratch region to DRAM, wipe scratch, fill it back — must match.
5. **Corners.** `len = 0` (immediate `done`, nothing moved), `len = 1`, unaligned bases, a
   `len` that isn't a multiple of 64, and a transfer that reaches a high SRAM address (exercises
   the §7 address-width decision).
6. **Handshake asserts.** `busy` frames the whole transfer, `done` is a single-cycle pulse, and
   the DMA never re-pulses `sram.start` while `sram.busy` is high.

---

## 9. Build order

1. **v1 byte-serial FSM** (§3–§4) against the §8 bench — correctness first.
2. **Rewire `tpu_top`** (§7): DMA between scalar dispatch, scratchpad `dma_*`, and a
   `sram_controller` instance; resolve `host_mem_*`.
3. **Widen the DRAM address** (§7) so the full SRAM is reachable.
4. **Line-buffered mode** (§6) for throughput, then optional prefetch/double-buffer.
