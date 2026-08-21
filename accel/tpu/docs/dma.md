# DMA Engine

The block that moves bytes between **DRAM** (the CMOD A7's external SRAM, driven by
[`sram_controller`](../rtl/sram.sv)) and the on-chip [scratchpad](scratchpad.md), without the
scalar unit copying each byte itself. It is the hardware behind the `ReadMemory` /
`WriteMemory` ISA ops ([scalar_unit.md](scalar_unit.md) §Comms/Memory).

**Status: built.** [`rtl/dma.sv`](../rtl/dma.sv) implements §3–§4 and is wired into
`tpu_top.sv` between the scalar unit's dispatch, the scratchpad's `dma_*` port and a
`sram_controller`; [`tb/dma_tb.sv`](../tb/dma_tb.sv) is the §9 bench. §5's **transpose
mode** is built on top of it. §7's throughput work is built too, though not as the
line buffer this document originally planned — the width adapter moved into
[`sram.sv`](../rtl/sram.sv) as a *range* request and the DMA became a stream client.
**1 clock/byte on fills, 2 on spills**, against ~8 before.

---

## 0. What a DMA is (and why we need one)

**DMA = Direct Memory Access.** It's a small dedicated state machine whose only job is to copy
a block of memory from one place to another *on its own*, so the main processor (here, the
[scalar unit](scalar_unit.md)) doesn't have to spend an instruction per byte.

> **Since updated.** Dispatch arrives from `cmd_dma.sv`'s command queue rather
> than from the scalar unit's wire bundle, and the geometry (`len`, `tcols`,
> `tsrow`, `tdrow`) travels *in the command* instead of in config registers --
> so a transfer carries its own length and the stale-`len` pitfall is gone by
> construction. The engine below is otherwise unchanged, plus a fill skid buffer
> and grant handshakes on both scratchpad ports so a transfer can overlap
> compute. See [picorv32_migration.md](picorv32_migration.md) §3 and §5.

The pattern everywhere in this TPU is **issue-and-wait**: the scalar unit hands a unit a
command (`start` + operands), then blocks until that unit raises `done`. The MXU and VPU
already work this way. The DMA is the same idea applied to a *memory copy*:

> "Copy `len` bytes from DRAM address X into scratchpad address Y, tell me when you're done."

The scalar unit issues that one command and goes to sleep; the DMA grinds through the bytes and
pulses `done`. That's the whole contract. Everything below is *how* it grinds through the bytes.

### Why it's not just a `for` loop

The two memories the DMA bridges have **completely different shapes**:

| Memory                | Width per access | Timing                             | Size (this board)   |
| --------------------- | ---------------- | ---------------------------------- | ------------------- |
| Scratchpad (BRAM)     | **64 bytes**     | 1 cycle, synchronous read          | 64 KB (`ADDR_W=16`) |
| DRAM (external SRAM)  | **1 byte**       | 1 clock/byte read, 2/byte written  | 512 KB (`ADDR_W=19`)|

So a DMA transfer is fundamentally a **width + rate adapter**: it takes wide, fast scratchpad
words and serializes them into narrow byte streams to the SRAM (and vice-versa). That
mismatch is the entire reason this is a real block and not a wire.

The rate column used to read *~3 cycles/byte, `start`/`done`*, because the controller took
one request per byte. It now takes one request per **range** and streams it, which is what
§7 is about.

---

## 1. Where it sits

```
   scalar_unit  ── dispatch (dma_start, dma_write, addrs, len / busy, done) ──►  ┌─────────┐
                                                                                 │   DMA   │
   scratchpad  ◄── dma_* port (64B synchronous read/write, byte-strobed) ──────► │  engine │
                                                                                 │  (FSM)  │
   sram_ctrl   ◄── start/we/addr/len/stride + byte streams / busy/done ────────► └─────────┘
       │
       └── FPGA pins ──► external SRAM chip (constraints/constraings-cmod.xdc)
```

The DMA is a **master on two buses at once**. It is a *slave* to the scalar unit (it accepts a
command) and a *master* to both the scratchpad and the SRAM controller (it drives their
requests):

- **Scalar side** — [`scalar_unit.sv`](../rtl/scalar_unit.sv) exposes
  `dma_start / dma_write / dma_scratch_addr / dma_dram_addr / dma_len / dma_busy / dma_done`,
  plus the §5 transpose inputs, and `tpu_top.sv` wires them straight to the engine.
- **Scratchpad side** — [`scratchpad.sv`](../rtl/scratchpad.sv) has a dedicated `dma_*` port
  (`DMA_BYTES=64` wide, synchronous read valid next cycle, byte-strobed write), which the
  engine owns; the host reaches memory over the UART instead.
- **DRAM side** — [`sram_controller`](../rtl/sram.sv): one `start`/`busy`/`done` request per
  **range** (`addr` + `len` + `stride`), `we` selects read vs. write, 19-bit address, 8-bit
  data, with a `dout_valid` strobe per byte read and a ready/valid stream for bytes written.
  `tpu_top` arbitrates it between the DMA (while a program runs) and the UART host (while
  idle, at `len = 1`).

---

## 2. The command (scalar-unit interface)

Latched on `dma_start` (one-cycle pulse), held until `dma_done`. This is fixed by the scalar
unit — the DMA consumes it as-is.

| Signal             | Dir | Width  | Meaning                                                    |
| ------------------ | --- | ------ | ---------------------------------------------------------- |
| `dma_start`        | in  | 1      | begin a transfer (pulse)                                   |
| `dma_write`        | in  | 1      | **1 = spill** scratch → DRAM · **0 = fill** DRAM → scratch |
| `dma_scratch_addr` | in  | 16     | scratchpad byte address (base)                             |
| `dma_dram_addr`    | in  | 16*    | DRAM byte address (base) — *see §8 note on width*          |
| `dma_len`          | in  | 16     | number of bytes to move                                    |
| `dma_transpose`    | in  | 1      | **transposed addressing** (the `.t` instruction flag) — §5 |
| `dma_tcols`        | in  | 16     | §5 source row length, elements (`cfg tcols`)               |
| `dma_tsrow`        | in  | 16     | §5 source row stride, bytes (`cfg tsrow`)                  |
| `dma_tdrow`        | in  | 16     | §5 destination row stride, bytes (`cfg tdrow`)             |
| `dma_busy`         | out | 1      | transfer in progress                                       |
| `dma_done`         | out | 1      | transfer complete (single-cycle pulse)                     |

`dma_write` mirrors the ISA: `WriteMemory` (spill) sets it, `ReadMemory` (fill) clears it. The
handshake is identical to MXU/VPU dispatch, so the scalar unit's `U_DMA` wait state
(`sel_done = dma_done`) already knows how to block on it.

`dma_transpose` comes from the instruction's `flags[0]`, not from the config file; the three
geometry registers come from the config file. That split is deliberate — see §5.3.

---

## 3. Data movement: the two flows

The engine walks a byte counter `i` from `0` to `len-1`. Each byte touches both memories; the
order depends on direction. The DRAM side is not a per-byte transaction any more — it is a
**range** (§7), issued once per source row, and the bytes of that range arrive or are asked
for one per handshake.

### Fill — DRAM → scratchpad (`dma_write = 0`, `ReadMemory`)

Per range: `sram.start=1, we=0, addr = this row's DRAM base, len = this row's byte count,
stride = 1`. Then, for each byte:

1. **Take the byte** — `sram.dout_valid` pulses with `sram.dout` holding it, one byte per
   clock.
2. **Scratchpad write** — in that same clock, via a one-hot strobe:
   `dma_waddr = scratch_addr + off`, `dma_wdata[7:0] = byte`, `dma_wstrb = 1` (lane 0 only).

The scratchpad write port takes one every clock, so nothing buffers and nothing needs to be
aligned. The range ends with `sram.done`, which the controller raises on the same clock as
the last `dout_valid`.

### Spill — scratchpad → DRAM (`dma_write = 1`, `WriteMemory`)

Per range: `sram.start=1, we=1, addr = this row's DRAM base, len, stride` (`tdrow` in
transposed mode, else 1). Then, two clocks per byte, pipelined against the controller's
two-clock write beat:

1. **Scratchpad read** — `dma_re=1, dma_raddr = scratch_addr + off`; `dma_rdata` is valid the
   *next* cycle (synchronous-read contract); take lane 0.
2. **Offer it** — `sram.din = byte`, `sram.din_valid = 1`; the controller takes it on
   `sram.din_ready`, which it raises on the last clock of the beat before. A byte that is
   *not* taken immediately (only possible at `CLOCKS_PER_ACCESS > 0`) is held in a one-byte
   register rather than re-read, because the scratchpad only promises its read data for the
   one cycle after `re`.

The SRAM is still the bottleneck — 1 clock/byte read and 2 written, against the scratchpad's
1 — but it is now a stream rather than a sequence of transactions, and the engine's own
sequencing no longer adds to it.

---

## 4. FSM

The engine's whole memory is small: one byte counter (`i`), the 2-D walk (`col`/`row` and
their two running offsets), and a one-byte holding register for the spill path. Everything
the DMA does is a loop that hands the controller one **row** at a time and then moves that
row's bytes one per handshake. The states exist because the two memories answer on different
clocks — the scratchpad one cycle after `re`, the SRAM one byte per beat — so a "copy one
byte" step is still spread over a couple of states on the spill side; on the fill side it
collapsed to nothing, because a byte arriving and a byte being stored are the same clock.

```
IDLE ──dma_start──► (i=0; col=row=0; busy=1)
  │
  ├─ if len == 0 ──────────────────────────────────────────────────► DONE
  │
  └─► ISSUE : sram.start=1, we=wr, addr=<row base>,                  ◄──┐
  │           len=min(tcols, len-i), stride=<1 or tdrow>                │
  │                                                                     │
  ├─ wr == 0  (FILL: DRAM → scratch) ───────────────────────────────┐   │
  │     FILL       : on sram.dout_valid — dma_we=1, wstrb=1,        │   │
  │                  waddr=scratch_addr+off, wdata=sram.dout ; i++  │   │
  │                  on sram.done — row finished ───────────────────────┤
  │                                                                 │   │
  ├─ wr == 1  (SPILL: scratch → DRAM) ──────────────────────────────┤   │
  │     SPILL_RD   : dma_re=1, raddr=scratch_addr+off               │   │
  │     SPILL_SEND : din=dma_rdata[7:0], din_valid=1 ;              │   │
  │                  on sram.din_ready — i++ ; more in this row     │   │
  │                  ? SPILL_RD : SPILL_END                         │   │
  │     SPILL_END  : wait sram.done (the last write commits) ───────────┤
  │                                                                 │   │
  │                       (i < len) ────────────────────────────────────┘
DONE : dma_done=1 (one cycle), busy=0 ──► IDLE  ◄──────────────────┘
```

In linear mode `tcols` defaults to `dma_len`, so the loop runs exactly once: one range for
the whole transfer.

### What each state means

**Shared**

- **`IDLE`** — the resting state. The DMA sits here doing nothing, `busy=0`, waiting for
  `dma_start`. The command inputs are *not* copied: the scalar unit holds them stable until
  `dma_done` (issue-and-wait), so the engine reads them where it needs them and only has to
  zero its own counters here.

- **`ISSUE`** — one cycle, once per source row. It pulses `sram.start` with this row's DRAM
  base address, its byte count (`min(tcols, len - i)` — so a `len` that is not a whole number
  of rows simply makes the last range short), and its stride. The controller latches all of
  it on that edge, and is guaranteed to be idle then: the previous range's `done` is what
  brought us back here.

- **`DONE`** — the finish line. Pulses `dma_done` high for exactly one cycle and drops `busy`,
  which is the signal the scalar unit's `U_DMA` wait state is blocking on. Then it falls back to
  `IDLE`, ready for the next command.

**Fill loop (`wr = 0`, DRAM → scratchpad)** — one state, because the two sides run at the
same rate:

- **`FILL`** — every `sram.dout_valid` is a byte, and it is written into the scratchpad on
  that same clock: `dma_we=1`, `dma_waddr = scratch_addr + off`, the byte on lane 0 of
  `dma_wdata`, and a one-hot `dma_wstrb = 1` so *only* that byte is written and the other 63
  lanes of the wide word are left untouched. The walk advances on the same edge, so the
  address is always the one belonging to the byte in hand. The row ends on `sram.done`, which
  lands on the same clock as its last `dout_valid` — `tb/dma_tb.sv` asserts exactly that,
  because the counter arithmetic here depends on it.

**Spill loop (`wr = 1`, scratchpad → DRAM)** — two states, because the scratchpad answers a
cycle late and that cycle is exactly the controller's turnaround:

- **`SPILL_RD`** ("scratchpad read, issue") — request the byte: `dma_re=1`,
  `dma_raddr = scratch_addr + off`. The scratchpad's read is **synchronous** — the data is
  *not* on `dma_rdata` this cycle, it appears next cycle.
- **`SPILL_SEND`** — the data is valid now, so offer it: `sram.din = dma_rdata[7:0]`,
  `sram.din_valid = 1`. The controller raises `din_ready` on the last clock of the beat
  before, which is precisely this clock, so the offer is taken with no stall and the pair of
  states costs two clocks per byte — the write beat's own length. If it is *not* taken (a
  longer beat, i.e. `CLOCKS_PER_ACCESS > 0`), the byte goes into a one-byte holding register
  and is re-offered from there; `dma_rdata` is only promised for the one cycle.
- **`SPILL_END`** — the row's last byte has been handed over but not yet committed. Park
  until `sram.done`, then issue the next row or finish.

### What is left of the wait states

Two of the DMA's neighbors still answer *later* than the cycle you ask them, but only one of
them now costs anything:

1. **The scratchpad read** returns data one cycle after `dma_re` (the synchronous-read
   contract every other unit relies on). That cycle is `SPILL_RD`, and it is free: it is
   spent inside the two-clock write beat rather than in addition to it.
2. **The SRAM controller** used to take several cycles per *access* and report each one with
   `done`. It now reports once per *range*, so the per-byte wait is gone entirely on the fill
   side and hidden on the spill side. `start` still must not be re-pulsed while `sram.busy`
   is high — the range would be lost — which is why `ISSUE` is only ever entered from `IDLE`
   or from a completed range.

Two more small points from the diagram:

- **`len == 0`** short-circuits `IDLE → DONE` so an empty transfer still produces a proper
  `done` handshake instead of hanging or underflowing the counter.
- **Symmetry.** Fill is *(SRAM read → scratch write)* and spill is *(scratch read → SRAM
  write)* — the same two-access shape with source and destination swapped. Only the counters,
  the range dispatch, the `done` handshake, and `IDLE` are shared between the two loops.

---

## 5. Transpose mode (`rdmem.t` / `wrmem.t`)

The byte *count* and the FSM are unchanged. Only the two address generators differ: the
engine walks the transfer as a 2-D loop and applies opposite orders to the two sides.

```
    for r = 0, 1, 2, ...            # rows, until dma_len bytes have moved
      for c = 0 .. tcols-1          # the inner counter
         read   source      at  src + r*tsrow + c        # row-major
         write  destination at  dst + c*tdrow + r        # transposed
```

**"Source" is direction-relative, and that is the whole trick.** On a fill the source is DRAM
and the destination is the scratchpad; on a spill it is the other way round. One convention
covers both directions because transposing on the way *out* of an `[R][C]` tensor and
transposing on the way *in* to a `[C][R]` one are the same permutation. So there is exactly
one rule to remember: **the source is read row-major, the destination is written transposed.**

### 5.1 Why the two strides are not just `C` and `R`

`tsrow` lets the source be a **column slice of a wider tensor**. That is not generality for
its own sake — it is the case attention actually has. Q, K and V come out of one fused
`[T][3d]` projection, so V is `d` columns of a `3d`-byte row, and a transfer that assumed
dense rows would need the slice copied out first. With `tsrow = 3d` and `tcols = d` the slice
is read in place.

`tdrow` is a free stride rather than a hard-wired row count for the mirror-image reason: the
result can land inside a wider destination (one head's `V^T` written into a `[H·d][T]` block).
For a plain dense transpose it is simply `R`, the source's row count.

The row count `R` is never a parameter. `dma_len` already fixes it — the transfer ends when
the byte count is met, wherever that falls. A `dma_len` that is not a whole number of rows
therefore stops part-way through the last one rather than faulting, and writes nothing past
it.

### 5.2 Zero means "not set"

Each parameter falls back to the value that makes the mode degenerate gracefully rather than
collapse every address onto `0` — the same convention `mxu.sv` uses for its strides, adopted
for the same reason (`macro_ops.md` §4.0):

| Register | 0 means | Effect |
| --- | --- | --- |
| `tcols` | `dma_len` | one row: a **strided scatter/gather**, which is the same generator seen end-on |
| `tsrow` | `tcols` | dense source rows |
| `tdrow` | `1` | dense destination |

With all three unset, `rdmem.t` is bit-for-bit a plain `rdmem`. A missing `setcfg` costs you a
copy, not a scribble.

### 5.3 Why the mode is an instruction flag and the geometry is not

`matmul_t` exists as a separate opcode because `matmul`'s 2-bit flags field was full
(`macro_ops.md` §4.0). The DMA ops had room — `rdmem` is `RS`-form and `wrmem` is `SS`-form,
so `flags[0]` was free on both — so the mode costs no opcode.

It has to be *in the instruction* either way. Config registers are not cleared between runs,
so a mode inferred from "the stride registers are nonzero" would let one program's leftover
geometry silently rearrange the next program's plain `rdmem`. That is precisely the bug the
MXU hit when `strided_matmul.tpu` ran before `relu_layer.tpu`. A plain `rdmem`/`wrmem` ignores
all three registers outright.

### 5.4 What it costs, and what it replaces

Two 16-bit counters (`col`, `row`), two running-sum offset registers, and a mux on each
address — no multipliers, since both offsets accumulate. Clocks per byte are unchanged: the
SRAM beat still dominates, and the permutation is free inside it. That survived the move to
ranges only because the controller takes a `stride`; a transposed spill is one range per
source row, `tcols` bytes at `tdrow` apart, and measures 2.03 clocks/byte at `tcols = 128`
against the 2.00 a dense spill gets.

What it replaces is `transpose_i8`, the byte-move loop that was the only way to transpose
before: `rdmem`/`wrmem` with `len = 1` around a two-deep software loop, **two dispatches per
byte** — about 8k of them for a `[24][128]` V, each paying full scalar-unit dispatch overhead
on top of the same SRAM access. One `.t` transfer is one dispatch.

The source and destination must be **distinct regions**: a transpose is not safe in place, and
nothing checks that for you. On-chip, `wrmem.t` + `rdmem` (or `wrmem` + `rdmem.t`) turns a
scratchpad tensor into its transpose via DRAM in two dispatches. A scratchpad-to-scratchpad
path would halve that traffic, but it needs the DMA to own both scratchpad ports at once and
is a separate change.

See [`examples/transpose_dma.tpu`](../../tpulang/examples/transpose_dma.tpu) for both
directions on a fused QKV block.

---

## 6. Address & timing facts to respect

- **Single clock domain.** The SRAM is asynchronous but its controller is clocked on the same
  `clk` as the DMA and scratchpad, so there is **no clock-domain crossing** here — unlike the
  comms link ([comms.md](comms.md) §4). That keeps v1 simple: no async FIFOs.
- **Arbitration is real now** (this bullet described v1). The engine takes
  `scratchpad_rgnt`/`scratchpad_wgnt` back from the arbiter and holds when
  denied; fill bytes wait in a four-entry skid buffer and, if that fills,
  `sram_dout_ready` stops the DRAM read stream rather than dropping a byte.
  The original v1 reasoning: the scalar unit blocks on `dma_done` (issue-and-wait), so
  while the DMA runs the MXU/VPU are idle and nobody else touches the scratchpad. The
  scratchpad's "DMA fills idle cycles" note ([scratchpad.md](scratchpad.md) §4) is about a
  future overlapped mode; v1 has the memory to itself.
- **Byte granularity handles any alignment / length.** Every byte still lands at its own
  absolute address on both sides, so unaligned bases and a `len` that isn't a multiple of 64
  just work — no first/last-word masking anywhere. That is the property §7 was careful not to
  give up.
- **A range is contiguous; a transposed transfer is not.** Ranges are therefore issued one
  per *source row*, and a row is what `tcols` says it is. In linear mode `tcols` defaults to
  `dma_len`, so there is exactly one range.

---

## 7. Throughput: ranges, not line buffers

**Built, and not the way this section originally planned it.** The plan was to buffer a
64-byte scratchpad line and drain it to the SRAM as 64 back-to-back byte transactions. What
was actually done inverts that: the *controller* learned to take a whole range, and the DMA
became a stream client. Same goal, less machinery, and none of the alignment bookkeeping the
line buffer would have needed.

The reason the line buffer looked necessary was a mis-attribution. The scratchpad was never
the problem — its port already takes one byte per clock at any alignment. The cost was in
`sram_controller`, which sequenced IDLE/ACCESS/WAIT per **byte**, and in the four DMA states
wrapped around each of those: 8 clocks per byte on a fill and 9 on a spill, to move a byte
the chip itself answers in 10 ns — an eighth of one clock.

So [`sram.sv`](../rtl/sram.sv) now takes `addr` + `len` + `stride` and streams:

| | before | after |
| --- | --- | --- |
| fill (DRAM → scratchpad) | ~8 clocks/byte | **1** |
| spill (scratchpad → DRAM) | ~8 clocks/byte | **2** (the write beat; see `sram.sv`) |
| `wrmem.t` spill | ~8 clocks/byte | **2.03** measured, at `tcols=128` |

Measured end to end on the whole adder model through `tb/tpu_top_tb.sv`, same program and
same golden vectors, byte-identical results: **1,226,722 → 541,590 clocks (2.27x)**. The
model is not purely memory-bound, which is why the whole-program figure is smaller than the
per-byte one.

`stride` is what keeps a transposed spill on the fast path. `wrmem.t` writes its destination
column-major, so consecutive bytes of a range are `tdrow` apart in DRAM; without a stride
those would be one range per byte and the mode would have gained nothing.

**What is still open.** The scratchpad's 64-byte width is genuinely unused — a line-buffered
fill would still cut its access count by 64x, but that access is free today, so it buys
nothing until something else wants the port. The real next step is **overlap**: prefetch the
next range while the current one drains, so SRAM traffic hides under compute instead of
blocking on `dma_done`. That needs the scalar unit to stop blocking, not more DMA states.

---

## 8. Open questions / decisions

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

## 9. Validation plan

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
   the §8 address-width decision).
6. **Handshake asserts.** `busy` frames the whole transfer, `done` is a single-cycle pulse,
   the DMA never pulses `sram.start` while `sram.busy` is high (the range would be lost), and
   `sram.done` on a fill always lands with a `sram.dout_valid` (the fill loop's byte count
   depends on it). All three are standing assertions in the bench, not one-off checks.
8. **Rate.** The four shapes the model actually moves — a 4 KB linear fill, a 4 KB linear
   spill, and the two transposing spills (`V^T` and a head's `A^T`) — verified byte-for-byte
   and then held to a clocks-per-byte bound derived from the controller's beat length, so a
   regression that is merely *slow* fails the bench rather than passing it quietly.
9. **Both beat lengths.** `make TEST=dma sim` and `make TEST=dma IVFLAGS=-DCPA=2 sim`:
   `CLOCKS_PER_ACCESS > 0` is the only configuration that exercises the spill path's holding
   register, because at the default the offer is always taken immediately.
7. **Transpose (§5).** Both directions against an explicit `[r][c] -> [c][r]` reference written
   as the permutation, not as a second address generator — a shared bug in the two would
   otherwise cancel. Square and non-square; a strided source slice (`tsrow > tcols`, the V case)
   and a strided destination; a `len` that is not a whole number of rows, which must stop
   mid-row and write nothing past it; the zero-geometry case, which must stay a plain copy; and
   a spill/fill round trip, which must be the identity. Every *linear* case in the bench runs
   with the geometry registers at 0, so they also prove an unflagged copy ignores them.

---

## 10. Build order

1. ~~**v1 byte-serial FSM** (§3–§4) against the §9 bench.~~ **done**
2. ~~**Rewire `tpu_top`** (§8): DMA between scalar dispatch, scratchpad `dma_*`, and a
   `sram_controller` instance.~~ **done** — the host reaches memory over the UART, so
   `host_mem_*` is gone.
3. ~~**Transpose mode** (§5), with the geometry in `cfg tcols/tsrow/tdrow`.~~ **done**
4. **Widen the DRAM address** (§8) so the full SRAM is reachable. Not done — the scalar unit
   still emits 16 bits, so a program addresses the low 64 KB.
5. ~~**Line-buffered mode** (§7) for throughput~~ — **done as ranges instead** (§7):
   `sram.sv` takes `addr`/`len`/`stride`, the DMA streams it, 1 clock/byte on fills and 2 on
   spills. Prefetch/double-buffer is still open and now needs a non-blocking dispatch more
   than it needs DMA states.
6. Optional: a **scratchpad-to-scratchpad** path, which would take the DRAM round trip out of
   an on-chip transpose (§5.4).
