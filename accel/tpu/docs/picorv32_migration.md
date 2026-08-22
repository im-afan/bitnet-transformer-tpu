# Migrating the scalar unit to PicoRV32 + macro-op dispatch

**Status: phases 0-5 are done, including the model. The scalar unit,
`assembler.py`, `gen_vectors.py`, `torch_ref.py`, `pytpu.py` and every
`examples/*.tpu` have been deleted — the CPU is the only producer, and
[`../fw/adder.c`](../fw/adder.c) is the whole four-layer int4 model in 518
commands and 1544 bytes of RISC-V. `adder_export.py` came back as the firmware
kernel's host: it derives the requant table from a checkpoint, compiles the
kernel natively against it, and scores the trace on `iss.py`. Phase 6 (a second
architecture as a new `.c` file) is the remaining one.**

| Phase | State |
| --- | --- |
| 0 | **done** - counters reworked (§9.5); `swait` supplemented by `idlec`/`qfull`/`ovlap` |
| 1 | **done** - `cmd_queue.sv` + `cmd_{mxu,vpu,dma}.sv`; `scalar_unit.sv` packs its config registers into commands and pushes them |
| 2 | **done** - scratchpad grants, VPU stall, DMA skid buffer + `sram.sv` backpressure, queue depth 8 |
| 3 | **done** - no interpreter (§8.1): `fw/tpu.h` gains `-DTPU_TRACE`, `fw/mock/` emits the trace from a native build, `iss.py` gains `exec_command`/`run_trace`, `fw_vectors.py` turns a trace into golden images + an expected command trace, and `fw_matmul_tb.sv` monitors `p_cmd_*` and diffs both |
| 4 | **done (RTL)** - `cpu_subsys.sv`: PicoRV32 + AXI4-Lite + the MMIO command aperture, alongside the scalar unit. The C toolchain now exists too (`../fw/`, §8): `matmul.c` (hardware tile walk) and `matmul_loop.c` (the same product, tile grid walked in C) are the kernels and `host/run_fw_matmul.py` runs either. Both **build** (Homebrew `riscv64-elf-gcc` 16.2.0, 248 and 352 bytes of the 16 KB) and both **pass in simulation** through `tb/fw_matmul_tb.sv` (`make fw`) — see §9.6. Not yet run on the board. |
| 4 (rest) | **done** - `tpu.h` has builders for all three units (VPU + transposing DMA), and the model is written against them: `ffn.c`, `mha.c`, then `adder.c` — four transformer layers and the output head, 518 commands, 439 917 clocks, byte-exact against the ISS through `make fw FWPROG=adder`. Row-major weights swapped which attention operand needs the transpose; see that file's header |
| 5 | **done** - `scalar_unit.sv` and the whole tpulang toolchain deleted; `tpu_top.sv` has one producer, one S-port requester and no imem/cfg path; `iss.py` kept for its op bodies with the instruction decoder stripped |
| 6 | not started - a second architecture as a new `.c` file and no RTL change |

What passes today, on the tree as it stands:

| Test | Result |
| --- | --- |
| `make TEST=mxu` / `vpu` / `scratchpad` / `sram` | 208 / 357 / 50 / 4872 checks, 0 errors |
| `make cosim` (the real host driver against the RTL) | 11 of 11 |
| `make TEST=dma` (incl. new contention tests) | 16 737 checks, 0 errors |
| `make TEST=cmd_queue` (new) | 123 checks, 0 errors |
| ~~`make examples`~~ | **gone with phase 5** - the ten `.tpu` kernels and the assembler that built them were deleted. `make fw FWPROG=...` over the four C kernels is what covers the dispatch plane now |
| `make uart` (two programs, one link, no reset between) | 469 checks, 0 errors |
| `make TEST=cpu_smoke` (new) | 68 checks, 0 errors - PicoRV32 boots, pushes two DMA commands, 64-byte round trip byte-exact |
| `make fw` - the C firmware through `tpu_top`, image via `FW_INIT`, golden vectors + expected command trace from the kernel's own native trace (§8.1) | **539 / 574 checks, 0 errors** on `matmul` / `matmul_loop`; halts after 2 210 / 2 587 clocks. Checks the DRAM image *and* the command stream, 5 / 12 commands matched |
| `make fw FWPROG=ffn` / `mha` | 0 errors - the feed-forward block and one attention head, the first VPU and transposing-DMA commands from firmware |
| `make fw FWPROG=adder` - **the whole model** | **526 879 checks, 0 errors**, halts after 439 917 clocks; 518 of 518 commands matched |
| ~~`make model [LAYERS=n]`~~ | **gone with phase 5** - `tpu_top_tb.sv` drove the scalar unit. `make fw FWPROG=adder` replaces it and checks strictly more (the command stream as well as the image) |
| `make fwsweep` - both kernels over 11 shapes, vectors regenerated per shape | 22 of 22, 0 failures (largest: 260 commands) |
| `make all` | 15 of 15 |

`make examples` is the one that matters most: between them the ten kernels cover
every path a dispatch can take through the new plane - single-tile `matmul`,
software tiling, the hardware tile loop and its config strides, `vecmatmul` and
its geometry command, runtime `setcfgr`, the transposing DMA, and DRAM above
64 KB. All byte-identical to what `iss.py` produced, unmodified.

Two make targets were added for this, and they are the loop worth using:
`make examples` (all ten kernels, a few minutes) and `make model [LAYERS=n]`
(the whole adder kernel; `LAYERS=1` is ~182 k clocks and exercises every code
path the shipped four-layer one does). A long run prints nothing until it halts,
so `tpu_top_tb` also gained `+HEARTBEAT` — a frozen PC with a unit stuck busy is
a hang, a moving one is just slow. That distinction is not academic: it is what
localized the one real bug found during this work, a broken tile loop that
looked exactly like a slow simulation for forty minutes.

### What the command plane actually cost

The four-layer run is the like-for-like comparison against §0's baseline, same
program, same vectors, same 8x8 geometry:

| | baseline | with the command plane | delta |
| --- | --- | --- | --- |
| run | 690 705 | **693 107** | **+2 401 (+0.35%)** |
| MXU | 405 433 | 404 409 | -1 024 |
| VPU | 52 160 | 51 648 | -512 |
| DMA | 226 198 | 227 820 | +1 622 |
| residual = issue overhead | 6 914 (1.00%) | **9 229 (1.33%)** | +2 315 |

Read that as three separate effects, because they pull in different directions:

- **Issue overhead is +2 315 clocks**, from 1.00% of the run to 1.33%. That is
  the real cost of the plane: a dispatch now spends a clock in `S_PUSH` per
  command (two for a matmul, which pushes geometry as well), plus two more in
  `S_RQRD`/`S_RQLATCH` when its `{m0,n}` operand is an address, plus a clock in
  the unit's front end. It is smaller than §9's estimate for the *CPU* producer
  (~2.7%) for the obvious reason: the scalar unit packs a command from registers
  in one clock, where firmware pays four stores.
- **The units got faster**, by 1 536 clocks between them, because the requant
  literal deleted a scratchpad round trip from every requantizing dispatch.
- **The DMA got slower**, by 1 622 clocks, because a fill now drains its skid
  buffer before the range is declared finished. That is the price of being able
  to overlap at all, and it is paid per range — which is why the transposing
  transfers (one range per row) carry most of it.

Net **+0.35%**, byte-identical output. That is the number to hold against the
~1.45x that §9.3 says the overlap is worth, none of which is claimed yet: the
scalar unit still waits after every dispatch, and `ovlap` reads 0.

**One measured side effect worth recording.** `tiled_matmul_hw` (two matmuls)
spends 578 clocks in the MXU against the 582 in `macro_ops.md` §8's phase-0
table. The 4 clocks are exactly the two states per requantizing matmul that the
`{m0,n}` literal deleted - the array no longer stops to fetch its requant word
over the C port before draining the result buffer.

Two measurements worth recording from the UART run, because they are what says
phase 1 preserved the old behaviour: `ovlap = 0` and `qfull = 0` on both
programs. The scalar unit still waits after every dispatch, so no two units ever
run at once and the queue never backs up - the command plane is in the path and
is provably not yet being used for anything. `idlec` reads 92 of 526 clocks on
`relu_layer`, which is exactly `run - (mxu + vpu + dma)`.

**What is deliberately still true:** the tpulang ISA is unchanged. Same opcodes,
same config registers, same programs, same golden vectors, same `iss.py`. The
config file did not go away - it moved from being *read by the units* to being
*packed into commands by the producer*, which is what removes the stale-config
hazard while leaving every existing program running.

Goal: write kernels for model architectures other than the adder in a real language, without
writing a compiler. Replace `scalar_unit.sv` with a PicoRV32 core; replace the global config
register file with self-contained 128-bit macro-ops the CPU pushes into per-unit command
queues over AXI4-Lite.

This supersedes [`macro_ops.md`](macro_ops.md) §1's rejection of PicoRV32 ("it costs LUTs we
don't have at 92% utilization and replaces the working, bit-exact `iss.py` chain"). The first
premise is stale — the design is **13 342 of 20 800 LUTs (64%)** today, not 92%
([`synth.md`](synth.md) §5). The second is still live and §8 is the answer to it.

---

## 0. The numbers, up front

Measured baseline, four-layer adder model through `tpu_top_tb`
(`../../tpulang/adder_kernel.md` §7.7):

| | clocks | share |
| --- | --- | --- |
| whole run | 690 705 | 100% |
| MXU | 405 433 | 58.7% |
| DMA | 226 198 | 32.8% |
| VPU | 52 160 | 7.6% |
| **residual = scalar unit issuing instructions** | **6 914** | **1.0%** |

The residual is the interesting one. The scalar unit's FSM is `FETCH → DECODE → EXEC`, three
clocks per instruction, so 6 914 clocks is **≈2 300 dynamic instructions** — which
independently matches a hand count of the kernel (≈563 per layer × 4, plus prologue). Of
those, ≈540 are dispatches and ≈1 760 are `li`/`adds`/`cmps`/`blt`/`setcfg` overhead.

Estimated after migration (§9 shows the arithmetic):

| | clocks | share of run |
| --- | --- | --- |
| CPU issue, today | 6 914 | 1.0% |
| CPU issue, PicoRV32 + 128-bit macro-ops | **≈19 000** | **2.7%** |
| ...worst topology (fetch also over AXI4-Lite) | ≈29 000 | 4.0% |

**So instruction overhead costs 1–3% of the run, and the queue that the overhead pays for is
worth ≈1.5× (§9.3).** That asymmetry is the whole argument. It inverts at `T=1` decode, which
is where the format width starts to matter (§9.4).

---

## 1. What binds today

Three limits, none of which are hardware:

| Limit | Value | Where it bites |
| --- | --- | --- |
| Instruction memory | 1024 words | one pre-macro-op quantized layer was 861 of them |
| Scalar registers | 31, no spilling | `gen_adder_model.py` peaks at **29 of 31** live in the head loop |
| No compiler | — | `adder_model.tpu` is 901 lines for one model; `pytpu.py` is 528 lines of staged emitter to make that writable |

The current kernel is 226 words, so IMEM is not binding *for this model*. It binds again for
anything with more structure — MoE, GQA with `heads_per_q > 1`, a KV cache, more than one
attention variant in one program.

Global config registers are the other tax. 18 assigned indices, **62 static `setcfg` in
`adder_model.tpu`**, and the failure mode is silent: `macro_ops.md` §4.0 records a real bug
where one program's leftover `arow`/`crow`/`wcol` corrupted the next program's plain `matmul`.
The fix there was a second opcode (`matmul_t`) that ignores the registers — a workaround for
global mutable state, not a removal of it.

What C buys, concretely: gcc does register allocation and the stack, so the 31-register
ceiling goes away; 16 KB of firmware is ~4 000 RV32I instructions against 1024 words today;
functions and structs make `linear()`, `attention_head()`, `dyt_residual()` reusable across
kernels; and the "compiler" the project has to maintain is a ~200-line MMIO header.

---

## 2. Scope

**Removed.** The config-register inputs on `mxu.sv` / `vpu.sv` / `dma.sv`, and the
scratchpad round trip for the requant `{m0,n}` word (`mxu.sv` lost two FSM states,
`vpu.sv` lost two). `scalar_unit.sv` and its config file survive as a *producer* —
phase 5 retires them, and until the firmware toolchain exists they are what keeps
the `.tpu` suite running.

**Added.** PicoRV32 + firmware BRAM, an AXI4-Lite fabric, one command decoder + queue per
unit, a status/sync block, a scratchpad AXI window, and backpressure on the scratchpad's
arbitrated ports (§5).

**Unchanged.** The PE array, the VPU lane datapath, the requant fixed point, `sram.sv`,
`dma.sv`'s range engine, the scratchpad's banking, the UART link, and **every numeric
convention in `isa.md` §A.5** — ternary packing, `{m0,n}`, the int8/int32 stride contract.
The units keep their existing `start`/`busy`/`done` handshakes; only where their operands come
from changes.

**Out of scope.** The inter-TPU LINK, changing the array geometry, and any change to what the
model computes.

---

## 3. The macro-op format

One **128-bit command**, four RV32 stores, uniform across all three units. §9.2 prices the
alternatives; uniform 128 costs about 1% of total runtime against a tightly packed
variable-width encoding, and is worth it.

A command is four 32-bit words, `word0` first: `cmd[31:0]`, `cmd[63:32]`, `cmd[95:64]`,
`cmd[127:96]`. The header lives in the **low** bits of `word0`, not the high bits of the
128 — so a producer builds it with an `ori` against a small constant rather than a shift,
and the first store carries it:

| `word0` bits | Field | Meaning |
| --- | --- | --- |
| 7..0 | `op` | command selector, per-unit namespace |
| 15..8 | `flags` | `.acc` / `.rq` / `tiled` / `write` / `.t`, per op |
| 31..16 | — | the first operand field (an address, in every command that has one) |

As built (`cmd_mxu.sv`, `cmd_vpu.sv`, `cmd_dma.sv`):

| Command | `op` | `word0[31:16]` | `word1` | `word2` | `word3` |
| --- | --- | --- | --- | --- | --- |
| `MXU_GEOM` | `0x01` | `arow` | `wcol:crow` | `tlen(6) : ntiles(8) : ktiles(8)` | — |
| `MXU_MM` | `0x02` | `out` | `wgt:act` | `{n,m0}` | — |
| `VPU_OP` | `0x01` | `dst` (`flags[4:0]` = the `VOP_*` code) | `src1:src0` | `{n,m0} : vlen(10)` | — |
| `VPU_GEOM` | `0x02` | `vrows` | `vrow0:vcols` | `vcrow:vrow1` | — |
| `DMA_MOVE` | `0x01` | `spad` (`flags[0]`=write, `flags[1]`=`.t`) | `dram(19)` | `tcols:len` | `tdrow:tsrow` |

An unknown `op` is discarded with a simulation message rather than executed, so a stale
command stream fails loudly instead of running the wrong instruction — the same principle
as the ISA's retired-opcode holes.

Three things fall out of the current ISA:

**The requant word travels in the command, not in the scratchpad.** `{m0,n}` is 12 + 4 = 16
bits, exactly the width of the scratchpad address that currently points at it. Same command
budget, one less indirection, one less scratchpad read per dispatch — and it deletes the `RQW`
block, the 16 words the host stages per layer, and every `li rq_word` in the kernel.

**MXU geometry does not fit alongside MXU addresses.** Full field count for a `matmul_t` is
`op/flags` + three 16-bit addresses + `{m0,n}` + `tlen`/`ktiles`/`ntiles` +
`arow`/`crow`/`wcol` = **146 bits**. Squeezing to 128 needs 12-bit strides and 6-bit tile
counts and lands at exactly 128 with zero margin, which is not worth doing. Split it instead:

| Command | Fields | Used |
| --- | --- | --- |
| `MXU.GEOM` | `tlen`(6), `ktiles`(8), `ntiles`(8), `arow`(16), `crow`(16), `wcol`(16) | 70 of 112 |
| `MXU.MM` | `out`(16), `act`(16), `wgt`(16), `{m0,n}`(16), flags `.acc`/`.rq` | 64 of 112 |
| `VPU.OP` | `dst`(16), `src0`(16), `src1`-or-`{m0,n}`(16), `vlen`(10), macro-op strides | 58–130 |
| `DMA.MOVE` | `spad`(16), `dram`(19), `len`(16), `tcols`(16), `tsrow`(16), `tdrow`(16) | 99 of 112 |

`VPU.OP` needs a second entry only for `vecmatmul` (`vrows`/`vcols`/`vrow0`/`vrow1`/`vcrow` =
80 more bits). Nothing in the shipped kernel issues `vecmatmul`, so the long form is a
`VPU.GEOM` companion on the same pattern as the MXU's, not a special case in the fast path.

**Sticky geometry is queue state, not global state — and that is the actual fix.** `MXU.GEOM`
persists until the next one, but it flows through the MXU's own in-order queue, so its scope
is a position in that unit's command stream rather than a register any other unit or any
earlier program can have written. The C driver caches the last geometry it emitted and skips
the `GEOM` command when nothing changed; in the current kernel that collapses 14 matmuls to
**10 GEOM + 14 MM per layer** (the three QKV projections share one geometry, and `Wo`/`W1`/`W2`
reuse it after the head loop).

The staleness check therefore lives in ~10 lines of the driver header, where it is testable,
instead of in the programmer's head — which is where `macro_ops.md` §4.0's bug lived.

**Commit-on-last-word.** Each unit's command port is a 4-word aperture; the write to offset
`0xC` enqueues. Four stores are atomic with respect to the queue with no separate trigger
store, and a partial write is detectable.

---

## 4. Queues and synchronization

One FIFO per unit, depth 4–8 entries. **A full queue stalls the store** (native `mem_ready`
low / AXI `BVALID` withheld), so flow control needs no software check — the CPU throttles
itself naturally and only *data* dependencies need explicit waits.

Sync is by sequence number, not by "is the unit idle":

| Register | Meaning |
| --- | --- |
| `ISSUED[unit]` | commands accepted into the queue |
| `RETIRED[unit]` | commands completed |
| `SPACE[unit]` | free entries (advisory; the stall is the real mechanism) |

Waiting is `while (RETIRED[u] < my_seq);` — poll cost is `lw` + `bltu` ≈ 8 clocks. Waiting on
a *specific* command rather than on unit idleness is what makes double-buffering expressible:
"start the DMA for layer L+1's weights, then wait only for the DMA that filled layer L's."

**Every cross-unit dependency becomes software's responsibility.** Issue-and-wait made them
automatic; queues do not. Two mitigations, both cheap:

- The driver ships a **strict mode** where each dispatch is followed by its own wait. A kernel
  in strict mode is cycle-for-cycle equivalent to today's model, so it is the bring-up
  configuration and the correctness baseline; relaxing it is a separate, measurable step.
- A **sim-only range tracker** in `tpu_top_tb`: record the scratchpad byte range each command
  writes, and assert on a read that overlaps an in-flight write from another unit. This is the
  same shape as the exclusivity assertion `scratchpad.sv` already carries, and it catches the
  entire class of bug at the point of failure rather than as a wrong logit.

---

## 5. The scratchpad exclusivity invariant is the real blocker

`scratchpad.sv`'s header states it plainly: six readers are muxed onto one read port and four
writers onto one write port by fixed priority, with **no stall handshake**, and this is exact
only because

> the scalar unit is issue-and-wait: a dispatch op parks it in `S_WAIT` until the unit reports
> done, so MXU, VPU and DMA activity never overlaps.

Queues delete that premise. There is a simulation assertion that will fire loudly, which is
the good case; the bad case is silently dropped accesses on hardware.

What overlap actually needs:

- Reads and writes are separately arbitrated, and the case that matters most — a DMA **fill**
  (writes) behind MXU **streaming** (`A_rd`/`W_rd` reads) — collides only on the MXU's `C`
  writeback cycles. A DMA **spill** reads and collides directly.
- `dma.sv` touches the scratchpad **once per byte** (`FILL` drains `sram_dout`, `SPILL_RD`
  addresses one byte), so its port occupancy during a transfer is high, not sparse.
- The DMA can absorb a stall — it is already gated on a 1 byte/clock SRAM handshake — but
  `sram_dout_valid` has no `ready`, so absorbing one needs a small skid buffer.

Recommendation: **give MXU and VPU un-stallable priority, add a grant handshake to the DMA and
scalar/CPU ports only, and put a 2–4 entry skid buffer in `dma.sv`.** MXU and VPU FSMs are
untouched. Estimated 150–300 LUTs. This is a prerequisite for the §9.3 win, and it is worth
doing on its own merits even without the CPU change.

---

## 6. PicoRV32 instantiation and bus topology

Core config (`picorv32` parameters):

| Parameter | Setting | Why |
| --- | --- | --- |
| `BARREL_SHIFTER` | 1 | shifts drop from 4–14 clocks to 3; address math uses them constantly |
| `ENABLE_FAST_MUL` | 1 | DSP multiplier; the alternative is 40 clocks per `mul`, and tile-offset math multiplies |
| `ENABLE_DIV` | 0 | no kernel needs it |
| `ENABLE_COUNTERS` | 1 | `rdcycle` gives the firmware self-profiling for free |
| `COMPRESSED_ISA` | 1 | ~30% smaller firmware; costs LUTs — make it the first thing dropped if area is tight |
| `ENABLE_IRQ` | 0 | polling `RETIRED` is simpler and the latency does not matter |
| `ENABLE_REGS_DUALPORT` | 1 | default; do not turn off |

Two topologies. The difference is ~2% of total runtime (§9.2), so start with the simple one.

**A — `picorv32_axi`, everything on one AXI4-Lite bus.** Firmware RAM, command ports, status
block and the scratchpad window are all AXI4-Lite slaves. Simplest, matches the request, one
interface to document. Cost: every instruction fetch pays the AR/R handshake, so CPI goes from
~4 to ~6.

**B — native `picorv32` + tightly coupled RAM + a native→AXI4-Lite bridge on the MMIO
aperture.** Fetch and data at zero wait states; only the ~4 stores per dispatch see AXI. Costs
one small bridge and one address decoder.

Recommendation: **build A, keep B in the back pocket.** Switch when the starvation counter
(§9.5) says CPU issue is over ~10% of the run — i.e. when decode or a faster datapath makes it
matter, not before.

Address map:

| Base | Size | Contents |
| --- | --- | --- |
| `0x0000_0000` | 16 KB | firmware: code + data + stack, BRAM, host-loadable |
| `0x8000_0000` | 16 B | MXU command port (commit on `+0xC`) |
| `0x8000_0100` | 16 B | VPU command port |
| `0x8000_0200` | 16 B | DMA command port |
| `0x8000_0300` | 64 B | `ISSUED`/`RETIRED`/`SPACE` per unit, `DONE`, perf counters |
| `0x9000_0000` | 64 KB | scratchpad, 32-bit word window onto the existing `S_rw` port |

The scratchpad window replaces `loads`/`stores` and is worth more than the instruction it
replaces: it lets a kernel compute requant words at run time, read back a reduction, and
— with an argmax loop in C — end the program with a token id instead of 2 KB of int32 logits.
Embedding lookup could follow, which would remove the host from the decode loop entirely.

A DRAM window is deliberately **not** mapped. The SRAM is byte-wide at 1 clock/byte; CPU
access to it would be a trap. DRAM stays the DMA's.

---

## 7. Host protocol and boot

The UART protocol survives almost intact ([`uart_host.md`](uart_host.md), `uart_interface.sv`):

| Command | Today | After |
| --- | --- | --- |
| `'I'` | write instruction BRAM, 10-bit address | **13-bit** word address: 12 index bits + `boot_pc[12]` selecting IMEM or firmware RAM |
| `'G'` | pulse `host_run` with `boot_pc` | same bit selects; set = release CPU reset. As built the reset vector is the fixed `PROGADDR_RESET = 0`, so the index bits are ignored for the CPU |
| `'R'`/`'W'` | SRAM read/write while idle | unchanged |
| `'T'` | perf counter block | unchanged shape, different counters (§9.5) |

`done` comes from `ebreak` → PicoRV32's `trap` output, latched. The core is held in reset
while `!busy`, which preserves the existing "host may only touch memory while idle"
arbitration in `tpu_top.sv` unchanged.

One regression to expect: 16 KB of firmware at 115 200 baud is ~1.4 s to load, against ~80 ms
for 226 words today. Load only the used portion (the `.bin` length, not the whole aperture)
and the edit loop stays interactive.

---

## 8. Toolchain and verification

This is the part `macro_ops.md` §1 was right to worry about. `iss.py` is bit-exact with the
RTL and `gen_vectors.py` turns it into the golden vectors `tb/` replays; a RISC-V core cannot
be modelled by `assembler.py` + `iss.py` as they stand.

**Move the verification contract from the instruction stream to the command stream.** The
macro-op trace — the ordered list of 128-bit commands per unit — becomes the ISA boundary.
Three producers must agree on it:

1. **The firmware kernel, compiled natively against a mock `tpu.h`.** Not an RV32IM
   interpreter — see §8.1. `tpu.h` already funnels every MMIO access through two inline
   functions, so `-DTPU_TRACE` swaps those for a trace emitter and the *actual* `.c` source
   becomes the producer, with its real control flow, compiled by the host compiler.
2. **`iss.py`, re-fronted.** Its `_matmul` / `_vpu` / `_dma` bodies are the golden numerics and
   are kept **verbatim**; only the decode front end changes, from 32-bit words to 128-bit
   commands. The scratchpad/DRAM images it produces are unchanged in meaning.
3. **RTL.** A sim-only monitor on the arbitrated command write (`tpu_top.sv`'s `p_cmd_*`,
   which is where both producers converge) dumps the trace from the testbench.

`gen_vectors.py` becomes 1 → 2 → expected images, as before. `tpu_top_tb` then checks *two*
things instead of one: the memory image, and 3 against 1. That second check is strictly new
capability — a mismatch localizes immediately to either the CPU/firmware or the datapath,
which is a better debug loop than the current one, not a worse one.

`torch_ref.py` and `adder_export.py` are unaffected: they compare final tensors, not
instructions.

### 8.1 Why there is no RV32IM interpreter

The original plan here was a ~400-line Python RV32IM interpreter that would run the firmware
image and capture its MMIO stores. It was dropped, and the reasoning is worth keeping because
it applies to anything else that wants to "model the CPU".

**The interpreter solves a problem you only have if you insist the producer must be a
binary.** The command stream is a pure function of the kernel's control flow, and that control
flow is ordinary C — `matmul.c` is integer arithmetic and five builder calls. Every
target-specific thing in the whole firmware lives in two inline functions:

| `tpu.h` | native build does |
| --- | --- |
| `tpu_push` (four volatile stores) | append `(unit, w0, w1, w2, w3)` to a trace |
| `tpu_wait` (two volatile loads + spin) | emit a barrier marker |
| `TPU_DONE` | nothing |

Everything else — `tpu_dma`, `tpu_mxu_geom`, `tpu_mxu_mm`, and every builder added later — is
plain arithmetic over `uint32_t` and comes along for free. `uint32_t` wraps identically on
x86-64 and RV32, and device addresses are `uint32_t` rather than pointers, so the arithmetic
matches without an ABI argument.

**A hand-written Python mirror of each kernel was considered and rejected** for the same
reason the interpreter was: it puts a transcription between the source and the test. Two
implementations maintained by one person drift, and if the mirror is written by reading the C
then a misunderstanding of intent is duplicated into both — the two "independent" producers
agree on the bug and the test passes. Compiling the real `.c` has the simplicity of the mirror
with none of the drift, and it preserves phase 6's promise: a new architecture stays *one* new
`.c` file, not a `.c` plus a `.py`.

**What this gives up**, honestly: RV32-specific execution semantics. A `div` under
`ENABLE_DIV(0)`, a stack overflow, a linker-script mistake — native compilation runs all of
those happily where the core faults. Every one is loud in the RTL run (a trap halts the core,
a bad image fails the compare), so they are *detected*; they are just localized less precisely
than an interpreter would. That is the trade: sharp diagnosis of rare bugs, for deleting 400
lines and a class of drift.

**What must not be dropped with it** is producer 3. The interpreter's real payoff was never
"run the firmware" — it was the command-by-command diff that localizes a failure to
producer-vs-datapath. That benefit is orthogonal to how the reference trace is produced, so it
survives here for free, but only if the RTL monitor is built and the reference emits a
*trace*, not just final images. A reference that emits images only puts you back in the
pre-phase-3 debug loop: image mismatch, no idea which command.

What dies: `assembler.py` (618 lines), `pytpu.py` (528), the `.tpu` language, every
`examples/*.tpu` and its golden vectors, and `isa.md` §§3–6 and A.2–A.4. That is the honest
price and it should be paid deliberately — see the phase plan, which keeps the scalar unit
alive alongside PicoRV32 through phase 4 precisely so the existing suite stays a live
regression while the new path is proven.

In the tree: `accel/tpu/fw/` — `start.S`, `link.ld`, `tpu.h` (the MMIO aperture and one
builder per command), `bin2hex.py`, one `.c` per kernel, and a Makefile that autodetects
the cross prefix. `riscv-none-elf-gcc` (xPack) works on Windows and macOS; Homebrew's
`riscv64-elf-gcc` is the shorter path on macOS and is what the images in the tree were
built with (16.2.0, binutils 2.47, `brew install riscv64-elf-gcc`, bottled — no source
build). Nothing outside the compiler is needed: `-nostdlib` means the newlib the formula
drags along is never linked, so the multilib set does not matter either.

The `-march` is **`rv32ic_zmmul`**, not `rv32imc`: `cpu_subsys.sv` builds the core with
`ENABLE_FAST_MUL(1)` but `ENABLE_DIV(0)`, so `mul` is real and `div`/`rem` decode as an
illegal instruction and trap. `-mabi=ilp32 -nostdlib -ffreestanding`, no libc and no
libgcc — a kernel that wants division needs the RTL parameter first. (A gcc older than 12
has no `zmmul`; `rv32ic` works there if the kernel multiplies nothing at run time.)

Two kernels so far, on the same `[8x32] @ [32x16]` problem and the same addresses, so
`host/run_fw_matmul.py` checks either: `matmul.c` issues one `MXU_MM` with `tiled` and
`ktiles/ntiles = 4/2`, and `matmul_loop.c` walks that grid in C as 8 single-tile
dispatches, `.acc` across the contraction. The pair mirrors
`examples/tiled_matmul_hw.tpu` vs `examples/tiled_matmul.tpu`, and is the cheapest way to
measure what the hardware tile loop buys once the producer is a CPU — the software version
pays 8 dispatch issues plus an int32 C round trip through the scratchpad per contraction
tile, against one issue and one store per output tile. `matmul_loop.c` still sets the
`tiled` flag: it selects the configured strides over the single-tile constants, and at
counts of 1 the hardware loop runs one pass. Neither has been executed yet.

---

## 9. Performance notes — instruction overhead

### 9.1 Where the estimate comes from

Per-layer dispatch inventory of `adder_model.tpu`, counted from the source:

| Unit | dispatches / layer | clocks / layer | clocks per dispatch |
| --- | --- | --- | --- |
| MXU | 14 | ≈101 358 | ≈7 240 |
| VPU | 112 | ≈13 040 | ≈116 |
| DMA | 9 | ≈56 550 | ≈6 283 |
| **total** | **135** | ≈170 950 | |

PicoRV32's published cycle counts: ALU reg/imm and not-taken branch 3, taken branch 5, load 5,
store 5, `jal` 3, `jalr` 6 — average CPI ≈ 4. *Verify these against the version you vendor
in.*

Cost of one 128-bit command = 4 stores (5 clocks each) plus operand setup. In a chunk loop
three of the four words are loop-invariant and live in registers, so the body is 4 `sw` + one
`addi` ≈ 23 clocks; straight-line sites that materialize constants are ≈32. Use ≈26.

| Per layer | commands | clocks |
| --- | --- | --- |
| MXU (10 `GEOM` + 14 `MM`, after driver caching) | 24 | ≈770 |
| VPU | 112 | ≈2 900 |
| DMA | 9 | ≈290 |
| loop counters, pointer math, branches (~250 instr @ 3.5) | — | ≈875 |
| **total** | 145 | **≈4 650** |

Four layers plus prologue/epilogue ≈ **19 000 clocks**, against 6 914 today. **2.7× more issue
cycles; 1.0% → 2.7% of the run; +1.8% wall clock under strict mode.**

### 9.2 What the format choices cost

| Choice | Δ clocks | Δ run |
| --- | --- | --- |
| uniform 128-bit vs. a packed 64-bit VPU short form | +5 700 | +0.8% |
| MXU without driver-side geometry caching (24 → 28 commands/layer) | +400 | +0.06% |
| AXI4-Lite store handshake, +2 clocks per store (topology A vs B) | +4 600 | +0.7% |
| AXI4-Lite instruction fetch, CPI 4 → 6 (topology A vs B) | +10 000 | +1.4% |

Every row is under 1.5%. **The simplest version of every choice is affordable at this
arithmetic intensity** — which is the useful conclusion, because it means the design can be
built the obvious way and optimized later against a measurement rather than a guess.

The one row worth taking anyway is geometry caching: it costs ~10 lines of driver and removes
a whole class of "did I set `crow` for this dispatch?" bug.

### 9.3 What the queue buys — the reason to do this

DMA is **226 198 clocks, 32.8% of the run, entirely serialized behind compute** because
issue-and-wait cannot express anything else. Per layer, DMA (56 550) is smaller than MXU
(101 358), so with a command queue and a double-buffered weight window nearly all of it hides:

| | clocks | |
| --- | --- | --- |
| today | 690 705 | |
| DMA fully hidden behind MXU+VPU | ≈458 000 | **1.51×** |
| ...minus the ≈19 000 of CPU issue | ≈477 000 | **1.45×** |

Three preconditions, all real:

- Scratchpad backpressure (§5). Without it, overlap is silent corruption.
- A second weight window: `WBUF` is 12 288 bytes and the scratchpad map runs to `0xCC40`,
  leaving ~13.2 KB free — just enough, with nothing to spare. Fetching `Wq`/`Wk`/`Wv`
  separately instead of as one 12 KB block relaxes it.
- The kernel has to actually issue the prefetch and wait on the right sequence number. This is
  the part that is now expressible and was not before.

**Instruction overhead costs 1–3%; the mechanism that adds it is worth ~45%.**

### 9.4 Where the overhead stops being second-order

Issue cost is per-dispatch and roughly constant; unit work scales with the tensor. So the
ratio moves against the CPU whenever tensors shrink or the datapath speeds up:

| Regime | CPU share of run |
| --- | --- |
| today, `T=32` prefill | ≈2.7% |
| datapath 4× faster (bigger array, faster DMA) | ≈11% |
| `T=1` incremental decode | ≈5–8%, and rising with kernel bookkeeping |

Decode is the interesting one. VPU dispatches per layer fall 112 → ~14 (the chunk loops go
from 8 iterations to 1) and MXU work collapses toward the weight-load bound — `macro_ops.md`
§9.3 puts decode MXU utilization at ≈`1/(ROWS+COLS)` ≈ 6%. Issue cost falls much less than the
work does. The tightest issue-rate case is already visible today: a VPU dispatch is ≈116
clocks of work against ≈26 clocks of issue, only **4.5× headroom**, and that is the one place
where a short command form or a deeper queue would earn its keep. MXU has 113× headroom and
never will.

### 9.5 What to measure

`perf_counters.sv` keeps working — it watches unit `busy` signals, not the scalar unit. But
`su_wait_cycles` becomes meaningless and should be replaced by the counter that actually
answers this section's question:

| Counter | Answers |
| --- | --- |
| `starved[unit]` — unit idle **and** its queue empty | is the CPU keeping up? this is *the* instruction-overhead metric |
| `qfull[unit]` — CPU stalled on a full queue | is the queue deep enough? |
| `overlap` — cycles with ≥2 units busy | is the §9.3 win being realized? |

**`qfull` as built does not answer its question.** `tpu_top.sv` samples
`p_cmd_we & p_cmd_full`, but *both* producers already gate their own write enable with
`!cmd_full` — `scalar_unit.sv:522` and `cpu_subsys.sv:177` — so the term is unsatisfiable
and the counter reads 0 by construction, not by observation. Every "`qfull = 0`" in this
document is therefore evidence of nothing. The fix is to sample the stall rather than the
accepted write (`state == S_PUSH && cmd_full`, and the equivalent commit-pending term in
`cpu_subsys`); until then, use the queue level or `idlec` and treat `qfull` as unwired.

Plus `rdcycle` in the firmware, which makes per-section timing a `printf`-free two-line
measurement inside the kernel.

### 9.6 What the C producer actually cost, measured

First run of the CPU as a producer of *compute*, `tb/fw_matmul_tb.sv` on the
`[8x32] @ [32x16]` problem. Both kernels produce byte-identical C, checked against a
reference computed in the testbench:

| clocks | `matmul.c` (one `matmul_t`) | `matmul_loop.c` (grid walked in C) |
| --- | --- | --- |
| run | **2 077** | **2 454** |
| MXU | 289 | 392 |
| ...of which weight load | 80 | 80 |
| DMA | 1 420 | 1 420 |
| `idlec` = CPU issue | **368 (17.7%)** | **642 (26.2%)** |
| `ovlap` | 0 | 0 |

Four things fall out, and only the last one is a surprise:

- **`run - (mxu + dma) == idlec` exactly**, and `ovlap` is 0, which says the firmware's
  `tpu_wait` fences are doing what they claim: no two units ever run at once, so this is
  the strict-mode baseline §4 asks for, not an overlapped run.
- **The hardware tile loop is worth 103 MXU clocks here** (289 vs 392), and the FSM state
  histogram says exactly where, with nothing left over:

  | `mxu.sv` state | `matmul` | `matmul_loop` | Δ |
  | --- | --- | --- | --- |
  | `S_LOAD` | 80 | 80 | 0 |
  | `S_RUN` | 192 | 192 | 0 |
  | `S_WB_ACC_RD` | 0 | 48 | **+48** |
  | `S_WB_WRITE` | 16 | 64 | **+48** |
  | `S_DONE` | 1 | 8 | +7 |
  | **busy** | **289** | **392** | **+103** |

  `LOAD` and `RUN` are *identical* — 8 weight tiles at 10 clocks and 8 contraction passes
  at 24, whoever walks the grid. All 103 clocks are writeback. `S_WB_WRITE` is one clock
  per token row: the hardware loop drains only when an **output** tile is finished
  (`NTILES*M` = 16 rows), the software one drains at the end of every dispatch
  (`NTILES*KTILES*M` = 64). The 48 extra rows are intermediate contraction partials. Worse,
  six of the eight dispatches carry `.acc`, so each of their rows first spends a clock in
  `S_WB_ACC_RD` reading the running int32 row back over the C port — `(KTILES-1)*NTILES*M`
  = 48 more. The hardware loop never enters that state at all: `k_tile_first` selects
  load-vs-add *inside* `result_buf`, so accumulating across the contraction costs zero
  cycles and zero scratchpad accesses.

  In closed form, software tiling costs **`2*KTILES - 1` times** the writeback of the
  hardware loop (7x here), and it scales with the contraction depth while the hardware
  loop does not. It is also 1 536 bytes read + 1 536 written on the arbitrated C port that
  the hardware loop never issues — free today at `ovlap = 0`, contention once units
  overlap. This is what `macro_ops.md` §4.2 meant by "the int32 C traffic disappears
  rather than being made faster", now measured from both sides.
- **289 MXU clocks matches the scalar unit's** `tiled_matmul_hw` at 578 for two matmuls.
  Same array, same work — the producer does not touch it.
- **Issue overhead is 18–26% of the run, not the ~2.7% §9.1 estimated.** That is not a
  contradiction — §9.4 says issue cost is per-dispatch and roughly constant while unit
  work scales with the tensor, and this problem is 2 000 clocks against the adder model's
  690 000. The share is what a 2 000-clock kernel looks like, not a regression. §9.7
  sweeps the shape and separates the two effects properly.

The 368 clocks are not divisible by hand: they cover CPU boot, 20 MMIO stores, three
poll-loop exits and the `DONE` write. The **marginal** number is clean, though, because
the two kernels differ by exactly seven `MXU_MM` commands and a loop: **+274 clocks for
+7 commands ≈ 39 clocks each** — and that is only the *exposed* part; §9.7 measures the
whole ~85 and shows what hides the rest.

### 9.7 How issue overhead scales with the problem

§9.4 predicted that issue cost is per-dispatch and roughly constant while unit work scales
with the tensor. `tb/run_fw_sweep.sh` (`make fwsweep`) measures it: both kernels rebuilt at
each shape, both run through `fw_matmul_tb.sv`, `cmds` taken from the queues' own `issued`
counters.

| shape M x K x N | kernel | run | MXU | DMA | `idlec` | cmds | MXU/dispatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8x8x8 | `matmul` | 1 039 | 43 | 604 | **392** | 5 | 43 |
| | `matmul_loop` | 1 039 | 43 | 604 | **392** | 5 | 43 |
| 8x16x16 | `matmul` | 1 755 | 153 | 1 228 | **374** | 5 | 153 |
| | `matmul_loop` | 1 828 | 188 | 1 228 | **412** | 8 | 47 |
| 8x32x32 | `matmul` | 3 520 | 577 | 2 572 | **371** | 5 | 577 |
| | `matmul_loop` | 4 413 | 784 | 2 572 | **1 057** | 20 | 49 |
| 8x64x64 | `matmul` | 8 294 | 2 241 | 5 644 | **409** | 5 | 2 241 |
| | `matmul_loop` | 11 562 | 3 200 | 5 644 | **2 718** | 68 | 50 |
| 8x128x128 | `matmul` | 22 537 | 8 833 | 13 324 | **380** | 5 | 8 833 |
| | `matmul_loop` | 35 222 | 12 928 | 13 324 | **8 970** | 260 | 50 |
| 16x32x32 | `matmul` | 5 994 | 737 | 4 876 | **381** | 5 | 737 |
| | `matmul_loop` | 6 720 | 1 136 | 4 876 | **708** | 20 | 71 |
| 32x32x32 | `matmul` | 10 911 | 1 057 | 9 484 | **370** | 5 | 1 057 |
| | `matmul_loop` | 11 782 | 1 840 | 9 484 | **458** | 20 | 115 |

**The hardware tile walk makes CPU cost O(1) in the problem.** `matmul.c` issues five
commands whatever the shape, and its `idlec` is 370–409 clocks across a **205x** range of
array work (43 → 8 833) and a 4x range of M. The spread is not a trend — it is gcc
materializing different address constants. Two commands describe a 16x16 grid exactly as
well as a 1x1 one, which is the whole argument for `MXU.GEOM` + `tiled`.

**The C loop is O(KTILES*NTILES), and the CPU — not the array — is the bottleneck there.**
Instrumenting the queue at 8x128x128: `mxu_level` **never exceeded 1**, and **not one push
landed while the MXU was busy**. The two strictly alternate, so the CPU's own per-dispatch
cost is recoverable as `(mxu + idlec - baseline) / dispatches`: **91** clocks at 8x32x32,
**87** at 8x64x64, **84** at 8x128x128. Call it ~85 to build and push one 128-bit command —
13 instructions, four AXI4-Lite stores, and every instruction fetch over that same bus
(topology A, §6). Against §9.1's estimate of ~26 for a chunk-loop dispatch, the factor is
the fetch path, not the command format.

**What that costs the run is `max(0, cpu - array)` per dispatch**, which is why the M axis
matters more than the tile axis:

| M | MXU/dispatch | CPU/dispatch | exposed per dispatch |
| ---: | ---: | ---: | ---: |
| 8 | 50 | ~85 | 34  (`(8970-380)/255`) |
| 16 | 71 | ~85 | 22  (`(708-381)/15`) |
| 32 | 115 | ~85 | **6**  (`(458-370)/15`) |

At M=32 the array's per-tile work passes the CPU's issue time and software tiling becomes
**free in CPU terms** — the crossover is around M ~ 24 on this 8x8 array. It is not free
overall: §9.6's writeback penalty is untouched by M, and is the whole of the remaining
871 clocks at 32x32x32.

Three consequences:

- **A deeper queue would not help anything here.** Depth is 8 and the level never reached
  2. The queue is not the constraint; ~85 clocks per command is.
- **The 1.56x gap at 8x128x128** (22 537 vs 35 222) splits 68% CPU issue (+8 590) and 32%
  array writeback (+4 095). At small M, issue cost — not the C traffic of §9.6 — is the
  larger half of the argument for `matmul_t`.
- **DMA is ~58% of the hardware-tiled run at every shape**, and none of it overlaps
  (`ovlap = 0`). That is §9.3's 1.45x, still unclaimed and still the largest single item
  on the table.

### 9.8 The whole model on the C producer, measured

`make fw FWPROG=adder` — four transformer layers plus the output head, 518
commands, `tb/fw_matmul_tb.sv` checking every DRAM byte *and* every command word
against `iss.py`:

| | clocks | share |
| --- | ---: | ---: |
| whole run | 439 917 | 100% |
| MXU | 206 361 | 46.9% |
| DMA | 131 168 | 29.8% |
| VPU | 84 352 | 19.2% |
| **`idlec` = CPU issuing commands** | **18 035** | **4.1%** |

**Do not read this against §0's 690 705.** That baseline is a *different model* —
the ternary `d=128, f=128` kernel on the scalar unit — and this is the int4
`d=64, f=256` one on the CPU. The only number that transfers is the last row.

**Issue overhead is 4.1%**, against §9's estimate of ≈2.7% for the CPU producer.
The gap is not a surprise and is not the command format: §9.7 measured ~85
clocks to build and push one command, and 518 × 85 = 44 030 clocks of *CPU*
time, of which only 18 035 is exposed — the rest hides under array work. It sits
above the estimate because this kernel's average dispatch is much smaller than
the sweep's: 96 of the 128 commands per layer are VPU passes over 512 elements
(32 clocks of work each), so the CPU is genuinely the critical path across the
elementwise stretches and genuinely idle across the matmuls.

Two things follow for anyone trying to shrink it, and neither is the queue:

- **`qfull` and `ovlap` are both 0.** Nothing is overlapped, because the kernel
  fences after every cross-unit dependency — `tpu_wait` before each of the four
  weight refills, around the K transpose, and between the array and the vector
  unit at each attention step. Every one of those is a real dependency, so
  removing them needs double-buffering (a second weight window, a second int32
  temp), not a looser barrier.
- **The VPU is 19.2% of the run**, against 7.6% on the retired ternary kernel,
  and its cost tracks element count almost exactly: 196 608 elements over 84 352
  clocks is 0.43 clocks each, i.e. ~7 clocks per 16-lane chunk with per-command
  overhead a rounding error. The FFN's `relu → requant` pair is **a third** of
  that traffic on its own — 32 commands and 16 384 elements per layer to compute
  a ReLU — and it is that size only because `relu` writes int32 and nothing
  narrows on the writeback path (vpu.md). A fused `relu.rq` would delete half
  those commands and all of the int32 traffic, worth ~6% of the whole run: the
  largest item on this table that is a *design* choice rather than arithmetic.

---

## 10. Area and timing

| | LUTs | FFs | BRAM | DSP |
| --- | --- | --- | --- | --- |
| today (post-route, `synth.md` §5) | 13 342 | 7 543 | 33 | 31 |
| − `scalar_unit` | −1 076 | −257 | −1 | −3 |
| + PicoRV32 (barrel shifter, fast mul, counters, compressed) | +1 300…2 000 | +600…900 | — | +2…4 |
| + firmware BRAM, 16 KB | — | — | +4 | — |
| + AXI4-Lite fabric + decode | +150…250 | +100 | — | — |
| + 3 command decoders + queues (depth 8 × 128b, SRL FIFOs) | +400…700 | +200 | — | — |
| + scratchpad grants + DMA skid buffer | +150…300 | +100 | — | — |
| **estimate** | **14 300…15 500 (69–75%)** | ≈8 400 (20%) | **36 (72%)** | ≈33 (37%) |

Timing is not a risk: WNS is **+26.2 ns against an 83.3 ns period** at 12 MHz, and PicoRV32
closes far above 100 MHz on this part. The board's only oscillator is 12 MHz, so the CPU
inherits three times the slack it needs.

**Get real numbers with `mode=ooc module=picorv32` before committing to any of this.**
`synth.md` §5's own estimates were wrong by 3× on `scratchpad`.

---

## 11. Phases

Each phase leaves the tree green, and the first two deliver value with **no CPU work at all**.

| Phase | Work | Leaves green because |
| --- | --- | --- |
| 0 | Add `starved`/`qfull`/`overlap` counters; re-measure the baseline | pure addition, as `macro_ops.md` phase 0 was |
| 1 | Command decoders + queues in front of MXU/VPU/DMA. `scalar_unit` **packs its config registers into commands** and pushes them, still waiting after each. | behavior identical; same golden vectors, same ISS, same `.tpu` programs. The command interface is validated before anything else moves. |
| 2 | Scratchpad grant handshake + DMA skid buffer (§5); queue depth > 1; a `push`-without-wait instruction flag; double-buffer the weight window | **this is where the 1.45× lands**, still on the existing scalar unit and existing toolchain |
| 3 | Mock `tpu.h` (`-DTPU_TRACE`) so the real `.c` compiles natively into a command trace (§8.1); `iss.py` re-fronted onto command traces; RTL command-trace monitor; `gen_vectors.py` rewired | the trace contract is proven against the *existing* scalar unit's commands before a CPU exists |
| 4 | PicoRV32 + firmware BRAM + AXI4-Lite, arbitrated onto the same command ports **alongside** `scalar_unit`. Port `adder_model` to C; check its command trace against the `.tpu` version's, command for command. | both producers live; the `.tpu` suite is still a regression. Costs 1 076 LUTs of duplication, affordable at 64%. |
| 5 | Retire `scalar_unit`, `assembler.py`, `pytpu.py`, `examples/*.tpu`; rewrite `isa.md` around the command format | only after 4 has been byte-identical for a while |
| 6 | The point of the exercise: a second architecture (GQA with `heads_per_q > 1`, or KV-cache decode) as a new `.c` file and no RTL change | — |

Phase 4 is the only irreversible one, and the trace-diff in it is what makes it verifiable
rather than a rewrite you hope is equivalent.

---

## 12. Decide before starting

1. **Do phases 1–2 alone get you enough?** They deliver the 1.45× and the stale-config fix
   without touching the toolchain. If the real complaint is "kernels are slow", stop there. If
   it is "I cannot write a second kernel", carry on — that is a compiler problem, not a
   hardware one, and only phase 4 fixes it.
2. **Command width.** 128 uniform is the recommendation and costs ~0.8% against a packed
   encoding. Locking it now matters because it is in the trace format, the RTL decoders and
   the driver header simultaneously.
3. **Sticky geometry.** Accept per-unit sticky state as queue-ordered (recommended), or make
   every command fully self-contained at 256 bits and pay ~1.5% more. The second is genuinely
   simpler to reason about; it is a taste call about whether "no state at all" is worth 4 extra
   stores per matmul.
4. **How much firmware RAM.** 16 KB is 4 RAMB36 and ~4 000 instructions. 32 KB doubles the
   BRAM cost to 8 tiles (33 → 40 of 50) and the UART load to ~2.8 s.
5. **`COMPRESSED_ISA`.** On, unless the phase-4 area measurement says otherwise. It is the
   cheapest thing to give back.
6. ~~**The Windows RISC-V toolchain.**~~ Settled: `fw/Makefile` takes any bare-metal
   RISC-V gcc and autodetects the prefix (xPack `riscv-none-elf-`, Homebrew
   `riscv64-elf-`, or a self-built one). It is still a new binary dependency in a repo
   whose only dep is `torch`, which is why `tb/fw_smoke.hex` stays hand-encoded — the CPU
   testbench must not be gated on an installed compiler.
7. **Does the CPU get the scratchpad window?** Yes in this plan, and it is what opens on-chip
   argmax and embedding lookup — but it is also a fourth requester on an arbitrated port, so
   it depends on §5 landing first.
