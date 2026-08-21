# Migrating the scalar unit to PicoRV32 + macro-op dispatch

**Status: the RTL is built and passing. Phases 0-2 and 4 are in the tree; 3 and
5-6 (the Python toolchain and the retirement of the old front end) are not.**

| Phase | State |
| --- | --- |
| 0 | **done** - counters reworked (§9.5); `swait` supplemented by `idlec`/`qfull`/`ovlap` |
| 1 | **done** - `cmd_queue.sv` + `cmd_{mxu,vpu,dma}.sv`; `scalar_unit.sv` packs its config registers into commands and pushes them |
| 2 | **done** - scratchpad grants, VPU stall, DMA skid buffer + `sram.sv` backpressure, queue depth 8 |
| 3 | not started - the RV32IM interpreter, `iss.py`'s trace front end, `gen_vectors.py` rewiring |
| 4 | **done (RTL)** - `cpu_subsys.sv`: PicoRV32 + AXI4-Lite + the MMIO command aperture, alongside the scalar unit. No firmware toolchain yet, so kernels in C are phase 3's dependency, not this one's. |
| 5-6 | not started - retiring `scalar_unit.sv`/`assembler.py`/`.tpu`, then a second architecture |

What passes today, on the tree as it stands:

| Test | Result |
| --- | --- |
| `make TEST=mxu` / `vpu` / `scratchpad` / `sram` | 168 / 318 / 50 / 4872 checks, 0 errors |
| `make cosim` (the real host driver against the RTL) | 11 of 11 |
| `make TEST=dma` (incl. new contention tests) | 16 737 checks, 0 errors |
| `make TEST=cmd_queue` (new) | 123 checks, 0 errors |
| `make examples` (new) - all ten example kernels vs. their golden vectors | **10 passed, 0 failed** |
| `make uart` (two programs, one link, no reset between) | 469 checks, 0 errors |
| `make TEST=cpu_smoke` (new) | 68 checks, 0 errors - PicoRV32 boots, pushes two DMA commands, 64-byte round trip byte-exact |
| `make model` - the adder model at `layers=1` through `tpu_top_tb` | **14 336 checks, 0 errors**, halts at pc=225 after 181 898 clocks |
| `make model LAYERS=4` - the shipped kernel | **26 624 checks, 0 errors**, halts at pc=225 after 693 107 clocks |
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
| `'I'` | write instruction BRAM, 10-bit address | write firmware BRAM, **12–14 bit address** |
| `'G'` | pulse `host_run` with `boot_pc` | release CPU reset; `boot_pc` becomes the reset vector |
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

1. **Firmware under a Python RV32IM interpreter.** MMIO stores are captured as commands.
   ~400 lines, well-trodden, no external dependency; it replaces `assembler.py`'s 618. (An
   external simulator — `spike`, `renode` — is the alternative if a dependency is acceptable.)
2. **`iss.py`, re-fronted.** Its `_matmul` / `_vpu` / `_dma` bodies are the golden numerics and
   are kept **verbatim**; only the decode front end changes, from 32-bit words to 128-bit
   commands. The scratchpad/DRAM images it produces are unchanged in meaning.
3. **RTL.** A sim-only monitor on each command FIFO write dumps the trace from `tpu_top_tb`.

`gen_vectors.py` becomes 1 → 2 → expected images, as before. `tpu_top_tb` then checks *two*
things instead of one: the memory image, and 3 against 1. That second check is strictly new
capability — a mismatch localizes immediately to either the CPU/firmware or the datapath,
which is a better debug loop than the current one, not a worse one.

`torch_ref.py` and `adder_export.py` are unaffected: they compare final tensors, not
instructions.

What dies: `assembler.py` (618 lines), `pytpu.py` (528), the `.tpu` language, every
`examples/*.tpu` and its golden vectors, and `isa.md` §§3–6 and A.2–A.4. That is the honest
price and it should be paid deliberately — see the phase plan, which keeps the scalar unit
alive alongside PicoRV32 through phase 4 precisely so the existing suite stays a live
regression while the new path is proven.

New in the tree: `accel/tpu/fw/` — `start.S`, a linker script, `tpu.h` (the MMIO driver:
command builders, `sync`, strict mode), one `.c` per kernel, and a Makefile.
`riscv-none-elf-gcc` (xPack) is the practical Windows toolchain; `-march=rv32imc
-mabi=ilp32 -nostdlib -ffreestanding`, no libc.

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

Plus `rdcycle` in the firmware, which makes per-section timing a `printf`-free two-line
measurement inside the kernel.

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
| 3 | Python RV32IM interpreter; `iss.py` re-fronted onto command traces; RTL command-trace monitor; `gen_vectors.py` rewired | the trace contract is proven against the *existing* scalar unit's commands before a CPU exists |
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
6. **The Windows RISC-V toolchain.** xPack `riscv-none-elf-gcc` is the path of least
   resistance, but it is a new binary dependency in a repo whose only dep is `torch`. Confirm
   it installs before phase 3, not during phase 4.
7. **Does the CPU get the scratchpad window?** Yes in this plan, and it is what opens on-chip
   argmax and embedding lookup — but it is also a fourth requester on an arbitrated port, so
   it depends on §5 landing first.
