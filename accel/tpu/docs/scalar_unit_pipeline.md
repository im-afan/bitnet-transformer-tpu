# Plan — Pipelining the Scalar Unit

Status: **proposal / not yet implemented**. Target file `rtl/scalar_unit.sv`; everything
else in `rtl/` stays untouched. This plan converts the current multi-cycle FSM into a
2-stage pipeline that retires one instruction per cycle, **without changing the ISA, the
unit handshakes, or the top-level port contract**.

---

## 1. What we have today

`scalar_unit.sv` is a non-overlapped FSM: `S_IDLE → S_FETCH → S_DECODE → S_EXEC →
{S_LOAD | S_WAIT | S_FETCH}`. Every instruction pays the full fetch+decode latency
before it does anything.

| Class                                                     | States                              | Cycles  |
| --------------------------------------------------------- | ----------------------------------- | ------- |
| scalar ALU (`adds`/`subs`/`muls`/`li`/`setcfg`/`cmps`/`stores`) | FETCH, DECODE, EXEC             | **3**   |
| `branch` / `jmp` (taken or not)                            | FETCH, DECODE, EXEC                 | **3**   |
| `loads`                                                    | FETCH, DECODE, EXEC, LOAD           | **4**   |
| dispatch (`matmul`, VPU ops, `rdmem`/`wrmem`, `wrneigh`)   | FETCH, DECODE, EXEC, WAIT×N         | **3+N** |
| `wait`                                                     | FETCH, DECODE, EXEC, WAIT×N         | **3+N** |

Two structural facts make this cheap to fix:

- The register file is a combinational-read / synchronous-write array, so a value written
  at the end of cycle *n* is readable in cycle *n+1*.
- The instruction memory read is already a registered BRAM read — that register can *be*
  the pipeline register instead of being copied into a separate `ir`.

And one fact makes it worth fixing: the tile loops and softmax/LayerNorm micro-sequences in
`scalar_unit.md` §5 and `isa.md` §8 are dominated by scalar address arithmetic. A typical
inner loop body (`vecadd` + 4 pointer bumps + `cmps` + `branch`) costs **18 cycles of
scalar overhead** on top of the VPU's own latency. The overhead is 3× larger than it needs
to be.

The clock is not the constraint: post-route WNS on the CMOD A7 build is **35.9 ns against
an 83.3 ns period (12 MHz)**, and the critical path is `u_mxu/wb_t_reg → scratchpad`, not
the scalar unit. There is no Fmax problem to solve here — this is purely a CPI problem, and
that fact is what lets us pick the simplest pipeline that works.

---

## 2. Goals / non-goals

**Goals**

1. CPI 1 for scalar ALU ops, `li`, `setcfg`, `cmps`, `stores`, and untaken branches.
2. CPI 2 for `loads` and taken branches/`jmp`.
3. Dispatch cost drops to `1 + N` (issue cycle + wait until `done`).
4. Zero architectural change: `iss.py` stays the golden model, `assembler.py` and every
   program in `tpulang/examples/` keep working, and the existing golden vectors must
   reproduce **bit-identically**.
5. Zero change to the module's port list or to any unit handshake.

**Non-goals (explicitly out of scope — see §9)**

- Run-ahead / scoreboarded overlap of scalar work with an in-flight MXU/VPU/DMA op. This is
  the *other* thing `scalar_unit.md` §2 calls an optimization, and it is a much larger
  change with real correctness consequences.
- Raising the clock, splitting the execute stage further, branch prediction, delay slots.

---

## 3. Chosen microarchitecture: 2-stage `F | DX`

```
        ┌──────────── F ────────────┐ ┌────────────── DX ──────────────┐
        │ pc_f → imem[] (sync BRAM) │ │ decode · RF read · ALU · branch │
        │                           │ │ resolve · unit dispatch · WB    │
        └───────────────────────────┘ └─────────────────────────────────┘
             imem output register  ===  the IR / pipeline register
```

**Why 2 stages and not 3 (`IF | ID | EX`).** A 3-stage split would put RF read in `ID` and
the ALU in `EX`, which immediately requires (a) an EX→ID forwarding network and (b) a
2-cycle branch penalty. It buys nothing: the combined decode + RF-read + 32-bit
add/multiply path is far inside a 83 ns budget, and even at 100 MHz it would have ~30 ns of
headroom against a path that today measures under 47 ns end-to-end through the *MXU*. The
2-stage design has **no forwarding logic at all** (§5.1) and the minimum possible branch
penalty. If the core is ever re-clocked with an MMCM, splitting `DX` is the obvious next
step and the hazard analysis below is the thing that would have to be redone.

### 3.1 State

Replaces `state`, `state_n`, `pc`, `ir`, `sel_unit`.

| Name          | Width     | Meaning                                                                 |
| ------------- | --------- | ----------------------------------------------------------------------- |
| `pc_f`        | `IMEM_AW` | address presented to the instruction BRAM this cycle (the fetch PC)      |
| `imem_rdata`  | 32        | BRAM output register — doubles as the IR of the instruction in `DX`      |
| `dx_pc`       | `IMEM_AW` | architectural PC of the instruction in `DX`; drives `pc_dbg`             |
| `dx_valid`    | 1         | `DX` holds a real instruction (0 during the post-branch bubble)          |
| `issued`      | 1         | this dispatch has already pulsed its unit's `start`                      |
| `mem_wait`    | 1         | this `loads` has already asserted `s_re`; `s_rdata` is valid this cycle  |
| `running`     | 1         | a program is executing (replaces `state != S_IDLE && != S_HALT`)         |
| `halted`      | 1         | `halt` retired (replaces `state == S_HALT`)                              |
| `flag_eq/lt`  | 1 each    | unchanged                                                                |

`sel_unit` becomes **combinational** — decoded from the instruction sitting in `DX`, which
is stable for its whole occupancy. This is what fixes bug §7.1.

### 3.2 Control

Three signals drive everything.

- **`retire`** — `dx_valid` AND the instruction is finished this cycle. Per class:
  - scalar ALU / `li` / `setcfg` / `cmps` / `stores` / `branch` / `jmp` → always
  - `loads` → only when `mem_wait` is set (i.e. its second cycle)
  - dispatch and `wait` → only when `issued` is set AND `sel_done`
  - `halt` → never (it deasserts `running` instead)
- **`advance`** — `!dx_valid || retire`. When set, `DX` accepts a new instruction next
  cycle, `pc_f` steps, and the BRAM read is enabled.
- **`redirect`** — a taken `branch` or a `jmp` retiring this cycle; forces `pc_f` to the
  target and clears `dx_valid` for the next cycle.

Per cycle, when `advance`: `imem_rdata ← imem[pc_f]`, `dx_pc ← pc_f`, `dx_valid ← fetch is
valid`, `pc_f ← redirect ? target : pc_f + 1`. When `!advance`, all four hold — the BRAM
read enable is what preserves the IR through a stall, so **no separate `ir` register is
needed**. On `redirect`, `dx_valid ← 0` for one cycle (the wrong-path word already in
flight is discarded).

### 3.3 Cycle diagrams

Steady state (`li`, `adds`, `adds`):

```
cyc     0        1        2        3
F     P0       P1       P2       P3
DX     ·       I(P0)    I(P1)    I(P2)      ← 1 instruction retired per cycle
```

Taken branch at P0 → T (the 1-cycle bubble is the whole penalty):

```
cyc     0        1        2        3
F     P0       P1       T        T+1
DX     ·      branch   bubble    I(T)
                 ↑ resolves here, redirects pc_f
```

`loads` at P0 (`s_re` in the first cycle, writeback in the second):

```
cyc     0        1        2        3
F     P0       P1       P1       P2
DX     ·      loads    loads    I(P1)
              s_re↑    s_rdata→rf
```

Dispatch at P0 (`start` is one cycle wide; operands held stable by DX occupancy):

```
cyc     0        1        2        3      ...    k      k+1
F     P0       P1       P1       P1            P1      P2
DX     ·      matmul   matmul   matmul        matmul   I(P1)
             start↑                                ↑ mxu_done
```

The dispatch minimum of 2 cycles (issue + at least one check of `sel_done`) is
**deliberate** and exactly reproduces today's `S_EXEC → S_WAIT` behaviour, including for the
`nb_done = 1'b1` comms stub in `tpu_top.sv`, which retires a `wrneigh` in 2 cycles either
way. Retiring in the issue cycle would be wrong for any unit whose `done` is still asserted
from a previous op.

### 3.4 Resulting cost

| Class                   | Now     | After   |
| ----------------------- | ------- | ------- |
| scalar ALU / `stores`   | 3       | **1**   |
| `cmps`                  | 3       | **1**   |
| untaken `branch`        | 3       | **1**   |
| taken `branch` / `jmp`  | 3       | **2**   |
| `loads`                 | 4       | **2**   |
| dispatch / `wait`       | 3 + N   | **1 + N** |

The `vecadd` + 4 pointer bumps + `cmps` + `branch` loop body goes from **21 + N** to
**8 + N** cycles — a 2.6× reduction in scalar overhead, and proportionally more on the
short-vector VPU sequences where N is small.

---

## 4. What stays exactly the same

Everything below is load-bearing for other blocks and must not drift:

- **Port list.** No signal added, removed, or re-widened.
- **`busy` / `done`.** `busy = running`, `done = halted`. `tpu_top.sv` uses `busy` to
  arbitrate the `sram_controller` between the DMA and the UART host, and `uart_interface`
  uses it as `core_busy` to NAK commands mid-run. `busy` must still rise the cycle after
  `host_run` and fall the cycle after `halt` retires.
- **`pc_dbg`.** Maps to `dx_pc` — the PC of the instruction currently executing, which is
  what `pc` means today (it only increments at retire). Both `tpu_top_tb` and
  `tpu_top_uart_tb` print it at halt; the value at halt is the `halt` instruction's own PC,
  unchanged.
- **`*_start` pulses are exactly one cycle wide** and are never asserted while the target
  unit is busy.
- **Operand buses are held stable for the whole dispatch.** `dma.sv` explicitly does *not*
  latch its operands ("the scalar unit holds the operands stable until `dma_done`"), so the
  dispatching instruction must remain in `DX` for the op's full duration. It does.
- **The scratchpad exclusivity invariant** (`scratchpad.sv` §"THE EXCLUSIVITY INVARIANT"):
  the priority mux is exact only because the scalar unit is issue-and-wait, so `s_re`/`s_we`
  never overlap another unit's port activity. A pure fetch/execute pipeline preserves this —
  `loads`/`stores` still only fire while no unit is running. Note that `s_rdata` is a *view
  of a shared output register*, so the one-cycle `loads` sampling window matters; the
  `mem_wait` flag makes that window explicit. The simulation assertion already in
  `scratchpad.sv` will catch it loudly if this is ever violated.
- **`iss.py` remains the golden model.** It is architectural, not cycle-accurate ("issue-and-wait:
  dispatch already ran atomically"), so a correct pipeline produces an identical final
  scratchpad image.

---

## 5. Hazard analysis

### 5.1 RAW on the register file — no forwarding needed

Only one instruction is ever in `DX`, and `DX` both reads and writes the RF. Instruction
*i+1* reads combinationally in cycle *n+1*; instruction *i* wrote synchronously at the end
of cycle *n*. Back-to-back `li r1,5` / `adds r2,r1,r1` is correct with zero forwarding
logic. Same argument covers `flag_eq`/`flag_lt` (`cmps` → `branch`) and the config file
(`setcfg` → dispatch reading `cfg[CFG_VLEN]`).

### 5.2 Load-use — resolved by the unconditional 1-cycle `loads` occupancy

`s_rdata` arrives one cycle after `s_re`. Rather than a load-use interlock plus forwarding,
`loads` simply occupies `DX` for two cycles and writes back in the second. Consequences:

- No consumer can ever observe a stale value.
- The single RF write port is never contended: a `loads` writeback and a following ALU
  writeback can never fall in the same cycle.
- Cost: `loads` is CPI 2 unconditionally, even when the next instruction is independent.
  Given how rare `loads` is in the emitted programs (it exists to bridge computed
  scales/addresses in from the scratchpad), a conditional interlock is not worth the extra
  comparator and second write port. Revisit only if profiling says otherwise.

### 5.3 Control — 1-cycle penalty, irreducible

The branch resolves in `DX`, which is the earliest cycle its instruction word exists in a
readable register; the fall-through fetch is already in flight and gets squashed via
`dx_valid`. There is no earlier point at which a redirect could be computed in a 2-stage
machine, so the 1-cycle taken-branch penalty is the floor without a next-PC predictor. **No
delay slot** — that would change the ISA and force matching changes in `assembler.py`,
`iss.py`, and `isa.md`, for one cycle in loop-closing branches.

### 5.4 Structural

- **IMEM port**: fetch vs. host write. Host writes are only accepted while `!running`, so
  they never coincide with a fetch of a running program.
- **RF write port**: see §5.2 — one writer per cycle by construction.
- **Scratchpad S port**: `s_re`/`s_we` are each asserted for exactly one cycle and, per §4,
  never concurrently with another requester.

### 5.5 New hazard introduced by tighter issue — unit readiness

Today, ≥3 cycles separate a dispatch's retire from the next `start`. After pipelining that
gap can be **1 cycle**, so it matters whether a unit is ready to accept `start` immediately
after asserting `done`. Verified against the RTL:

| Unit | `done` source            | Next state after done | Accepts `start` in |
| ---- | ------------------------ | --------------------- | ------------------ |
| MXU  | `state == S_DONE`        | `S_DONE → S_IDLE`     | `S_IDLE` only      |
| VPU  | `state == S_DONE`        | `S_DONE → S_IDLE`     | `S_IDLE` only      |
| DMA  | `state == DONE`          | `DONE → IDLE`         | `IDLE` only        |

All three hold `done` for exactly one cycle and are back in idle the following cycle — which
is the earliest cycle the pipeline can present the next `start`. So back-to-back dispatch is
safe **as built**, but it is now a load-bearing property rather than an accident. Add an
assertion (simulation-only) that no `*_start` is ever asserted while the corresponding
`*_busy` is high, and cover it with a directed test (§8, test 7).

---

## 6. Signal-level change list for `rtl/scalar_unit.sv`

Nothing below the "instruction field decode" comment changes semantically; most of the file
is reusable as-is.

**Delete**
- The `state_t` enum, `state`/`state_n`, and the whole next-state `always_comb`.
- The `pc` and `ir` registers and the `S_DECODE: ir <= imem_rdata` latch.
- The `sel_unit` register.

**Add**
- The registers in §3.1 and the `retire`/`advance`/`redirect` control in §3.2.
- A read enable (`advance`) on the `imem_rdata <= imem[pc_f]` assignment. This infers a
  7-series BRAM with `EN`; confirm the RAMB count is unchanged post-synthesis (§8.5).

**Rewrite (mechanical `state ==` → new predicate)**

| Today                                        | After                                                   |
| -------------------------------------------- | ------------------------------------------------------- |
| `(state == S_EXEC) && (opc == OP_MATMUL)`     | `dx_valid && !issued && (opc == OP_MATMUL)`              |
| `(state == S_EXEC) && is_vpu_op(opc)`         | `dx_valid && !issued && is_vpu_op(opc)`                  |
| `(state == S_EXEC) && (WRMEM \|\| RDMEM)`     | `dx_valid && !issued && (WRMEM \|\| RDMEM)`              |
| `(state == S_EXEC) && (opc == OP_WRNEIGH)`    | `dx_valid && !issued && (opc == OP_WRNEIGH)`             |
| `s_re = (state == S_EXEC) && (opc==OP_LOADS)` | `dx_valid && !mem_wait && (opc == OP_LOADS)`             |
| `s_we = (state == S_EXEC) && (opc==OP_STORES)`| `dx_valid && (opc == OP_STORES)`                         |
| `imem_we && state == S_IDLE`                  | `imem_we && !running`  *(also fixes §7.2)*               |
| `cfg_we && state == S_IDLE`                   | `cfg_we && !running`   *(also fixes §7.2)*               |
| `rf_we` in `S_EXEC` / `S_LOAD`                | `dx_valid` ALU commit / `dx_valid && mem_wait` for loads |
| `busy = state != S_IDLE && != S_HALT`         | `busy = running`                                         |
| `done = state == S_HALT`                      | `done = halted`                                          |

**Unchanged**: the opcode `localparam` block, `VOP_*`, `CFG_*`, field decode wires, the RF
and config arrays, the `vpu_op` decode `always_comb`, all dispatch operand routing assigns,
`is_vpu_op` / `is_dispatch`, `br_taken`, and the ALU result mux.

---

## 7. Two pre-existing bugs this restructure should fix

Both are latent today and both are naturally corrected by the rewrite, so fixing them here
costs nothing. Each needs its own regression test.

### 7.1 `wait` ignores its unit selector

`sel_unit <= dispatch_unit(opc)` is evaluated for `OP_WAIT` too, and `dispatch_unit` has no
`OP_WAIT` arm, so it falls to `default: U_MXU`. Every `wait` therefore blocks on `mxu_done`
regardless of `flags`, contradicting `isa.md` A.4 (`wait unit: mxu=0b00, vpu=0b01,
dma=0b10, link=0b11`) and the opcode comment in the RTL itself. `wait.vpu` after a VPU op
will hang until an unrelated matmul completes. It has never been caught because `iss.py`
treats `wait` as a no-op under issue-and-wait, so no golden vector exercises it and the
example programs don't emit it.

**Fix**: make `sel_unit` combinational — `flags` for `OP_WAIT`, `dispatch_unit(opc)`
otherwise.

### 7.2 Instruction memory cannot be reloaded after a program halts

`imem_we` and `cfg_we` are gated on `state == S_IDLE`, but after `halt` the FSM parks in
`S_HALT` and only leaves on `host_run`. Meanwhile `uart_interface` gates on `core_busy`,
which is *low* in `S_HALT` — so the UART host happily ACKs a program-load command whose
writes are then silently dropped. The current e2e flow loads once and runs once, so it never
shows up.

**Fix**: gate on `!running`, which is low in both idle and halted.

---

## 8. Verification plan

The regression bar is: **identical final scratchpad image on every existing test**, plus
new directed coverage for the things a pipeline can break that an architectural test can't
see.

### 8.1 Baseline first (do this before touching the RTL)

Record, for each program in `tpulang/examples/`, the halt time from `make uart PROG=...`
and the `post_synth_utilization` / `post_route_timing` numbers. These are the before-numbers
for the speedup claim and the area/timing no-regression check.

### 8.2 New `tb/scalar_unit_tb.sv`

There is no unit-level testbench for the scalar unit today — every test goes through
`tpu_top`. Add one, with behavioural stubs: a 1-cycle-latency scratchpad model on the S
port, and four unit stubs with programmable latency exposing the real `start`/`busy`/`done`
protocol. Inspect the RF via hierarchical reference (`dut.rf[n]`) — Icarus supports it.

Directed tests:

1. **Throughput** — a run of independent `adds` retires one per cycle (`pc_dbg` advances
   every cycle).
2. **Back-to-back RAW** — `li r1,5` / `adds r2,r1,r1` / `adds r3,r2,r2` → `r3 == 20`.
3. **Load-use** — `loads r1,[r2]` / `adds r3,r1,r1` reads the loaded value, and `loads`
   takes exactly 2 cycles.
4. **Flags RAW** — `cmps` immediately followed by `branch` for all four conditions, both
   polarities; check taken = 2 cycles, untaken = 1.
5. **Branch shadow squash** — put a `li r9, 0xBEEF` immediately after a taken branch and
   assert `r9` is unmodified. This is the single most important pipeline-specific test.
6. **Dispatch protocol** — `start` asserted for exactly one cycle; the operand buses
   (`mxu_act_addr` etc.) stable from issue through `done`; retire on the `done` edge.
7. **Back-to-back dispatch** — two dispatches to the same unit with no instruction between
   them; assert `start` is never seen while `busy` (covers §5.5).
8. **`wait` on each of the four units** — regression for §7.1.
9. **`setcfg` → immediate dispatch** — the dispatch sees the new `vlen`/`tlen`.
10. **`stores` → `loads`** to the same address.
11. **Halt / restart / reload** — `busy`, `done`, `pc_dbg` at halt; then write new
    instruction words while halted and re-run — regression for §7.2.

### 8.3 Existing end-to-end regressions (must pass unmodified)

- `make TEST=tpu_top` — backdoor-loaded program against `tb/vectors/`.
- `make uart` — full UART path against `tb/vectors_uart/`.
- `make uart PROG=examples/<each>.tpu` for every example, since the golden vectors are
  regenerated from `iss.py` per program and are exactly the architectural contract.

Any pipeline bug that corrupts a value shows up here; §8.2 is what localises it.

### 8.4 Cycle-count reporting

Add a halt-time `$display` of the elapsed cycle count to `tpu_top_tb` and
`tpu_top_uart_tb`, then compare against §8.1 to confirm and quantify the speedup on real
programs rather than on the synthetic loop estimate in §3.4.

### 8.5 Synthesis

Re-run the CMOD A7 flow. Check: WNS still passes with wide margin; LUT/FF delta is small
(the FSM is replaced by a handful of flags, so expect roughly break-even); and **`imem`
still infers as BRAM** — the added read enable is the one change that could plausibly push
it to distributed RAM. Compare RAMB counts in `post_synth_utilization.rpt`.

---

## 9. Explicitly deferred: run-ahead / decoupled dispatch

`scalar_unit.md` §2 and §7 float letting the scalar unit continue address arithmetic while
a matmul drains. That is a **separate project** and should not be bundled here, because it
breaks three things this plan is careful to preserve:

1. **The scratchpad exclusivity invariant.** `scratchpad.sv` states that the priority mux is
   exact *because* issue-and-wait means the scalar unit's `s_re`/`s_we` never overlap MXU /
   VPU / DMA port activity. Run-ahead makes overlap the normal case and demands real
   arbitration with a stall handshake — which none of the compute units have.
2. **Unlatched dispatch operands.** `dma.sv` reads its address/length inputs throughout the
   transfer. Run-ahead requires latching operands at issue (in the DMA, or in a dispatch
   register in the scalar unit).
3. **The ISS as golden model.** `iss.py` executes each dispatch atomically. Overlap is only
   safe if the program's dependences are expressed with explicit `wait`s — which means a
   real hazard model (scratchpad region scoreboard, or programmer-declared) and a matching
   change to `isa.md` §A.6.

The right sequencing is: land this pipeline, fix `wait` (§7.1) so the synchronisation
primitive actually works, get the cycle-count baseline from §8.4 — and *then* evaluate
whether overlap is worth the arbitration cost, with numbers in hand.

Also deferred, in rough order of appeal: a load-use interlock instead of the unconditional
`loads` stall (§5.2); an instruction prefetch queue (free during dispatch stalls, and a
prerequisite for run-ahead anyway); splitting `DX` if the core is ever re-clocked.

---

## 10. Phasing

| Phase | Work                                                                                     |
| ----- | ---------------------------------------------------------------------------------------- |
| 0     | Baseline: cycle counts per example program, current synthesis reports (§8.1)              |
| 1     | Rewrite `scalar_unit.sv` to `F | DX` (§3, §6), including the two bug fixes (§7)           |
| 2     | New `tb/scalar_unit_tb.sv` (§8.2)                                                         |
| 3     | Full regression: `make all`, `make uart` across all examples (§8.3, §8.4)                 |
| 4     | Synthesis check (§8.5)                                                                    |
| 5     | Docs: `scalar_unit.md` §2, `isa.md` §A.6 FSM sketch and per-op retire rules, the header comment in `scalar_unit.sv`, and the `S_EXEC`/`S_WAIT` references in `scratchpad.sv`'s exclusivity-invariant comment |

Phases 1–4 are one atomic change: the RTL rewrite is not independently shippable from its
tests.

## 11. Risks

| Risk                                                                | Mitigation                                                              |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Branch shadow not squashed → silent wrong-path execution             | §8.2 test 5, plus every e2e program has loops                            |
| Unit not ready for a `start` 1 cycle after `done`                    | Verified in §5.5; assertion + §8.2 test 7                                |
| `s_rdata` sampled in the wrong cycle (shared output register)        | `mem_wait` makes the window explicit; §8.2 test 3                        |
| `imem` read enable changes BRAM inference                            | §8.5 RAMB count comparison                                               |
| `busy` timing shift breaks UART/SRAM arbitration in `tpu_top`        | `busy = running` preserves both edges; `make uart` covers the whole path |
| Stale `d_ir` during a bubble driving a `unique case`                 | All enables qualified by `dx_valid`; keep `default:` arms                |
