# Macro-op ISA (CISC) — design note

> ## ⚠️ `softmax` and `layernorm` are out of scope as of the VPU trim
>
> This document was written when the model was softmax attention + LayerNorm. It is not.
> `model/transformer.py` is **ReLU attention, DyT normalization, ReLU feed-forward**, and
> the VPU was cut down to the ops that model actually issues.
>
> - **`softmax` (phase 5) was built, validated, and has now been removed** — with `EXP`,
>   `REDUCEMAX`, `REDUCESUM`, `SCALAR_DIV`, `SM_EXP`, both activation ROMs and `cfg
>   vscalar`. §5.1 and everything about it below is history, not the current ISA.
> - **`layernorm` (phase 6) is not merely unbuilt, it is not planned**, and §5.2's `rsqrt`
>   LUT with it. DyT replaced LayerNorm in the model, and DyT needs no macro op at all: it
>   is `requant` with a symmetric clip (opcode `0x21`, `dyt`), fused into the narrow the
>   residual add already required.
>
> Everything else here — `setcfgr`, the MXU config strides, `matmul_t`, `vecmatmul`, the
> config-register model, the phase/measurement tables — is current and load-bearing.
> See [vpu.md §Removed ops](vpu.md#removed-ops) for the removal table and measured area.

**Status: phases 0–4 built and validated; 5 built then removed; 6 dropped; 7 not started.**
See §8 for the phase table.

Built and passing (`tb/` 13/13, `make cosim` 11/11, `torch_ref.py` all kernels):

| Piece | Where | Test |
| --- | --- | --- |
| Per-run performance counters | `rtl/perf_counters.sv` (replaces `cycle_timer.sv`) | `tpu_top_uart_tb` structural + per-program checks |
| `setcfgr` (register → config) | `scalar_unit.sv` `0x1C` | `examples/setcfgr.tpu` |
| MXU config strides | `mxu.sv` `a_row`/`c_row`/`w_col` | `examples/strided_matmul.tpu` |
| `matmul_t` hardware tile loop | `mxu.sv` `0x1D` | `examples/tiled_matmul_hw.tpu` |
| `vecmatmul` | `vpu.sv` `VOP_VECMATMUL`, opcode `0x1E` | `examples/vecmatmul.tpu` |
| ~~`softmax`~~ | *removed* — was `vpu.sv` `VOP_SOFTMAX`/`VOP_SM_EXP`, opcode `0x20` | *(was `vpu_tb` + `examples/softmax_rows.tpu`; both deleted)* |

Measured instruction counts, same problem in each case:

| Kernel | Software | Macro-op |
| --- | --- | --- |
| 8×32 @ 32×16 tiled matmul | 70 words | **25** words — and that does it *twice* (int32 and requantized) |
| Attention score block | 44 words (`vpu_matmul.tpu`) | **17** words (`vecmatmul.tpu`) |
| ~~Softmax over a row block~~ | *8 words per row + a scalar bridge* | *18 words for 5 rows* — measured before the op was removed; kept as the record of what a fused macro op bought |

Not built: the full-layer rewrite (§8 phase 7) and weight double-buffering (§4.5).
`layernorm` + the `rsqrt` LUT (§5.2) are **dropped**, not pending — see the banner.

## 1. Why

The ISA today is RISC-shaped: one instruction is one array pass or one vector pass, and
everything above that — tile loops, and at the time softmax and LayerNorm — is expanded in
the program. That choice has three measured costs:

| Symptom | Evidence |
| --- | --- |
| Instruction memory is the binding constraint | One quantized layer = **861 of 1024** words (`README.md` roadmap §7); `tiled_matmul.tpu` alone is 191 lines |
| Programs are not hand-writable at layer scale | the since-deleted `softmax_row.tpu` was 72 lines for **one row**; `vpu_matmul.tpu` is 129 lines for one score block |
| The datatype contract is hand-enforced and silently breakable | `softmax_row.tpu:64-65` — `sadd` wrote int32 (4-byte stride), `exp` read int8 (1-byte stride), with nothing checking it. Still true of any int8/int32 pairing; that example is gone |

The fix is not a different control processor (see the PicoRV32 discussion — it costs LUTs
we don't have at 92% utilization and replaces the working, bit-exact `iss.py` chain). The
fix is to **raise the work-per-instruction**, which is what TPU v1 did: ~12 instructions,
CISC, >10 cycles each on average, with `MatrixMultiply` taking a variable-size operand and
sequencing the tiling in hardware.

Goal: a transformer layer that is **40–60 instructions and hand-writable**, with
`assembler.py` staying a table lookup rather than growing into a compiler.

## 2. Scope — what changes and what does not

**Unchanged.** The scalar unit stays (issue-and-wait, same FSM, same 32-bit encoding). The
PE array, the VPU lane datapath, the scratchpad, DMA, the UART link, and every numeric
convention in `isa.md` §A.5 (ternary packing, requant, Q15 `sdiv`) are untouched. The
existing primitive opcodes all stay valid.

**Changed.** Three things gain internal sequencers:

- **MXU** — an outer tile loop over the contraction (`k`) and output (`n`) axes, with
  configurable row strides. New opcode; the existing single-tile `matmul` is retained.
- **VPU** — three macro ops (`softmax`, `layernorm`, `vecmatmul`) implemented as a wrapper
  FSM that issues the existing `VOP_*` passes internally.
- **Config file** — five new registers for the matmul geometry, two for the VPU.

**Explicitly out of scope.** Removing branches (`beq`/`jmp` still drive the layer loop),
the inter-TPU LINK, and any change to the host protocol.

## 3. Opcode and config budget

Both have room; check this before adding anything else.

- **Opcode field is 6 bits = 64 slots.** Used: `0x00–0x1B` and `0x1F`. Free: `0x1C–0x1E`
  and the whole of `0x20–0x3E`. No pressure.
- **`flags` is only 2 bits and `matmul` already uses both** (`acc`, `rq`). A "tiled" bit
  does not fit — hence a new opcode rather than a flag.
- **`cfg` was `CFG_AW=4` = 16 registers**, of which `0..3` are assigned (`tlen`, `vlen`,
  `len`, `scalar`). Twelve free; this proposal takes eleven (ten below, plus `vcrow`).
  *Since built:* the DMA's transpose geometry (`dma.md` §5) needed three more, so `CFG_AW`
  is now **5** — 32 registers, `0..17` assigned. Widening cost only the register file; the
  `dst` field was already 8 bits.
- **`setcfg` is immediate-only** (`CFG` form: `cfg[dst] = zero_extend(imm16)`). There is no
  register→config path, so no config value can vary at runtime. §9.2 needs one; see
  `setcfgr` below.
- **`vpu_op` is 4 bits and codes `0..12` are used** — only 3 free, which the three macro
  ops exactly consume. **Widen `vpu_op` to 5 bits now** (a port-width change in `vpu.sv`,
  `scalar_unit.sv`, `tpu_top.sv` — three lines) rather than landing at exactly full.

### New config registers

| Idx | Name | Width | Meaning |
| --- | --- | --- | --- |
| 4 | `ktiles` | 8 | contraction tiles = `K / ROWS` |
| 5 | `ntiles` | 8 | output tiles = `N / COLS` |
| 6 | `arow` | 16 | activation row stride in bytes (`= K`) |
| 7 | `crow` | 16 | result row stride in bytes (`= N*4`, or `N` when requantizing) |
| 8 | `wcol` | 16 | weight column stride in bytes (`= K*2/8`) |
| 9 | ~~`vscalar`~~ | — | **retired** with `softmax`, the only op that read it. Index left vacant so 10–17 keep their meaning |
| 10 | `vrows` | 16 | `vecmatmul` **query** rows |
| 11 | `vcols` | 16 | `vecmatmul` **key** rows — the second, independent count |
| 12 | `vrow0` | 16 | `vecmatmul` `src0` row stride in bytes |
| 13 | `vrow1` | 16 | `vecmatmul` `src1` row stride in bytes |

Plus `vcrow` at 14 (`vecmatmul`'s dst row stride, distinct from the MXU's `crow`). 15..17
later went to the DMA transpose geometry (`dma.md` §5); 18..31 are free. `vscalar` is
deliberately *not* `cfg3 scalar` — that one is the MXU's requant word, and the two are live
simultaneously inside a layer.

`vrows` and `vcols` are separate because `vecmatmul` is **not square** in general: decode
attends one query row against `t+1` key rows (§9.2). `vpu_matmul.tpu`'s square `T × T`
block is the prefill special case, not the shape to design to.

### New scalar opcode: `setcfgr`

```
setcfgr  name, rN      cfg[name] = rN        ; opcode 0x1C, new CFGR form
```

The only genuinely new instruction outside the macro ops, and the only new assembler
*form* (`dst` = cfg index, `src0` = register). The config file already has a write port
from `setcfg`; this muxes its data source from the register file instead of the sign-
extended immediate — on the order of 20 LUTs.

Without it, nothing computed at runtime can reach a config register, which makes every
macro op's geometry a compile-time constant. That is fine for prefill and fatal for
decode.

## 4. MXU: hardware-tiled matmul

### 4.0 Why tiling is a separate opcode, not a flag

`matmul`'s `flags` field is 2 bits and `.acc`/`.rq` use both, so a "tiled" bit does not
fit. But there is a stronger reason than encoding pressure, found while building this:

**Config registers are not cleared between runs.** An earlier plan had the MXU key tiling
off "the stride registers are nonzero", with zero meaning the single-tile default. That
works for one program and breaks for two — running `strided_matmul.tpu` (which sets
`arow=16`, `crow=64`, `wcol=4`) and then `relu_layer.tpu` made the second program's plain
`matmul` read its operands at the first program's strides, corrupting every output byte.
The two-program `tpu_top_uart_tb` sequence catches this; a single-program test never would.

So `matmul_t` (opcode `0x1D`) reads the strides and `matmul` (`0x00`) ignores them
outright. The mode is a property of the instruction, not of leftover machine state. Within
`matmul_t` an unset (zero) stride still falls back to the single-tile value, so a missing
`setcfg` degrades to something sane rather than addressing everything at offset zero.

A related latent bug surfaced at the same time: the `cfg` array had **no reset**, so an
unread config register was X in simulation and undefined on hardware. Harmless while every
program set each register it used; not harmless once the MXU began reading `a_row`/`c_row`/
`w_col` on every dispatch. `scalar_unit.sv` now resets the whole config file.

### 4.1 The problem with the current address generation

Three strides are hardcoded and all three assume the operand *is* one tile:

| `mxu.sv` | Current | Needs to be |
| --- | --- | --- |
| `A_raddr = act_a + arq*ROWS` (L240) | row stride `ROWS` | `cfg arow` (`= K`) |
| `RES_STRIDE = COLS*4` (L125) | row stride `COLS*4` | `cfg crow` (`= N*4`) |
| `W_raddr = wgt_a + wreq*WCOL_BYTES` (L236) | column stride `ROWS*2/8` | `cfg wcol` (`= K*2/8`) |

`WCOL_BYTES` remains correct as the *intra-column k-tile offset* — a k-tile advances
`ROWS*2/8` bytes down within a column. So the weight address for tile `(k,n)`, column `c`
is:

```
W_raddr = wgt_a + (n*COLS + c) * cfg.wcol + k * WCOL_BYTES
```

### 4.2 Loop order — and the win that falls out of it

Order the tiles **`n` outer, `k` inner**:

```
for n in 0 .. ntiles-1:
    clear resbuf
    for k in 0 .. ktiles-1:
        LOAD  weight tile (k,n)
        RUN   stream A[:, k*ROWS : (k+1)*ROWS], accumulate into resbuf
    WB    write resbuf once, requantized if cfg.rq
```

Because `resbuf` already holds `MAXT × COLS` int32 partials, **the partial sums never leave
the chip across the `k` loop.** That deletes the `.acc` readback path for intermediate
tiles entirely, which is worth more than the instruction-count saving:

- No int32 `C` staging buffer in the scratchpad — at `COLS=8`, a `T×K` contraction
  currently needs `T*COLS*4` bytes of int32 partials round-tripping per tile.
- The `C_rw` port is idle during the whole `k` loop, so it stops contending on the
  scratchpad's arbitrated read port.
- The three-way `matmul` / `matmul.acc` / `matmul.acc.rq` choreography that the programmer
  hand-manages today (`isa.md` §6) becomes internal and unmissable.

Keep the `accumulate` flag for its *other* use — adding into a pre-existing `C` buffer
(cross-`M` accumulation, or a second call into the same output).

### 4.3 The `MAXT` limit — decide this explicitly

`TOK_W = 6`, so `MAXT = 64` and `t_len` is 6 bits: **at most 63 token rows per dispatch.**
For the adder model that is fine (`EQUALS_POS = 15`, sequences ~20–24 tokens), so the
initial implementation can leave `M` untiled and simply document the ceiling.

If you want generality, the `M` loop is a third counter and costs nothing extra in the
datapath — but it does *not* need `resbuf` to grow, since each `M`-tile drains before the
next begins. Recommend: implement `M` tiling in the ISS from the start (free), leave it out
of RTL v1, and make the assembler reject `tlen > 63`.

### 4.4 Cost estimate

Three counters (`k`, `n`, and the existing token counter), three stride multipliers folded
into the address adders, and a `resbuf`-clear state. No change to the PE array, the skew
buffers, or `requant8`. Rough guess **200–400 LUTs**; `mxu` is already 10,845 of the
design's 19,227, and you are at 92%. **Get a real number with
`mode=ooc module=mxu` before and after** — the `synth.md` §5 estimates were wrong by 3×
on `scratchpad`, in the good direction, but do not assume that twice.

### 4.5 Optional phase 2: weight double-buffering

With hardware tiling, the `k` loop alternates LOAD (COLS cycles) and RUN (~`T + ROWS`
cycles) with no overlap. Double-buffering the PE weight registers lets tile `k+1` load
while tile `k` streams — this is TPU v1's weight FIFO, and it is the single largest
throughput lever the tiled matmul unlocks. Cost is `ROWS*COLS*2` bits of extra weight
register: **128 FFs at the board's 8×8 geometry**, which is nothing. FFs are at 48%.

Do this *after* the tiled matmul is correct, and measure it — it is a clean before/after
roofline result.

## 5. VPU macro ops

All are a **wrapper FSM** that issues the existing `VOP_*` passes internally and holds the
scalar intermediates in registers. The lane datapath and `V_rw` are unchanged. (The LUTs
and the reciprocal divider this once also relied on no longer exist — only `vecmatmul`
survives, and it needs neither.) Reduction results stay in internal registers instead of
round-tripping through the scratchpad, which is what kills the `loads`/`subs`/`stores`
bridge idiom (`isa.md` §4.5) from programs.

### 5.1 `softmax dst, src, tmp` — REMOVED

> Built, validated, and then removed along with the VPU ops it was made of. The model uses
> ReLU attention. Retained below as the design record; **this is not the current ISA.**

Four passes over a row of `cfg vlen` elements, repeated `cfg vrows` times:

| Pass | Reads | Computes | Writes |
| --- | --- | --- | --- |
| 1 | `src` int8 | `m = max_i x[i]` | internal register |
| 2 | `src` int8 | `exp_lut[requant(x[i] - m, cfg.vscalar)]` | `tmp` **int8** |
| 3 | `tmp` int8 | `D = Σ e[i]` | internal register |
| 4 | `tmp` int8 | `round(e[i] * 2^15 / D)` (Q15) | `dst` int32 |

Pass 2 is the important one: it **fuses `sadd` + `requant` + `exp` into one pass that
writes narrow.** That is what makes pass 3's int8 read correct by construction, and it is
precisely the mismatch `softmax_row.tpu:64-65` got wrong. The `{m0,n}` word lands
the operand in the exp LUT's canonical 1/16 input scale (`vpu.sv` header, §Activation
LUTs) — an obligation currently on the programmer, with nothing checking it.

`tmp` is an explicit third operand because the int8→int32 width expansion in pass 4 would
otherwise clobber unread data. *Optimization worth noting, not doing first:* walking pass 4
**backwards** makes it safe in place (the int32 write at byte `4i` is above every unread
int8 at `j < i`), removing the `tmp` operand at the cost of a reverse-order streaming mode.

### 5.2 `layernorm dst, src, tmp` — DROPPED

> Never built, and no longer planned. `model/transformer.py` replaced LayerNorm with DyT
> (`hardtanh(alpha*x, -1, 1)`), which needs no macro op, no `rsqrt`, no square and no
> divide: it is `requant` with a symmetric clip (`dyt`, `0x21`), fused into the narrow the
> residual add already performed. The proposal below is kept for the reasoning about
> `rsqrt` and about keeping γ/β out of an instruction, which would apply again if a
> variance-based norm ever came back.

The model *used to use* `nn.LayerNorm`, which needs
`1/sqrt(var)` — and **the VPU has no square root.** `sdiv` gives `1/d`, not `1/sqrt(d)`.

Recommendation: **add an `rsqrt` LUT** as a third 256-entry int8 ROM, generated by
a `tpulang/luts.py` exactly like the since-deleted `gelu`/`exp` tables, loaded via a new `RSQRT_INIT` parameter. Same
pattern, same canonical-input-scale caveat, negligible area (`gelu_rom` and `exp_rom` are
already there and `vpu` uses 0 BRAM — they infer as LUTROM).

Passes: `Σx` → mean (multiply by the compile-time `1/L` as a `{m0,n}`), then `x - mean`
into `tmp`, then `Σ(x-mean)²` → var, then `rsqrt_lut[var]`, then scale.

**Keep γ and β out of the macro-op.** They are per-channel *vectors*, and folding them in
pushes the operand count past what the encoding holds. Emit them as a following
`vecemul` + `vecadd` pair — two extra instructions per norm, four per layer. Not worth
distorting the instruction format for.

### 5.3 `vecmatmul dst, src0, src1`

The MXU is **ternary-weight × int8-activation** (`mxu.sv` header). Attention's `Q@K^T` and
`P@V` are activation × activation, so they cannot touch the MXU at all — which is why
`vpu_matmul.tpu` exists and is 129 lines. This is the single biggest instruction-count
item in a layer and the CISC matmul of §4 does nothing for it.

`vecmatmul` wraps the existing `VOP_DOT` datapath in a two-level counter:

```
S[t][s] = Σ_d src0[t][d] * src1[s][d]      for t in cfg.vrows, s in cfg.vcols
```

contracting over `cfg vlen`, with `src0`/`src1` row strides from `cfg vrow0`/`cfg vrow1`
(these are **not** equal to `vlen` when a head is a slice of a wider `[T][heads*head_dim]`
tensor). K is read row-major, so **`K^T` is never materialized** — the transpose is
implicit in the loop order, exactly as `vpu_matmul.tpu` already documents. Output is int32
at `cfg crow` stride.

That implicit transpose covers `Q@K^T` and **not** `P@V`, which contracts over keys — V's
*row* axis — and so needs a real `V^T` whichever way round you write it. That is the DMA's
job rather than this op's: `wrmem.t`/`rdmem.t` ([dma.md](dma.md) §5) do it in one dispatch.

`vrows` and `vcols` are independent so decode (`vrows=1`, `vcols=t+1`) and prefill
(`vrows=vcols=T`) are the same instruction. Both come from `setcfgr` when they vary.

This is `vrows² ` dot products of length `vlen`, at `LANES = 8` on the board geometry
(`vpu_bytes=32`). It is slow — but it is *already* slow, one instruction at a time, and
collapsing it removes ~120 instruction words.

## 6. Toolchain changes

### `assembler.py`

Purely additive — new `SPECS` entries and new `CFG_NAMES`. No new *form* is needed:
`matmul_t` is `RRR` (dst/act/weight, geometry from cfg), and all three VPU macro ops are
`RRR`. This is the point: the assembler stays a table.

Add one validation that does not exist today: **reject `tlen > 63`** (`TOK_W` limit, §4.3).

### `iss.py`

The tiled matmul makes the ISS **simpler, not harder.** `_matmul` currently models one
tile; the macro version is one atomic operation over the full `M×K×N` shape. Bit-exactness
is preserved regardless of tile order because int32 accumulation is associative (including
under two's-complement wraparound), so the ISS does **not** need to replicate the hardware's
`n`-outer/`k`-inner order to match.

The VPU macro ops must model the pass structure exactly — specifically pass 2's fused
requant-then-LUT and its **int8** intermediate — because that *is* the numerics. Do not
model `softmax` as float and round at the end.

### `pytpu.py`

Its whole reason for existing (docstring: "instruction memory is 1024 words, so anything
shaped like a tensor loop has to stay a loop") is substantially relieved. It stays useful
for the layer/geometry loop, but the tile loops it exists to emit move into hardware.

## 7. Observability — this becomes mandatory

Hardware tiling **hides the tile loop from the program**, so you can no longer infer
weight-load vs. stream vs. drain from an instruction trace. Today `cycle_timer` counts one
number: total `busy` cycles for a run. That was already too coarse for a roofline; after
this change it is the *only* thing you'd have.

Add alongside `cycle_timer`, surfaced on the UART `'T'` command:

| Counter | Source | Answers |
| --- | --- | --- |
| `mxu_busy_cycles` | `mxu.busy` | MXU utilization |
| `mxu_load_cycles` | `mxu.state == S_LOAD` | weight-load overhead — and the phase-2 double-buffer payoff |
| `vpu_busy_cycles` | `vpu.busy` | how much of the layer is attention/softmax rather than GEMM |
| `dma_busy_cycles` | `dma.busy` | memory-bound vs. compute-bound — the roofline's x-axis |
| `su_wait_cycles` | `scalar_unit.state == S_WAIT` | control overhead, i.e. what issue-and-wait actually costs |

Five 32-bit counters and their UART readback: ~200 FFs, ~150 LUTs. This is the cheapest
work in this document and the only item that directly produces the analysis the project
exists for. **Build it first** — it also gives you honest before/after numbers for
everything else here.

## 8. Migration plan

Each phase leaves the tree green.

| Phase | Work | Status |
| --- | --- | --- |
| 0 | Cycle counters (§7) + `'T'` command extension | **done** — six counters, `'T'` returns a 6-word block, word 0 still the run length |
| 1 | `vpu_op` widened to 5 bits; new cfg registers + `setcfgr` (§3) wired through `scalar_unit`/`tpu_top`; assembler + ISS know the names | **done** — no behavior change; whole suite still green |
| 2 | MXU strides from cfg (§4.1), still single-tile | **done** — `tiled_matmul.tpu` passes unmodified; `strided_matmul.tpu` proves the strides are consulted |
| 3 | MXU tile loop + `matmul_t` opcode (§4.2) | **done** — `tiled_matmul_hw.tpu`, checked against an independent Python matmul over the full K |
| 4 | `vecmatmul` | **done** — `vecmatmul.tpu`, deliberately non-square (12×20) |
| 5 | `softmax` | built and validated, then **removed** — the model does not use softmax |
| 6 | `rsqrt` LUT + `layernorm` | **dropped** — DyT replaced LayerNorm; `dyt` (`0x21`) needs no macro op |
| 7 | Full layer rewritten; re-measure against the phase-0 baseline | not started |
| 8 | Optional: MXU weight double-buffering (§4.5) | not started |

**Phase-0 baseline**, for phase 7 to measure against (8×8 array, 100 MHz sim clock):

| Program | run | mxu | mload | vpu | dma | swait |
| --- | --- | --- | --- | --- | --- | --- |
| `relu_layer.tpu` | 1968 | 37 | 10 | 7 | 1856 (94%) | 1905 (97%) |
| `vector_add.tpu` | 6457 | 0 | 0 | 21 | 6400 (99%) | 6424 |
| `tiled_matmul.tpu` (software) | — | 616 | 160 (26% of MXU) | 0 | 7328 | 7981 |
| `tiled_matmul_hw.tpu` (2 matmuls) | — | 582 | 160 | 0 | 8864 | 9451 |

Two things fall straight out and neither was visible before phase 0. Every kernel at this
size is **DMA-bound** — 94–99% of the run is the byte-at-a-time SRAM path, so the array
geometry is nowhere near the limiter yet. And **26% of MXU time is weight loading**
(`mload`/`mxu`), which is the §4.5 double-buffering opportunity quantified: it is worth
having, and it is worth roughly a quarter of MXU time, not more.

**Keep every primitive opcode.** `matmul` and the primitive VPU ops stay in the ISA at
their current encodings, so `tiled_matmul.tpu`, `vpu_matmul.tpu` and
their golden vectors remain a live regression suite the whole way through. That is what
makes each phase verifiable rather than a rewrite you hope is equivalent.

## 9. Risks and open questions

1. **Area.** 92% LUTs, and `mxu` is 56% of them. Phases 3 and 8 both grow it. If §4.4 comes
   back over budget, the `synth.md` §5 knobs are `TOK_W` first (the `resbuf[0:63][0:7]`
   int32 buffer is ~16k FFs alone — and §4.3 says you only need 24 tokens, so `TOK_W=5`
   halves it for free), then 64-byte operand alignment to delete the barrel rotates.
   **`TOK_W = 5` is probably the cleanest way to pay for this whole proposal.**

2. **Decode with a KV cache — the layout is a non-issue; the *runtime config* is the real
   requirement.**

   Preallocate `K[MAX_T][d]` and `V[MAX_T][d]` at fixed scratchpad bases and write token
   `t`'s row at `base + t*d`. Then:

   - **The write** is a projection matmul with `t_len=1` whose `out_addr` is a register
     (`adds` from a step counter). Registers already hold addresses — nothing new.
   - **The read** is rows `0 .. t` of a fixed row-major tensor at a constant stride `d`.
     Contiguous, and exactly the `base + count + stride` shape §4.1 and §5.3 already
     express. Nothing ragged about it.
   - **Capacity** is not a constraint: at `d = 128`, even `MAX_T = 64` is 8 KB each for K
     and V against a 64 KB scratchpad.
   - **Causal masking comes free**, and is already the intended approach — iterate `j ≤ i`
     rather than materializing the `-1e9` mask (`scalar_unit.md` §5).

   What decode *does* require is that `vcols = t+1` change every step. Config registers are
   `setcfg`-immediate-only today, so **`setcfgr` (§3) is the actual prerequisite**, not any
   change to the address generators. It is ~20 LUTs. Add it in phase 1 and decode stays
   open at essentially no cost — this is much cheaper than the "decide now, expensive to
   retrofit" tradeoff an earlier draft of this section claimed.

   The genuinely rigid thing is narrower: the strides are *runtime* values but the **tensor
   layout convention** (activations row-major int8, weights column-major 2-bit packed) is
   compiled into the address arithmetic. A layout change is a retrofit; a shape change is
   a `setcfgr`.

3. **Decode is weight-load bound, and that is a result rather than a defect.** With
   `t_len = 1`, each projection pays a full `LOAD` (COLS cycles) plus pipeline drain
   (~ROWS cycles) to push one token row, so MXU utilization is roughly `1/(ROWS+COLS)` —
   about 6% at the board's 8×8. The §4.5 weight double-buffer does not help, because there
   is no activation stream to hide the load behind.

   This is the standard prefill-vs-decode arithmetic-intensity split, and it is precisely
   the kind of thing the phase-0 counters (§7) exist to show. Do not design around it —
   measure it, and let `mxu_load_cycles` versus `mxu_busy_cycles` make the point on a
   roofline.

4. **`rsqrt` LUT precision.** A 256-entry int8 table over the variance range may be too
   coarse for LayerNorm to match `nn.LayerNorm` acceptably. Validate against `torch_ref.py`
   **before** committing the RTL. Fallback: keep `layernorm` decomposed and accept the
   instruction cost — it is 2 per layer, not 2 per row.

5. **`vecmatmul` cost.** It removes ~120 instruction words but does not make attention
   faster; it is still `vrows²` VPU dot products. If phase-0 counters show attention
   dominating `vpu_busy_cycles`, the real answer is an int8×int8 mode in the MXU, which is
   a much larger change (the PE is a select+conditional-negate adder, not a multiplier)
   and is out of scope here. Measure before deciding.

6. **`softmax` operand count.** Three addresses plus `cfg vscalar` is the ceiling of what
   the encoding holds. If a macro op ever needs a fourth address, that is the signal the
   instruction format — not the op — needs revisiting.
