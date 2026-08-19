# VPU — Vector Unit

SIMD unit for every pointwise / reduction operation that is not a ternary-weight matmul.
See [README](README.md) §3 for the high-level role.

## Scope

The VPU owns every pointwise / reduction op the shipped model needs, and **only** those.
It was cut down to that set: `GELU`, `EXP`, `SQUARE`, `ELEMENT_MUL`, `SCALAR_MUL`,
`SCALAR_ADD`, `SCALAR_DIV`, `REDUCEMAX`, `REDUCESUM` and the `SOFTMAX` macro op were
removed, along with the two activation ROMs and the restoring divider. See
[Removed ops](#removed-ops).

What is left:

- **Activations:** ReLU. (The FFN is `ReLU`, not GELU — `model/transformer.py`.)
- **Elementwise arithmetic:** vector add — the residuals `X + O` and `X1 + F`.
- **Narrowing:** `REQUANT` (int32 → int8) and `DYT` (the same rescale, symmetric clip),
  which is how DyT normalization costs no pass of its own.
- **The mask and the ReLU, but no longer the attention matmuls.** There is **no
  softmax**: the model uses ReLU attention, so `relu(S + mask)` is a `vecadd` against
  an int8 mask and a `RELU`, both here. `QK^T` and `PV` used to be here too, as the
  `VECMATMUL` macro op — a serial `DOT` per (query, key) pair, and the slowest thing
  in the shipped kernel. They are now on the [MXU](mxu.md), because K and V are
  **ternary activations**: the array multiplies an int8 activation by a ternary
  weight, so ternarizing the operand that lands on the weight side is all it took.
  `TQUANT` is the op that does that. The output head followed — `Model.fc` is a
  `TernaryLinear` too — so `VECMATMUL` has **no caller in the shipped kernel at all**,
  and neither does `DOT`. Both are kept: an int8 × int8 matmul is one model shape away,
  and dropping them means dropping the whole reduction path (accumulator, fold, scalar
  store) plus the counter block around it, which is an area decision to make on purpose
  rather than a deletion.

## Operations

The op is selected by the 5-bit `vpu_op` field driven by the scalar unit, matching the
`VOP_*` values in `scalar_unit.sv` and `vpu.sv`.

| `vpu_op` | Name        | Kind        | Operands used            | Result                                       |
| -------- | ----------- | ----------- | ------------------------ | -------------------------------------------- |
| `0`      | `DOT`       | reduction   | `src0`, `src1`           | scalar `Σ src0[i]·src1[i]` → `dst`           |
| `1`      | `ADD`       | elementwise | `src0`, `src1`           | `dst[i] = src0[i] + src1[i]`                 |
| `3`      | `RELU`      | elementwise | `src0`                   | `dst[i] = max(src0[i], 0)`                   |
| `10`     | `REQUANT`   | requant     | `src0`(int32), `scalar`  | `dst[i] = clip((src0[i]·m0 + rnd) >> n)`     |
| `13`     | `VECMATMUL` | macro op    | `src0`, `src1`, geometry | `dst[t][s] = Σ_d src0[t][d]·src1[s][d]`      |
| `16`     | `DYT`       | requant     | `src0`(int32), `scalar`  | `dst[i] = clip±127((src0[i]·m0 + rnd) >> n)` |
| `17`     | `TQUANT`    | requant     | `src0`(**int8**), `scalar` | `dst[i] = clip±1((src0[i]·m0 + rnd) >> n)`, 2 bits wide |

The gaps (`2`, `4`–`9`, `11`, `12`, `14`, `15`) are retired codes, not free encoding
space — see [Removed ops](#removed-ops). They are left vacant so a stale binary decodes
to an unknown op rather than a different one.

`DOT` is the only reduction: it collapses the vector to one int32 scalar at `dst`. Every
other op produces a same-length vector. `DOT` is also the sole reason `vecdot` still
exists as an instruction — no shipped kernel issues it, but it is `VECMATMUL`'s inner
primitive, so the datapath is mandatory and the opcode costs one decode arm.

### Removed ops

These were implemented and are now gone. They served a softmax-attention, LayerNorm,
GELU-FFN model; `model/transformer.py` is ReLU attention, DyT normalization and a ReLU
feed-forward, so by the time the shipped kernel
([`adder_model.tpu`](../../tpulang/examples/adder_model.tpu)) was written not one of
them had a caller.

| Removed | Was for | Went with it |
| --- | --- | --- |
| `GELU` (4), `EXP` (6) | GELU FFN; softmax's `exp` | both 256×int8 ROMs, `rtl/luts/*.hex`, `accel/tpulang/luts.py`, the `GELU_INIT`/`EXP_INIT` parameters through `tpu_top.sv` and the board wrapper |
| `SQUARE` (5) | LayerNorm variance | — |
| `ELEMENT_MUL` (9), `SCALAR_MUL` (2), `SCALAR_ADD` (11) | LayerNorm/softmax broadcasts | — |
| `SCALAR_DIV` (12) | softmax's `Σexp`, LayerNorm variance | the shared restoring divider, `recip_R`/`div_*` state, the `S_RECIP` state, and the `RECIP_Q`/`DIV_Q` parameters |
| `REDUCEMAX` (7), `REDUCESUM` (8) | softmax/LayerNorm statistics | the **max fold** in the reduction path — `acc` now always opens at zero, since `DOT` is the only reduction left |
| `SOFTMAX` (14), `SM_EXP` (15) | the fused row-wise softmax macro op | its whole four-pass sequencer (`sm_*`), and `cfg vscalar` (index 9), the config register only it read |

Two ISA-level consequences. `cfg vscalar` is **retired but its index is not reused**:
renumbering `vrows`…`tdrow` would silently repoint every `setcfg` in every existing
program. And `vpu_scalar` is now unconditionally the third register operand, because
`SOFTMAX` — the one op whose three registers were `dst`/`src`/`tmp`, leaving no slot for
the requant word — was the reason that routing had a special case at all.

Measured on the trimmed unit, out-of-context synthesis of `vpu` alone
(`build.tcl mode=ooc module=vpu`, xc7a35t, ROWS=COLS=8):

| | LUTs | FFs | DSPs |
| --- | --- | --- | --- |
| before | 10012 | 897 | 90 |
| after | **5162** | **667** | **32** |
| | −48.4% | −25.6% | −64.4% |

The DSP collapse is the headline and is not mysterious: `SCALAR_MUL`, `ELEMENT_MUL`,
`SQUARE` and `SCALAR_DIV`'s reciprocal multiply each wanted a per-lane multiplier, and
there are 16 lanes. What remains is `DOT`'s 8×8 product and the two narrowing ops'
`acc32 × m0`.

**What coming back would cost.** `softmax` is the one to think about before deleting
anything further: `model/transformer.py` still carries the comment *"revert to softmax
if training bad"* next to its ReLU attention, and restoring it means restoring `EXP`
(and so the ROM, `luts.py` and the `$readmemh` plumbing), `REDUCEMAX`, `REDUCESUM`,
`SCALAR_DIV` with its divider, `SM_EXP`, the four-pass sequencer and `cfg vscalar` —
i.e. essentially this entire table. It is all recoverable from git history, but it is
not a one-line revert. LayerNorm would additionally want `SQUARE` back.

### Datatypes and requant

Compute ops read **int8** operands, accumulate in **int32**, and write their **int32**
result straight to scratchpad — the VPU does *not* narrow on the writeback path.
Narrowing int32 → int8 is the job of the explicit **`REQUANT`** instruction, which
applies the BitNet fixed-point rescale

```
dst[i] = clip_int8( (src0[i] * m0 + (1 << (n-1))) >> n )
```

the same arithmetic as [`requant.sv`](../rtl/requant.sv), with the per-tensor multiplier
`m0` (`M0_W` bits) and shift `n` (`N_W` bits) packed into the 32-bit word at the
`scalar` address (`m0` in the low bits, `n` above). Keeping requant a separate op lets
the scalar unit hold the residual stream at full int32 precision and requantize **only
where an int8 activation is actually consumed** — e.g. right before an MXU matmul —
instead of clipping after every pointwise op. The result scale is chosen entirely by
`m0`/`n`, so a producer emits its result already in the consumer's scale. That is the
only mechanism left for reconciling a residual add whose operands differ in scale:
`ADD` takes two int8 operands at one scale, so the producer's `REQUANT` has to land on
it (`adder_kernel.md` §4 — this is what pins `RQ_O` and `RQ_F`).

### DyT (`DYT`)

`DYT` is the same instruction with the clip moved: `-127` instead of `-128`.

```
dst[i] = clip_±127( (src0[i] * m0 + (1 << (n-1))) >> n )
```

That one constant is the whole op, and the reason it exists is that it turns a
*normalization* into an instruction the datapath already had. DyT
([arxiv 2503.10622](https://arxiv.org/abs/2503.10622)) is
`hardtanh(alpha·x, -1, 1)` with one learned scalar — the drop-in LayerNorm
replacement `model/transformer.py` uses. Pin the output scale to `1/127` and
fold `alpha` into the multiplier, and the saturation of a narrow *is* the
hardtanh: every value the clip catches is exactly one hardtanh would have
flattened to ±1. The clip has to be symmetric because hardtanh is odd; int8's
`-128` would put the saturated end at `-1.0079`.

The consequence for a kernel is that DyT costs **zero extra passes**: a
normalization always follows a residual add, and that add's result was going to
be narrowed from int32 to int8 anyway. See
`accel/tpulang/adder_kernel.md` §4 and `examples/adder_model.tpu`, where the two
`dyt`s replace two `requant`s and nothing else changes.

Nothing checks that the output scale really is `1/127`. A `DYT` whose word
targets some other scale is a rescale with an odd clip, not a hardtanh — the
scale contract is the compiler's.

### Ternary quantize (`TQUANT`)

The third op on the same shifter, and the only one whose destination is not int8:

```
dst[i] = clip_±1( (src0[i] * m0 + (1 << (n-1))) >> n )      # 2 bits per element
```

Two differences from `REQUANT`, and both are the point:

- **the source is int8**, not int32. `TQUANT` narrows an activation a requant already
  produced, so it does not read the accumulator width. It is the one op in the unit
  where `src0_is32` and `dst_is8` are genuinely independent predicates rather than two
  names for the same thing.
- **the destination is 2 bits**, written in the MXU's own weight encoding
  (`00` = 0, `01` = +1, `11` = −1 — [scratchpad.md](scratchpad.md) §2), four elements
  to a byte. So the result is not "a ternary tensor that then needs packing"; it *is*
  a packed weight block, addressable by `matmul_t` with no pass in between.

That is what lets an **activation be a weight operand**, which is the whole reason the
op exists. The array multiplies int8 × ternary and cannot multiply int8 × int8 at all,
so attention's `Q @ K^T` and `P @ V` could only run on it once K and V were ternary.
`accel/tpulang/examples/adder_model.tpu` ternarizes both, one `tquant` loop each, and
in exchange both attention matmuls became `matmul_t` dispatches.

Two constraints a kernel has to respect:

- **`vlen` must be a multiple of 4.** The write strobe is per byte and a byte holds
  four trits. A tail that does not fill its byte writes `00` — trit 0 — into the
  remaining slots rather than preserving what was there.
- **the destination pointer advances a quarter as fast as the source.** A `CHUNK`-long
  pass reads `CHUNK` bytes and writes `CHUNK/4`.

Nothing checks that the multiplier is a sensible ternary threshold, exactly as with
`DYT`. A word that is too large makes every element ±1 and a word that is too small
makes every element 0; both are legal instructions and neither is a matmul worth
running.

## Interface

The VPU is a slave of the [scalar unit](scalar_unit.md): the scalar unit dispatches one
op and blocks on `done` (issue-and-wait, scalar_unit.md §2). The VPU owns the `V_rw`
scratchpad port for the duration of the op.

### Control (scalar-unit side)

| Signal      | Dir | Width     | Meaning                                             |
| ----------- | --- | --------- | --------------------------------------------------- |
| `vpu_start` | in  | 1         | pulse: latch operands and begin (ignored if busy)   |
| `vpu_op`    | in  | 5         | operation selector (table above)                    |
| `vpu_src0`  | in  | `ADDR_W`  | byte address of source vector 0                     |
| `vpu_src1`  | in  | `ADDR_W`  | byte address of source vector 1 (binary ops)        |
| `vpu_scalar`| in  | `ADDR_W`  | byte address of the `{m0,n}` word (`REQUANT`/`DYT`) |
| `vpu_dst`   | in  | `ADDR_W`  | byte address of the destination                     |
| `vpu_vlen`  | in  | 10        | vector length in elements (config reg, ≤ 1023)      |
| `vpu_busy`  | out | 1         | high from `start` until the op retires              |
| `vpu_done`  | out | 1         | one-cycle pulse when the result is fully written    |
| `vpu_mm_busy`| out| 1         | `vpu_busy` **and** the op is `VECMATMUL` (perf only)|

`vpu_mm_busy` is not part of the dispatch handshake — the scalar unit never reads it.
It exists so `perf_counters.sv` can separate the macro op's clocks from the primitive
ops' (`tpu_top.sv`'s counter 6, `vmm`). `vpu_busy` alone conflates them, and they cost
very differently: a pointwise op retires `LANES` elements per chunk, while `vecmatmul`
pays a whole `S_RD0`…`S_WB` round trip per (row, col) pair. See macro_ops.md §7.

All addresses are byte offsets into the scratchpad and are captured on `vpu_start`, so
the scalar unit may reuse its register file immediately after issue.

### Scratchpad `V_rw` port (SIMD)

The lane count is int32-limited: `LANES = SCRATCHPAD_W/4`, where `SCRATCHPAD_W` is the
port width **in bytes**. `tpu_top`'s default `VPU_BYTES = 64` (512-bit) gives
`LANES = 16`, and that is the width this doc's examples use.

> **The Cmod A7 build is half that.** `boards/cmod_a7/board.tcl` sets `VPU_BYTES = 32`
> (256-bit), so the synthesized VPU has **`LANES = 8`**. Everything below holds with 8
> substituted for 16; only the chunk count changes, since a vector longer than `LANES` is
> streamed either way. The VPU is 4.5k LUTs and 48 of the design's 67 DSPs even at 8 lanes
> (`synth.md` §5), which is why it is not 16.

int32 operands/results use the full width (`LANES·32` bits); int8
operands/results occupy the low `LANES` bytes of an access, so int8 and int32 byte
strides differ (an int8 chunk advances `LANES` bytes, an int32 chunk `LANES·4`). The
port is a single logical read/modify/write port, so reads and writes are issued on
separate cycles by the FSM.

| Signal     | Dir | Width           | Meaning                                             |
| ---------- | --- | --------------- | --------------------------------------------------- |
| `V_re`     | out | 1               | read enable (address valid this cycle)              |
| `V_raddr`  | out | `ADDR_W`        | read byte address                                   |
| `V_rdata`  | in  | `SCRATCHPAD_W·8`| one int8 chunk, valid the cycle **after** `V_re`    |
| `V_we`     | out | 1               | write enable                                        |
| `V_waddr`  | out | `ADDR_W`        | write byte address                                  |
| `V_wdata`  | out | `SCRATCHPAD_W·8`| int8 chunk (or int32 scalar in the low 4 bytes)     |
| `V_wstrb`  | out | `SCRATCHPAD_W`  | per-byte write strobe (partial tail / scalar write) |

The memory contract matches the scalar unit's: synchronous read, data valid one cycle
after `V_re`; writes complete in the asserting cycle under `V_wstrb`.

### Handshake and streaming

```
   start                            done
     │   read src0   read src1        │
 idle ──► RD0 ──► RD1 ──► EXEC ──► ... ──► idle
              (per LANES-element chunk, looped ceil(vlen/LANES) times)
```

A vector longer than `LANES` is streamed in chunks: the FSM reads a chunk of each
operand, computes all lanes in one cycle, and either writes the chunk back (elementwise
/ requant) or folds it into a running int32 accumulator (reduction). Source/destination
pointers advance by the element stride per chunk (§Scratchpad port); a partial final
chunk is masked by `V_wstrb` and by a lane-active predicate so out-of-range lanes never
contribute to a reduction. The `SCALAR_MUL` multiplier / `REQUANT` `{m0,n}` word is read
once, on the first chunk, and held. When the last chunk
retires — writing the final elementwise chunk, or the single int32 reduction scalar to
`dst` — the VPU pulses `vpu_done` for one cycle and drops `vpu_busy`.

