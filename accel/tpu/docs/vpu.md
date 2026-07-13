# VPU — Vector Unit

SIMD unit for every pointwise / reduction operation that is not a ternary-weight matmul.
See [README](README.md) §3 for the high-level role.

## Scope

The VPU owns:

- **Activations:** ReLU, GELU (the FFN uses GELU; see `Expert`/`nn.Sequential` in
  `model/transformer.py`).
- **Elementwise arithmetic:** vector add (residuals `O + X`, `X + ff(X)`), vector ×
  scalar (requant scales), vector dot product.
- **Reductions:** sum / max over a vector — the primitives for LayerNorm statistics and
  softmax.
- **Attention scores.** Because `QK^T` and `AV` multiply *activations by activations*
  (not ternary weights) with small dims (`head_dim=16`, `T≤32`), they run here as batched
  int8 dot products rather than on the [MXU](mxu.md). Softmax (the `+mask`, `exp`,
  normalize of `mha_torch`) is a VPU reduction + activation sequence.

## Microarchitecture

`NLANE` int8/int32 ALU lanes fed from a single wide scratchpad read (the `V_rw` port,
512 bits ⇒ `NLANE = 16` int32 or 64 int8). One scratchpad access supplies all lanes in
lockstep — true SIMD, no per-lane addressing.

```
scratchpad V_rw (512b)
        │
   ┌────┴────┬────────┬─── … ──┐
  lane0    lane1    lane2     laneN
 [add/sub][mul  ][cmp/max][act LUT]   ← each lane: add, sub, mul, compare, shift
        │
   reduction tree (sum / max)  ──► scalar out
        │
   writeback V_rw
```

Each lane has: int32 adder, small multiplier (for `×scalar` and dot products — this is
the one place multipliers are unavoidable), comparator (max), and a shifter for
requant. A shared **reduction tree** across lanes produces sum/max for dot products,
softmax denominators, and LayerNorm mean/variance.

## Operations

The op is selected by the 4-bit `vpu_op` field driven by the scalar unit. Codes 0–4
match the `VOP_*` values the scalar unit already emits (`scalar_unit.sv`); 5–9 extend
the set for the softmax / LayerNorm / attention micro-sequences that the ISA builds out
of these primitives.

| `vpu_op` | Name          | Kind        | Operands used             | Result                                  |
| -------- | ------------- | ----------- | ------------------------- | --------------------------------------- |
| `0`      | `DOT`         | reduction   | `src0`, `src1`            | scalar `Σ src0[i]·src1[i]` → `dst`      |
| `1`      | `ADD`         | elementwise | `src0`, `src1`            | `dst[i] = src0[i] + src1[i]`            |
| `2`      | `SCALAR_MUL`  | elementwise | `src0`, `scalar`          | `dst[i] = src0[i] × scalar`             |
| `3`      | `RELU`        | elementwise | `src0`                    | `dst[i] = max(src0[i], 0)`             |
| `4`      | `GELU`        | elementwise | `src0`                    | `dst[i] = gelu_lut[src0[i]]`           |
| `5`      | `SQUARE`      | elementwise | `src0`                    | `dst[i] = src0[i]²` (LayerNorm var)     |
| `6`      | `EXP`         | elementwise | `src0`                    | `dst[i] = exp_lut[src0[i]]` (softmax)  |
| `7`      | `REDUCEMAX`   | reduction   | `src0`                    | scalar `max_i src0[i]` → `dst`          |
| `8`      | `REDUCESUM`   | reduction   | `src0`                    | scalar `Σ src0[i]` → `dst`             |
| `9`      | `ELEMENT_MUL` | elementwise | `src0`, `src1`            | `dst[i] = src0[i] · src1[i]`           |
| `10`     | `REQUANT`     | requant     | `src0`(int32), `scalar`   | `dst[i] = clip((src0[i]·m0 + rnd) >> n)`|

`REDUCEMAX`/`REDUCESUM`/`DOT` collapse the whole vector to one int32 scalar; the other
compute ops produce a same-length vector. `GELU`/`EXP` are 256-entry int8→int8 LUTs
indexed by the signed input byte, generated on the host from the reference activation
(`$readmemh` at init).

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
the scalar unit hold the residual stream and softmax/LayerNorm temporaries at full int32
precision and requantize **only where an int8 activation is actually consumed** — e.g.
right before an MXU matmul — instead of clipping after every pointwise op. The result
scale is chosen entirely by `m0`/`n`, so a producer emits its result already in the
consumer's scale (this is how residual adds with differing operand scales are
reconciled: rescale-then-add via `SCALAR_MUL`/`REQUANT`, then `ADD`).

## Interface

The VPU is a slave of the [scalar unit](scalar_unit.md): the scalar unit dispatches one
op and blocks on `done` (issue-and-wait, scalar_unit.md §2). The VPU owns the `V_rw`
scratchpad port for the duration of the op.

### Control (scalar-unit side)

| Signal      | Dir | Width     | Meaning                                             |
| ----------- | --- | --------- | --------------------------------------------------- |
| `vpu_start` | in  | 1         | pulse: latch operands and begin (ignored if busy)   |
| `vpu_op`    | in  | 4         | operation selector (table above)                    |
| `vpu_src0`  | in  | `ADDR_W`  | byte address of source vector 0                     |
| `vpu_src1`  | in  | `ADDR_W`  | byte address of source vector 1 (binary ops)        |
| `vpu_scalar`| in  | `ADDR_W`  | byte address of the broadcast scalar (`SCALAR_MUL`) |
| `vpu_dst`   | in  | `ADDR_W`  | byte address of the destination                     |
| `vpu_vlen`  | in  | 10        | vector length in elements (config reg, ≤ 1023)      |
| `vpu_busy`  | out | 1         | high from `start` until the op retires              |
| `vpu_done`  | out | 1         | one-cycle pulse when the result is fully written    |

All addresses are byte offsets into the scratchpad and are captured on `vpu_start`, so
the scalar unit may reuse its register file immediately after issue.

### Scratchpad `V_rw` port (512-bit SIMD)

The canonical lane count is int32-limited: `LANES = SCRATCHPAD_W/4` (512 bit ⇒
`LANES = 16`). int32 operands/results use the full width (`LANES·32` bits); int8
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

