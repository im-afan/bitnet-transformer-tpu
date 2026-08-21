> ## Dispatch now goes through per-unit command queues
>
> Every unit is fronted by `cmd_mxu.sv` / `cmd_vpu.sv` / `cmd_dma.sv`: a 128-bit
> command queue that carries the operands and geometry a dispatch needs, instead
> of the unit reading a global config register file at `start`. Producers push
> commands; the units pop them in order. See
> [picorv32_migration.md](picorv32_migration.md) §3-§5 for the format and the
> reasoning, and note the two consequences this document predates:
>
> * **The units can run at once.** Scratchpad access is arbitrated with real
>   grants rather than the exclusivity invariant that used to make arbitration
>   free (`scratchpad.sv`).
> * **There are two producers.** `scalar_unit.sv` still runs tpulang and still
>   waits after each dispatch, so nothing below about the ISA changes; PicoRV32
>   (`cpu_subsys.sv`) is the other, and pushes the same commands from firmware.

# MXU — Matrix Unit

Weight-stationary systolic array that computes the ternary-weight matmuls of the model.
See [README](README.md) §2 for the high-level role and [scratchpad.md](scratchpad.md)
for the memory it reads/writes.

## 1. Scope

The MXU handles every **ternary-weight × int8-activation** matmul in the transformer —
which, since ternary K/V and a ternary output head, means *every matmul in the model*:

| Op             | Shape (per token tile) | Notes                                 |
| -------------- | ---------------------- | ------------------------------------- |
| Wq, Wk, Wv, Wo | `T×128 @ 128×128`      | attention projections                 |
| Q @ Kᵀ         | `T×32 @ 32×T`          | per head; K is a ternary *activation*  |
| P @ V          | `T×T @ T×32`           | per head; V likewise                  |
| FF fc1         | `T×128 @ 128×512`      | tiled over output columns             |
| FF fc2         | `T×512 @ 512×128`      | tiled over the contraction            |
| output head    | `T×128 @ 128×16`       | `Model.fc` is ternary; 16 = 13 padded |

The last row is why the head is padded: this array stores a whole `COLS`-wide output
tile, so a 13-column result would have its second tile overrun each row into the next
one's first three words. `adder_model.tpu` rounds the head up to 16 columns and the
host stages the top three as zero trits (`adder_kernel.md` §2.6).

The attention rows are recent too. `Q@K^T` and `P@V` are
activation × activation, which this array cannot do — so they ran on the [VPU](vpu.md) as
`vecmatmul`, one serial dot product per output element, and were the slowest thing in the
shipped kernel. The fix was to make the operand that lands on the *weight* side ternary:
K in `Q@K^T`, V in `P@V`, both narrowed by `tquant` straight into this array's packed
2-bit weight layout. Q and the attention matrix stay int8, which is what the activation
side wants anyway. See `accel/tpulang/adder_kernel.md` §2.5.

There is no softmax to compute anywhere: attention is ReLU, so the only thing between the
two matmuls is a `vecadd` against an int8 mask and a `relu`, both on the VPU.

## 2. Why a ternary systolic array is cheap

A conventional PE needs a multiplier. With ternary weights `w ∈ {−1, 0, +1}`, the
multiply degenerates into a **select + conditional negate**:

```
w = +1  → acc += a
w =  0  → acc += 0        (pass-through)
w = −1  → acc -= a
```

No DSP slices, no multiplier — each PE is an int8→int32 add/sub plus a 2-bit weight
register. That is what makes a large array feasible on a low-end FPGA and is the whole
reason the model is quantized to ternary.

## 3. Array geometry

Parameterized `ROWS × COLS` array; suggested default **`128 × 128`** so the adder
model's `d=128` contraction and 128-wide projections map to a single tile with no
inner tiling. On a smaller board, drop to `64×64` or `32×32` and tile (§6).

- **Rows** = contraction dimension (`d`). Each row streams one activation element per
  clock into the array.
- **Cols** = output features. Each column holds one stationary weight column and
  accumulates one output element per token.
- Dataflow: activations flow **left→right**, partial sums flow **top→bottom**
  (classic weight-stationary OS/WS hybrid — weights stationary, activations stream,
  each column produces a full dot product).

### PE

```
          a_in (int8) ─────────────┐
                                    ▼
weight reg (2-bit) ──► [ sel ± ] ──►(+)──► acc (int32) ──► psum_out (down)
                                    ▲
          psum_in (int32, from PE above)
                        a_out (int8) ──► next PE (right)
```

Each PE registers `a_in` for one cycle before passing it right, creating the systolic
skew. `acc = psum_in ± a_in` (or `psum_in` when `w=0`).

## 4. Operation phases

A `Matmul(act_addr, weight_addr, out_addr)` runs in three overlapping phases:

1. **Weight load.** Stream the `d × COLS` ternary tile from scratchpad into the PE
   weight registers, 2-bit packed. Takes ~`COLS` cycles; overlaps the drain of the
   previous tile so back-to-back matmuls hide load latency.
2. **Feed.** Read one activation column (`d` int8 = 1024 bits) per clock from the
   activation banks and inject it **staggered** across rows — row *i* is delayed *i*
   cycles so all contributions to a given output element line up. This is why the
   scratchpad must deliver a full column per clock.
3. **Drain.** Column sums fall out the bottom after `ROWS + COLS + T` cycles and are
   written to `out_addr` as int32, one result row per token.

Latency for one `T×d @ d×COLS` tile ≈ `ROWS + COLS + T` cycles; throughput is one
output column per clock once the pipeline is full.

## 5. Numerics

- Accumulate in **int32**. Worst case per output element is `d` terms of
  `±127`; `128 × 127 ≈ 16 k`, well inside int32 — no saturation needed mid-accumulate.
- **Requantize on store.** BitNet-style: the reference folds a per-tensor
  `scale = mean(|W|)` into the ternary weights, so the int32 accumulator is multiplied
  by `scale` (and any activation scale) then rounded/clipped back to int8 on the way to
  scratchpad. The rescale is a fixed-point `clip((acc*M0 + round) >> N)` (same math as
  `requant.sv` / the VPU's `VOP_REQUANT`); the per-tensor `{M0, N}` word is read once
  from `scalar_addr`. It is applied by the store-path requantizer, **not** inside the PEs.
  Requant is gated by the `requant` input: assert it to narrow the store to int8; leave
  it clear to write int32 — the mode intermediate contraction tiles use so `accumulate`
  can keep running int32 partials in the result bank (§6), with only the final tile
  asserting `requant`.
- Bias add (projections have bias) is folded into the requantizer as an int32 add before
  rounding.

Validate bit-exactly against `TernaryLinear` in `model/transformer.py`: quantize `W`
with `RoundClip(W/scale)`, run `x @ w_quant`, compare.

## 6. Tiling (arrays smaller than the matmul)

For `COLS_array < N_out` (e.g. fc1's 512 outputs on a 128-wide array): iterate output
tiles, reloading weights each tile, accumulating each tile's results independently.
For `ROWS_array < d` (e.g. fc2's 512 contraction on a 128-tall array): split the
contraction, keep a **running int32 partial** in the result bank, and add successive
row-tiles into it before requantizing. The scalar unit emits the tile loop; each tile is
one hardware `Matmul`.

## 7. Interface

| Signal        | Dir | Width     | Meaning                              |
| ------------- | --- | --------- | ------------------------------------ |
| `start`       | in  | 1         | begin matmul (from scalar unit)      |
| `act_addr`    | in  | addr      | activation tile base                 |
| `weight_addr` | in  | addr      | ternary weight tile base             |
| `out_addr`    | in  | addr      | result base (int32, or int8 if requant) |
| `scalar_addr` | in  | addr      | requant `{M0, N}` word (used if requant) |
| `t_len`       | in  | 6         | number of token columns (`T`)        |
| `accumulate`  | in  | 1         | add into existing int32 result (tiling) |
| `requant`     | in  | 1         | narrow store int32→int8 via `{M0, N}` |
| `busy`        | out | 1         | high during load/feed/drain          |
| `done`        | out | 1         | pulse when result fully written      |

The store uses a per-byte `C_wstrb`, so an int8 requant row (COLS bytes) writes only its
lanes and does not clobber the neighbouring int32-width result bytes.

## 8. Open questions

- Final `ROWS × COLS` — depends on LUT/FF budget of the target board.
- Skew buffering: dedicated shift registers vs. reusing the activation bank read
  address staggering.
- Whether to fuse the `Wo` output residual-add (`O + X`) into the requantizer or leave
  it to the VPU.
