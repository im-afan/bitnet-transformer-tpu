# VPU — Vector Unit

SIMD unit for every pointwise / reduction operation that is not a ternary-weight matmul.
See [README](README.md) §3 for the high-level role.

## 1. Scope

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

## 2. Microarchitecture

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

## 3. Activation functions

- **ReLU:** `max(x, 0)` — a comparator per lane, no table.
- **GELU:** piecewise-linear or small LUT approximation of the exact GELU, indexed by
  the int8 input. A `256`-entry int8→int8 LUT per lane group is cheap in BRAM and exact
  enough for the quantized model; validate the max abs error against `F.gelu` on the
  golden vectors.
- **exp** (for softmax): shared LUT + fixed-point normalize, not a per-lane resource.

## 4. Compound operations

Higher-level ISA ops decompose into VPU micro-sequences driven by the scalar unit:

| ISA op        | VPU sequence                                                        |
| ------------- | ------------------------------------------------------------------- |
| `VectorDot`   | lanewise multiply → reduction-tree sum → requantize                 |
| `VectorAdd`   | lanewise add → writeback                                            |
| `VectorMul`   | lanewise multiply by broadcast scalar → writeback                   |
| `ReLU`        | lanewise `max(·,0)` → writeback                                     |
| `GeLU`        | lanewise LUT → writeback                                            |
| softmax*      | max-reduce → subtract → exp-LUT → sum-reduce → reciprocal-multiply  |
| LayerNorm*    | sum-reduce (mean) → var → rsqrt → normalize → affine (γ, β)         |

*Softmax and LayerNorm are not single ISA opcodes in the overview; they are emitted by
the scalar unit as sequences of the primitives above. LayerNorm (`norm1`, `norm2`) stays
higher-precision internally — accumulate the mean/variance in int32/fixed-point.

## 5. Numerics

- Multiply-accumulate in **int32**; requantize to int8 on writeback via the lane
  shifter (matching the MXU store path so tensors stay in a consistent int8 domain).
- Softmax and LayerNorm carry fixed-point intermediates to avoid catastrophic
  cancellation; only the final result is requantized. Pin the fractional-bit count
  against the golden model's outputs.

## 6. Interface

| Signal      | Dir | Width  | Meaning                                   |
| ----------- | --- | ------ | ----------------------------------------- |
| `start`     | in  | 1      | begin op                                  |
| `op`        | in  | 4      | ALU/activation/reduction selector         |
| `src0/src1` | in  | addr   | vector operand bases                      |
| `scalar`    | in  | addr   | scalar operand (for `×scalar`, bias)      |
| `dst`       | in  | addr   | result base                               |
| `vlen`      | in  | 10     | vector length in elements                 |
| `busy/done` | out | 1      | status back to scalar unit                |

For `vlen > NLANE`, the VPU strip-mines internally: successive wide reads until the
vector is consumed.

## 7. Open questions

- `NLANE` vs. MXU size — balance so the FFN's GELU doesn't bottleneck matmul throughput.
- GELU LUT vs. polynomial: measure LUT BRAM cost against a 2-segment PWL on the target
  board.
- Whether attention `QK^T`/`AV` deserve a dedicated small array instead of the generic
  dot-product path once `T` grows toward the context limit.
