# Scalar Unit & ISA

The control processor of the TPU. It fetches instructions, dispatches work to the
[MXU](mxu.md) and [VPU](vpu.md), drives DMA and the [comms](comms.md) link, and performs
scalar arithmetic on values in memory. See [README](README.md) §5.

## 1. Role

- **Control.** Sequences a transformer layer: projection matmuls → attention → residual
  add → LayerNorm → FFN → residual add → LayerNorm. It expands high-level structure into
  the primitive ops the MXU/VPU understand, including the **tile loops** for matmuls
  larger than the array (see [mxu.md](mxu.md) §6) and the micro-sequences for softmax /
  LayerNorm (see [vpu.md](vpu.md) §4).
- **Dispatch & sync.** Issues `start` to a unit, waits on its `done`, manages hazards
  between units sharing a scratchpad bank group.
- **Scalar math.** Add / multiply / compare on single int32 values in memory (loop
  counters, addresses, requant scales, softmax denominators).

It is a small in-order engine — think microcontroller, not out-of-order core — reading a
program from a dedicated **instruction memory** (BRAM), separate from the scratchpad.

## 2. Execution model

```
fetch ─► decode ─► dispatch ─┬─► MXU   (Matmul)
                             ├─► VPU   (vector ops)
                             ├─► DMA   (Read/WriteMemory)
                             └─► LINK  (WriteNeighbor)
   scalar ALU handles add/mul/branch inline
```

- **In-order issue.** One instruction issues per decode; compute ops run asynchronously
  on their unit while the scalar unit continues with independent scalar/address work.
- **Synchronization.** A dependent instruction blocks on the producer's `done`
  (scoreboard on unit-busy). Simplest v1: issue-and-wait (no overlap). Optimization:
  allow the scalar unit to run ahead on address math while a matmul drains, since the
  matmul's operands/results are known.
- **Control flow.** Scalar compare + branch drives tile loops and the per-layer / per-
  token loops. Loop bounds (`T`, tile counts) come from config registers set by the host.

## 3. Instruction set

> **The authoritative instruction reference is [isa.md Appendix A](isa.md#appendix-a--encoding--opcode-reference)** —
> exact 32-bit encoding, the full opcode map (with hex values and tpulang mnemonics),
> per-op semantics, and the shared numeric conventions (ternary packing, requant, `sdiv`
> Q15); the rest of [isa.md](isa.md) is the programmer's guide to writing tpulang. The
> tables below are the conceptual overview; where they differ from `isa.md`, `isa.md`
> (which tracks `rtl/scalar_unit.sv`, `tpulang/assembler.py`, and `tpulang/iss.py`) wins.

All **math** ops address **scratchpad**; all **comms/memory** ops address **DRAM** except
where noted. Fixed-size operands are implied by config registers (array size, `d`, `T`).

### Math

| Instruction | Operands                          | Semantics                                              |
| ----------- | --------------------------------- | ------------------------------------------------------ |
| `Matmul`    | `act_addr, weight_addr, out_addr` | `out = act @ weight` (ternary weight), on the MXU      |
| `VectorDot` | `vec0, vec1, out`                 | `out = Σ vec0 · vec1`, on the VPU reduction tree       |
| `VectorMul` | `vec, scalar, out`                | `out = vec × scalar` (broadcast scalar)                |
| `VectorAdd` | `vec0, vec1, out`                 | `out = vec0 + vec1` (residuals)                        |
| `ReLU`      | `vec, out`                        | `out = max(vec, 0)`                                    |
| `GeLU`      | `vec, out`                        | `out = gelu(vec)` (VPU LUT)                            |
| `VectorEMul`| `vec0, vec1, out`                 | `out[i] = vec0[i]·vec1[i]` (elementwise)              |
| `Square`    | `vec, out`                        | `out[i] = vec[i]²` (LayerNorm variance)               |
| `Exp`       | `vec, out`                        | `out = exp(vec)` (VPU LUT; softmax)                   |
| `ReduceMax` | `vec, out`                        | `out = max_i vec[i]` → scalar                         |
| `ReduceSum` | `vec, out`                        | `out = Σ_i vec[i]` → scalar                           |
| `Requant`   | `vec(int32), param, out(int8)`    | `out = clip((vec·m0+rnd)>>n)`, `{n,m0}=param` (VPU)    |
| `ScalarAdd` | `vec, scalar, out`                | `out[i] = vec[i] + scalar` (broadcast; `x−max`/`x−mean`)|
| `ScalarDiv` | `vec, scalar, out`                | `out[i] = round(vec[i]·2¹⁵ / scalar)` (Q15; softmax/LN)|

### Comms / Memory

| Instruction     | Operands                         | Semantics                                          |
| --------------- | -------------------------------- | -------------------------------------------------- |
| `WriteNeighbor` | `neighbor, my_addr, nb_addr`     | push `my_addr` (DRAM) → neighbor's DRAM via link   |
| `WriteMemory`   | `scratch_addr, dram_addr`        | DMA scratchpad → DRAM (spill)                       |
| `ReadMemory`    | `dram_addr, scratch_addr`        | DMA DRAM → scratchpad (fill)                        |

### Scalar (control)

Not enumerated in the overview but required to drive loops; minimal set:

| Instruction        | Semantics                              |
| ------------------ | -------------------------------------- |
| `AddS r, a, b`     | scalar int32 add                       |
| `MulS r, a, b`     | scalar int32 multiply                  |
| `CmpS a, b`        | set flags                              |
| `Branch cond, tgt` | conditional jump (loop control)        |
| `SetCfg reg, imm`  | load a config register (`T`, tile cnt) |
| `Wait unit`        | block until unit's `done`              |

## 4. Encoding

Fixed 32-bit instructions: `[ opcode:6 | dst:8 | src0:8 | src1:8 | flags:2 ]`, with an
`imm16` form (`[ opcode:6 | dst:8 | imm16:16 | flags:2 ]`) for `li`/`setcfg`/`branch`/
`jmp`. The three 8-bit fields name scalar registers whose *contents* are the byte
addresses handed to a unit. Opcode space is small (<32), leaving room for future fused ops
(e.g. `MatmulRequant`, `LayerNorm`). This is now concrete — the field layout, opcode
values, and forms are specified in full in [isa.md Appendix A.2–A.4](isa.md#a2-instruction-word).

## 5. Worked example — one attention block

Emitted micro-program for `MultiHeadAttention.forward` (`d=128`, `head_dim=16`, `T` tokens):

```
; projections (MXU), X already in scratchpad @X
Matmul  @X, @Wq, @Q          ; T×128 @ 128×128
Matmul  @X, @Wk, @K
Matmul  @X, @Wv, @V
; attention scores per head (VPU), head_dim=16
loop h in 0..q_heads:
  loop i in 0..T:            ; query token
    loop j in 0..i:          ; causal: keys 0..i
      VectorDot @Q[h,i], @K[h,j], @S[i,j]   ; QK^T / sqrt(d) folded into scale
    ; softmax over row S[i, 0..i]:
    ;   ReduceMax @S[i] -> m;  ScalarAdd @S[i], -m -> shifted (x-max)
    ;   Exp (LUT) -> e;  ReduceSum @e -> D;  ScalarDiv @e, D -> @P[i]
    ScalarDiv  @e[i], @D, @P[i]   ; p = e / Σe, in Q15 (no separate @recip needed)
    loop j in 0..i:
      VectorMul @P[i,j], @V[h,j], @acc      ; weighted V
      VectorAdd @acc, @A[h,i], @A[h,i]
Matmul  @A, @Wo, @O          ; output projection
VectorAdd @O, @X, @O         ; residual O + X
```

The scalar unit generates the loop indices and address arithmetic; MXU/VPU do the math.
(Attention here follows `mha_torch`; the causal mask is realized by only iterating
`j ≤ i`, which is cheaper than materializing the `-1e9` mask on hardware.)

## 6. Config registers

Host-visible, set before a run: `T` (tokens), `d`, `f`, array `ROWS/COLS`, tile counts,
per-tensor requant `scale`s, layer count, neighbor-present bitmap. These make the same
bitstream run different problem sizes without resynthesis.

## 7. Open questions

- Degree of scalar/compute overlap (issue-and-wait vs. run-ahead scoreboarding).
- Whether the instruction stream is generated on the host (compiler in `host/`) and
  loaded, vs. a fixed micro-ROM per layer type. Leaning host-generated for flexibility.
- Fused opcodes (`MatmulRequant`, `Softmax`, `LayerNorm`) to cut instruction count and
  scalar-unit overhead in the inner loops.
- Interrupt vs. poll for comms doorbells and DMA completion.
