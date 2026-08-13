# tpunn — running `model/transformer.py` on the TPU

> **Superseded — kept for the reasoning, not the design.** This plan was written
> against a model with LayerNorm, GELU, softmax, biases and `f=512`, and against
> an ISA with no `matmul_t`, no `vecmatmul` and no transposing DMA. Both have
> changed: the model is now bias-free, norm-free and uses ReLU attention, and
> the macro-ops run the tile loops in hardware. The result is that IMEM stopped
> being the binding constraint, so the ~82 per-op programs below collapse to
> **one**.
>
> §2's claim that "the hardware DMA only emits a 16-bit DRAM address" was also
> wrong, and load-bearing: the SRAM is 512 KB, the DMA and the UART host both
> address all 19 bits, and it was `scalar_unit.sv` that truncated a *program's*
> DRAM address to 16 bits. §3's wording ("DRAM *visible to a program*") is the
> accurate one. That truncation has since been removed, so the 64 KB budget
> this whole document is built around no longer exists.
>
> See [`adder_kernel.md`](adder_kernel.md) and
> [`examples/adder_model.tpu`](examples/adder_model.tpu) for what was actually
> built; §2 there compares the two directly. The parts of this document that
> still hold are the constraint table (§3, minus the DRAM row), the numerics
> discussion (§5) and the verification structure (§9).

**Status: plan, for review.** Nothing below is built yet. Read [`pytpu.md`](pytpu.md) (the
emitter), [`README.md`](README.md) (the language) and
[`../tpu/docs/isa.md`](../tpu/docs/isa.md) (the target) first; this document is only the
layer that turns a checkpoint into TPU programs.

---

## 1. The goal, stated precisely

Take `model/saved/colab_ternary_mha_int8act.pt` — the shipped `adder_ternary_vanilla`
(d=128, f=512, 4 layers, 4 heads, head_dim=32, vocab 13, T=32) with calibrated int8
activation scales — and execute its forward pass on the TPU, verified in the ISS.

Measured baseline, so we know what we are aiming at (256 random addition problems, T=32):

```
ternary weights + int8 activations, everything else float:
    exact-sequence accuracy  98.4%      token accuracy  99.9%
```

That is the number the integer pipeline has to stay close to. It is also the ceiling — the
weights are already ternary in the checkpoint, so ternarization costs nothing more.

**"On the TPU" means:** every matmul, the attention (both matmuls, the mask, the softmax),
GELU, both LayerNorms, every residual and every bias runs as TPU instructions on TPU
memory. Two things stay on the host, for reasons that are structural rather than
convenient:

| Host-side | Why |
| --- | --- |
| token embedding + positional encoding | the ISA has no gather; a table lookup is not a tensor op. The host produces `X0[T,128] int8` and that is the program's input. |
| `argmax` over 13 logits | `redmax` returns the max, not its index. The TPU produces the logits; the host reads the winner. |

Everything between those two is the TPU's.

---

## 2. Three ways to do this, and which one to pick

The question was: current pytpu, a torch compiler, or one giant `.tpu`.

| Approach | Verdict |
| --- | --- |
| **One giant `.tpu`** | **Impossible.** IMEM is 1024 words. The deleted `pytpu/PLAN.md` budgeted *one* layer of the *demo* config at ~896 words with hand-tuned templates and said so explicitly: a second layer does not fit. Four layers of the real config is not close. |
| **A torch compiler** (trace `Model.forward`, lower ATen to tpulang) | Right shape, wrong size. It needs an IR, a quantization pass, a memory planner and a lowering for every op — and then still has to solve the 1024-word problem underneath. Months of work to get to the same place. |
| **pytpu + a primitive library + a host sequencer** | **Pick this.** It is the giant-`.tpu` idea with the one change that makes it fit: *don't emit one program, emit ~80 small ones and let the host run them in order.* |

The whole trick is in that last row. `pytpu.md` §7 already names it — "Host-side sequencing
across programs: the real fix for *one layer exceeds 1024 words* — emit N programs, let DRAM
carry state." It costs almost nothing to build, because the ISS already supports it for
free: `tpu.run(words)` does not reset `tpu.mem` or `tpu.dram`, so calling it repeatedly on
one `TPU` object *is* a sequenced multi-program run. On the board, `run_program.py` already
does load-program → write-DRAM → run → read-DRAM; sequencing is that loop, N times.

Once programs are per-op, three constraints that dominated every previous design evaporate:

- **IMEM stops mattering.** The largest primitive is ~110 words against 1024. No fusion, no
  peeling, no branch-on-`kt` cleverness to save instructions.
- **64 KB of DRAM stops mattering.** The full model's ternary weights are 192 KB and the
  hardware DMA only emits a 16-bit DRAM address (`tpu_top.sv` line 31 — zero-extended to
  `MEM_ADDR_W`), so 64 KB is a hard ceiling on both the board *and* the ISS. With per-op
  programs the host stages only what the next op reads. The biggest single working set is
  W1 (16 KB) + its input (4 KB) + its output (16 KB) = 36 KB. It fits, at the real `d=128`,
  with no change to the RTL and no change to `iss.py`.
- **Debugging stops being archaeology.** Every intermediate tensor is `wrmem`'d and read
  back, so a wrong answer is localized to one op by construction (§9 leg B).

The cost is DMA round-trips between ops, and UART traffic on the board. §12 says what to do
about that later; neither affects correctness.

```
checkpoint.pt ──export──► int8 / ternary tensors + scales
                              │
model graph (Python) ─────────┼──► per-op pytpu Program ──► .tpu ──► assembler ──► words
                              │                                                      │
                              └──► host stages DRAM bytes ─────────────────► ISS ────┤
                                                                    (or UART/board)  │
                                   integer reference (pure torch) ◄── compare ───────┘
```

---

## 3. Hard constraints

Everything below is downstream of this. Sources are `iss.py`, `assembler.py`, `tpu_top.sv`,
`luts.py`.

| Limit | Value | Consequence here |
| --- | --- | --- |
| Instruction memory | 1024 words | Non-issue: largest primitive ~110 words. |
| DRAM visible to a program | **64 KB** (16-bit address, board and ISS alike) | The host stages a per-op working set. Drives §8. |
| Scratchpad | 64 KB | Never close; primitives use fixed small buffers. |
| MXU array | ROWS = COLS = 8 | Ternary matmuls tile K and N to multiples of 8. 128/512/384 all qualify. |
| `cfg tlen` | ≤ 63 | T = 32 tokens fit in **one** MXU pass, so MTILES = 1. |
| `cfg vlen` | ≤ 1023 | Longest row is f = 512. No chunking needed. |
| `li` immediate | 0..65535 accepted | Any 16-bit address `li`s directly. ≥ 0x8000 lands as a negative int32; every use masks mod 2¹⁶, so it is harmless — but keep loop *bounds* positive. |
| Scalar ALU | `adds` `subs` `muls` only | No shift, no divide. Drives the LayerNorm construction in §6.8. |
| requant | `clip_i8((acc·m0 + 2ⁿ⁻¹) >> n)`, m0 < 4096, n ≤ 15 | Scale ratios from 1/32768 to ~4095. Ample. |
| `gelu`/`exp` LUTs | fixed `in_scale = 1/16`; out 1/16 and 1/127 | **The tightest numerical constraint in the model.** §5. |
| ISS step limit | 100 000 default | Raise it per program if a primitive loops per element. |

Two semantics that cause most tpulang bugs, restated because every primitive below is
shaped by them:

- **VPU ops read int8 at stride 1 and write int32 at stride 4.** `requant` is the only op
  that narrows back. Every value a VPU op consumes must have been requantized to int8 first.
- **Reductions (`redmax`, `redsum`, `vecdot`) write one int32 to the scratchpad**, and
  broadcast scalars (`sadd`, `vecmul`, `sdiv`, `requant`) read their scalar from a
  scratchpad address. Reduction → broadcast is `reduce → loads → scalar math → stores`.

---

## 4. What gets built

```
accel/tpulang/tpunn/            (imports assembler/iss/luts/pytpu from the parent,
                                 the same sys.path dance torch_ref.py uses)
  quant.py      float -> int8 / ternary packing; {m0,n} selection; scale bookkeeping
  export.py     checkpoint -> a QuantModel: every weight as bytes, every scale as a float
  calibrate.py  one float forward over a calibration batch -> activation scales + clip rates
  prims.py      the eight primitives, each a function that fills a pytpu Program
  runtime.py    Device: stage(addr, bytes) / run(words) / read(addr, n). ISS now, UART later
  graph.py      the model forward, written as a sequence of primitive calls
  ref.py        the exact integer reference — the same pipeline in pure torch int64
  test_tpunn.py the driver: unit tests, full-forward equality, task accuracy
```

Estimate ~1500 lines. `prims.py` and `ref.py` are half of it.

**`pytpu.py` needs no changes.** It already emits everything these primitives use.

**`iss.py` needs no changes.** Per-op programs are why.

---

## 5. Numerics — the part that decides whether this works

Every tensor carries a compile-time `scale` (a float; `real ≈ int · scale`). The host, not
the hardware, keeps producer and consumer scales consistent. Nothing but ints ever reaches
the device.

- **Weight scales are free.** Ternary weights are already `{-1,0,1}·absmean` in the
  checkpoint, so `absmean` is just a factor in the following requant's `m0`.
- **Activation scales come from calibration**, not from guessing: `calibrate.py` runs the
  float model over a batch with hooks at exactly the points in §7's graph, and sets
  `scale = absmax / 127`. Where the checkpoint already carries an `act_scale` for a
  `TernaryLinear` input, that scale is used directly, so the integer pipeline enters each
  matmul at the scale the model was calibrated for.
- **Two scales are not ours to choose.** `gelu` and `exp` assume `in_scale = 1/16`, burned
  into the bitstream. So the requant feeding each must land there, and its `±7.94` domain is
  a real ceiling on the FF hidden state and on the post-max attention scores.
  `calibrate.py` **reports the clip rate at both sites** rather than assuming it is zero.
  If it is not small, that is a finding, not a bug to paper over.
- **`1/sqrt(head_dim)` is not an op.** It folds into the `m0` of the requant between
  `S32` and the exp scale.
- **Biases are added in int8 after the requant**, not in int32 before it. Adding before
  would mean the ternary matmul writes int32 to DRAM, and `[32,512] int32` = 64 KB does not
  fit next to its weights (§3). The integer reference does the identical thing, so
  exactness (§9 leg B) is unaffected; the accuracy cost shows up in leg C, where it belongs.

---

## 6. The primitive library

Eight primitives. Each is a Python function `(p: Program, **params) -> None` that emits into
a fresh `pytpu.Program`, reads its inputs from DRAM and writes its outputs to DRAM. No
register and no cfg register is live across a primitive — every one sets `tlen`/`vlen`/
`len`/`scalar` before use, because stale config is the ISA's classic silent bug.

The design of 6.1–6.8 is lifted from the deleted `pytpu/PLAN.md` §7, which worked these out
carefully; the change is that they are pytpu functions instead of `{{param}}` text
templates, and each gets its own program instead of being inlined into a shared one.

| # | Primitive | Contract | ~words |
| --- | --- | --- | --- |
| 6.1 | `matmul_ternary` | `C[M,N] int8 = requant(A[M,K] int8 @ W[K,N] ternary)` | 90 |
| 6.2 | `matmul_i8` | `O[H,M,N] int32 = A[H,M,K] @ B[H,N,K]ᵀ`, fully strided | 55 |
| 6.3 | `transpose_i8` | `B[N,M] = A[M,N]ᵀ`, int8 — now one `wrmem.t` | 10 |
| 6.4 | `requant_block` | `dst[N] int8 = requant(src[N] int32)` | 25 |
| 6.5 | `add_rows` | `C[R,N] int8 = requant(A[R,N] + B[R,N])`, B row-stride may be 0 | 30 |
| 6.6 | `gelu_block` | `dst[N] int8 = requant(gelu(requant(src)))` | 25 |
| 6.7 | `softmax_rows` | `P[R,L] int8 @ 1/128 = softmax(S[R,L] int8 @ 1/16)` | 48 |
| 6.8 | `layernorm_rows` | `Y[R,L] int8 = γ·(x−μ)/σ + β` | 110 |

**6.1 `matmul_ternary`** — `examples/tiled_matmul.tpu` generalized, which
[`examples/gen_tiled_matmul.py`](examples/gen_tiled_matmul.py) has already done in pytpu and
proven byte-identical to the hand-written original. Two changes from that example: **A and C
are plain row-major, not tile-major** (the VPU primitives have no notion of tiles, so a
tile-major output could not feed `softmax_rows` or `layernorm_rows` without a relayout — the
A tile is gathered with T fills at DRAM stride K and the C tile scattered the same way), and
W stays tile-major `[nt][kt]` because `quant.pack_ternary` lays it out offline. `matmul` on
the first K tile, `matmul.acc` on the middle, `matmul.acc.rq` on the last, so the int32
accumulation runs the full K before a single requant.

**6.2 `matmul_i8`** — `examples/vpu_matmul.tpu` generalized to a batched, fully strided form:
one `vecdot` per `(h, m, n)` with `vlen = K`. Attention needs both of its matmuls this way
because neither operand is ternary. Explicit strides are what let one template serve
`S = Q Kᵀ` (whose Q is a strided sub-block of the fused `[T,384]` QKV result — each
`(head, token)` row is still 32 contiguous bytes, which is all `vecdot` needs, so Q/K/V never
have to be un-interleaved), `A = P Vᵀ`, **and** the final `fc` head with int8 weights.

**6.3 `transpose_i8`** — `vecdot` needs both operands contiguous along the contraction axis,
and `P @ V` contracts over keys, so V must be key-contiguous.

*Superseded by hardware.* This was written when there was no transpose op and no strided
vector load, and used the fact that **DMA with `len = 1` is a byte move to an arbitrary
address**: loop over elements, `rdmem` one byte from `A + m·N + n`, `wrmem` it to
`B + n·M + m`. Tiny in instructions, correct, and brutally slow — two dispatches per byte.

The DMA now transposes natively ([`../tpu/docs/dma.md`](../tpu/docs/dma.md) §5), so the
primitive is **one `wrmem.t`** (or `rdmem.t`, whichever side the tensor is already on) with
`cfg tcols/tsrow/tdrow` set: ~10 words instead of 27, and one dispatch instead of ~8k for a
`[32,128]` V. `tsrow` reads the V columns straight out of the fused `[T,384]` QKV block, so
step 7 of §7 no longer needs V un-interleaved first either. The ~30 k ISS steps quoted in §3
go with it.

**6.5 `add_rows`** — one primitive covers three jobs, because a row-stride of 0 on the second
operand is a broadcast: residual add (stride N), bias add (stride 0), attention mask
(stride 0 per head). The causal+padding mask is an int8 tensor of `0` and `−128`; adding
`−128` at the exp scale drives an entry to the clip floor, which is `x = −8`, where the exp
table stores 0. **The mask is data, not an instruction.**

**6.7 `softmax_rows`** — the `isa.md` §5 sequence made type-correct, which is exactly what
`examples/softmax_row.tpu` gets wrong (and is registered in `torch_ref.UNSUPPORTED` for):

```
redmax  m        ; int32 scalar in scratchpad
loads / subs / stores            ; nega = -m, staged back as a broadcastable scalar
sadd    shift32 <- row, nega     ; int32, <= 0
requant shift8  <- shift32 {1,0} ; int8; clips at -128 == x = -8, where exp -> 0
exp     e32     <- shift8        ; int32 at exp's out_scale 1/127
requant e8      <- e32   {1,0}   ; int8
redsum  D       <- e8            ; int32 scalar
sdiv    p32     <- e8, D         ; int32 Q15: round(e_i * 2^15 / D)
requant p8      <- p32   {1,8}   ; int8 at the pinned probability scale 1/128
```

Every `requant` there exists because the next op reads int8. None is decoration.

**6.8 `layernorm_rows` — divide and sqrt with neither a shift nor a divide.** The interesting
one, and the reason a naive "just lower LayerNorm" plan fails. Per row of length L:

1. `redsum` → `S = Σ xᵢ`; `μ = trunc(S/L)` in scalar code (below).
2. `sadd` by `−μ` → int32; `requant {1,0}` → `c8` (int8).
3. `vecdot(c8, c8)` → `SS = Σ(xᵢ−μ)²` as one int32. **`vecdot`, not `square`+`redsum`** —
   `square` writes int32 and `redsum` reads int8, so the obvious spelling is precisely the
   type error above.
4. `R = ⌊√SS⌋` in scalar code (below).
5. `sdiv(c8, R)` then `requant` to the output scale. Normalizing by `R = √(Σc²)` instead of
   by `σ = √(Σc²/L)` leaves a constant `√L`, folded into that requant's `m0` at compile time.
   Nothing at run time ever divides by L or √L in the vector path.
6. `vecemul` by γ → requant → `vecadd` β → requant.

The scalar unit has `adds`, `subs`, `muls`, `cmps` and nothing else, so neither a restoring
sqrt (needs `bit >>= 1`) nor Newton (needs a divide) can be written directly. The way
through: **the scratchpad is the shifter.**

```
build:  b = 1; for k in 0..15: stores tbl+4k, b; b = b + b      ; powers of two, ascending
walk:   q = 0; ptr = tbl + 60
        loop: loads b, ptr                                      ; ...read back descending
              t = q + b
              if f(t) <= v: q = t                               ; muls + cmps + branch
              ptr -= 4; repeat while ptr >= tbl
```

Writing the sequence to memory ascending and reading it back descending *is* the right shift
the ISA does not have. One table, built once; the walk is instantiated twice with `f(t)=t·L`
(giving `⌊v/L⌋`, binary long division) and `f(t)=t·t` (giving `⌊√v⌋`), ~12 instructions each,
exact for `v < 2³¹`. Here L = 128 and `SS ≤ 128·127² ≈ 2.1e6`, so `R ≲ 1450` and the widest
intermediate is the first probe `(2¹⁵)² = 2³⁰` — inside int32.

**This requires one addition to pytpu**: `p.loop` is a counted loop and there is no `p.if_`
(`pytpu.md` §7 defers it deliberately). The bit walk needs a data-dependent branch. The
minimal change is `p.cmps(a, b)` plus a `with p.if_(cond):` context manager emitting
`cmps`/`b<cc>`/label — about 20 lines, and the first case that actually justifies it.

**Precision, stated rather than buried:** `c8` is int8, so a row entry more than 127
quantization steps from its mean is clipped. That is what an int8 LayerNorm is; the integer
reference clips identically, so exactness still holds and the cost lands in leg C.

---

## 7. The model as a program sequence

`Transformer.forward` is `X1 = norm1(X + attn(X))`, `Y = norm2(X1 + ff(X1))`, post-norm,
`use_moe=False`, activation `nn.GELU`. Per layer, 20 programs:

| # | op | primitive | out shape / dtype |
| --- | --- | --- | --- |
| 1 | `QKV = X·[Wq\|Wk\|Wv]` | `matmul_ternary` | `[32,384]` int8 |
| 2 | `+ bias_qkv` | `add_rows` (stride 0) | `[32,384]` int8 |
| 3 | `S32 = Q·Kᵀ` per head | `matmul_i8` | `[4,32,32]` int32 |
| 4 | requant into exp scale, `1/√32` folded | `requant_block` | `[4,32,32]` int8 |
| 5 | `+ causal/pad mask` | `add_rows` (stride 0) | int8 |
| 6 | `P = softmax(S8)` | `softmax_rows` | int8 @ 1/128 |
| 7 | `Vᵀ` | `transpose_i8` (one `wrmem.t`) | `[4,32,32]` int8 |
| 8 | `AO32 = P·Vᵀᵀ` per head | `matmul_i8` | `[32,128]` int32 |
| 9 | requant | `requant_block` | `[32,128]` int8 |
| 10–11 | `O = AO·Wo`, `+ bias_o` | `matmul_ternary`, `add_rows` | `[32,128]` int8 |
| 12 | `X1 = X + O` | `add_rows` | int8 |
| 13 | `N1 = norm1(X1)` | `layernorm_rows` | int8 |
| 14–15 | `H = N1·W1`, `+ bias1` | `matmul_ternary`, `add_rows` | `[32,512]` int8 |
| 16 | `HG = gelu(H)` | `gelu_block` | `[32,512]` int8 |
| 17–18 | `F = HG·W2`, `+ bias2` | `matmul_ternary`, `add_rows` | `[32,128]` int8 |
| 19 | `X2 = N1 + F` | `add_rows` | int8 |
| 20 | `Y = norm2(X2)` | `layernorm_rows` | int8 |

Then once: `logits = Y·fcᵀ + fc_bias` via `matmul_i8` (the head is *not* ternary in the
checkpoint, so it is quantized to int8 and run on the VPU) + `add_rows`.

**Q, K and V come from one fused matmul.** `[Wq|Wk|Wv]` is a single `[128,384]` ternary
weight, so the projection is one program instead of three. It costs nothing in addressing
because `matmul_i8` takes explicit strides (§6.2). It is also what keeps the working set
inside 64 KB.

**82 programs per forward.** With `kv_heads = 4` and `heads_per_q = 1`, the 5-D layout of
`mha_torch` collapses to `[T, 4, 32] = [T, 128]`, so head `h` is a contiguous 32-byte slice
of each 128-byte row — no reshape anywhere in the sequence.

**Not modelled, deliberately:** dropout (inference only, `model.eval()`).

---

## 8. Memory: the host owns DRAM

Each program is a pure function on a DRAM working set. Before running program *k*, the host
writes exactly the bytes it reads; afterwards it reads back exactly the bytes it wrote. No
tensor has to survive across programs, so the 64 KB space is re-used freely and its layout is
recomputed per program by a small bump `Arena` (named allocations, hard limit, so an overflow
names the tensor that broke the budget instead of wrapping an address).

Largest working sets, all comfortably inside 64 KB:

| program | in | weights | out | total |
| --- | --- | --- | --- | --- |
| `H = N1·W1` | 4 KB | 16 KB | 16 KB | 36 KB |
| `F = HG·W2` | 16 KB | 16 KB | 4 KB | 36 KB |
| `QKV = X·Wqkv` | 4 KB | 12 KB | 12 KB | 28 KB |
| `AO32 = P·Vᵀᵀ` | 4+4 KB | — | 16 KB int32 | 24 KB |

On the board this is the real cost of the design: ~200 KB of weight bytes cross the UART per
forward, ~20 s at 115200 baud. That is a throughput problem, not a correctness problem, and
§12 says what to do about it. In the ISS staging is a `bytearray` slice assignment.

---

## 9. Verification

Four legs, weakest to strongest. Legs A and B are the ones that say *correct*; C says
*accurate*; D says *the hardware agrees*.

**Leg A — per-primitive unit tests.** Each of the eight primitives, on random inputs, against
a small numpy/torch integer reference. Bugs are 50× cheaper to find here than in a
full forward, and this is where the LayerNorm bit-walk and the strided `matmul_i8` get
hammered.

**Leg B — whole-model bit-exactness.** `ref.py` implements the identical integer pipeline in
pure torch `int64` — same `{m0,n}` words, same LUTs (imported from `luts.py`, since the ROM
contents are a hardware fact), same flooring shifts, and its own `trunc_div`/`isqrt` written
the obvious way as a check on the table walk. It shares no arithmetic with `iss.py`. Because
every op is its own program, **all 82 intermediates are compared, byte for byte, zero
tolerance** — so a mismatch names the op that broke rather than the forward that did.
`ref.py` does share the memory map and the scale choices with `graph.py`; that is a real
limit on its independence, because those *are* the declaration of what the program means.

**Leg C — task accuracy.** 256 random addition problems end to end, reporting
exact-sequence and token accuracy against the 98.4% / 99.9% float-with-fake-quant baseline
in §1, plus the clip rates at the two LUT sites. **Reported, not asserted** — an int8
LayerNorm and a `1/16` exp table will not match float exactly, and choosing a threshold in
advance would be pretending otherwise. Leg B is what says the program is right; leg C says
what quantization costs.

Leg C runs the *reference*, not the ISS, for the 256-problem sweep. That is legitimate
precisely because leg B has already proved the two byte-identical on full forwards — and it
matters, because a full ISS forward is ~25 M Python-level integer multiplies (~3000 MXU
passes/layer at ROWS=COLS=8) and takes minutes. Leg B runs on a handful of sequences; leg C
runs on 256. If the ISS turns out to be the bottleneck even for leg B, the contingency is a
numpy inner loop in `iss._matmul` — same arithmetic, so bit-exactness is preserved.

**Leg D — hardware, later.** `gen_vectors.emit_*` already writes the three `$readmemh` files
`tpu_top_tb.sv` consumes, so any single generated program can be run in RTL simulation with
no testbench change. `runtime.py`'s `Device` interface has one ISS implementation and one
UART implementation against `host/tpu_uart.py`; the same `graph.py` drives both. That is the
payoff of §2's structure, and it is out of scope for this plan's acceptance.

---

## 10. Milestones

**M0 — the integer reference, alone, before any tpulang** (~1 day). `export.py` +
`calibrate.py` + `ref.py`, run over 256 problems in pure Python. This answers the only
question that can kill the project — *does the fully-integer model still do addition?* — and
answers it with no TPU code written. It also produces the clip rates at the `gelu` and `exp`
sites, which is where I expect trouble if there is any. **Everything after this is
mechanical; this is the one step with a real unknown, so it goes first.**

**M1 — primitives.** `prims.py` + `runtime.py`, leg A green for all eight. Includes the
`p.if_` addition to pytpu (§6.8).

**M2 — one layer.** Programs 1–20 of §7 for layer 0, leg B green on all 20 intermediates.

**M3 — the whole model.** All 4 layers + the head; leg B green on a handful of sequences,
leg C reported on 256.

**M4 — write-up.** A section in [`README.md`](README.md), and this file updated from *plan*
to *built* with the numbers that actually came out.

---

## 11. Risks, and the two questions I'd like answered

1. **The `1/16` LUT domain is the tightest thing in the model** (§5). Attention scores after
   the max-subtract and the FF hidden state both have to live inside `±7.94` or they clip,
   and that ceiling is in the bitstream, not in the program. M0 measures it. If the FF
   hidden state clips badly, the fix is a `gelu` LUT regenerated at a different `in_scale` —
   which invalidates every bitstream and every compiled program, so it is a decision, not a
   patch. **Q: is regenerating the LUTs on the table if M0 says it is needed?**
2. **Biases in int8 after the requant, not int32 before** (§5). Cheap and it keeps the
   working set inside 64 KB. The precise alternative is splitting each matmul over column
   blocks so the int32 output fits — roughly 4× the programs for that op. M0 measures what
   the int8 bias actually costs. **Q: acceptable, or should I plan for the column-split from
   the start?**
3. **Sequence length.** This plan uses T = 32, which is `train.py`'s context length and fits
   `cfg tlen` in one MXU pass. `test_inference.py` uses 64, which needs MTILES = 2 — supported,
   just slower. Assuming 32 unless told otherwise.
4. **ISS runtime** — mitigated by §9's split of leg B and leg C, with the numpy contingency
   behind it. Low risk, but it is the thing most likely to make the loop annoying to work in.

---

## 12. Deliberately out of scope

| Not now | What would trigger it |
| --- | --- |
| Fusing ops into per-layer programs | The obvious 2–3× win once correctness is settled. Needs a liveness pass over the op list — the first real step toward being a compiler. |
| Resident weights / a DRAM parameter block | The answer to §8's 20 s of UART per forward on the board: load each primitive once, drive it from a parameter block in scratchpad. Needs `loads` from a `rdmem`'d block into registers — supported today, just unnecessary until the board path matters. |
| Autoregressive decode | Not needed: the answer tokens sit at fixed positions after `EQUALS_POS = 15`, so one forward pass produces the whole answer. |
| Embedding and `argmax` on device | §1. Would need a gather and an argmax the ISA does not have. |
| ~~A transposing DMA mode~~ | **Built** — `rdmem.t`/`wrmem.t`, `dma.md` §5. `transpose_i8` (§6.3) is now one dispatch. |
| A torch compiler | §2. `graph.py` being a plain Python function *is* the honest version of this at this scale. |
