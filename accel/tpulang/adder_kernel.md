# adder_kernel — running `adder_ternary_vanilla` on the TPU

[`examples/adder_model.tpu`](examples/adder_model.tpu) is the whole shipped
ternary adder model — four transformer layers and the output head — as **one
program, one run**. This document is the byte-level contract the host has to
satisfy to drive it.

> **Status: built and verified; the kernel is correct and the checkpoint is not
> int8-quantizable.** The program assembles to 196 words and runs to completion
> in both the ISS and the RTL. It matches an independent PyTorch implementation
> byte-for-byte (24 992 elements), the RTL matches the ISS on all 26 240 output
> bytes, and on the *real* checkpoint the ISS matches the reference logits
> exactly. But the shipped model scores **96.09% exact-sequence in float and
> 0.00% with int8 activations** — a property of the model (no LayerNorm), not of
> this kernel, established four ways in §7.6. Not yet run on the board.

Read [`README.md`](README.md) (the language), [`../tpu/docs/isa.md`](../tpu/docs/isa.md)
(the target) and [`../tpu/docs/macro_ops.md`](../tpu/docs/macro_ops.md) (the
macro-ops this leans on) first.

This supersedes [`tpunn.md`](tpunn.md), which was written against an earlier
model (LayerNorm, GELU, softmax, biases, `f=512`) and an earlier ISA (no
`matmul_t`, no `vecmatmul`, no transposing DMA). §2 says what changed.

---

## 1. The model, exactly as it is today

`model/transformer.py::adder_ternary_vanilla` — `d=128`, `f=128`, `layers=4`,
`q_heads=kv_heads=4`, `head_dim=32`, `vocab=13`, `use_ternary=True`,
`use_bias=False`. Sequence length `T=32` (`train.py --max_tokens`). Inference,
so dropout is off.

```
for L in 0..3:
    Q, K, V = Wq(X), Wk(X), Wv(X)            # ternary, no bias
    S       = Q @ K^T / sqrt(head_dim)       # per head
    P       = relu(S + causal_mask)          # ReLU attention, NOT softmax
    A       = P @ V                          # [T, 4*32] -> [T, 128]
    X       = X + Wo(A)                      # attention residual
    X       = X + W2(relu(W1(X)))            # feed-forward residual
logits = X @ fc.weight^T                     # fc is int8, not ternary
```

There is **no LayerNorm** — `norm1`/`norm2` are commented out — and **no bias**
anywhere. `mha_torch` ignores the padding mask it is handed, so the only mask
is the causal one.

Two things stay on the host, structurally rather than by convenience:

| Host-side | Why |
| --- | --- |
| token embedding + positional encoding | the ISA has no gather. The host produces `X0[T,128] int8`; that is the program's input. |
| `argmax` over 13 logits | `redmax` returns the max, not its index. The TPU produces the logits. |

Everything between those two is the TPU's.

---

## 2. One program, and the change that made it one

The four layers' ternary weights are 96 KB packed (`Wq|Wk|Wv` is 128×384 trits
= 12 KB; `Wo`, `W1`, `W2` are 128×128 = 4 KB each). Until now that was
unreachable from a program, for a reason that turned out not to be about memory
at all:

| Stage | Address width | Where |
| --- | --- | --- |
| SRAM chip | **19 bits = 512 KB** | `tpu_top.sv` `MEM_ADDR_W = 19` (Cmod A7, 512K×8) |
| host over UART | 19 bits | `uart_interface #(.ADDR_W(MEM_ADDR_W))`; `tpu_uart.py`'s `R`/`W` carry a 24-bit field checked against `2^19` |
| DMA engine | 19 bits | `dma.sv` |
| **program** | ~~16 bits~~ → **19 bits** | `scalar_unit.sv` — was `r_src1[ADDR_W-1:0]`, `ADDR_W = 16` |

`tpu_top.sv` then zero-extended that 16-bit address straight back to 19 on the
way into the DMA. So the chip had 512 KB, the host could write all of it, and a
`rdmem`/`wrmem` could reach the first eighth — which is what made a 64 KB
program window look like a hardware ceiling. `iss.py` modelled the program's
view (`dram = bytearray(1 << addr_w)`), which hid it further.

### What changed

| File | Change |
| --- | --- |
| `rtl/scalar_unit.sv` | new `MEM_ADDR_W = 19` parameter; `dma_dram_addr` port widened; the `rdmem`/`wrmem` address now takes `r_src*[MEM_ADDR_W-1:0]` while `dma_scratch_addr` stays `ADDR_W`. LINK's `nb_*_addr` deliberately left at `ADDR_W` — no comms block is attached. **`li` now zero-extends** (below). |
| `rtl/tpu_top.sv` | `dma_dram_addr` declared at `MEM_ADDR_W`; `MEM_ADDR_W` passed into `scalar_unit`; the zero-extend at the DMA instantiation is now a direct connect. |
| `iss.py` | new `mem_addr_w = 19` field; `dram` sized off it; new `_d()` DRAM mask beside the scratchpad's `_a()`; `_dma` and the `OP_RDMEM`/`OP_WRMEM` dispatch mask each side in its own space; `OP_LIS` zero-extends. |
| `assembler.py` | `li` is range-checked unsigned; a negative immediate is now an error naming the `li`/`subs` idiom instead of silently delivering `imm & 0xFFFF`. |
| `tb/tpu_top_tb.sv`, `tb/tpu_top_uart_tb.sv` | `DEPTH` (the span scanned for input/expected DRAM bytes) sized off `MEM_ADDR_W` instead of `ADDR_W` (below). |
| `tb/tpu_top_tb.sv` | `dram_wr_byte`/`dram_rd` backdoor widened to `MEM_ADDR_W` (below). Also `WATCHDOG_NS` and `NO_VCD` overrides, since a full-model run is ~2.4 M clocks and dumps 400+ MB of waveform at the default settings. |
| `gen_vectors.py` | `MEM_ADDR_W` constant; DRAM pokes and `check_reference` use `_d()`; `TPU(...)` passes `mem_addr_w`. |
| `torch_ref.py`, `host/run_program*.py`, `examples/gen_tiled_matmul.py` | pass `mem_addr_w=gv.MEM_ADDR_W`. |

### Three things this broke, found by running it

None was predicted. The common shape is worth naming up front, because it cost
three separate debugging rounds: **`ADDR_W` was being used for things that are
DRAM, not scratchpad**, and every one of them was invisible while a program
could only reach the low 64 KB — where the two widths agree. A name like
`DEPTH` or `dram_wr_byte(addr)` does not say which memory it means, and none of
these failed loudly the first time.

The first is the reason `li` changed:

**`li` had to stop sign-extending.** `li r, 0x8000` put `0xFFFF8000` in the
register. Masked to the scratchpad's 16 bits that wraps back to `0x8000`, so it
had always worked; masked to a 19-bit DRAM address it becomes `0x78000`. That
silently broke `vpu_matmul.tpu`, whose `S8` tensor sits at DRAM `0x8000` — the
reference reported *no output byte at 0x08000* because the program had written
0x78000 instead. So the claim that "every existing program addresses below
64 KB" was wrong: it holds for `0..0x7FFF`, not for `0x8000..0xFFFF`. Zero
extension makes a register hold what was written, which is what an address
wants; nothing in the tree used a negative `li`, and every `cmps` is on a small
positive loop counter, so nothing else moved.

**The testbenches could not *see* DRAM above 64 KB.** `in_bytes`/`exp_bytes` are
*DRAM* byte maps but were sized `1 << ADDR_W`. `$readmemh` reported "address out
of range" for every high expectation, dropped it, and the run-scanner then found
nothing there — so a program writing high DRAM scored as passing with none of it
checked. Fixing the span took `highmem_dma`'s program-A check count from 114 to
306. **A silently-skipped expectation is worse than a failing one.**

**And `tpu_top_tb`'s backdoor could not *reach* it either.** `dram_wr_byte` and
`dram_rd` took `[ADDR_W-1:0]`, and both call sites truncated to match. So the
preload poked every weight byte above 64 KB to `addr & 0xFFFF`, and the readback
checked `0x21ff5` by reading `0x1ff5`. The full model came back 26 169 / 26 240
wrong — all of them `got xx`, i.e. *never written*, which is the signature of a
wrong address rather than wrong arithmetic. The comment above those tasks
documented the assumption that had just stopped being true: "tensor addresses
(< 2^ADDR_W) index directly into the wider DRAM".

`tpu_top_uart_tb` was unaffected because it stages and reads over the real UART
`W`/`R` commands, which have carried a 19-bit address all along — which is
exactly why `highmem_dma` passed there while the backdoor TB was still broken.

### What it bought

| | before | after |
| --- | --- | --- |
| runs per forward | 5 (layer ×4, then the head) | **1** |
| `X` between layers | DRAM round trip each time | resident in the scratchpad |
| weight traffic per forward | 96 KB, re-staged every forward | 96 KB **once**; weights are read-only and stay in SRAM |

That last row is the real payoff. Weights never change, so after one ~9 s
upload at 115200 baud, each subsequent forward uploads only the 4 KB `X` and
reads back 1.7 KB of logits — roughly half a second instead of nine.

### Versus `tpunn.md`

| | `tpunn.md` (pre-macro-op) | here |
| --- | --- | --- |
| binding constraint | **IMEM** — one layer was 861 of 1024 words | none: 196 of 1024, for *four* layers plus the head |
| programs per forward | ~82, one per op | **1** |
| tile loops | expanded in the program | `matmul_t` / `vecmatmul` run them in hardware |
| transpose | a byte-at-a-time DMA loop | one `wrmem.t` |

The ops that plan had to design around are simply gone: no LayerNorm means no
`rsqrt` LUT and no scalar bit-walk for integer division and square root (§6.8
there — the hardest piece in the whole plan); no softmax means the `exp` LUT's
fixed `1/16` input scale, which that document called "the tightest numerical
constraint in the model", does not constrain anything here. ReLU attention is
one `relu` instruction. **The model got easier to lower at the same time the
ISA got stronger.**

---

## 3. Memory maps

### DRAM (512 KB available; the top byte used is `0x21FFF`, i.e. 136 KB)

Globals live below `0x5000` so each fits a 16-bit `li`. The four layer blocks
sit above and are reached through the `wl` base register.

| Addr | Bytes | Tensor | Written by |
| --- | --- | --- | --- |
| `0x0000` | 4096 | `X0` `[32][128]` int8 — embedded + PE'd input | host |
| `0x1000` | 1024 | causal mask `[32][32]` int8 | host |
| `0x1400` | 4096 | `A` `[32][128]` int8 — scratch | device |
| `0x2400` | 4096 | `V^T` `[128][32]` int8 — scratch | device |
| `0x3400` | 1664 | `W_fc` `[13][128]` int8 | host |
| `0x3C00` | 1664 | logits `[32][13]` **int32** — **the result** | device |
| `0x5000 + L*0x7400` | 29696 | layer `L`'s block, below | |

Inside layer `L`'s block:

| Offset | Bytes | Tensor |
| --- | --- | --- |
| `+0x0000` | 12288 | `[Wq\|Wk\|Wv]` `[128][384]` trits, col-major 2-bit |
| `+0x3000` | 4096 | `Wo` `[128][128]` trits |
| `+0x4000` | 4096 | `W1` `[128][128]` trits |
| `+0x5000` | 4096 | `W2` `[128][128]` trits |
| `+0x6000` | 52 | this layer's 13 requant `{m0,n}` words |
| `+0x6400` | 4096 | `X` after this layer — a **checkpoint**, device-written |

The requant words live inside the layer block so the program can reload them
into one fixed scratchpad slot each iteration, which keeps every
`setcfg scalar` a compile-time immediate instead of needing `setcfgr`. The
per-layer `X` spill is for verification (§7), not a data path — `X` itself
stays in the scratchpad the whole run.

### Scratchpad (64 KB; top byte `0xC833`)

A 12 KB weight window (refilled four times per layer), the 12 KB fused QKV
block, and small fixed buffers — 51 KB of 64, laid out in the program's `.equ`
block. The window is sized by the largest single fill (`[Wq|Wk|Wv]`), which is
why the three projections share one `rdmem`.

---

## 4. Numerics — the part the host has to get right

Every tensor carries a compile-time scale: integer `v` means real `v · s`. The
host, not the hardware, keeps producer and consumer scales consistent. Nothing
but integers reaches the device.

`requant` computes `clip_i8((acc·m0 + 2^(n-1)) >> n)`, so it implements a
multiplier `m ≈ m0/2^n` with `m0 < 4096` and `n ≤ 15` — representable range
`2^-15` to `4095`. Pick the largest `n ≤ 15` for which `round(m·2^n) ≤ 4095`.

Ternary weights are already `trit · absmean` in the checkpoint
(`TernaryLinear.forward`: `RoundClip(w/scale) * w.abs().mean()`), so `absmean`
is never a tensor — it is just a factor in the following requant's `m`.

### The 13 words, per layer, in block order

`a_M` is matrix `M`'s `absmean`; `s_T` is tensor `T`'s scale.

| # | Slot | Op | `m = m0/2^n` |
| --- | --- | --- | --- |
| 0 | `RQ_Q` | `X @ Wq → Q` | `s_x · a_q / s_q` |
| 1 | `RQ_K` | `X @ Wk → K` | `s_x · a_k / s_k` |
| 2 | `RQ_V` | `X @ Wv → V` | `s_x · a_v / s_v` |
| 3 | `RQ_S` | `Q@K^T int32 → S8` | `s_q · s_k / (sqrt(32) · s_s)` |
| 4 | `RQ_ID` | mask clamp | **`{m0=1, n=0}`** |
| 5 | `RQ_P` | `relu(S8) → P8` | `s_s / s_p` — **`{1,0}`** is natural |
| 6 | `RQ_A` | `P@V int32 → A8` | `s_p · s_v / s_a` |
| 7 | `RQ_O` | `A8 @ Wo → O` | `s_a · a_o / s_x` ← **must land on `s_x`** |
| 8 | `RQ_X1` | `X + O → X1` | `s_x / s_x1` |
| 9 | `RQ_H` | `X1 @ W1 → H` | `s_x1 · a_1 / s_h` |
| 10 | `RQ_HR` | `relu(H) → HR` | `s_h / s_hr` — **`{1,0}`** is natural |
| 11 | `RQ_F` | `HR @ W2 → F` | `s_hr · a_2 / s_x1` ← **must land on `s_x1`** |
| 12 | `RQ_X2` | `X1 + F → X2` | `s_x1 / s_x2` |

Four constraints, stated flat because they are where a plausible host will get
it wrong:

- **`vecadd` adds int8 operands, so a residual add is only meaningful if both
  sides share a scale.** That pins `RQ_O` to `s_x` and `RQ_F` to `s_x1`. Those
  two matmul outputs do not get to choose their own scale, so they are the two
  most likely places to clip. Measure them.
- **`1/sqrt(head_dim)` is not an op.** It folds into `RQ_S`.
- **Q, K and V get separate requant words** even though they come from one
  weight block, because each has its own `absmean`. Fusing them into a single
  48-tile `matmul_t` would force one multiplier on all three; three 16-tile
  dispatches cost six instructions and keep each projection's scale its own.
- **The mask needs no scale.** It is `-128` at masked positions and `0`
  elsewhere; `S8 - 128 ≤ -1` for every `S8` in int8 range, so a masked entry is
  negative *whatever* `s_s` is, and ReLU takes it to exactly zero. Exact, not a
  tolerance.

### Where the scales come from

**All of them are calibrated; the checkpoint supplies none.** `TernaryLinear`
carries an `act_scale` buffer that would give four of them per layer, but in
`colab_ternary_mha_small_nobias.pt` all 24 are `NaN` — the model was never run
through `calibrate_activations`. [`adder_export.py`](adder_export.py) therefore
derives every scale itself, from one instrumented float forward over a
calibration set disjoint from the evaluation set (`absmax / 127`, the same
symmetric-int8 convention).

Two of the thirteen are not the obvious thing, and both were found by measuring
rather than by deriving:

- **`s_s` is calibrated on the scores that survive, not on the widest score.**
  `S8` feeds `relu(S8 + mask)`, so a large *negative* score contributes nothing:
  it clips to `-128` and ReLU takes it to exactly 0, which is the correct
  answer. On this checkpoint layer 3's widest score is `1.46e6` while the widest
  *surviving* one is `4.6e3` — 300×, i.e. ~8 of the 8 bits spent on values that
  are discarded. Pinning `s_s` to the post-ReLU range is also what makes `RQ_P`
  the `{1,0}` identity this design expected all along.
- **The residual pair's shared scale is *not* widened to cover the addend.**
  `vecadd` adds int8 operands, so `O` lives at `s_x` whether or not `s_x` is
  wide enough — and on this checkpoint `|O|` reaches 55× `|X|`. Widening `s_x`
  to `max(|X|,|O|)/127` removes the `RQ_O` clipping §8.3 predicted, but it is a
  net loss, because `X` is the same tensor the Q/K/V/W1 projections read and it
  collapses to ~2 of 127 levels. Measured over 64 problems: tight 63.18% token,
  widened 44.63%, and a rescaled second copy of `X` for the add (one extra
  requant slot plus a VPU pass per residual) 69.73%. The kernel keeps the tight
  scale and reports the clip rate.

### Byte layouts

**Ternary weight `W[K][N]`** — column-major, 2 bits per weight, `00 → 0`,
`01 → +1`, `11 → −1`. Column `n` occupies `K·2/8 = 32` bytes:

```
trit[k][n] = round(w[k][n] / (absmean + eps)).clip(-1, 1)     # w is [in, out]
byte  base + n*32 + k//4,  bits 2*(k % 4)
```

`TernaryLinear.w` is already `[in_dim, out_dim]`, so no transpose is needed.
In the fused block, columns `0..127` are `Wq`, `128..255` `Wk`, `256..383` `Wv`.

**Causal mask** `[32][32]` int8 row-major: `0` where `s <= t`, `-128` above.

**Activations** `[T][D]` int8 row-major. **`W_fc`** `[13][128]` int8 row-major
— `nn.Linear` already stores `[out, in]`, exactly the layout `vecmatmul`
contracts over. Its quantization scale is irrelevant to an argmax, so any
per-tensor `absmax/127` will do.

---

## 5. The host loop

Weights are read-only, so staging splits into a one-time part and a per-input
part:

```python
# ---- once ----------------------------------------------------------------
for L in range(4):
    base = 0x5000 + L * 0x7400
    dram[base + 0x0000 : base + 0x6000] = pack_ternary(model.layers[L])   # 24 KB
    dram[base + 0x6000 : base + 0x6034] = requant_words(L)                # 13 int32
dram[0x1000:0x1400] = causal_mask
dram[0x3400:0x3A80] = quantize_i8(model.fc.weight)
load_program(adder_model_words)

# ---- per input -----------------------------------------------------------
dram[0x0000:0x1000] = quantize(embed(tokens) + positional_encoding(T), s_x[0])
run()
logits = int32_at(dram, 0x3C00, (32, 13))
answer = logits[EQUALS_POS + 1:].argmax(-1)          # host
```

In the ISS that is literal: `tpu.run(words)` resets neither `tpu.mem` nor
`tpu.dram`, so the weights persist across calls exactly as they do in SRAM. On
the board, `host/run_program.py` already does load-program → write-DRAM → run →
read-DRAM; the per-input part is just the last three.

**One forward is a single pass — no autoregressive decode.** `EQUALS_POS = 15`
is fixed, so the answer tokens sit at known positions and one pass produces the
whole answer.

---

## 6. Costs — measured

**Instructions: 196 words of 1024** (`python assembler.py examples/adder_model.tpu`).
IMEM has stopped being a constraint, and so has DRAM.

**Arithmetic per layer.** 1536 MXU tile passes (3×256 for the projections, 256
each for `Wo`/`W1`/`W2`) ≈ 3.1 M MACs, plus 8 `vecmatmul` blocks ≈ 0.26 M.

**ISS: a full four-layer forward plus the PyTorch reference and the byte
comparison takes ~19 s.** That is far better than the "minutes" estimated
before it was run, and it means leg-B verification is an edit-run loop rather
than a batch job. The numpy contingency for `iss._matmul` is not needed.

**Staging.** 105 296 bytes in (weights dominate), 26 240 bytes of device output
— the four per-layer `X` checkpoints, the final `V^T` and `A`, and the logits.
On the board that is a one-time ~9 s upload at 115200, then 4 KB in and 1.7 KB
out per forward (~0.5 s), since the weights are read-only and stay resident.

---

## 7. Verification — what has been run

Everything below has been executed. The one remaining item is §7.6, which needs
a real checkpoint rather than synthetic weights.

| # | Check | Result |
| --- | --- | --- |
| 1 | `python torch_ref.py` — every example vs PyTorch | **7/7 pass**, exact |
| 2 | `tb/` RTL suite, 13 testbenches | **12 pass**; `uart_transmitter_tb` is a pre-existing no-op (its whole body is commented out) |
| 3 | `make cosim` — host driver vs RTL over a simulated UART | **11/11 pass** |
| 4 | `assembler.py examples/adder_model.tpu` | **196 words** of 1024 |
| 5 | Full model, ISS vs PyTorch, exact | **0 of 24 992 elements differ** — 4 layer checkpoints (4096 int8 each), `V^T`, `A`, and 416 int32 logits |
| 6 | `highmem_dma` + `vpu_matmul` through the RTL over the UART (`make uart`) | **5446 checks, 0 errors** — this is what covers the widened address and the `0x8000` case in hardware |
| 7 | Full model through the RTL (`tpu_top_tb`, backdoor DRAM) | **26 240 checks, 0 errors** — halts at pc=195 after 23.58 ms simulated |
| 8 | Real checkpoint, ISS vs the reference (`adder_export.py --iss-check`) | **0/416 logits differ** |
| 9 | Real-checkpoint task accuracy | float 96.09% / int8 **0.00%** exact-sequence — see below |

**§7.7 is a long simulation.** The model is ~188 KB of byte-serial DMA at ~14
clocks/byte plus the MXU/VPU work — 2.36 M clocks, 23.58 ms at 100 MHz, with
every unit busy. That is long enough in Icarus to need the watchdog raised
(`-DWATCHDOG_NS`) and the waveform dump off (`-DNO_VCD`; without it the run
writes a 400+ MB VCD before the old 2 ms watchdog even fires). Both are now
options on `tpu_top_tb.sv`, defaults unchanged:

```bash
cd accel/tpu/tb
python ../../tpulang/gen_vectors.py -p ../../tpulang/examples/adder_model.tpu -o vectors_model
iverilog -g2012 -o tpu_top_model.vvp \
  -DPROG_FILE='"vectors_model/tpu_prog.hex"' \
  -DSPAD_IN_FILE='"vectors_model/tpu_spad_in.hex"' \
  -DSPAD_EXP_FILE='"vectors_model/tpu_spad_exp.hex"' \
  -DWATCHDOG_NS=400000000 -DNO_VCD \
  ../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv \
  ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv ../rtl/uart_interface.sv \
  ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/perf_counters.sv \
  tpu_top_tb.sv
vvp tpu_top_model.vvp
```

What it adds over §7.5 is narrow but real: `gen_vectors` exports the ISS's
result as the golden image, so §7.5 pins the *arithmetic* against an
independent implementation, and §7.6 exercises the RTL's widened DRAM path on a
small program. §7.7 is the two together at full scale — and it is what caught
the backdoor-width bug in §2, which neither of the others could see.

**Together these are legs A–D of the original plan.** The chain is closed: an
independent PyTorch implementation agrees with the ISS, and the RTL agrees with
the ISS, on the whole model, byte for byte. What is *not* yet established is
accuracy on the real task — that is §7.6 below, and it needs a checkpoint
rather than synthetic weights.

**§7.5 is the leg that matters**, and it is genuinely independent: `torch_ref
.adder_model` is written from `model/transformer.py`, not from the kernel's
instruction order. Where the kernel is clever, the reference is not — it
computes `A` the obvious way and transposes, rather than reproducing the
operand swap that lets one `.t` DMA scatter each head; and it applies the mask
as the same two operations the kernel claims are equivalent to
`relu(S + -1e9)`, so the claim is tested rather than assumed. The two share the
DRAM layout (read from the program's own `.equ` table) and nothing else.

**The comparison is not vacuous.** With the synthetic operands in
`gen_vectors.ADDER_RQ`, the four layer checkpoints use 43–124 distinct int8
values each at **0% saturation**, `V^T` and `A` cover 246 and 226 distinct
values with 2% and 16% saturation (so the requant clip path is exercised
without dominating), and the logits take 350 distinct int32 values. Getting
there took tuning — see §8.6, which is a finding in its own right.

### §7.6 — the real checkpoint: the kernel is correct, the model is not quantizable

`python accel/tpulang/adder_export.py -n 256 -c 128 --iss-check`, on
`colab_ternary_mha_small_nobias.pt`:

| | exact-sequence | token |
| --- | --- | --- |
| float model | **96.09%** | 99.76% |
| int8 activations, this kernel | **0.00%** | 63.23% |

**The ISS cross-check passed: 0 of 416 logits differ** between the reference and
`adder_model.tpu` run on the real weights. So this is a measurement of the
kernel, not of a model of it, and the gap is not an implementation bug.

Four experiments locate the cause, and it is not the kernel, the calibration, or
the requant words (whose worst relative error is 0.87%):

1. **Fake-quantizing in pure float reproduces it**: 0.00% / 71.00%, matching the
   integer pipeline. Nothing integer-specific is involved.
2. **Any single site is fatal on its own.** Quantizing only `X1` → 0.00%
   exact-sequence; only Q/K/V → 3.12%; only `A` → 1.56%. Only the network input
   survives (100%). This is not one bad tensor.
3. **The model tolerates ±10% *multiplicative* jitter on every tensor at 100%
   exact-sequence** — 100× more error than int8 injects. So it is not fragile to
   perturbation in general.
4. **The difference is that quantization error is absolute, and it zeroes the
   bulk of every tensor.** A per-tensor scale is pinned by the maximum, so what
   matters is `median/max` — and the usual outlier check misses it entirely,
   because the *top* of these distributions is well behaved (`max/p99.9` is only
   1.1–2.5) while the *middle* falls away:

   | site | median/max | fraction → exactly 0 |
   | --- | --- | --- |
   | L0 `X` | 0.101 | 1.3% |
   | L2 `A` | 0.00026 | 83.5% |
   | L3 `X` | 0.0015 | 62.4% |
   | L3 `A` | 0.00000 | **91.6%** |

**This is §8.1 with a mechanism.** With no LayerNorm, a few entries per tensor
run away while the typical entry does not, and nothing renormalizes the ratio —
so it compounds with depth, from 1.3% of layer 0's residual stream rounding to
zero to 62% of layer 3's. The `1.46e6` score maximum against a `4.6e3` surviving
maximum is the same phenomenon inside attention.

**The remedy is per-channel (or per-token) activation scales, which this ISA
cannot express**: `requant` takes one `{m0,n}` per dispatch, i.e. strictly
per-tensor, and `matmul_t.rq` reads its word from a single `cfg scalar`
address. So the options are (a) retrain with normalization — the configuration
`tpunn.md` §1 measured at **98.4% exact-sequence with ternary weights and int8
activations**, on this same task, which is the strongest evidence that
LayerNorm is what is missing; or (b) add a per-channel requant path to the VPU,
a much larger change than anything here.

**What this does and does not say.** It does not say the kernel is wrong — three
independent checks say it is right, including on these very weights. It says
this *checkpoint* cannot be served in int8 activations by this hardware. The
clip rates §8.3 asked for, for completeness: `RQ_O` 8.24% mean (L0 14.63%,
L2 17.68%), `RQ_F` 0.00%.

---

## 8. Risks

1. **No LayerNorm means nothing bounds the residual stream — and this is now
   the confirmed cause of §7.6's result, not a prediction.** Two independent
   measurements:

   *Synthetic.* While tuning §7's requant words, moving a single word by two
   shifts (`RQ_A` `{1,6}`→`{1,8}`) took the model from 45% of `A` saturated at
   ±127 to **99.3% of `A` exactly zero**. Each layer's requant multiplies the
   residual stream by a constant, so an error compounds geometrically instead of
   being renormalized away.

   *Real.* On the shipped checkpoint the same absence shows up as within-tensor
   dynamic range that widens with depth: the median entry falls from 10% of the
   tensor maximum at layer 0 to 0.15% at layer 3, so 62% of layer 3's residual
   stream and 92% of its attention output round to exactly zero under a
   per-tensor int8 scale. Float 96.09% exact-sequence becomes 0.00%.

   This is the single most consequential property of the model for this
   hardware, and it is a training decision, not a kernel one.
2. **ReLU attention has no denominator.** Softmax's rows sum to 1; `relu(S)`'s
   do not, and the row sum grows with the number of attended keys, so `A`'s
   magnitude varies strongly with token position. A single per-tensor `s_a` has
   to cover both ends. If 7.6 shows heavy clipping, the honest fix is per-row
   scaling, which the ISA cannot broadcast cheaply — so measure before
   designing anything.
3. **`RQ_O` and `RQ_F` are pinned by the residual adds** (§4) and are the two
   requants with no freedom left. Clip rates there are the diagnostic.
4. **`cfg vlen` maxes at 1023 and a `[32][32]` block is 1024 elements.**
   `vlen = 1024` reads as zero and the dispatch silently does nothing — no
   error, just a missing result. This is why every elementwise pass is chunked
   at 512, and anything added here has to respect it.
5. **~~The widened DRAM address is untested~~** — now covered by
   [`highmem_dma.tpu`](examples/highmem_dma.tpu), which spills linearly and
   transposed to `0x10100`/`0x10800` and fills back from high DRAM, in both the
   ISS and the RTL. It is deliberately built so the predicted failure mode
   (aliasing back into the low window) surfaces as a *missing* byte rather than
   a wrong one. Keep it in the suite: it is the only thing in the tree that
   addresses DRAM above 64 KB apart from the model itself.

6. **`ADDR_W` vs `MEM_ADDR_W` is now a live distinction, and the tree still
   conflates them in places.** Three separate spots used the scratchpad width
   for a DRAM quantity (§2), each invisible until a program actually addressed
   past 64 KB, and each failing *silently* — a dropped `$readmemh` line, a
   wrapped backdoor poke — rather than loudly. Two more are known and
   deliberately left: `scalar_unit.sv`'s LINK `nb_*_addr`, and the DMA's
   transpose offsets, which accumulate in `SCRATCHPAD_ADDR_W` bits and so
   confine a `.t` transfer to a 64 KB window *relative to its base*. Both are
   fine today. When touching anything that names a DRAM address, check the width
   rather than the name.

7. **Synthetic test operands need mixing, not index formulas.** The other
   `gen_vectors` builders use patterns like `(i*3 + j) % 9 - 4`, which are fine
   for an 8×8 tile. At model scale a ternary weight of `(k*31 + n*17) % 3` is a
   period-3 lattice in both indices, and contracting it over K=128 against an
   equally regular activation made **97% of the products cancel** — a
   bit-exactness check that would have passed on almost all zeros. `_mix()` in
   `gen_vectors.py` exists for that reason. Anything added at this scale should
   check its own value distributions before trusting a green run.

---

## 9. Deliberately not done

| Not now | What would trigger it |
| --- | --- |
| Batching | `cfg tlen ≤ 63` covers `T=32`, so a batch needs an outer loop and ~4× the activation DRAM. With weights resident, this is the cheap axis if throughput matters — the weights, not the activations, dominate. |
| Autoregressive decode | Not needed: `EQUALS_POS` is fixed, so one pass gives the whole answer. `setcfgr` already exists if it ever is. |
| Embedding / `argmax` on device | No gather, and `redmax` returns the value not the index (§1). |
| Widening LINK's `nb_*_addr` | They are nominally DRAM addresses but no comms block is attached (`nb_done` is tied high). Widen alongside `comms.md`. |
| Fusing the weight fills | Four `rdmem`s per layer at 4–12 KB each. Double-buffering them against the MXU is `macro_ops.md` §4.5 territory and wants the phase-0 counters to justify it first. |
