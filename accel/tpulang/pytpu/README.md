# pytpu — a primitives-based tpulang program generator

Writes [tpulang](../README.md) programs by composing **compute primitives** —
hand-written, parameterized `.tpu` templates — from Python. The output is a single
ordinary `.tpu` file that the existing toolchain handles unchanged: `assembler.py`
assembles it, `iss.py` runs it, `tpu_top_tb.sv` executes the vectors it emits.

```bash
python accel/tpulang/pytpu/examples/transformer_layer.py            # build + verify
python accel/tpulang/pytpu/examples/transformer_layer.py --map      # + memory map
python accel/tpulang/pytpu/examples/transformer_layer.py --emit-vectors
```

That builds **one quantized transformer layer** (§4), runs it in the ISS, and checks all
sixteen intermediates against an exact integer reference. Current state:

```
861 / 1024 instruction words  (84%)
[OK] QKV S32 S8 SM8 P8 Vt AO32 AO8 O X1 N1 H HG F X2 Y   0 values differ
Y  max |err| = 0.105  mean |err| = 0.030  (3.8% of mean |Y|)   corr = 0.9993
```

[`PLAN.md`](PLAN.md) is the design document — the constraints table in its §2 is the thing
to read before writing a new primitive.

---

## 1. Why templates instead of codegen

This is a *composer*, not a compiler. There is no IR, no scheduler, no cross-block register
allocator. The tiling lives inside each template, where a human wrote it and can read it,
so the hard-won loop nests from [`examples/tiled_matmul.tpu`](../examples/tiled_matmul.tpu)
get **reused** rather than re-derived by codegen nobody can debug against a bitstream.

The cost is that the ISA has no call/return, so every instantiation is **inlined**: two
`matmul_ternary` calls are two copies of the loop nest. Instruction memory (1024 words) is
what this design spends, and `Program` reports and enforces that budget.

---

## 2. The template language

Two additions to ordinary tpulang, both chosen so a mistake is a hard error.

**`{{NAME}}` — parameters.** Declared in the template's header comment, so the file is
self-describing and `primitives.py` does not restate the list. An unsubstituted placeholder
or an unused parameter is an error in both directions.

**`$name` — instance-local symbols.** tpulang has one flat namespace and redefining a
symbol is an `AsmError`, so every `$name` is rewritten to `<prefix>_name` per instantiation.
`$` is the sigil because it is a *syntax error* in the assembler's expression evaluator — an
unmangled one cannot reach a program that assembles.

```tpu
; @primitive add_i8
; @param A      DRAM address of A[N], int8
; @param CHUNK  elements per pass (<= 1023, divides N)
; @clobbers r1-r13, cfg vlen/len

.equ $NCHUNK {{N}} // {{CHUNK}}
.reg $a r1
$loop:
    cmps $i, $nch
    bge  $done
    ...
```

### The calling convention

1. **No register is live across a block** — all state between primitives is in memory. This
   is what makes inlining safe without a register allocator.
2. **No config register is live across a block** — each primitive `setcfg`s `tlen`/`vlen`/
   `len` before use. Stale config is this ISA's most common silent bug, and this rule kills
   it by construction.
3. **Inputs come from DRAM, outputs go to DRAM**, at one address valid in both spaces (so a
   fill is `rdmem a, a`). Costs redundant DMA between adjacent blocks; buys composability in
   any order, and every intermediate becomes host-visible for debugging. `matmul_ternary` is
   the exception — it *streams* tiles, so its operands may exceed the scratchpad.

---

## 3. The primitive library

| Primitive | Does | Words |
| --- | --- | --- |
| [`matmul_ternary`](lib/matmul_ternary.tpu) | `C[M,N] = requant(A[M,K] int8 @ W[K,N] ternary)` on the MXU, tiled | 83 |
| [`matmul_i8`](lib/matmul_i8.tpu) | `O[H,M,N] int32 = A @ Bᵀ` on the VPU, batched and fully strided | 53 |
| [`transpose_i8`](lib/transpose_i8.tpu) | `B = Aᵀ` int8, via `len = 1` DMA byte moves | 27 |
| [`requant_block`](lib/requant_block.tpu) | int32 → int8 over N elements | 24 |
| [`add_i8`](lib/add_i8.tpu) | `C = requant(A + B)` — residual adds *and* the causal mask | 26 |
| [`gelu_block`](lib/gelu_block.tpu) | `gelu` LUT + exit requant | 23 |
| [`softmax_rows`](lib/softmax_rows.tpu) | row softmax, int8 @ 1/16 → int8 @ 1/128 | 46 |
| [`layernorm_rows`](lib/layernorm_rows.tpu) | `γ(x−μ)/σ + β` with integer divide and sqrt | 100 |

Three are worth reading for the ISA lessons in them:

- **`matmul_i8`** gives every axis an explicit byte stride, which is what lets one template
  serve both attention matmuls even though Q and K are *strided sub-blocks* of the fused
  `[T, 3d]` QKV matrix. A head's `(head, token)` row is `head_dim` contiguous bytes, which
  is all `vecdot` needs, so heads are never un-interleaved and the head loop costs no
  instructions.
- **`transpose_i8`** exists because `vecdot` needs both operands contiguous along the
  contraction axis, and `P @ V` contracts over *keys*. The ISA has no transpose and no
  strided vector load — but a DMA with `len = 1` is a byte move between two arbitrary
  addresses, and that is the whole mechanism.
- **`layernorm_rows`** needs a divide and a square root, and the scalar ALU is `adds`,
  `subs`, `muls`, `cmps` — no shift, no divide. The way through is that **the scratchpad is
  the shifter**: build the powers of two by repeated doubling, `stores` them ascending, then
  `loads` them back descending. One table drives both a binary long division (`f(t) = t·L`)
  and a bit-by-bit sqrt (`f(t) = t·t`), ~12 instructions each.

Everywhere: **VPU ops read int8 and write int32, and `requant` is the only op that goes back
the other way.** Every requant in these templates is there because the *next* op reads int8.
Getting that wrong is what makes [`softmax_row.tpu`](../examples/softmax_row.tpu) unrunnable.

---

## 4. The sample program

[`examples/transformer_layer.py`](examples/transformer_layer.py) builds the post-norm decoder
layer of `model/transformer.py`:

```
QKV  = X·[Wq|Wk|Wv]   matmul_ternary    S32  = Q·Kᵀ per head    matmul_i8
S8   = requant(S32)   requant_block     SM8  = S8 + mask        add_i8
P    = softmax(SM8)   softmax_rows      Vᵀ                      transpose_i8
AO32 = P·Vᵀᵀ per head matmul_i8         AO8  = requant(AO32)    requant_block
O    = AO8·Wo         matmul_ternary    X1'  = X + O            add_i8
X1   = norm1(X1')     layernorm_rows    H    = X1·W1            matmul_ternary
HG   = gelu(H)        gelu_block        F    = HG·W2            matmul_ternary
X2   = X1 + F         add_i8            Y    = norm2(X2)        layernorm_rows
```

**Scales are calibrated, not guessed.** The float layer runs first — over the *ternarized*
weights, since those are a hardware fact rather than a quantization error — and each
tensor's int8 scale is its observed absmax / 127. Every `{m0,n}` word then follows from the
producer's and consumer's scales.

**Two scales are not free.** `gelu` and `exp` are 256-entry ROMs that assume an input scale
of `1/16` and never check it, so the requants feeding them are made to land there (which
costs nothing — that requant had to happen anyway). The example *measures* how much of each
distribution the LUT's `±7.94` domain clips instead of assuming it is nothing.

Deliberate simplifications, all printed rather than hidden: demo geometry (`d=32, f=64, 2
heads, 8 tokens`) rather than `adder_ternary_vanilla`'s `d=128, f=512` — the real layer's
ternary weights alone are 48 KB against the ISS's 64 KB DRAM space; no linear bias (the MXU
has no bias path); no dropout.

---

## 5. Verification

```
python accel/tpulang/pytpu/examples/transformer_layer.py
```

runs three legs:

1. **It assembles and fits** — word count asserted against 1024, per-block breakdown printed.
2. **Exact integer reference** ([`examples/layer_ref.py`](examples/layer_ref.py)) — the whole
   pipeline recomputed in plain Python integers from the bytes actually loaded, compared to
   the bytes actually spilled, with **zero tolerance**. It shares no arithmetic with
   `iss.py`; its `_trunc_div` and `_isqrt` are written the obvious way as a check on the
   template's table walk. It *does* share the address map and the `{m0,n}` words with the
   composer, which is a real limit on its independence.
3. **Float reference** — reported, never asserted. int8 activations, a 32-wide integer
   LayerNorm and a 256-entry exp table do not reproduce float arithmetic, and a threshold
   here would encode nothing but today's seed. Leg 2 says the program is *correct*; leg 3
   says what quantization *costs*.

`--emit-vectors` additionally writes `tpu_prog.hex` / `tpu_spad_in.hex` /
`tpu_spad_exp.hex` in `gen_vectors.py`'s format, so `tpu_top_tb.sv` can run a generated
layer with no testbench change.

---

## 6. Known limits

- **A second layer does not fit** in 1024 words. The lever is hoisting the four
  `matmul_ternary` instances into one driven by a parameter block in scratchpad (~250 words
  back), and that is the natural next step rather than a rewrite.
- **No fill/spill elision** between adjacent blocks (convention rule 3). The obvious 2–3×
  cycle win; needs a liveness pass, which is the first real step toward being a compiler.
- **`transpose_i8` is one DMA pair per byte.** Delete it the day a transposing DMA lands.
- **The ISS sizes DRAM and the scratchpad from one `addr_w`**, so a simulated program lives
  in 64 KB. That, not the generator, is what stands between this and the real `d = 128`
  layer; the board itself has 512 KB.
- **`run_program.py` cannot yet run a generated program** — it builds its input image via
  `gen_vectors.build_image`, which dispatches on a program's `.equ` constants, and a
  generated program brings its own image instead. The testbench path (§5) works today.
