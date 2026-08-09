# pytpu — a primitives-based tpulang program generator

**Status: plan.** This document is the design; the code below it in this directory is
the implementation. Read [`../README.md`](../README.md) §5 first — it reserves the name
`pytpu` for "the Python layer above tpulang", and this is that layer, built the concrete
way rather than as a general compiler.

---

## 1. What this is, and what it deliberately is not

**Is:** a *composer*. A small library of **compute primitives**, each one a hand-written,
hand-tuned `.tpu` template with named parameters. Python instantiates them with concrete
shapes and addresses, concatenates the rendered text into one `.tpu` source, and hands
that to the existing `assembler.py` → `iss.py` → testbench/board pipeline. Nothing in the
existing toolchain changes.

**Is not:** a compiler. There is no IR, no scheduler, no register allocator across
primitives, no automatic tiling or fusion. The tiling lives inside each template, where a
human wrote it and can read it. That is the point: the hard-won loop nests in
`examples/tiled_matmul.tpu` stay hand-written tpulang and get *reused*, instead of being
re-derived by codegen that nobody can debug at 2 a.m. against a bitstream.

The trade is that each instantiation is **inlined** — the ISA has no call/return (no link
register), so two `matmul_ternary` calls cost two copies of the loop nest. Instruction
memory, not scratchpad, is what this design spends. §9 budgets it.

```
 primitives/*.tpu  ──render(params)──►  one generated .tpu  ──assembler.py──► words
      (templates)         ▲                    │                                │
                          │                    │                            iss.py
   examples/*.py  ────────┘                    ▼                                │
   (the composer: shapes,             tb/vectors/*.hex  ◄──────────────────── golden
    memory map, quantization)          run_program.py                          image
```

---

## 2. Hard constraints (these shape every decision below)

Taken from `tpu_top.sv` parameters, `iss.py`, and `docs/isa.md` §A. Everything in this
design that looks arbitrary is downstream of this table.

| Limit | Value | Source | Consequence for the generator |
| --- | --- | --- | --- |
| Instruction memory | **2¹⁰ = 1024 words** | `IMEM_AW = 10` | The binding budget. `Program` asserts on it and prints a per-block breakdown. |
| Scratchpad | 2¹⁶ B = 64 KB | `ADDR_W = 16` | Address arithmetic wraps mod 2¹⁶ in both RTL and ISS. |
| DRAM (in the ISS) | also 2¹⁶ B | `iss.TPU.dram = bytearray(depth)` | The board has 512 KB (`MEM_ADDR_W = 19`) but the ISS shares one `addr_w`, so a program that must be *simulated* lives inside 64 KB. This caps the demo config, not the generator. |
| MXU array | ROWS = COLS = **8** | `gen_vectors.py` asserts it; `tpu_top_tb.sv` DUT | Ternary matmuls tile K and N to multiples of 8. |
| `cfg tlen` | 6 bits, ≤ 63 | `iss._matmul` masks `0x3F` | Tokens per MXU pass. |
| `cfg vlen` | 10 bits, ≤ 1023 | `iss._vpu` masks `0x3FF` | Elementwise blocks longer than 1023 must be **chunked** — every block primitive loops over chunks. |
| `li` immediate | 16-bit | `assembler._imm16` | Any address fits, but ≥ 0x8000 lands in the register as a negative int32. It still works (all uses are masked mod 2¹⁶), and the arena keeps below 0x8000 anyway. |
| Registers | r0..r31, r0 = 0 | `REG_AW = 5` | Plenty *within* a primitive; §5 forbids liveness *across* primitives. |
| Scalar ALU | `adds` `subs` `muls` only | ISA §A.4 | **No shift, no divide.** Drives the isqrt construction in §7.8. |
| `gelu` / `exp` | 256-entry int8 LUTs at fixed scales | `luts.py` | `in_scale = 1/16` for both; `out_scale` 1/16 (gelu) and 1/127 (exp). An operand must be *requantized into* that scale first — the hardware never checks. |
| ISS step limit | 100 000 default | `iss.TPU.run` | A looping layer blows past it; `Program.run` raises it explicitly. |

Two more that are semantics, not sizes, and cause most of the bugs in this ISA:

- **VPU ops read int8 (stride 1) and write int32 (stride 4).** `requant` is the *only* op
  that goes back the other way. So every value that a VPU op is going to consume must have
  been requantized to int8 first. `softmax_row.tpu` is broken in exactly this way and is
  the cautionary example (see `torch_ref.UNSUPPORTED`).
- **Reductions (`redmax`, `redsum`, `vecdot`) write one int32 to the *scratchpad*, not to a
  register,** and broadcast scalars (`sadd`, `vecmul`, `sdiv`, `requant`) read their scalar
  from a *scratchpad address*. Getting a reduction into a broadcast is
  `reduce → loads → scalar math → stores`.

---

## 3. Files

```
accel/tpulang/pytpu/
  PLAN.md              this document
  README.md            usage, written after the code works
  template.py          render a .tpu template: {{param}} substitution + $local mangling
  memory.py            Arena (bump allocator) + Tensor / Scalar descriptors
  quant.py             float -> int8 / ternary packing, {m0,n} selection
  program.py           Program: collect blocks -> one .tpu -> assemble -> ISS -> images
  primitives.py        the registry: one Python wrapper per template, with validation
  lib/                 the templates themselves
    matmul_ternary.tpu   requant_block.tpu    softmax_rows.tpu
    matmul_i8.tpu        add_i8.tpu           layernorm_rows.tpu
    transpose_i8.tpu     gelu_block.tpu
  examples/
    transformer_layer.py  the sample program (§8): calibrate, compose, run, verify
    layer_ref.py          the exact integer reference for it (§9, leg 2)
```

`pytpu/` imports `assembler`, `iss` and `luts` from its parent directory the way
`torch_ref.py` already does (`sys.path.insert` on the parent, no packaging of tpulang).

---

## 4. The template language

A primitive is a `.tpu` file with two additions, both chosen so the file still *reads* as
tpulang and so a stray one is a hard error rather than silent nonsense.

### 4.1 `{{NAME}}` — parameter substitution

Textual, before assembly. A parameter is an int (rendered as decimal), a hex address
(rendered `0x1234`), or a raw string (used for instruction flag suffixes). Any `{{NAME}}`
left unsubstituted is an error; any parameter passed but not used is an error. Both
directions checked — a typo'd parameter name must not silently take a default.

### 4.2 `$name` — instance-local symbols

tpulang has one flat namespace for `.equ` names, `.reg` aliases and labels, and redefining
one is an `AsmError`. Two instantiations of the same primitive would therefore collide on
every symbol. So: **any symbol written `$name` is rewritten to `<prefix>_name`**, where
`prefix` is the block's unique instance name (`mm0`, `softmax1`, …).

`$` is chosen because it is a syntax error inside the assembler's expression evaluator, so
an unmangled `$` cannot survive into a program that assembles.

```tpu
; lib/example.tpu
.equ $ROWS  {{ROWS}}          ; -> .equ mm0_ROWS 8
.reg $abuf  r1                ; -> .reg mm0_abuf r1
$kloop:                       ; -> mm0_kloop:
    jmp $kloop
```

### 4.3 Header metadata

The first comment block declares the parameters, so the template is self-describing and
`primitives.py` does not restate them:

```tpu
; @primitive matmul_ternary
; @param   M          token rows of A            (multiple of TT)
; @param   K          contraction length          (multiple of 8, >= 16)
; @default TT         8
; @clobbers r1-r26, cfg tlen/len/scalar
```

`template.py` parses `@param` / `@default` and uses them to check the call.

### 4.4 The calling convention

Three rules, enforced by convention and documented in every template's `@clobbers` line:

1. **No register is live across a block.** A primitive may use r1..r31 freely. All state
   between primitives lives in memory. (This is what makes inlining safe without a
   register allocator.)
2. **No config register is live across a block.** Every primitive `setcfg`s `tlen`/`vlen`/
   `len`/`scalar` before each use. Stale config is the #1 silent bug in this ISA
   (`isa.md` §"Gotchas") and this rule kills it by construction.
3. **A primitive reads its inputs from DRAM and writes its outputs to DRAM**, at the
   address given by its parameters, with the *identical-address convention*: every tensor
   has one address that is valid in both DRAM and scratchpad, so a primitive `rdmem`s what
   it reads and `wrmem`s what it writes at that same address.

Rule 3 costs redundant DMA round-trips between adjacent primitives. It buys three things
worth more than the cycles right now: primitives compose in any order, every intermediate
tensor is host-visible for debugging (it was `wrmem`'d, so it appears in `dram_written`),
and a block can be lifted out and tested alone. Eliding redundant fill/spill is the obvious
later optimization and is listed in §11. The one exception is `matmul_ternary`, which
*streams* tiles through small fixed scratch buffers exactly as `tiled_matmul.tpu` does —
its inputs may be far larger than the scratchpad.

---

## 5. Python API

```python
from pytpu import Program, Arena, primitives as P

prog = Program(name="attn_layer")
dram = Arena(base=0x0000, limit=0x8000)          # tensors: one address, both spaces
scratch = Arena(base=0x8000, limit=0x10000)      # primitive-private buffers

X   = dram.tensor("X",  (T, D), dtype="int8",  scale=1/64)
Wq  = dram.tensor("Wq", (D, D), dtype="ternary")
Q   = dram.tensor("Q",  (T, D), dtype="int8",  scale=...)

P.matmul_ternary(prog, a=X, w=Wq, c=Q, rqw=rq_q, scratch=scratch)
P.softmax_rows(prog, src=S8, dst=Pb, rows=H*T, length=T, scratch=scratch)
...

prog.finalize()                    # -> source text, words; asserts <= 1024 words
img = prog.host_image()            # {addr: byte} of every input tensor's contents
out = prog.run_iss(inputs=img)     # {addr: byte} of everything wrmem'd
prog.emit_vectors("../../tpu/tb/vectors_layer")   # the three $readmemh files
```

- **`Arena`** — a bump allocator with alignment and a hard limit; every allocation is
  named, so `prog.memory_map()` prints a table and an overflow names the tensor that broke
  the budget rather than producing a wrapped address.
- **`Tensor`** — `name, shape, dtype ∈ {int8, int32, ternary}, addr, scale`. `scale` is a
  compile-time float only (never in TPU memory), used by `quant.py` to pick `{m0,n}`. Byte
  size follows from dtype: int8 → `prod(shape)`, int32 → `4·prod(shape)`, ternary →
  column-major 2-bit packed, `cols · (rows·2/8)` per tile.
- **`Program`** — holds ordered `(name, rendered_text, word_count)` blocks plus a prologue
  of `.equ` address constants. `finalize()` concatenates, assembles once, and reports:

```
block                     words   cum
qkv/matmul_ternary_q         54    54
qkv/matmul_ternary_k         54   108
...
                            ----
                             761 / 1024 words  (74%)
```

---

## 6. Numerics and quantization policy

Every tensor carries a compile-time `scale` (a float: `real = int · scale`). The composer,
not the hardware, is responsible for making producer and consumer scales agree.

- **`{m0, n}` selection** (`quant.choose_requant(scale_in, scale_out)`): pick `n ≤ 15` and
  `m0 = round(2ⁿ · scale_in / scale_out)` subject to `m0 < 2¹² = 4096` (`M0_W`), maximizing
  `n` for precision. Emitted as one int32 word `(n << 12) | m0`, exactly
  `gen_vectors.requant_word`.
- **Requant is `clip_int8((acc·m0 + (1 << (n-1))) >> n)`** — an *arithmetic* (flooring)
  shift, and `m0` is unsigned. `quant.py` reuses `iss.TPU.requant8`'s semantics rather than
  restating them.
- **LUT scales are fixed, not chosen.** `gelu` and `exp` both assume `in_scale = 1/16`. So
  `gelu_block` and `softmax_rows` each begin with a requant *into* 1/16 whose `{m0,n}` is
  derived from the input tensor's scale. This is the single most likely source of a
  "runs fine, wrong numbers" bug, so both templates document it at the top.
- **Ternary packing**: `00 = 0, 01 = +1, 11 = −1`, column-major, 2 bits per element, and —
  for `matmul_ternary` — laid out **tile-major `[nt][kt]`** exactly as
  `gen_vectors.build_matmul_image` does. `quant.pack_ternary` produces that layout from a
  `[K, N]` array so the packer and the loop nest cannot drift apart.

---

## 7. The primitive library

Eight primitives. Each row's "words" is an estimate of the emitted instruction count,
which is what §9 budgets against.

| # | Primitive | Signature | Words |
| --- | --- | --- | --- |
| 7.1 | `matmul_ternary` | `C[M,N] int8 = requant(A[M,K] int8 @ W[K,N] ternary)` | ~84 |
| 7.2 | `matmul_i8` | `O[H,M,N] int32 = A[H,M,K] @ B[H,N,K]ᵀ`, fully strided | ~54 |
| 7.3 | `transpose_i8` | `B[N,M] = A[M,N]ᵀ`, int8, strided both sides | ~27 |
| 7.4 | `requant_block` | `dst[N] int8 = requant(src[N] int32)` | ~25 |
| 7.5 | `add_i8` | `C[N] int8 = requant(A[N] int8 + B[N] int8)` | ~27 |
| 7.6 | `gelu_block` | `dst[N] int8 = requant(gelu(src[N]))` | ~25 |
| 7.7 | `softmax_rows` | `P[R,L] int8 (1/128) = softmax(S[R,L] int8 @ 1/16)` | ~48 |
| 7.8 | `layernorm_rows` | `Y[R,L] int8 = γ·(x−μ)/σ + β` | ~110 |

### 7.1 `matmul_ternary` — the MXU workhorse

`examples/tiled_matmul.tpu` generalized: `MTILES = M/TT`, `KTILES = K/8`, `NTILES = N/8`,
with `M`, `K`, `N` and the DRAM bases as parameters. Three nested loops; `matmul` on the
first K tile, `matmul.acc` on the middle ones, `matmul.acc.rq` on the last, so the int32
accumulation runs the full K before a single requant. Scratchpad footprint is fixed at
`TT·8 + 16 + TT·8·4` bytes regardless of problem size.

**A and C are plain row-major, not tile-major.** `tiled_matmul.tpu` stores its activations
in the same tile-major order it walks, so a tile is one contiguous DMA. That is cheaper,
and it is the wrong choice here: the VPU primitives have no notion of tiles, so a
tile-major matmul output could not be fed to `softmax_rows` or `layernorm_rows` without a
relayout. Instead the A tile is *gathered* (T fills of 8 bytes at DRAM stride K) and the C
tile *scattered* the same way — two short loops, ~14 extra words, and in exchange every
activation in a generated program is plain row-major and any primitive can consume any
other's output. W keeps the tile-major packing, because it is a weight and `quant.pack_ternary`
lays it out offline.

The K flavour is selected by branching on `kt` rather than by three copies of the tile-load
code (which is how the example does it), saving ~25 words per instantiation — worth it when
four instantiations have to fit in 1024 words.

Constraints: `K % 8 == 0`, `N % 8 == 0`, `M % TT == 0`, `TT ≤ 63`, `KTILES ≥ 2` (the
program has a distinct init tile and a distinct requant tile — inherited from the example,
and asserted in `primitives.py` with that reason in the message).

### 7.2 `matmul_i8` — the VPU matmul, for activation × activation

`examples/vpu_matmul.tpu` generalized to a batched, fully strided form, because attention
needs *both* of its matmuls this way and neither can use the MXU (neither operand is
ternary):

```
O[h·OH + m·OM + n·4] = Σ_k A[h·AH + m·AM + k] · B[h·BH + n·BN + k]
```

one `vecdot` per `(h, m, n)` with `vlen = K`. The explicit strides are what make the same
template serve both:

| use | A | AH | AM | B | BH | BN | K | O | OH | OM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `S = Q Kᵀ` per head | QKV+0 | `dh` | `3d` | QKV+`d` | `dh` | `3d` | `dh` | S32 | `T·T·4` | `T·4` |
| `A = P Vᵀ` per head | P | `T·T` | `T` | Vᵀ | `dh·T` | `T` | `T` | AO32 | `dh·4` | `d·4` |

Note the first row: Q's head slice is *not* contiguous — it is a strided sub-block of the
**fused `[T, 3d]` QKV matrix** — but each `(head, token)` **row** is `dh` contiguous bytes,
which is all `vecdot` needs. So neither the heads nor Q/K/V ever have to be un-interleaved
out of the layout the MXU produced. The head loop is inside the template, so head count
costs zero instructions.

Staging is explicit (`IN0`/`IN1`, either may be 0 bytes) rather than derived from the
strides, because with strided sub-block operands there is no way to derive the right fill
span from the arithmetic parameters alone.

### 7.3 `transpose_i8`

`vecdot` requires both operands contiguous along the contraction axis. For `P @ V` the
contraction is over keys, so `V` must be key-contiguous, i.e. `Vᵀ[d][s]` — and the ISA has
no transpose and no strided vector load.

The trick: **DMA with `len = 1` is a byte move to an arbitrary address.** The primitive
loops over elements doing `rdmem` (1 byte, from `A + m·N + n`) then `wrmem` (1 byte, to
`B + n·M + m`) through a single scratch cell. `M·N` iterations of a 7-instruction body —
slow in cycles, tiny in instructions, and correct. Flagged in §11 as the thing to replace
first if a transposing DMA mode ever lands.

### 7.4–7.6 The block primitives

All three are the same skeleton: walk `N` elements in chunks of `vlen ≤ 1023`, doing
fill → op → spill per chunk, so they work for any `N`.

- `requant_block` — int32 → int8 with a `{m0,n}` word supplied by the composer.
- `add_i8` — `vecadd` (int8 + int8 → int32) then `requant` back to int8. Serves both the
  residual add **and** the attention mask: a causal mask is an int8 tensor of `0` and
  `−128`, and adding `−128` before the max-subtract drives the entry to the clip floor,
  which is exactly `x = −8` in the exp table's scale, where `exp` stores 0. The mask is
  therefore data, not a special instruction.
- `gelu_block` — requant into the LUT's `1/16` input scale, `gelu`, requant back out. The
  table is scale-preserving (`out_scale = in_scale = 1/16`, saturation unreachable), so the
  exit requant only has to return to the consumer's scale.

### 7.7 `softmax_rows`

Per row, the `isa.md` §5 sequence made type-correct — which is the part `softmax_row.tpu`
gets wrong, so this primitive is the corrected reference:

```
redmax  m   <- row              int32 scalar in scratchpad
loads/subs/stores               nega = -m, staged back as a broadcastable scalar
sadd    shift32 <- row, nega    int32:  x_i - max  (<= 0)
requant shift8  <- shift32      int8, {1,0}: clips at -128 == x = -8, exp -> 0
exp     e32     <- shift8       int32 in exp's out_scale 1/127
requant e8      <- e32          int8, {1,0}
redsum  D       <- e8           int32 scalar
sdiv    p32     <- e8, D        int32 Q15: round(e_i * 2^15 / D)
requant p8      <- p32          int8, {1, 8}: probability scale 1/128
```

Every `requant` there exists because the next op reads int8 — none is decoration. The
input is expected **already in the exp table's `1/16` scale and already masked**; the
composer inserts a `requant_block` and an `add_i8` ahead of it. Output scale is `1/128`,
pinned, so the following `matmul_i8` knows its input scale at compile time.

### 7.8 `layernorm_rows` — divide and sqrt with neither a shift nor a divide

The interesting one. Per row of length `L`:

1. `redsum` → `S = Σ x_i`; then `μ = trunc(S / L)` in **scalar** code (below).
2. `sadd` row by `−μ` → `c32_i = x_i − μ` (int32); `requant {1,0}` → `c8` (int8).
3. `vecdot(c8, c8)` → `SS = Σ (x_i − μ)²` as one int32. Note this is `vecdot`, not
   `square` + `redsum`: `square` writes int32 and `redsum` reads int8, so the obvious
   spelling is exactly the type error §2 warns about. `vecdot` does both halves in one op
   and stays in int32 throughout.
4. `R = ⌊√SS⌋` in scalar code (below).
5. `sdiv(c8, R)` → `y32_i = round(c8_i · 2¹⁵ / R)`, then `requant` to the output scale.
   Normalizing by `R = √(Σc²)` rather than by `σ = √(Σc²/L)` leaves a constant factor `√L`,
   which is folded into that requant's `m0` at compile time (`quant.layernorm_requant`).
   Nothing at run time ever divides by `L` or by `√L` in the *vector* path.
6. `vecemul` by `γ` (int8) → requant → `vecadd` `β` (int8) → requant.

**Divide and isqrt with `adds`/`subs`/`muls`/`cmps` only.** The scalar unit has no shift
and no divide, so neither the restoring-sqrt bit walk (needs `bit >>= 1`) nor Newton (needs
a divide) can be written directly. The way through: *the scratchpad is the shifter.*

```
build   b = 1; for k in 0..15: stores tbl+4k, b; b = b + b     ; powers of two, ascending
walk    q = 0; ptr = tbl + 60
        loop: loads b, ptr                                     ; ...read back descending
              t = q + b
              if f(t) <= v: q = t                              ; muls + cmps + branch
              ptr = ptr - 4;  repeat while ptr >= tbl
```

One table, built once; the walk is then instantiated twice per row with `f(t) = t·L`
(giving `⌊v/L⌋`, binary long division) and `f(t) = t·t` (giving `⌊√v⌋`). ~12 instructions
each, exact for `v < 2³¹`. Writing the descending bit sequence to memory and reading it back
*is* the right shift the ISA does not have.

Because `μ` comes out of a real division, `L` does **not** have to be a power of two — an
earlier draft of this design divided by `L` with a requant shift, which would have required
it. The demo still uses `d = 32`; the constraint is simply gone.

For the demo config `SS ≤ L·127² ≈ 5.2·10⁵`, so `R ≲ 720`, and the widest intermediate is
the first probe `(2¹⁵)² = 2³⁰` — inside int32.

Precision, stated rather than buried: `c8` is int8, so a row entry more than 127
quantization steps from its mean is clipped. That is what an int8 LayerNorm is; the integer
reference clips identically, so exactness still holds.

---

## 8. The sample program: one quantized transformer layer

`examples/transformer_layer.py` builds the post-norm decoder layer of
`model/transformer.py` (`Transformer.forward`), quantized end to end:

```
QKV    = X·[Wq|Wk|Wv]                     matmul_ternary         (MXU, fused)
S      = Q·Kᵀ / per-head                  matmul_i8              (VPU)
S8     = requant(S) to exp scale 1/16     requant_block
S8     = S8 + causal_mask                 add_i8
Pb     = softmax(S8)                      softmax_rows           -> scale 1/128
Vt     = Vᵀ                               transpose_i8
AO     = Pb·Vtᵀ / per-head                matmul_i8              (VPU)
AO8    = requant(AO)                      requant_block
O      = AO8·Wo                           matmul_ternary         (MXU)
X1     = X + O                            add_i8                 (residual)
X1     = layernorm(X1, γ1, β1)            layernorm_rows         (norm1)
H      = X1·W1                            matmul_ternary
H      = gelu(H)                          gelu_block
F      = H·W2                             matmul_ternary
X2     = X1 + F                           add_i8                 (residual)
Y      = layernorm(X2, γ2, β2)            layernorm_rows         (norm2)
```

That is `norm2(X1 + ff(X1))`, `X1 = norm1(X + attn(X))`, with `ff = W2·gelu(W1·)` — the
shipped `adder_ternary_vanilla` layer, whose `use_moe=False` and whose activation is
`nn.GELU`.

**Q, K and V come from one fused matmul.** `[Wq|Wk|Wv]` is a single `[d, 3d]` ternary
weight, so the projection is one `matmul_ternary` instead of three — worth 168 instruction
words out of 1024, which is the difference between fitting and not. It costs nothing in
addressing because `matmul_i8` takes explicit strides: Q is the `[T, 3d]` result at row
stride `3d` and offset 0, K at offset `d`, V at offset `2d`.

**Scales are calibrated, not guessed.** The composer runs the float layer first (over the
*ternarized* weights — those are a hardware fact, not a quantization error) and sets each
tensor's scale to its observed absmax / 127. Two scales are not free: `gelu` and `exp`
assume `1/16`, so the requants feeding them are made to land there, and the example
*reports* how much of each distribution the LUT's `±7.94` domain clips rather than assuming
it is nothing.

**Simplifications, stated rather than hidden:** no linear bias (`TernaryLinear` has one; the
MXU has no bias path and four more `add_i8` blocks do not fit), and no dropout (inference).

### 8.1 Demo config, and why it is not `d=128`

| | model (`adder_ternary_vanilla`) | demo |
| --- | --- | --- |
| `d` | 128 | **32** |
| `f` | 512 | **64** |
| heads | 4 | **2** |
| `head_dim` | 32 | **16** |
| tokens `T` | 24 | **8** |

The real layer's ternary weights alone are `4·(128·128) + 2·(128·512)` trits = 48 KB
packed, against a **64 KB ISS DRAM** (§2) that also has to hold activations. The demo
config's weights are 2.5 KB and everything fits with room to spare. The generator itself is
shape-general — nothing below `d = 32` is special-cased — so the full layer becomes
reachable the moment the ISS's DRAM space is widened past the scratchpad's `addr_w`
(§11). The `.tpu` this emits would also run the big config on the board, which has 512 KB.

### 8.2 Memory map (demo config, DRAM = scratchpad addresses)

Two disjoint ranges, both below `0x8000` so every address `li`s to a *positive* register
value and no signed/unsigned reasoning is ever needed: **tensors** at `0x0000..0x4FFF` (one
address valid in both DRAM and scratchpad) and **scratch buffers** at `0x5000..0x7FFF`
(scratchpad only, never DMA'd).

| region | bytes | note |
| --- | --- | --- |
| `X` `[8,32]` int8 | 256 | layer input |
| `Wqkv` ternary `[32,96]` | 768 | tile-major `[nt][kt]`, 2-bit packed |
| `Wo [32,32]`, `W1 [32,64]`, `W2 [64,32]` ternary | 256 + 512 + 512 | |
| `γ1 β1 γ2 β2` int8 `[32]` | 128 | |
| `mask` int8 `[16,8]` | 128 | 0 / −128, causal |
| `QKV [8,96]`, `Vᵀ [32,8]` int8 | 768 + 256 | |
| `S32 [2,8,8]` int32; `S8`, `SM8`, `P8` int8 | 512 + 3×128 | |
| `AO32` int32; `AO8 O X1 N1 F X2 Y` int8; `H HG [8,64]` | 1024 + 7×256 + 2×512 | |
| requant `{m0,n}` words, one per rescale site | 4 × 14 | |
| scratch arena (primitive-private buffers) | ~7 KB | |

Roughly 8 KB of tensors — the 64 KB space is not the constraint at this size; instruction
memory is.

### 8.3 Instruction budget

| block | ×  | words | subtotal |
| --- | --- | --- | --- |
| `matmul_ternary` | 4 | 84 | 336 |
| `matmul_i8` | 2 | 54 | 108 |
| `layernorm_rows` | 2 | 110 | 220 |
| `softmax_rows` | 1 | 48 | 48 |
| `add_i8` | 3 | 27 | 81 |
| `requant_block` | 2 | 25 | 50 |
| `gelu_block` | 1 | 25 | 25 |
| `transpose_i8` | 1 | 27 | 27 |
| `halt` | | | 1 |
| | | **total** | **≈ 896 / 1024** |

It fits, with ~12% headroom — but a *second* layer does not, which is the honest statement
of what inlined primitives cost and is called out in §11. `Program.report()` prints the real
per-block numbers, and `finalize()` refuses to emit a program that overflows, naming the
budget rather than letting the assembler produce an image the BRAM cannot hold. The first
lever if it ever does overflow is hoisting the four `matmul_ternary` instances into one
driven by a parameter block in scratchpad: that trades ~250 words for a runtime dispatch
loop, and it is the natural next step rather than a rewrite.

---

## 9. Verification

Three independent legs, in increasing strength:

1. **It assembles and runs.** `assembler.py` + `iss.py`, with the word count asserted
   against 1024 and the ISS step limit raised. Catches template/mangling/shape errors.
2. **Integer reference** (`examples/layer_ref.py`). The *exact integer pipeline* in plain
   Python ints — same requant `{m0,n}`, same LUTs (imported from `luts.py`, since ROM
   contents are a hardware artifact, not a per-program choice), same flooring shifts, and
   its own `_trunc_div` / `_isqrt` written the obvious way as a check on the templates'
   table walk. It reads the bytes actually loaded and compares the bytes actually spilled,
   **exactly**, zero tolerance, in the style of `torch_ref.py`. It shares no arithmetic
   with `iss.py`. It does share the address map and the `{m0,n}` words with the composer,
   because those *are* the composer's declaration of what the program means — that is a
   real limit on its independence and is worth knowing when reading a green run.
3. **Float reference.** The same layer in float `model/transformer.py` terms, compared with
   a *reported* error rather than an assertion. int8 activations with a `1/16` exp table
   and a 32-entry LayerNorm will not match float closely, and pretending otherwise would be
   worse than useless — leg 3 says *how far off quantization puts us*, and only leg 2 says
   *whether the program is correct*. The plan is to report max/mean abs error and the
   argmax agreement, not to assert a threshold.

Plus emission: `prog.emit_vectors()` writes `tpu_prog.hex` / `tpu_spad_in.hex` /
`tpu_spad_exp.hex` in exactly `gen_vectors.py`'s format, so `tpu_top_tb.sv` can run a
generated layer with no testbench change.

---

## 10. Open questions for review

1. **`softmax_rows` masking placement.** This masks in int8 at the exp scale (§7.6) by
   adding `−128`, which clips to the table's floor. The alternative is masking the int32
   scores before the requant, which is more precise but needs an int32 add the VPU does not
   have (`vecadd` reads int8). I believe the int8 mask is right *and* cheaper; flagging it
   because it is a numerics decision, not a mechanical one.
2. **The `1/16` LUT domain is the tightest numerical constraint in the layer.** Attention
   scores and the FF hidden state both have to fit in `±7.94` or they clip, and that ceiling
   is burned into the bitstream, not chosen per program. The example measures and prints the
   clip rate at both sites; if a real calibrated model overflows there, the fix is a LUT
   regenerated at a different `in_scale`, which invalidates every bitstream and every
   compiled program (`luts.py` says so at the top). Worth deciding deliberately.
3. **Should `Program` emit one `.tpu` per layer or one per model?** Per-layer, because 1024
   words does not hold two.

## 11. Deferred, deliberately

- **Fill/spill elision** between adjacent primitives (§4.4 rule 3). The obvious 2–3×
  cycle win; needs a liveness pass over the block list, which is the first step toward
  being an actual compiler.
- **Hoisting repeated primitives into runtime-parameterized loops** (§8.3) — the answer to
  the 1024-word ceiling.
- **A transposing DMA mode**, which would delete `transpose_i8` (§7.3).
- **ISS DRAM wider than the scratchpad** (`iss.TPU` currently sizes both from one
  `addr_w`), which is what stands between this and simulating the real `d = 128` layer.
- **`run_program.py` integration.** It builds its input image via
  `gen_vectors.build_image`, which dispatches on a program's `.equ` constants — a generated
  program brings its own image instead. A small hook (accept a generated package directory
  of program + input + expected) makes generated layers runnable on the FPGA; the vector
  files in §9 already make them runnable in the *testbench* today.
- **The `.tpu` output is the artifact.** It is emitted with per-block comment banners and
  the composer's shapes in them, so it can be read, hand-edited, and assembled with the
  existing tools independently of Python. Keeping that readable is a constraint on the
  generator, not a nicety.
