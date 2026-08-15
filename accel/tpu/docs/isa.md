# Writing TPU Programs (tpulang)

A practical guide to writing programs for the TPU in **tpulang**, its assembly language.
It walks from the machine you're targeting, through the language and the handful of idioms
every kernel uses, to fully worked programs — then keeps a condensed encoding reference in
[Appendix A](#appendix-a--encoding--opcode-reference) for when you need the bits.

Read this to *write* code. For the *language spec* (grammar, directives, toolchain flags)
see the [tpulang README](../../tpulang/README.md); for the per-unit hardware see
[scalar_unit.md](scalar_unit.md), [mxu.md](mxu.md), and [vpu.md](vpu.md). Where any of them
disagree with this doc on encoding, [Appendix A](#appendix-a--encoding--opcode-reference)
(mirrored by `scalar_unit.sv`, `assembler.py`, and `iss.py`) is the contract.

```
   kernel.tpu  ──assembler.py──►  32-bit words  ──►  instruction BRAM  ──►  scalar_unit.sv
       │                              │
       │                              └──iss.py──►  golden scratchpad image
       └──── you write this ────┘
```

---

## 1. The machine you're programming

The program runs on the **scalar unit** — a small in-order microcontroller. It fetches one
instruction at a time, does the scalar/address math itself, and **dispatches** the heavy
ops to two accelerators, blocking on each until it finishes:

```
   scalar_unit  ── issue-and-wait ──►  MXU   (ternary matmul: act @ weight)
        │                          └─►  VPU   (SIMD vectors: add, relu, exp, reductions…)
        │
   scratchpad (on-chip BRAM)  ◄── every unit reads/writes here
        ▲
   DMA  │  moves bytes DRAM ⇄ scratchpad (rdmem / wrmem)
   DRAM ┘  (external; where the host stages tensors)
```

Four things drive how you write every program:

1. **Two memory levels.** External **DRAM** holds tensors the host stages in; the on-chip
   **scratchpad** (2¹⁶ bytes, byte-addressed) is the working memory. *All math addresses
   point into the scratchpad.* DMA (`rdmem`/`wrmem`) moves bytes between the two.

2. **Registers hold addresses, not data.** There are 32 registers `r0..r31` (`r0` is
   hardwired to 0). A compute op names registers, but reads their *contents* as the
   scratchpad byte address to operate on. So the universal idiom is **load an address into
   a register, then hand the register to the op**.

3. **Sizes live in config registers, not instructions.** A `matmul` doesn't carry "how
   many tokens"; a `vecadd` doesn't carry "how many elements". You set those once in the
   config registers (`tlen`, `vlen`, `len`) and the ops read them.

4. **Dispatch is issue-and-wait.** The hardware automatically blocks until a dispatched
   unit is done before running the next instruction, so you rarely manage concurrency by
   hand.

---

## 2. Anatomy of a program

Nearly every kernel has the same five-part shape. Here is the smallest complete one —
a residual add `C = A + B` ([`vector_add.tpu`](../../tpulang/examples/vector_add.tpu)):

```asm
; ---- 1. declare constants (addresses, sizes) and register aliases ----
.equ A  0x0000          ; input vector A (int32)   -- a scratchpad byte address
.equ B  0x0400          ; input vector B (int32)
.equ C  0x0800          ; output C = A + B
.equ N  64              ; elements
.equ VEC_BYTES  256     ; N int32 = N * 4

.reg a  r1              ; give registers readable names
.reg b  r2
.reg c  r3

; ---- 2. load addresses into registers ----
    li      a, A
    li      b, B
    li      c, C

; ---- 3. stage inputs DRAM -> scratchpad ----
    setcfg  len, VEC_BYTES
    rdmem   a, a
    rdmem   b, b

; ---- 4. compute ----
    setcfg  vlen, N         ; the VPU will process N elements
    vecadd  c, a, b         ; C[i] = A[i] + B[i]

; ---- 5. store result scratchpad -> DRAM, then stop ----
    setcfg  len, VEC_BYTES
    wrmem   c, c
    halt
```

Every example in [`tpulang/examples/`](../../tpulang/examples) follows this skeleton.
Internalize it and most of the work is deciding the addresses, sizes, and op sequence.

---

## 3. Language basics

**One instruction per line**, of the form:

```
[label:]  mnemonic[.flag...]  operand, operand, ...   [; comment]
```

- **Comments** start with `;` or `#` and run to end of line.
- **Labels** end in `:` and name the *word address* of the next instruction — branch/jump
  targets. They may stack (`loop: inner: op ...`).
- Mnemonics and `.flag` suffixes are case-insensitive; symbol names are case-sensitive.

**Two directives** set up names before you use them:

| Directive        | Effect                                                            |
| ---------------- | ----------------------------------------------------------------- |
| `.equ NAME expr` | define an integer constant (addresses, sizes, tile strides)       |
| `.reg NAME rN`   | alias a register, so you write `act` instead of `r1`              |

`.equ` values are full expressions over earlier constants, labels, and numbers, using
`+ - * // % << >> & | ^ ~` and parentheses — e.g. `.equ NELEM T * COLS`,
`.equ LAST TILES - 1`, `.equ WGT_BYTES COLS * (ROWS*2/8)`. Numbers are decimal or `0x` hex.
Compute these at assemble time rather than hand-carrying magic numbers.

**Three operand kinds**, picked by the instruction's form:

- **Register** — `rN` or a `.reg` alias. Holds a byte address (or a scalar value).
- **Immediate / expression** — a number or expression, used by `li`, `setcfg`, and as
  branch/jump targets (e.g. `li act, ACT + 0x40`).
- **Label** — a control-flow target.

---

## 4. The core idioms

### 4.1 Load an address, then use it

Because registers hold addresses, the pattern is always *materialize the address, pass the
register*:

```asm
.equ  ACT  0x0000
.reg  act  r1
    li      act, ACT            ; act now holds the byte address 0x0000
    matmul  out, act, wgt       ; MXU reads scratchpad starting at that address
```

`li` **zero-extends** a 16-bit immediate, so any scratchpad address fits directly. DRAM is
wider than 16 bits (`MEM_ADDR_W = 19`), so an address above 64 KB is *built* rather than
loaded — `li hi, 0x8000` then `adds hi, hi, hi` reaches 0x10000. See
[`highmem_dma.tpu`](../../tpulang/examples/highmem_dma.tpu).

Zero extension is why that works. Under the sign extension this instruction used to have,
`li r, 0x8000` put `0xFFFF8000` in the register: masked to the scratchpad's 16 bits that
wrapped back to `0x8000`, but masked to a 19-bit DRAM address it became `0x78000` — so
every DRAM address in `[0x8000, 0xFFFF]` silently aliased to the top of the chip. A
negative constant is now `li rN, K` followed by `subs rN, r0, rN`, and the assembler
rejects a negative `li` immediate rather than delivering `imm & 0xFFFF`.

### 4.2 Set sizes in config before you compute

A compute op needs to know how much data to touch, and that comes from config registers —
set them with `setcfg` (or the host presets them):

| Config  | Drives                                   | Set before      |
| ------- | ---------------------------------------- | --------------- |
| `tlen`  | MXU token-row count `T`                  | `matmul`        |
| `vlen`  | VPU vector length (elements)             | any VPU op      |
| `len`   | DMA byte count                           | `rdmem`/`wrmem` |
| `scalar`| scratchpad address of the requant `{m0,n}` word | `matmul.rq` |

`setcfg` takes a *zero-extended* 16-bit immediate. A common bug is reusing `len` for
different-sized tensors — set it again right before each DMA (the examples do). You only
re-`setcfg` when the value changes.

### 4.3 The identical-address DMA convention

`rdmem`/`wrmem` are a **real byte copy** between DRAM and the scratchpad, in the RTL
(`dma.sv`, wired into `tpu_top.sv`) and in the ISS alike; only `wrneigh` is still a
completing no-op. The examples nonetheless give every tensor the **same address in DRAM and
in the scratchpad**, so `rdmem a, a` / `wrmem a, a` pass the same register twice. That is a
convention, not a requirement: it keeps the two address maps from drifting apart, and it
makes the golden-vector flow legible, since `gen_vectors.py` seeds DRAM and reads back the
bytes `wrmem` wrote. Scratchpad-only scratch (softmax's intermediates, say) never needs DMA
at all.

The one place it cannot hold is a **transposing** transfer (§4.6): a transpose is not safe in
place, so its source and destination are necessarily different regions.

### 4.4 Lay tensors out the way the units expect

The MXU and VPU consume fixed byte layouts. Place your tensors to match (full detail in
[Appendix A.5](#a5-numeric-conventions)):

- **Activations** `A[t][i]`: int8, **row-major**, `base + t·ROWS + i`.
- **Weights** `W[i][j]`: ternary, **column-major, 2-bit packed** (`00→0`, `01→+1`,
  `11→−1`), `base + j·(ROWS·2/8)`.
- **Requant word** `{m0,n}`: one int32 — `m0` in the low 12 bits, `n` above.
- **VPU buffers**: ops read **int8** (1-byte stride) and write **int32** (4-byte stride);
  `requant` is the only op that narrows int32→int8. So an int8 buffer and an int32 buffer
  of the same length differ 4× in bytes — size and place them accordingly.

### 4.5 Bridge scalars between registers and vectors

Reductions (`redmax`, `redsum`, `vecdot`) write a single int32 *into the scratchpad*, not a
register. To use that value as a scalar in later scalar math, pull it into a register with
`loads`, and push a register value back with `stores`:

```asm
    redmax  maxa, x         ; scratch[MAXA] = max_i x[i]
    loads   t, maxa         ; t = that max (register)
    subs    t, r0, t        ; t = -max
    stores  nega, t         ; scratch[NEGA] = -max   (now a broadcastable scalar)
    sadd    shift, x, nega  ; shift[i] = x[i] - max
```

`sadd`/`vecmul`/`sdiv`/`requant` take their scalar/param from a *scratchpad* address (the
third operand's register), so a scalar destined for broadcast must live in the scratchpad —
hence the `stores`.

### 4.6 Transpose with the DMA, not with a loop

The VPU contracts over each operand's **contiguous** axis, so `P @ V` — which contracts over
keys, V's *row* axis — needs `Vᵀ`. There is no transpose op and no strided vector load, and no
loop order avoids it (asking for `Aᵀ` instead just wants `Vᵀ` from the other side).

The DMA does it, in one dispatch, with the `.t` flag on either direction. The source is read
row-major and the destination written transposed, over a geometry in three config registers
(full detail in [dma.md §5](dma.md#5-transpose-mode-rdmemt--wrmemt)):

```asm
    setcfg  tcols, D            ; D elements per source row...
    setcfg  tsrow, D3           ; ...taken from rows D3 bytes apart (a column slice)
    setcfg  tdrow, T            ; destination rows are T elements long
    setcfg  len,   T * D
    wrmem.t v, vt               ; scratchpad V slice -> DRAM Vᵀ
    rdmem.t qt, qkv             ; DRAM Q slice       -> scratchpad Qᵀ
```

`tsrow` is what lets the source be a **column slice of a wider tensor**, which is the case
that matters: Q/K/V come out of one fused `[T][3d]` projection, so nothing has to be
un-interleaved before the transpose. Zero means "not set" for all three (`tcols → len`,
`tsrow → tcols`, `tdrow → 1`), so an unconfigured `.t` degrades to a plain copy, and `tcols`
alone unset gives a strided scatter/gather. A plain `rdmem`/`wrmem` ignores the three
registers outright — the mode is in the instruction, so a stale geometry cannot reach it.

See [`transpose_dma.tpu`](../../tpulang/examples/transpose_dma.tpu).

---

## 5. The instruction toolbox

Grouped by what you reach for. Operands `rX` are registers holding scratchpad addresses
unless noted; full semantics/encoding in [Appendix A.3](#a3-opcode-map).

**MXU — matmul.** `matmul[.acc][.rq] rout, ract, rweight`: `out = act @ ternary-weight`,
int8×ternary, int32 accumulate. `.acc` adds into the existing int32 `C` buffer (for tiling
a contraction larger than the array); `.rq` requantizes int32→int8 on store using the
`{m0,n}` word at `cfg scalar`. Token rows come from `cfg tlen`.

**VPU — elementwise, activations, reductions.**

```
vecadd  d, a, b     d[i]=a[i]+b[i]        relu    d, a    max(a[i],0)
vecemul d, a, b     d[i]=a[i]*b[i]        gelu    d, a    gelu_lut[a[i]]
vecmul  d, a, s     d[i]=a[i]*scalar(s)   exp     d, a    exp_lut[a[i]]
sadd    d, a, s     d[i]=a[i]+scalar(s)   square  d, a    a[i]^2
sdiv    d, a, s     round(a[i]*2^15/s)    redmax  d, a    max_i a[i]  -> scalar
vecdot  d, a, b     Σ a[i]*b[i] -> scalar redsum  d, a    Σ a[i]      -> scalar
requant d, a, p     clip((a[i]*m0+rnd)>>n) int32->int8, {m0,n} at p
dyt     d, a, p     as requant, clipped to +-127 (DyT / hardtanh)
```

`dyt` is `requant` with a symmetric clip. DyT — `hardtanh(alpha*x, -1, 1)`, the
normalization `model/transformer.py` uses — needs no pass of its own on this
datapath: pin the output scale to `1/127`, fold `alpha` into the multiplier, and
the saturating narrow the caller already needed *is* the hardtanh. The floor is
`-127` rather than int8's `-128` because hardtanh is odd. See
[vpu.md](vpu.md) and `accel/tpulang/adder_kernel.md` §4.

All read `cfg vlen` elements. `gelu`/`exp` use fixed 256-entry LUTs loaded at init
(`accel/tpulang/luts.py`), which means their operands must **already be at the tables'
canonical input scale of 1/16** — put a `requant` in front. Nothing checks this; a
wrongly-scaled operand just reads the wrong table entry. See [vpu.md](vpu.md).

**DMA / comms** (byte length from `cfg len`):

```
rdmem[.t]  rscratch, rdram  DRAM -> scratchpad (fill)
wrmem[.t]  rscratch, rdram  scratchpad -> DRAM (spill)
wrneigh rmy, rnb, dir       push local DRAM -> neighbor DRAM  (dir: n|e|s|w)
```

`.t` transposes: the source is read row-major and the destination written transposed, over
`cfg tcols`/`tsrow`/`tdrow` — §4.6.

**Scalar & control:**

```
li     d, imm16     d = zero_extend(imm16)      adds  d, a, b   d = a + b
loads  d, raddr     d = scratch[raddr] (int32)  subs  d, a, b   d = a - b
stores raddr, rval  scratch[raddr] = rval        muls  d, a, b   d = a * b
setcfg name, imm16  cfg[name] = imm16            cmps  a, b      set eq/lt flags
setcfgr name, rN    cfg[name] = rN  (runtime)
jmp    label        pc = label                   halt            stop, raise done
branch cond, label  if cond: pc = label          wait  unit      block on unit done
beq/bne/blt/bge label   conditional jump (cond baked in: eq|ne|lt|ge)
```

---

## 6. Control flow: loops

Loops are `cmps` + a conditional branch, exactly like a tiny assembly CPU. Targets are
labels. The tiling loop from [`tiled_matmul.tpu`](../../tpulang/examples/tiled_matmul.tpu)
walks a contraction bigger than the array, accumulating partials in the MXU's int32 `C`
buffer:

```asm
    matmul  out, act, wgt       ; tile 0: C = A0 @ W0   (no acc, no requant)
    adds    act, act, astep     ; advance the tile base pointers
    adds    wgt, wgt, wstep
    li      i, 1

loop:
    cmps    i, last             ; while i < LAST:
    bge     final               ;   (flags set by cmps; bge takes them)
    matmul.acc  out, act, wgt   ;   C += Ai @ Wi
    adds    act, act, astep
    adds    wgt, wgt, wstep
    adds    i, i, one
    jmp     loop

final:
    matmul.acc.rq  out, act, wgt ; last tile: C += A_last@W_last, then requant int32->int8
```

Note the three-way flag split across the tiles — plain `matmul` initializes `C`, `.acc`
accumulates, and the final `.acc.rq` also narrows — and that `bge` reads the flags the
immediately preceding `cmps` set (`eq=0, ne, lt, ge`).

---

## 7. Synchronization

Because dispatch is **issue-and-wait**, the hardware already fences one compute op against
the next — you do *not* insert `wait` between two dependent VPU/MXU ops. You only need an
explicit `wait unit` when a **scalar** instruction (`loads`) reads a value a unit just
wrote, and even then the examples rely on the in-order blocking and use `loads` directly
after the reduction. Reach for `wait mxu|vpu|dma|link` only if you have reason to fence a
scalar read against an outstanding dispatch. `halt` ends the program and raises `done`; the
host can restart it.

---

## 8. Worked example: softmax over a score row

Softmax has no single instruction — you decompose it into VPU primitives, using the
scalar↔vector bridge from §4.5. From
[`softmax_row.tpu`](../../tpulang/examples/softmax_row.tpu):

```asm
    setcfg  vlen, N         ; N-element row
    ; ... li each address into its register, rdmem the input row X ...

    redmax  maxa, x          ; m       = max_i x[i]
    loads   t, maxa          ; t       = m
    subs    t, r0, t         ; t       = -m
    stores  nega, t          ; scratch = -m   (broadcastable)
    sadd    shift, x, nega   ; s[i]    = x[i] - m
    exp     expa, shift      ; e[i]    = exp(s[i])       (LUT)
    redsum  dena, expa       ; D       = Σ e[i]
    sdiv    prob, expa, dena ; p[i]    = round(e[i]*2^15 / D)   (Q15)
```

The pattern to absorb: a reduction lands in the scratchpad → `loads` it → do scalar math →
`stores` it back → a broadcast op (`sadd`) consumes it. The `x−m` shift is the standard
numerically-stable softmax, and `sdiv` produces a Q15 fixed-point probability whose scale is
a compile-time constant even though `D` is runtime.

---

## 9. Assemble, simulate, test

From `accel/tpulang/`:

```bash
python assembler.py prog.tpu                 # hex words -> stdout ($readmemh image)
python assembler.py prog.tpu -o prog.hex     # write the loadable image
python assembler.py prog.tpu --listing       # addr / word / flags / source, to stderr
```

`--listing` is the fastest way to sanity-check an encoding by eye. Programmatically,
`from assembler import assemble; assemble(src) -> list[int]`.

To check correctness against the golden model, `gen_vectors.py` assembles a program, runs it
on the bit-exact [ISS](../../tpulang/iss.py), and emits the three `$readmemh` files
(`program`, `inputs`, `expected`) that `tpu_top_tb.sv` replays on the RTL:

```bash
python gen_vectors.py                        # default: examples/relu_layer.tpu
python gen_vectors.py -p examples/softmax_row.tpu
```

The expected file holds exactly the bytes the program *wrote*, so the testbench checks that
program's outputs and nothing else — the loop that keeps hardware honest against the model.

---

## 10. Checklist & common pitfalls

- **Set `len` before every DMA** whose tensor size differs from the last one; likewise
  `vlen`/`tlen` before compute. Stale config is the most common silent bug.
- **int8 vs int32 strides.** VPU ops write int32 (4 bytes/elem); only `requant` narrows.
  An int32 output buffer needs 4× the bytes of the int8 input — leave room.
- **Weights are column-major, 2-bit packed** (`00/01/11 → 0/+1/−1`); activations are
  row-major int8. Mixing these up produces a plausible-looking but wrong matmul.
- **Reductions write to the scratchpad, not a register** — `loads` to get the scalar out.
- **Broadcast scalars must be in the scratchpad** (`sadd`/`vecmul`/`sdiv`/`requant` read the
  param from a scratchpad address), so `stores` a computed scalar before broadcasting it.
- **Identical DRAM/scratchpad addresses** per tensor, by convention (§4.3) — except for a
  `.t` transfer, whose source and destination must be *different* regions.
- **A `.t` transfer needs `tcols`/`tsrow`/`tdrow` set** (§4.6). They default to a plain copy
  rather than to garbage, so the symptom of forgetting is an untransposed result.
- **Branch conditions come from the preceding `cmps`** — don't let another flag-setting op
  slip between the `cmps` and its branch.
- **End with `halt`.** Without it the scalar unit runs off into whatever follows in IMEM.

---

# Appendix A — Encoding & opcode reference

The bit-level contract shared by `scalar_unit.sv` (decodes/executes), `assembler.py`
(emits), and `iss.py` (golden simulator). You rarely need this to write programs, but it is
the authority when the units disagree.

## A.1 Machine model

| State                  | Size                       | Notes                                            |
| ---------------------- | -------------------------- | ------------------------------------------------ |
| **PC**                 | `IMEM_AW` (10) bits        | word index into instruction memory               |
| **Registers** `r0..r31`| 32 × int32 (`REG_AW=5`)    | `r0` hardwired to 0 (writes ignored)             |
| **Config** `cfg0..cfg31`| 32 × int32 (`CFG_AW=5`)   | implied operands (lengths, requant addr); §A.4   |
| **Flags**              | `eq`, `lt`                 | set by `cmps`, consumed by `branch`              |
| **Instruction mem**    | 2¹⁰ × 32-bit BRAM          | host-loaded while idle; separate from scratchpad |
| **Scratchpad**         | 2¹⁶ bytes                  | working memory; all *math* addresses point here  |
| **DRAM**               | external                   | all *DMA/comms* addresses point here             |

Address arithmetic wraps mod 2¹⁶ (scratchpad depth), matching RTL and ISS.

## A.2 Instruction word

Every instruction is a fixed **32-bit** word, in one of two layouts selected by opcode:

```
 bits  31..26   25..18   17..10    9..2    1..0
       ┌──────┬────────┬────────┬────────┬───────┐
 RRR   │opcode│  dst   │  src0  │  src1  │ flags │   register-form ops
       └──────┴────────┴────────┴────────┴───────┘
       ┌──────┬────────┬─────────────────┬───────┐
 imm16 │opcode│  dst   │      imm16      │ flags │   li / setcfg / branch / jmp
       └──────┴────────┴─────────────────┴───────┘
                        17................2
```

| Field    | Bits    | Meaning                                                            |
| -------- | ------- | ----------------------------------------------------------------- |
| `opcode` | 31..26  | 6-bit operation selector (§A.3)                                   |
| `dst`    | 25..18  | destination register (low 5 bits) — or config index for `setcfg` |
| `src0`   | 17..10  | source-0 register                                                 |
| `src1`   | 9..2    | source-1 register                                                 |
| `flags`  | 1..0    | per-op modifier: matmul acc/rq, branch cond, wait/neighbor unit  |
| `imm16`  | 17..2   | 16-bit immediate (overlays `src0`+`src1`); sign/zero-extended per op |

Packing (`assembler.py`, matching `scalar_unit.sv`):

```
word     = opcode<<26 | dst<<18 | src0<<10 | src1<<2 | flags
word_imm = opcode<<26 | dst<<18 |     (imm16 & 0xFFFF)<<2 | flags
```

**Worked encoding.** `matmul.rq r3, r1, r2` (out=r3, act=r1, weight=r2, requant):

```
opcode=0x00  dst=3  src0=1  src1=2  flags=0b10 (.rq)
word = (0<<26)|(3<<18)|(1<<10)|(2<<2)|2 = 0x000C_040A
```

`li r1, 0x1000` (imm16 form):

```
opcode=0x14  dst=1  imm16=0x1000
word = (0x14<<26)|(1<<18)|(0x1000<<2)|0 = 0x5004_4000
```

## A.3 Opcode map

Opcodes are the 6-bit `OP_*` constants in `scalar_unit.sv` (mirrored by `SPECS` in
`assembler.py`, `OP_*` in `iss.py`). "scratch[a]" is the byte at scratchpad address `a`;
int32 is little-endian 4-byte, int8 a single signed byte.

| Opcode | Mnemonic  | Form   | Unit    | Semantics                                            |
| ------ | --------- | ------ | ------- | ---------------------------------------------------- |
| `0x00` | `matmul`  | RRR    | MXU     | `out = act @ ternary-weight`; `.acc`/`.rq` via flags |
| `0x01` | `vecdot`  | RRR    | VPU     | `dst = Σ src0·src1` (scalar result)                  |
| `0x02` | `vecmul`  | RRR    | VPU     | `dst[i] = src0[i]·scalar(src1)`                      |
| `0x03` | `vecadd`  | RRR    | VPU     | `dst[i] = src0[i] + src1[i]`                         |
| `0x04` | `relu`    | RR     | VPU     | `dst[i] = max(src0[i], 0)`                           |
| `0x05` | `gelu`    | RR     | VPU     | `dst[i] = gelu_lut[src0[i]]`                         |
| `0x06` | `wrmem`   | SS     | DMA     | scratch(src0) → DRAM(src1); `.t` transposes (§4.6)   |
| `0x07` | `rdmem`   | RS     | DMA     | DRAM(src0) → scratch(dst); `.t` transposes (§4.6)    |
| `0x08` | `wrneigh` | NEIGH  | LINK    | local DRAM(src0) → neighbor DRAM(src1), dir = flags  |
| `0x09` | `requant` | RRR    | VPU     | `dst[i] = clip((src0·m0 + rnd) >> n)` int32→int8     |
| `0x0A` | `vecemul` | RRR    | VPU     | `dst[i] = src0[i]·src1[i]` (elementwise)             |
| `0x0B` | `square`  | RR     | VPU     | `dst[i] = src0[i]²`                                  |
| `0x0C` | `exp`     | RR     | VPU     | `dst[i] = exp_lut[src0[i]]`                          |
| `0x0D` | `redmax`  | RR     | VPU     | `dst = max_i src0[i]` (scalar result)               |
| `0x0E` | `redsum`  | RR     | VPU     | `dst = Σ_i src0[i]` (scalar result)                 |
| `0x0F` | `sadd`    | RRR    | VPU     | `dst[i] = src0[i] + scalar(src1)` (broadcast)       |
| `0x10` | `adds`    | RRR    | scalar  | `dst = src0 + src1`                                  |
| `0x11` | `subs`    | RRR    | scalar  | `dst = src0 − src1`                                  |
| `0x12` | `muls`    | RRR    | scalar  | `dst = src0 × src1` (low 32 bits)                    |
| `0x13` | `cmps`    | SS     | scalar  | set `eq`/`lt` from `cmp(src0, src1)`                 |
| `0x14` | `li`      | RIMM   | scalar  | `dst = zero_extend(imm16)` (§4.1)                    |
| `0x15` | `setcfg`  | CFG    | scalar  | `cfg[dst] = zero_extend(imm16)`                      |
| `0x16` | `loads`   | RS     | scalar  | `dst = scratch[src0]` (int32)                        |
| `0x17` | `stores`  | SS     | scalar  | `scratch[src0] = src1` (int32)                       |
| `0x18` | `branch`  | BRANCH | control | `if cond(flags): pc = imm16`                         |
| `0x19` | `jmp`     | JMP    | control | `pc = imm16`                                         |
| `0x1A` | `wait`    | WAIT   | control | block until `flags`-selected unit is done            |
| `0x1B` | `sdiv`    | RRR    | VPU     | `dst[i] = round(src0[i]·2¹⁵ / scalar(src1))` (Q15)  |
| `0x1C` | `setcfgr` | CFGR   | scalar  | `cfg[dst] = r[src0]` (full 32 bits, no extension)   |
| `0x1D` | `matmul_t`| RRR    | MXU     | as `matmul`, strides from `cfg arow/crow/wcol` (§3)  |
| `0x1E` | `vecmatmul`| RRR   | VPU     | `dst[t][s] = Σ_d src0[t][d]·src1[s][d]` (macro op)  |
| `0x1F` | `halt`    | NONE   | control | stop; raise `done`                                   |
| `0x20` | `softmax` | RRR    | VPU     | row-wise softmax, Q15 result (macro op)             |
| `0x21` | `dyt`     | RRR    | VPU     | as `requant`, clipped to ±127 — DyT / hardtanh       |

> `sdiv` sits at `0x1B` (added after the `0x00–0x1A` block) — the numeric gap is
> intentional. The opcode field is 6 bits, so `0x22–0x3E` remain free; `halt` keeps `0x1F`
> even though allocation has continued past it.

## A.4 Instruction forms & selectors

Which fields are *live* depends on the opcode's form (unused fields are 0):

| Form     | `dst`    | `src0`             | `src1`      | `flags`     | Mnemonics                                     |
| -------- | -------- | ------------------ | ----------- | ----------- | --------------------------------------------- |
| `RRR`    | dst reg  | src0 reg           | src1 reg    | op modifier | matmul[_t], vecdot/mul/add/emul, sadd, sdiv, requant, dyt, vecmatmul, softmax, adds/subs/muls |
| `RR`     | dst reg  | src0 reg           | —           | —           | relu, gelu, square, exp, redmax, redsum       |
| `RS`     | dst reg  | src0 reg           | —           | `.t` (rdmem)| rdmem, loads                                  |
| `SS`     | —        | src0 reg           | src1 reg    | `.t` (wrmem)| wrmem, stores, cmps                           |
| `RIMM`   | dst reg  | ⟵ imm16 ⟶          |             | —           | li                                            |
| `CFG`    | cfg idx  | ⟵ imm16 ⟶          |             | —           | setcfg                                        |
| `CFGR`   | cfg idx  | src0 reg           | —           | —           | setcfgr                                       |
| `BRANCH` | —        | ⟵ imm16 (target) ⟶ |             | cond        | branch, beq/bne/blt/bge                       |
| `JMP`    | —        | ⟵ imm16 (target) ⟶ |             | —           | jmp                                           |
| `WAIT`   | —        | —                  | —           | unit        | wait                                          |
| `NEIGH`  | —        | src0 reg           | src1 reg    | direction   | wrneigh                                       |
| `NONE`   | —        | —                  | —           | —           | halt                                          |

**Selectors / flags:**

| Field         | Values                                                   |
| ------------- | -------------------------------------------------------- |
| matmul flags  | `flags[0]=.acc` (accumulate), `flags[1]=.rq` (requant)  |
| DMA flags     | `flags[0]=.t` (transpose, §4.6) on `rdmem`/`wrmem`      |
| branch cond   | `eq=0b00`, `ne=0b01`, `lt=0b10`, `ge=0b11`              |
| wait unit     | `mxu=0b00`, `vpu=0b01`, `dma=0b10`, `link=0b11`         |
| neighbor dir  | `n=0`, `e=1`, `s=2`, `w=3`                               |
| config index  | see the config-register table below                      |

**Config registers.** Host-presettable while idle (`cfg_we`), runtime-writable with
`setcfg` (16-bit immediate, zero-extended) and `setcfgr` (a register, full 32 bits).
Indices `18..31` remain unassigned and can be named `cfg18`…`cfg31`.

| Idx | Name      | Drives                                                        |
| --- | --------- | ------------------------------------------------------------- |
| 0   | `tlen`    | MXU token rows `T` (6 bits, so ≤ 63)                          |
| 1   | `vlen`    | VPU vector length in elements (≤ 1023)                        |
| 2   | `len`     | DMA / LINK byte count (16 bits)                               |
| 3   | `scalar`  | `matmul.rq` requant `{m0,n}` word address                     |
| 4   | `ktiles`  | MXU contraction tiles = `K / ROWS`                            |
| 5   | `ntiles`  | MXU output tiles = `N / COLS`                                 |
| 6   | `arow`    | MXU activation row stride, bytes (`= K`)                      |
| 7   | `crow`    | MXU result row stride, bytes (`= N*4`, or `N` requantized)    |
| 8   | `wcol`    | MXU weight column stride, bytes (`= K*2/8`)                   |
| 9   | `vscalar` | VPU macro-op `{m0,n}` word address                            |
| 10  | `vrows`   | VPU macro-op row count (`vecmatmul`: query rows)              |
| 11  | `vcols`   | `vecmatmul` key rows                                          |
| 12  | `vrow0`   | `vecmatmul` `src0` row stride, bytes                          |
| 13  | `vrow1`   | `vecmatmul` `src1` row stride, bytes                          |
| 14  | `vcrow`   | `vecmatmul` dst row stride, bytes (int32) — the MXU owns `crow` |
| 15  | `tcols`   | `rdmem.t`/`wrmem.t` source row length, elements (0 ⇒ `len`)   |
| 16  | `tsrow`   | `rdmem.t`/`wrmem.t` source row stride, bytes (0 ⇒ `tcols`)    |
| 17  | `tdrow`   | `rdmem.t`/`wrmem.t` destination row stride, bytes (0 ⇒ 1)     |

`vscalar` is deliberately separate from `scalar`: the MXU's requant word and a VPU
macro-op's are live at the same time inside a layer.

`CFG_AW` is **5**, not 4: the transpose geometry is three registers and only index 15 was
free. Widening the file cost nothing in the encoding — `dst` is 8 bits — and left 14 spare.

**`setcfg` vs `setcfgr`.** `setcfg` takes an immediate, so the value is fixed at assembly
time. `setcfgr cfgname, rN` takes it from a register, which is what lets a geometry vary at
run time — the case that matters is incremental decode, where the number of attended keys
grows by one per step. See [`setcfgr.tpu`](../../tpulang/examples/setcfgr.tpu), which
computes a length in registers and installs it.

## A.5 Numeric conventions

Shared verbatim by `mxu.sv`, `vpu.sv`/`requant.sv`, and the ISS — what makes hardware and
golden simulator agree byte-for-byte.

**Ternary weight packing.** Column-major, 2 bits/weight: `00 → 0`, `01 → +1`, `11 → −1`
(bit 0 = nonzero flag, bit 1 = sign). `ROWS·2/8` bytes per output column.

**Requant (`matmul.rq`, `requant`)** — BitNet fixed-point int32→int8 rescale:

```
rnd     = (n == 0) ? 0 : (1 << (n - 1))
shifted = (acc·m0 + rnd) >> n          # arithmetic (floor) shift
dst     = clip(shifted, -128, 127)
```

`m0` is a positive 12-bit (`M0_W`) multiplier, `n` a 4-bit (`N_W`) shift, packed into one
int32 (`m0` low, `n` above) at `cfg[scalar]` (matmul) or the `src1` address (`requant`).
`{m0=1, n=0}` is identity+clip (a plain integer matmul).

**Scalar divide (`sdiv`)** — a runtime divisor reciprocated once, reused per lane (Q15):

```
R = (d == 0) ? all-ones(2^32−1) : floor(2^31 / |d|)     # RECIP_Q = 31
q = round( (src0[i]·R) >> (31 − 15) )                    # DIV_Q = 15
dst[i] = (d < 0) ? −q : q
```

The result is always Q15, so its scale is a compile-time constant even though `d` is
runtime. A zero divisor saturates `R` (the compiler guarantees nonzero: `Σexp ≥ 1`).

**VPU datatypes.** Compute ops read int8, accumulate int32, write int32 (4-byte stride).
`requant` is the sole narrowing op (int32→int8, 1-byte stride). Reductions write one int32.

## A.6 Execution model

The scalar unit issues one instruction per decode, **in order**, under the v1
**issue-and-wait** model: a compute/comms dispatch asserts the unit's `start`, then the
scalar unit blocks in `S_WAIT` until that unit's `done` before retiring and advancing PC.
Independent scalar/address work does not currently overlap a dispatch (run-ahead
scoreboarding is a documented future step — see
[scalar_unit.md §2](scalar_unit.md#2-execution-model)).

FSM sketch (`scalar_unit.sv`): `FETCH → DECODE → EXEC → {LOAD | WAIT | FETCH}`.
- Single-cycle ops (scalar arith, `li`, `setcfg`, `cmps`, `branch`, `jmp`, `stores`) retire
  in `EXEC`.
- `loads` takes an extra cycle in `S_LOAD` for the synchronous scratchpad read.
- Dispatches (`matmul`, VPU ops, `rdmem`/`wrmem`, `wrneigh`) and `wait` sit in `S_WAIT`
  until the selected unit's `done`.
- `halt` enters `S_HALT`, drives `done`, and re-enters on the next `host_run`.

Because the ISS runs every dispatch **atomically** (read operands → compute → write back),
the final scratchpad image it produces is exactly what the cycle-accurate hardware must
reproduce — this is what `gen_vectors.py` exports as the golden test vectors.
