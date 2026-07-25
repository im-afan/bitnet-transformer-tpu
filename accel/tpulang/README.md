# tpulang & pytpu

**tpulang** is the TPU's assembly language: a human-writable text form that lowers
**1:1** onto the scalar unit's [instruction set](../tpu/docs/isa.md). A `.tpu` source
file assembles into the 32-bit machine words the TPU's instruction BRAM executes.
**pytpu** is a planned higher-level Python DSL that compiles *down* to tpulang; it is
sketched at the end of this doc and is not yet implemented.

```
   kernel.tpu  ──assembler.py──►  32-bit words  ──►  instruction BRAM  ──►  scalar_unit.sv
       │                              │
       │                              └──iss.py──►  golden scratchpad image
       └────────── (pytpu → tpulang, planned) ──────────┘
```

The toolchain (all in this directory):

| File                          | Role                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| [`assembler.py`](assembler.py) | `tpuasm` — assemble `.tpu` → machine words (hex/bin image or `list[int]`) |
| [`iss.py`](iss.py)            | instruction-set simulator; runs the words, bit-exact with the RTL    |
| [`gen_vectors.py`](gen_vectors.py) | ties both together: assemble + simulate → golden test vectors for `tpu_top_tb.sv` |
| [`examples/`](examples)       | annotated `.tpu` programs (vector add, tiled matmul, relu layer, softmax) |

For the machine-level encoding and per-opcode semantics, read the
[ISA reference](../tpu/docs/isa.md) alongside this — this doc is the *language* (syntax,
directives, conventions); that doc is the *target* (bits, opcodes, numerics).

---

## 1. Language

### 1.1 Lines

One instruction per line. A line is:

```
[label:]...  mnemonic[.flag...]  operand, operand, ...   [; comment]
```

- **Comments** start with `;` or `#` and run to end of line.
- **Labels** end in `:` and may stack (`a: b: op ...`); each resolves to the *word
  address* of the following instruction, for use as a branch/jump target.
- **Blank lines** and comment-only lines are ignored.
- Mnemonics and flag suffixes are case-insensitive; register aliases and symbol names are
  case-sensitive.

### 1.2 Directives

| Directive          | Effect                                                                 |
| ------------------ | ---------------------------------------------------------------------- |
| `.equ NAME expr`   | define an integer constant usable in later immediates/expressions      |
| `.reg NAME rN`     | define a register alias (`rN`, N in 0..31)                             |

Symbols (labels, `.equ` names, `.reg` names) share one namespace; redefining one is an
error. `.equ` values are computed with the expression evaluator below.

### 1.3 Operands

Three operand kinds, chosen by the instruction's form (see [ISA §7](../tpu/docs/isa.md#7-encoding-quick-reference)):

- **Register** — `rN` (0..31) or a `.reg` alias. `r0` reads as 0. Registers hold **byte
  addresses** (or scalar values); compute ops read the register *contents* as the
  scratchpad/DRAM address they operate on.
- **Immediate / expression** — a number or an expression over `.equ` constants and
  labels, e.g. `ACT + 0x40`, `T - 1`, `TILES * ATILE`. Numbers are decimal (`42`, `-3`)
  or hex (`0x1000`).
- **Label** — a branch/jump target, resolved to a word address.

**Expression evaluator.** Immediates are evaluated over a restricted, safe AST (no
general `eval`): integer literals, previously-defined `.equ` constants and labels,
parentheses, and the operators `+ - * // % << >> & | ^ ~` plus unary `+`/`-`. `li`
immediates are sign-checked to 16-bit signed range; `setcfg` immediates to 16-bit
unsigned.

### 1.4 Instruction set

Every mnemonic maps to exactly one opcode. Full semantics, encoding, and numerics are in
the [ISA reference](../tpu/docs/isa.md); this is the assembler-syntax summary. Operands
named `rX` are registers holding byte addresses unless stated.

**Compute / dispatch** (operands are registers holding *scratchpad* addresses):

```
matmul[.acc][.rq] rout, ract, rweight   MXU: out = act @ ternary-weight
                                          .acc accumulate into C (tiling)
                                          .rq  requant int32→int8 on store
                                          (requant {m0,n} word at cfg 'scalar')
vecdot   rdst, rsrc0, rsrc1              dst = Σ src0·src1        (scalar result)
vecadd   rdst, rsrc0, rsrc1              dst[i] = src0[i] + src1[i]
vecemul  rdst, rsrc0, rsrc1              dst[i] = src0[i] * src1[i]
vecmul   rdst, rsrc0, rscalar           dst[i] = src0[i] * scalar
sadd     rdst, rsrc0, rscalar           dst[i] = src0[i] + scalar   (broadcast)
sdiv     rdst, rsrc0, rscalar           dst[i] = round(src0[i]*2^15 / scalar)  (Q15)
relu     rdst, rsrc0                    dst[i] = max(src0[i], 0)
gelu     rdst, rsrc0                    dst[i] = gelu_lut[src0[i]]
square   rdst, rsrc0                    dst[i] = src0[i]^2
exp      rdst, rsrc0                    dst[i] = exp_lut[src0[i]]
redmax   rdst, rsrc0                    dst = max_i src0[i]      (scalar result)
redsum   rdst, rsrc0                    dst = Σ_i src0[i]        (scalar result)
requant  rdst, rsrc0, rparam            dst[i] = clip((src0*m0+rnd) >> n)  int32→int8
```

VPU vector length comes from `cfg 'vlen'`; MXU token count from `cfg 'tlen'`.

**Memory / comms** (operands are registers holding *DRAM/scratchpad* addresses; byte
length from `cfg 'len'`):

```
wrmem    rscratch, rdram                DMA scratchpad → DRAM (spill)
rdmem    rscratch, rdram                DMA DRAM → scratchpad (fill)
wrneigh  rmy, rnb, dir                  push local DRAM → neighbor DRAM
                                         dir: n|e|s|w or 0..3
```

**Scalar / control:**

```
adds     rdst, ra, rb                   rdst = ra + rb
subs     rdst, ra, rb                   rdst = ra - rb
muls     rdst, ra, rb                   rdst = ra * rb
cmps     ra, rb                          set flags from cmp(ra, rb)
li       rdst, imm16                    rdst = sign_extend(imm16)
loads    rdst, raddr                    rdst = scratch[raddr]        (int32)
stores   raddr, rval                    scratch[raddr] = rval        (int32)
setcfg   cfgname, imm16                 cfg[name] = zero_extend(imm16)
                                         cfgname: tlen|vlen|len|scalar (or cfgN)
branch   cond, label                    if cond (from prior cmps): pc = label
beq/bne/blt/bge label                   same, condition baked in (cond: eq|ne|lt|ge)
jmp      label                          pc = label
wait     unit                           block until unit done  (unit: mxu|vpu|dma|link)
halt                                    stop, raise done
```

---

## 2. Programming model

### 2.1 Registers hold addresses

The instruction fields name registers, but compute ops consume the register **contents**
as byte addresses. So the universal idiom is *load an address, then use it*:

```
.equ  ACT  0x0000
.reg  act  r1
    li      act, ACT        ; act (r1) now holds the byte address 0x0000
    matmul  out, act, wgt   ; MXU reads scratch starting at the address in r1
```

Tensor *sizes* are never in the instruction — set them in config first (`setcfg tlen`,
`setcfg vlen`, `setcfg len`) so a compute op knows how many elements/tokens/bytes to
touch.

### 2.2 Memory model

The TPU has two memory levels: external **DRAM** and on-chip **scratchpad**. Math ops
(MXU/VPU) address the scratchpad; DMA (`rdmem`/`wrmem`) and LINK (`wrneigh`) move bytes
between DRAM and scratchpad. A self-contained kernel therefore follows:

```
host --UART/DMA--> DRAM --rdmem--> scratchpad --compute--> scratchpad --wrmem--> DRAM
```

**The identical-address convention.** The current DUT (`tpu_top.sv`) has no DMA/LINK
engine attached — their `done` is tied high — and the [ISS](iss.py) models them as
no-ops with input tensors pre-placed in the scratchpad. To keep programs correct in *both*
the real byte-copying hardware and the no-op simulator, the examples give each tensor the
**same address in DRAM and scratchpad**. Then `rdmem a, a` / `wrmem a, a` are the identity
in simulation and a matched-address copy on hardware — both agree. `gen_vectors.py` only
checks the bytes the program actually *wrote*, so DRAM-only scratch never needs staging.

### 2.3 Data layout

The MXU and VPU consume fixed layouts (see [ISA §4](../tpu/docs/isa.md#4-instruction-reference)):

- **Activations** `A[t][i]`: int8, **row-major**, `base + t·ROWS + i`.
- **Weights** `W[i][j]`: ternary, **column-major, 2-bit packed** (`00→0, 01→+1, 11→−1`),
  `base + j·(ROWS·2/8)`.
- **Requant word** `{m0,n}`: one int32, `m0` in the low 12 bits, `n` above.
- **VPU buffers**: int8 operands (1-byte stride), int32 results (4-byte stride) — a
  buffer's byte size depends on which it is. `requant` is the only op that narrows
  int32→int8.

### 2.4 Synchronization

Execution is in-order **issue-and-wait**: the hardware already blocks on a dispatched
unit's `done` before the next instruction, so an explicit `wait` is only needed to fence
against a unit whose result a later *scalar* op reads. `halt` ends the program and can be
restarted by the host.

---

## 3. Worked examples

The [`examples/`](examples) programs are heavily commented and are the best next read.
Each stages its own operands over DRAM (§2.2) and ends in `halt`.

| Program                                   | Shows                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------- |
| [`vector_add.tpu`](examples/vector_add.tpu) | smallest VPU program: config + one elementwise op (residual `C=A+B`) |
| [`relu_layer.tpu`](examples/relu_layer.tpu) | one ternary layer `Y=requant(A@W)` + `relu` — the shape the adder model runs |
| [`tiled_matmul.tpu`](examples/tiled_matmul.tpu) | the tile loop: `li/adds/cmps/branch/jmp` drive `matmul` → `matmul.acc` → `matmul.acc.rq` |
| [`softmax_row.tpu`](examples/softmax_row.tpu) | a multi-op micro-sequence + scalar↔vector interplay (`redmax → sadd → exp → redsum → sdiv`) |

**Softmax, annotated** — no single instruction; the scalar unit decomposes it into VPU
primitives, using `loads`/`stores` to bring the reduction result back as a broadcast
scalar:

```
    redmax  maxa, x         ; scratch[MAXA] = max_i x[i]
    loads   t, maxa         ; t = max
    subs    t, r0, t        ; t = -max
    stores  nega, t         ; scratch[NEGA] = -max
    sadd    shift, x, nega  ; shift[i] = x[i] - max
    exp     expa, shift     ; e[i] = exp(shift[i])
    redsum  dena, expa      ; scratch[DENA] = Σ e[i]
    sdiv    prob, expa, dena ; p[i] = round(e[i]*2^15 / D)   (Q15)
```

---

## 4. Toolchain usage

**Assemble:**

```bash
python assembler.py prog.tpu                 # hex words → stdout ($readmemh)
python assembler.py prog.tpu -o prog.hex     # $readmemh-loadable image
python assembler.py prog.tpu --format bin -o prog.bin   # $readmemb text
python assembler.py prog.tpu --listing       # annotated addr/word/flags/source → stderr
```

Programmatically:

```python
from assembler import assemble
words = assemble(open("prog.tpu").read())     # -> list[int] (32-bit each)
```

**Simulate** (golden final scratchpad image; bit-exact with the RTL):

```python
from iss import TPU
tpu = TPU(rows=8, cols=8, addr_w=16)
# ... place input tensors in tpu.mem, preset tpu.cfg ...
tpu.run(words)                                # scratchpad = tpu.mem afterward
```

**Generate test vectors** for `accel/tpu/tb/tpu_top_tb.sv` (assemble + simulate → three
`$readmemh` files: program, inputs, expected outputs):

```bash
python gen_vectors.py                       # default: examples/relu_layer.tpu
python gen_vectors.py -p examples/softmax_row.tpu
```

The expected file contains exactly the bytes the program wrote, so the SystemVerilog
testbench checks precisely that program's outputs — the loop that keeps hardware honest
against the golden model.

---

## 5. pytpu (planned)

pytpu is a higher-level Python DSL intended to compile down to tpulang, so kernels can be
written in terms of tensors and primitives instead of hand-managed registers and
addresses. **It is not yet implemented** (`core.py` is a stub); the design intent:

### Tensor

Represents an int8 or ternary tensor in TPU memory. Fields:
- `dtype` (`int8`, `int2`): element type.
- `shape` (`Tuple[int]`): tensor shape.
- `scale` (`float`): quantization scale. **Not** stored in TPU RAM — a compile-time value
  only, used to pick requantization factors around matmuls.

### Scalar

A 32-bit value in memory: either an int32, or a `{M0, n}` pair (12-bit `M0`, 4-bit `n`)
encoding a requantization factor.

### Primitives

Tensor operations that map to a single TPU instruction (matmul, vector add/dot, ReLU, …).
Everything else lowers to a sequence of primitives — e.g. softmax → `redmax`, `sadd`,
`exp`, `redsum`, `sdiv` (exactly the [softmax example](examples/softmax_row.tpu)).

### Matmuls

The compiler tiles `(ternary × int8)` matmuls to the array size and expands
`(int8 × int8)` matmuls into VPU dot-product sequences.

### Memory management

The compiler inserts the `rdmem`/`wrmem` movement around primitives. Automatic fusion to
avoid redundant scratchpad traffic is a later consideration.
