# tpulang

**tpulang** is the TPU's assembly language: a human-writable text form that lowers
**1:1** onto the scalar unit's [instruction set](../tpu/docs/isa.md). A `.tpu` source
file assembles into the 32-bit machine words the TPU's instruction BRAM executes.

```
   kernel.tpu  ──assembler.py──►  32-bit words
       │                              │
       │                              ├──►  instruction BRAM  ──►  scalar_unit.sv
       │                              └──iss.py──►  golden scratchpad image
       └──torch_ref.py──►  independent PyTorch check
```

The toolchain (all in this directory):

| File                          | Role                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| [`assembler.py`](assembler.py) | `tpuasm` — assemble `.tpu` → machine words (hex/bin image or `list[int]`) |
| [`iss.py`](iss.py)            | instruction-set simulator; runs the words, bit-exact with the RTL    |
| [`luts.py`](luts.py)          | generates the VPU's `gelu`/`exp` activation ROMs (`../tpu/rtl/luts/*.hex`) at their canonical input scale; the ISS imports the same tables |
| [`gen_vectors.py`](gen_vectors.py) | ties both together: assemble + simulate → golden test vectors for `tpu_top_tb.sv` |
| [`torch_ref.py`](torch_ref.py) | PyTorch references for the examples — an independent check on the ISS *and* the FPGA |
| [`adder_export.py`](adder_export.py) | real checkpoint → integers. The calibration, the 14 requant words per layer and the integer pipeline all live in [`model/quant.py`](../../model/quant.py); this script stages them into the kernel's DRAM map and reports task accuracy against the float model. `--iss-check` runs the real weights through `adder_model.tpu` in the ISS to prove the numbers describe the kernel |
| [`examples/`](examples)       | annotated `.tpu` programs (vector add, relu layer, tiled matmul, VPU matmul, softmax) |

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

Three operand kinds, chosen by the instruction's form (see [ISA Appendix A.4](../tpu/docs/isa.md#a4-instruction-forms--selectors)):

- **Register** — `rN` (0..31) or a `.reg` alias. `r0` reads as 0. Registers hold **byte
  addresses** (or scalar values); compute ops read the register *contents* as the
  scratchpad/DRAM address they operate on.
- **Immediate / expression** — a number or an expression over `.equ` constants and
  labels, e.g. `ACT + 0x40`, `T - 1`, `TILES * ATILE`. Numbers are decimal (`42`, `-3`)
  or hex (`0x1000`).
- **Label** — a branch/jump target, resolved to a word address.

**Expression evaluator.** Immediates are evaluated over a restricted, safe AST (no
general `eval`): integer literals, previously-defined `.equ` constants and labels,
parentheses, and the operators `+ - * // % << >> & | ^ ~` plus unary `+`/`-`. `li` and
`setcfg` immediates are both range-checked to 16-bit **unsigned** (0..65535); a negative
one is an error rather than a silent `imm & 0xFFFF`, because both instructions
zero-extend. For a negative constant use `li rN, K` then `subs rN, r0, rN`.

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
dyt      rdst, rsrc0, rparam            as requant, clipped to ±127  (DyT / hardtanh)
```

VPU vector length comes from `cfg 'vlen'`; MXU token count from `cfg 'tlen'`.

`dyt` is `requant` with a symmetric clip, which is all DyT
(`hardtanh(alpha*x, -1, 1)`) costs on this datapath: pin the output scale to
`1/127`, fold `alpha` into the multiplier, and the clip of the narrow the caller
already needed *is* the hardtanh. See [`../tpu/docs/vpu.md`](../tpu/docs/vpu.md).

**Memory / comms** (operands are registers holding *DRAM/scratchpad* addresses; byte
length from `cfg 'len'`):

```
wrmem[.t] rscratch, rdram               DMA scratchpad → DRAM (spill)
rdmem[.t] rscratch, rdram               DMA DRAM → scratchpad (fill)
wrneigh  rmy, rnb, dir                  push local DRAM → neighbor DRAM
                                         dir: n|e|s|w or 0..3
```

`.t` **transposes**: the source is read row-major over `cfg 'tcols'` elements per row at
`cfg 'tsrow'` byte stride, and the destination is written transposed at `cfg 'tdrow'`.
Source and destination must be different regions. Each of the three registers falls back
to the value that makes the mode a plain copy when left at 0, so a forgotten `setcfg`
gives an untransposed result rather than a scribble. See
[`../tpu/docs/dma.md`](../tpu/docs/dma.md) §5 and
[`examples/transpose_dma.tpu`](examples/transpose_dma.tpu).

**Scalar / control:**

```
adds     rdst, ra, rb                   rdst = ra + rb
subs     rdst, ra, rb                   rdst = ra - rb
muls     rdst, ra, rb                   rdst = ra * rb
cmps     ra, rb                          set flags from cmp(ra, rb)
li       rdst, imm16                    rdst = zero_extend(imm16)
loads    rdst, raddr                    rdst = scratch[raddr]        (int32)
stores   raddr, rval                    scratch[raddr] = rval        (int32)
setcfg   cfgname, imm16                 cfg[name] = zero_extend(imm16)
                                         cfgname: tlen|vlen|len|scalar|ktiles|
                                         ntiles|arow|crow|wcol|vscalar|vrows|
                                         vcols|vrow0|vrow1|vcrow|tcols|tsrow|
                                         tdrow (or cfgN, N < 32)
setcfgr  cfgname, rN                    cfg[name] = rN   (runtime value)
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

**DMA is real.** `tpu_top.sv` now has a DMA engine + external SRAM controller
(`dma.sv`/`sram.sv`), and the [ISS](iss.py) matches it: `rdmem`/`wrmem` are real byte
copies over `cfg 'len'` between a **separate DRAM space** and the scratchpad. Inputs live
in DRAM, `rdmem` fills the scratchpad, and `wrmem` is what makes a result host-visible
again — so `gen_vectors.py` seeds DRAM with the inputs and checks the DRAM bytes the
program spilled with `wrmem` (the testbench does the same by backdoor-poking its DRAM
model). (LINK/`wrneigh` is still a no-op — no link engine is attached.)

**The identical-address convention.** The small examples give each tensor the **same
address in DRAM and in the scratchpad**, so a fill/spill is just `rdmem a, a` / `wrmem a,
a` — a tidy way to keep the address arithmetic obvious. This is only a convenience, *not*
a requirement: because the DMA truly copies, a scratch buffer and its DRAM tile can sit at
different addresses. [`tiled_matmul.tpu`](examples/tiled_matmul.tpu) uses that to **stream
tiles** — each `rdmem` pulls one tile from an advancing DRAM address into a small *fixed*
scratchpad buffer, so A, W and C can each be far larger than the scratchpad. A `.t`
transfer is the one case where the convention *cannot* hold: a transpose is not safe in
place, so its two regions are necessarily distinct.

### 2.3 Data layout

The MXU and VPU consume fixed layouts (see [ISA §4.4](../tpu/docs/isa.md#44-lay-tensors-out-the-way-the-units-expect)):

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
| [`tiled_matmul.tpu`](examples/tiled_matmul.tpu) | streams a big `A@W` tile-by-tile: nested M/N/K loops (`li/muls/adds/cmps/branch/jmp`) `rdmem` each tile into fixed buffers, then `matmul` → `matmul.acc` → `matmul.acc.rq` — scratchpad use is fixed regardless of matrix size |
| [`vpu_matmul.tpu`](examples/vpu_matmul.tpu) | a matmul with **no** ternary operand, so the MXU cannot help: attention's `S = Q@K^T` as a `T x T` nest of `vecdot`, contracting over the head dim — `K^T` is never materialised — then a per-row `requant` back to int8 for softmax |
| [`softmax_row.tpu`](examples/softmax_row.tpu) | a multi-op micro-sequence + scalar↔vector interplay (`redmax → sadd → exp → redsum → sdiv`) — illustrative only, see the caveat below |
| [`transpose_dma.tpu`](examples/transpose_dma.tpu) | `Vᵀ` and `Qᵀ` out of a fused `[T][3D]` QKV block in one dispatch each — `wrmem.t` (spill) and `rdmem.t` (fill), with `tsrow` reading a column slice in place |
| [`highmem_dma.tpu`](examples/highmem_dma.tpu) | DMA above the low 64 KB of DRAM — the only thing here that leaves the first window. Builds a `0x10000` base with `li`+`adds` (no immediate can name it) and spills there linearly and transposed |
| [`adder_model.tpu`](examples/adder_model.tpu) | **not an example — the real thing.** The entire `adder_ternary_vanilla` model in one program: four layers (3 ternary projections, both attention matmuls, the causal mask, ReLU attention, `Wo`, the feed-forward, both residuals with their `dyt` normalizations) plus the output head. 208 words |

`adder_model.tpu` is a full forward pass of the shipped adder checkpoint in a
single run. It was five runs until `scalar_unit.sv` stopped truncating a
`rdmem`/`wrmem` DRAM address to 16 bits: the SRAM is 512 KB and the UART host
always reached all of it, but a *program* could only name the low 64 KB, which
the model's 96 KB of weights do not fit in. With the full `MEM_ADDR_W` visible
the weights are staged once and stay resident, so a forward pass is a 4 KB
upload rather than a 96 KB one. The byte-level contract the host has to satisfy
— memory maps, the 14 requant words per layer and where their scales come from,
the weight packing — is in [`adder_kernel.md`](adder_kernel.md); §2 there has
the RTL/ISS change and §7 the verification order.

> **`softmax_row.tpu` does not compute a correct softmax.** It is not type-correct:
> `exp`, `redsum` and `sdiv` read int8 operands at stride 1, but the `sadd` and `exp`
> feeding them write int32 at stride 4. Two `requant` ops — one to reach the exp table's
> `1/16` `in_scale`, one after `exp` — are what
> [vpu.md](../tpu/docs/vpu.md#activation-luts) says belong there. Read it for the op
> sequence; don't trust its output. It is registered in `torch_ref.UNSUPPORTED` with that
> reason.
>
> **The exp LUT is no longer the blocker.** [`luts.py`](luts.py) generates it, the ISS
> defaults to it, and `cmod_a7_top.sv` loads it into the VPU ROM — so the program now
> assembles and runs; it just computes the wrong thing.

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

**Check a kernel against PyTorch.** The golden model is `iss.py`, so agreeing with it
proves consistency, not correctness — a kernel and the ISS can be wrong the same way.
[`torch_ref.py`](torch_ref.py) closes that gap: it decodes the tensors out of the DRAM
byte images, recomputes each kernel in plain PyTorch, and compares exactly (these
kernels are integer end to end, so there is no tolerance to allow). It calls neither
the ISS nor the RTL, and reads the DRAM layout from the program's own `.equ` symbol
table rather than restating it.

```bash
python torch_ref.py                          # every example with a reference, vs the ISS
python torch_ref.py -p examples/relu_layer.tpu
```

`relu_layer`, `vector_add`, `tiled_matmul`, `vpu_matmul`, `transpose_dma`,
`highmem_dma` and `adder_model` have references; `softmax_row` is registered as
unsupported with the reason in `torch_ref.UNSUPPORTED`.
The same references run against **real hardware** — `accel/tpu/host/run_program.py`
calls `torch_ref.verify()` on the bytes it reads back off the FPGA, so one command
checks the device against both the ISS and PyTorch.
