# pytpu — a staged Python emitter for tpulang

**Status: built.** The implementation is one file, [`pytpu.py`](pytpu.py), ~300 lines including
docstrings. Milestone 1 is [`examples/gen_tiled_matmul.py`](examples/gen_tiled_matmul.py).
Read [`README.md`](README.md) (the language) and [`../tpu/docs/isa.md`](../tpu/docs/isa.md)
(the target) first — this document only describes the Python layer on top.

```
  kernel.py  ──pytpu.py──►  kernel.tpu  ──assembler.py──►  words  ──iss.py──►  golden
   (you)      (emitter)      (generated,      (unchanged)          gen_vectors.py
                              readable)                            torch_ref.py
                                                                   run_program.py
```

Nothing downstream changes. pytpu's only output is ordinary tpulang text.

---

## 1. What this is, and what it is not

**Is:** a *staged emitter*. You write Python; it appends tpulang lines to a buffer. Python's
own control flow runs at **build time**; `p.loop(...)` emits a **run-time** loop out of
`cmps`/`bge`/`jmp` — the same loop nest that
[`examples/tiled_matmul.tpu`](examples/tiled_matmul.tpu) has by hand.

**Is not:** a compiler. No IR, no parser, no scheduler, no liveness analysis, no autotiling,
no fusion. Register allocation is a free list. If you want to know what the machine will do,
read the generated `.tpu` — it is line-for-line what executes.

**The problem it solves.** Full unrolling from Python ("emit one `matmul` per tile") is dead
on arrival: IMEM is **1024 words**. `tiled_matmul.tpu` computes an arbitrarily large
`[M,K]@[K,N]` in ~60 instructions because the tile counts live in *registers*, not in the
instruction stream. pytpu's job is to make that style writable from Python, not to replace it.

---

## 2. The one concept: two-level staging

Every value in a pytpu program is exactly one of two things.

| Kind | Python type | When it exists | Cost |
| --- | --- | --- | --- |
| **Build-time** | `int`, `Sym` | while the Python script runs | free — folds into an immediate |
| **Run-time** | `Reg` | while the TPU runs | one instruction per operation |

That single distinction decides everything else:

```python
for kt in range(KTILES):          # kt is an int   -> UNROLLED, KTILES copies emitted
    ...
with p.loop(KTILES) as kt:        # kt is a Reg    -> ONE copy + 4 words of loop overhead
    ...
```

Address arithmetic follows the same rule automatically, because `Reg` overloads the
operators:

```python
arow + kt * ATILE     # kt is int  -> Python folds it; one `adds` (or an immediate)
arow + kt * ATILE     # kt is Reg  -> emits `muls tmp, kt, ATILE` + `adds tmp, tmp, arow`
```

There is deliberately **no `unroll=` flag**. Python's `for` already *is* the unrolled form;
adding a second way to spell it would only invite confusion about which one you got.

---

## 3. Hard constraints

Everything in the design below is downstream of this table. `pytpu.py` checks the starred
ones and raises; the rest are yours to respect.

| Limit | Value | Source |
| --- | --- | --- |
| Instruction memory ★ | **1024 words** (`IMEM_AW = 10`) | `scalar_unit.sv`, `tpu_uart.py` |
| Scalar registers ★ | r0..r31, `r0` reads as 0 | `assembler.NUM_REGS` |
| Scratchpad / DRAM | 2¹⁶ B each (ISS); board DRAM is 512 KB | `iss.py`, `tpu_top.sv` |
| MXU array | ROWS = COLS = 8 | `gen_vectors.py` asserts it |
| `cfg tlen` | ≤ 63 (6 bits) | `iss._matmul` |
| `cfg vlen` | ≤ 1023 (10 bits) — longer blocks must be chunked | `iss._vpu` |
| `li` immediate ★ | 16-bit signed | `assembler._imm16` |
| Scalar ALU | `adds`, `subs`, `muls` **only** — no divide, no shift | ISA §A.4 |
| Control flow | `cmps` + `beq/bne/blt/bge` + `jmp label`; **no call/return, no indirect jump** | ISA §A.4 |
| ISS step limit | 100 000 (raise it for looping kernels) | `iss.TPU.run` |

Two semantic traps that cause most tpulang bugs — pytpu does **not** protect you from either,
by design (it emits, it does not typecheck):

- **VPU ops read int8 at stride 1 and write int32 at stride 4.** `requant` is the only op that
  narrows back. `softmax_row.tpu` is broken in exactly this way.
- **Reductions (`redmax`, `redsum`, `vecdot`) write one int32 to the *scratchpad*, not to a
  register.** Getting a reduction into a broadcast scalar is `reduce → loads → scalar math →
  stores`.

---

## 4. API surface

The whole thing. Anything not listed here is out of scope for v1 (§7).

### 4.1 Program

```python
p = Program("tiled matmul")        # optional header comment
p.equ("MTILES", 2)                 # -> `.equ MTILES 2`; returns a Sym (an int you can range())
p.emit("; anything at all")        # raw escape hatch, appended verbatim
src   = p.render()                 # -> the .tpu source text
words = p.assemble()               # -> list[int]; raises if len > 1024
p.save("examples/out/foo.tpu")     # render + write
```

`p.equ` matters more than it looks: `gen_vectors.program_kind()` and `torch_ref.select()`
dispatch on a program's `.equ` **names**. Emit `MTILES`, `ATILE`, `A`, `W`, `C`, `RQW` with
the names the hand-written program used and the entire existing verification pipeline works
on generated files with no changes.

### 4.2 Values

```python
r = p.const(0x1000)      # int or Sym or expression string -> Reg, `li` hoisted to the prologue
r = p.reg("acc", init=0) # a named Reg, optionally `li`-initialised
a + b, a - b, a * b      # -> a NEW Reg; the other side may be Reg, int, Sym or str
a += b, a -= b, a *= b   # -> mutates a in place, no new Reg
```

Every op below coerces a non-`Reg` operand through `p.const()`, so `p.matmul(CBUF, ABUF,
WBUF)` with plain ints just works — no manual `li` boilerplate anywhere.

**Use `+=` for address arithmetic.** `a + b` allocates a destination and never reclaims it, so
a four-term expression burns four registers where the hand-written code uses one. `+=` emits
`adds a, a, b` exactly like the examples do. It is safe by construction: on a *protected*
register (a `p.const` cache entry, a loop counter) it silently falls back to allocating, so it
cannot corrupt a shared value.

Build-time identities are folded, so peeling an iteration in Python is genuinely free:
`x + 0`, `0 + x`, `x - 0`, `x * 1` emit nothing and return `x`; `x * 0` returns `r0`. That is
what makes `tile(0)` in §4.5 cost zero address instructions.

### 4.3 Control

```python
with p.loop(N) as i:          # i = 0 .. N-1
with p.loop(1, KLAST) as kt:  # range-style: kt = 1 .. KLAST-1
with p.scope():               # reclaim temporaries without emitting a loop
p.halt()
p.wait("mxu")                 # mxu | vpu | dma | link
p.emit("; raw line"); p.comment("indented comment")
```

**v1 has loops but no `if`.** Every branchy case in the examples — `tiled_matmul`'s special
first (`matmul`) and last (`matmul.acc.rq`) K tiles — is build-time-known, so you *peel* it in
Python and the loop covers only the uniform middle. That is why `p.loop` takes a start.

### 4.4 Ops

One thin wrapper per mnemonic, generated from a table rather than hand-written (§5.5).

```python
p.cfg(tlen=T, len=ATILE, vlen=..., scalar=SRQW)     # -> setcfg lines

p.matmul(out, act, wgt, acc=False, rq=False)        # .acc / .rq suffixes
p.vecadd(dst, a, b);  p.vecemul(dst, a, b);  p.vecdot(dst, a, b)
p.vecmul(dst, a, s);  p.sadd(dst, a, s);     p.sdiv(dst, a, s)
p.requant(dst, src, param)
p.relu(dst, s);  p.gelu(dst, s);  p.exp(dst, s);  p.square(dst, s)
p.redmax(dst, s);  p.redsum(dst, s)

p.rdmem(scratch, dram, t=False)                     # length from cfg 'len';
p.wrmem(scratch, dram, t=False)                     #   t=True -> .t (transpose)
p.stores(addr, value);   v = p.loads(addr)          # loads RETURNS a new Reg
```

### 4.5 The whole of `tiled_matmul` in this API

Milestone 1, abridged — the real thing is
[`examples/gen_tiled_matmul.py`](examples/gen_tiled_matmul.py); compare against
[`examples/tiled_matmul.tpu`](examples/tiled_matmul.tpu) lines 100–191.

```python
p = Program("tiled matmul: C = A @ W, streamed one tile at a time")
T = p.equ("T", 4);  MTILES = p.equ("MTILES", 2);  KTILES = p.equ("KTILES", 4)
KLAST = p.equ("KLAST", KTILES - 1);  ATILE = p.equ("ATILE", T * ROWS)
...                                     # the rest of the .equ block, as in the .tpu

p.cfg(tlen=T, scalar=SRQW)
p.cfg(len=4); p.rdmem(SRQW, RQW)        # stage the requant word once

def tile(kt, **kw):                     # kt may be int (peeled) or Reg (looped)
    at = kt * ATILE;  at += arow
    wt = kt * WTILE;  wt += wcol
    p.cfg(len=ATILE); p.rdmem(ABUF, at)
    p.cfg(len=WTILE); p.rdmem(WBUF, wt)
    p.matmul(CBUF, ABUF, WBUF, **kw)

with p.loop(MTILES, name="mt") as mt:
    arow = mt * ASTRIDE_M;  arow += A
    with p.loop(NTILES, name="nt") as nt:
        wcol = nt * WSTRIDE_N;  wcol += W
        ctil = mt * NTILES;  ctil += nt;  ctil *= CTILE;  ctil += C

        tile(0)                                     # init the int32 accumulator
        with p.loop(1, KLAST, name="kt") as kt:
            tile(kt, acc=True)                      # C += this tile
        tile(KLAST, acc=True, rq=True)              # accumulate + requant

        p.cfg(len=CTILE); p.wrmem(CBUF, ctil)
p.halt()
```

`tile()` being called with an `int` for the peeled tiles and a `Reg` for the loop body, with no
change to its text, is the payoff of §2 — and with `kt = 0` the folds reduce `at` to `arow`
itself, so the peeled tile emits *no* address arithmetic at all.

---

## 5. Implementation

One file, `accel/tpulang/pytpu.py`, imports `assembler` from its own directory the way
`torch_ref.py` already does. Seven pieces, in order.

### 5.1 `Sym(int)` — ~6 lines

A named build-time constant. Subclasses `int`, so `range(MTILES)` and `MTILES * ATILE` work;
carries `.name` so an operand renders as `MTILES` instead of `2`. Arithmetic on a `Sym` yields
a plain `int` (the name is lost and the value folds) — that is intended. Use `p.equ` for
anything you want to keep a name.

### 5.2 `Reg` — ~35 lines

`idx`, `name`, `protected`, and a back-pointer to the `Program`. `__add__`/`__sub__`/`__mul__`
(plus the `__r*__` reflections) allocate a destination; `__iadd__`/`__isub__`/`__imul__` reuse
the left operand unless it is protected. All nine route through one `_bin`, which applies the
build-time identity folds first (§4.2). `__repr__` returns the alias name, which is how a `Reg`
renders into an operand list. No `__del__`; lifetime is scope-based (§5.3).

### 5.3 Register allocation — ~25 lines

A free list over r1..r31 (`r0` is the hardwired zero, and is what `p.const(0)` returns without
emitting anything). Two orthogonal properties, not one:

- **permanent** (lifetime) — `p.const()` results. Never reclaimed, because their `li` is in the
  prologue and their value must survive every scope.
- **protected** (mutability) — constants *and* loop counters. In-place operators refuse to
  clobber these and allocate instead. A counter is protected but not permanent: it dies with
  its loop.

Everything else is scoped: `p.scope()` (which `p.loop` wraps) snapshots the free list on entry
and restores it on exit, minus anything permanent allocated in between. A stack allocator with
one rule: *do not use a `Reg` outside the block that made it.* Running out raises `TpuError`
naming the last few allocations and suggesting `+=`.

### 5.4 `const` and the prologue — ~15 lines

`p.const(v)` caches on the *rendered* expression (so `ATILE` and a bare `32` stay distinct, and
two uses of `ATILE` share one register) and appends its `li` to a **separate prologue buffer** that
`render()` emits before the body. Hoisting matters: a constant first used inside a loop body
would otherwise re-execute its `li` every iteration, and would be undefined if the loop ran
zero times. `.equ`, `.reg` and `li` lines therefore all come out at the top, exactly like the
hand-written examples.

### 5.5 Ops from a table — ~30 lines

Do not hand-write fifteen near-identical methods. One dict of `name -> arity`, wrapped into
bound methods by a `setattr` loop at import time; the body coerces every argument through
`_val` and emits. `matmul` (flag suffixes) and `loads` (returns a `Reg`) are the two that do
not fit the shape and are written out. `p.cfg(**kw)` is a `for` over its keyword arguments.
A mnemonic in `assembler.SPECS` but absent from the table is a deliberate omission, listed with
its reason at the bottom of the file: `branch`/`jmp` (use `p.loop`), `li`/`setcfg` (use
`p.const`/`p.cfg`), `cmps` (no run-time `if` in v1), `wrneigh` (LINK is a no-op in RTL).

### 5.6 `loop` — ~20 lines

A `@contextmanager` emitting, with a per-program counter for unique labels:

```
    li      i, START
L3_top:
    cmps    i, bound          ; bound = p.const(STOP)
    bge     L3_end
    ...body...
    adds    i, i, one
    jmp     L3_top
L3_end:
```

Loop overhead is 4 words per nest level plus the counter; `bound` and `one` are ordinary cached
constants, so sibling loops with the same trip count share them. The body is indented one level
per nest so the generated source reads like the hand-written one.

### 5.7 `render` / `assemble` — ~20 lines

Header comment → `.equ` block → `.reg` block → prologue `li`s → body. `assemble()` calls
`assembler.assemble()` and raises if the program exceeds 1024 words, reporting the count.
Register names are uniquified with a suffix on collision, since tpulang has one flat namespace
for labels, `.equ` names and `.reg` aliases.

---

## 6. Milestones and acceptance

**M1 — reproduce `tiled_matmul`.** [`examples/gen_tiled_matmul.py`](examples/gen_tiled_matmul.py)
(§4.5) builds it and self-checks three ways:

```bash
python examples/gen_tiled_matmul.py            # from accel/tpulang/
python examples/gen_tiled_matmul.py --print    # + dump the generated source
```

1. **Behavioural.** Build the DRAM image with `gen_vectors.build_image()`, run both the
   generated words and the hand-written words in the ISS, and assert the two final DRAM images
   are byte-identical. This, not word-for-word equality, is the real bar — pytpu is free to
   pick different registers and to fold arithmetic the human wrote out.
2. **Size.** Assert the generated program is within 8 words of the hand-written 61. A blowout
   here means staging leaked and something unrolled that should not have.
3. **Reference.** `gen_vectors`' independent Python `A@W` must pass on the generated program,
   which it finds by `.equ` name alone.

Then, on the generated file, unchanged and with no pytpu involvement at all:

```bash
python gen_vectors.py -p examples/out/tiled_matmul_gen.tpu
python torch_ref.py   -p examples/out/tiled_matmul_gen.tpu
```

**M2 — a kernel that does not already exist by hand.** The obvious one is a ternary linear
layer over a real weight matrix from `model/` — `matmul` + `requant` + `relu`, tiled over N.
This is the first time pytpu earns its keep, because the shapes come from a checkpoint rather
than from a `.equ` block someone typed.

**M3 — write it up.** A short §5 in [`README.md`](README.md) pointing here, and a line in the
examples table.

---

## 7. Deliberately out of scope for v1

Listed so they do not creep in. Each is a separate, later decision.

| Not now | Why, and what would trigger it |
| --- | --- |
| `p.if_` / run-time conditionals | Every branchy case so far is build-time-known and peels in Python. Add it when a kernel needs a data-dependent branch. |
| Host-side sequencing across programs | The real fix for "one layer exceeds 1024 words": emit N programs, let DRAM carry state, drive them from `run_program.py`. Needs pytpu to exist first. |
| Resident routines + a DRAM parameter block | The workaround for the missing call/return: load parameterised routines once, `go(pc)` per call. Only worth it once IMEM reload latency actually hurts. |
| On-device fake call/return (return-ID + `cmps`/`beq` chain) | `jmp` is imm16-only, so real subroutines are impossible. Ugly; only if a body must be shared *within* one program. |
| Quantisation, weight packing, memory arena | Salvage from the deleted `pytpu/` tree (`quant.py`, `memory.py`) when M2 needs real weights. Not part of the emitter. |
| `.tpu` templates with `{{param}}` substitution | The previous design. Superseded: a Python function that emits lines composes better than text substitution, and does not need symbol mangling. |
| Automatic `vlen` chunking, layout checking, int8/int32 stride checking | pytpu emits; it does not typecheck. Revisit only if the same bug bites twice. |
