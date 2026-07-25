# TPU Instruction Set Reference

The authoritative reference for the TPU's 32-bit machine instructions — the code the
[scalar unit](scalar_unit.md) fetches and executes. This document defines the encoding,
the machine model, and the exact semantics of every opcode.

**Three implementations must agree on everything below**, and this doc is the contract
between them:

| Where               | File                                     | Role                                 |
| ------------------- | ---------------------------------------- | ------------------------------------ |
| Hardware            | [`rtl/scalar_unit.sv`](../rtl/scalar_unit.sv) | decodes and executes these words     |
| Assembler           | [`tpulang/assembler.py`](../../tpulang/assembler.py) | emits these words from [tpulang](../../tpulang/README.md) |
| Golden simulator    | [`tpulang/iss.py`](../../tpulang/iss.py) | reproduces the final memory state    |

For the *microarchitecture* that runs these instructions (FSM, dispatch, run-ahead) see
[scalar_unit.md](scalar_unit.md); this doc is the programmer's-model / encoding view.
For the human-writable assembly syntax that lowers 1:1 onto these opcodes, see the
[tpulang reference](../../tpulang/README.md).

---

## 1. Machine model

The scalar unit is a small in-order engine — a microcontroller, not an out-of-order core.
Programmer-visible state:

| State                | Size                    | Notes                                                        |
| -------------------- | ----------------------- | ------------------------------------------------------------ |
| **PC**               | `IMEM_AW` bits (10)     | word index into instruction memory                           |
| **Registers** `r0..r31` | 32 × int32 (`REG_AW=5`) | `r0` is hardwired to 0 (writes ignored)                      |
| **Config regs** `cfg0..cfg15` | 16 × int32 (`CFG_AW=4`) | implied operands (lengths, requant addr); §6                 |
| **Flags**            | `eq`, `lt`              | set by `CMPS`, consumed by `BRANCH`                           |
| **Instruction mem**  | 2¹⁰ × 32-bit BRAM       | separate from the scratchpad; host-loaded while idle         |
| **Scratchpad**       | 2¹⁶ bytes               | on-chip working memory; all *math* addresses point here      |
| **DRAM**             | external                | all *DMA / comms* addresses point here                       |

**Registers hold addresses, not tensors.** The three 8-bit operand fields in an
instruction name registers (low 5 bits used). A compute op reads the *contents* of those
registers to get the scratchpad/DRAM **byte addresses** it hands to its unit. So the
idiom everywhere is: load an address into a register with `LI`, then pass the register to
the op. Tensor *sizes* are not in the instruction — they come from config registers (§6):
`TLEN` (MXU token rows), `VLEN` (VPU vector length), `LEN` (DMA/LINK byte count).

Address arithmetic wraps mod 2¹⁶ (the scratchpad depth), matching the RTL and ISS.

---

## 2. Instruction encoding

Every instruction is a fixed **32-bit** word. There are two field layouts, selected by
opcode:

```
 bits  31..26   25..18   17..10    9..2    1..0
       ┌──────┬────────┬────────┬────────┬───────┐
 RRR   │opcode│  dst   │  src0  │  src1  │ flags │   register-form ops
       └──────┴────────┴────────┴────────┴───────┘
       ┌──────┬────────┬─────────────────┬───────┐
 imm16 │opcode│  dst   │      imm16      │ flags │   LI / SETCFG / BRANCH / JMP
       └──────┴────────┴─────────────────┴───────┘
                        17................2
```

| Field    | Bits    | Meaning                                                                 |
| -------- | ------- | ----------------------------------------------------------------------- |
| `opcode` | 31..26  | 6-bit operation selector (§4). Opcode space is < 32, leaving room for fused ops. |
| `dst`    | 25..18  | destination register index (low 5 bits) — or the config index for `SETCFG` |
| `src0`   | 17..10  | source-0 register index                                                 |
| `src1`   | 9..2    | source-1 register index                                                 |
| `flags`  | 1..0    | per-op modifier: matmul acc/rq, branch condition, wait/neighbor unit, direction |
| `imm16`  | 17..2   | 16-bit immediate (overlays `src0`+`src1`); sign- or zero-extended per op |

Packing (from `assembler.py`, matching `scalar_unit.sv`):

```
word      = opcode<<26 | dst<<18 | src0<<10 | src1<<2 | flags
word_imm  = opcode<<26 | dst<<18 |     (imm16 & 0xFFFF)<<2 | flags
```

Which operand fields are *live* depends on the opcode's **form** — a two-register op
still occupies 32 bits, it just ignores the unused fields. The forms, and which register
each field names, are given per-instruction in §5 and summarized in §7.

---

## 3. Opcode map

Opcodes are the 6-bit `OP_*` constants in `scalar_unit.sv` (mirrored by `SPECS` in
`assembler.py` and the `OP_*` map in `iss.py`). Mnemonics are the [tpulang](../../tpulang/README.md)
spelling.

| Opcode | Mnemonic  | Form    | Unit   | One-line semantics                                        |
| ------ | --------- | ------- | ------ | --------------------------------------------------------- |
| `0x00` | `matmul`  | RRR     | MXU    | `out = act @ ternary-weight`; `.acc`/`.rq` via flags      |
| `0x01` | `vecdot`  | RRR     | VPU    | `dst = Σ src0·src1` (scalar result)                       |
| `0x02` | `vecmul`  | RRR     | VPU    | `dst[i] = src0[i] × scalar(src1)`                         |
| `0x03` | `vecadd`  | RRR     | VPU    | `dst[i] = src0[i] + src1[i]`                              |
| `0x04` | `relu`    | RR      | VPU    | `dst[i] = max(src0[i], 0)`                                |
| `0x05` | `gelu`    | RR      | VPU    | `dst[i] = gelu_lut[src0[i]]`                              |
| `0x06` | `wrmem`   | SS      | DMA    | scratch(src0) → DRAM(src1)                                |
| `0x07` | `rdmem`   | RS      | DMA    | DRAM(src0) → scratch(dst)                                 |
| `0x08` | `wrneigh` | NEIGH   | LINK   | local DRAM(src0) → neighbor DRAM(src1), dir = flags       |
| `0x09` | `requant` | RRR     | VPU    | `dst[i] = clip((src0[i]·m0 + rnd) >> n)` int32→int8       |
| `0x0A` | `vecemul` | RRR     | VPU    | `dst[i] = src0[i] · src1[i]` (elementwise)                |
| `0x0B` | `square`  | RR      | VPU    | `dst[i] = src0[i]²`                                       |
| `0x0C` | `exp`     | RR      | VPU    | `dst[i] = exp_lut[src0[i]]`                               |
| `0x0D` | `redmax`  | RR      | VPU    | `dst = max_i src0[i]` (scalar result)                     |
| `0x0E` | `redsum`  | RR      | VPU    | `dst = Σ_i src0[i]` (scalar result)                       |
| `0x0F` | `sadd`    | RRR     | VPU    | `dst[i] = src0[i] + scalar(src1)` (broadcast)             |
| `0x10` | `adds`    | RRR     | scalar | `dst = src0 + src1`                                       |
| `0x11` | `subs`    | RRR     | scalar | `dst = src0 − src1`                                       |
| `0x12` | `muls`    | RRR     | scalar | `dst = src0 × src1` (low 32 bits)                         |
| `0x13` | `cmps`    | SS      | scalar | set `eq`/`lt` flags from `cmp(src0, src1)`                |
| `0x14` | `li`      | RIMM    | scalar | `dst = sign_extend(imm16)`                                |
| `0x15` | `setcfg`  | CFG     | scalar | `cfg[dst] = zero_extend(imm16)`                           |
| `0x16` | `loads`   | RS      | scalar | `dst = scratch[src0]` (int32)                             |
| `0x17` | `stores`  | SS      | scalar | `scratch[src0] = src1` (int32)                            |
| `0x18` | `branch`  | BRANCH  | control| `if cond(flags): pc = imm16`                              |
| `0x19` | `jmp`     | JMP     | control| `pc = imm16`                                              |
| `0x1A` | `wait`    | WAIT    | control| block until `flags`-selected unit is done                |
| `0x1B` | `sdiv`    | RRR     | VPU    | `dst[i] = round(src0[i]·2¹⁵ / scalar(src1))` (Q15)        |
| `0x1F` | `halt`    | NONE    | control| stop; raise `done`                                       |

> Opcodes `0x1C–0x1E` are unallocated. `sdiv` sits at `0x1B` (added after the
> `0x00–0x1A` block was assigned) — the numeric gap is intentional, not a typo.

---

## 4. Instruction reference

Semantics below are stated for the [ISS](../../tpulang/iss.py), which is bit-exact with
the [RTL](../rtl/scalar_unit.sv). "scratch[a]" is the byte at scratchpad address `a`;
int32 values are little-endian 4-byte reads/writes; int8 values are single signed bytes.

### 4.1 MXU

#### `matmul[.acc][.rq]  rout, ract, rweight`  — `0x00`, RRR

`out = act @ weight`, int8 activations × ternary weights, int32 accumulate. Fields:
`dst → out`, `src0 → act`, `src1 → weight` (all registers holding scratchpad byte
addresses). Sizes: `TLEN` token rows (`cfg[0]`, 6 bits); contraction dim = array `ROWS`;
output features = array `COLS`.

| Flag  | Bit       | Effect                                                                     |
| ----- | --------- | -------------------------------------------------------------------------- |
| `.acc`| `flags[0]`| accumulate into the existing int32 `C` buffer at `out` (tile loops, §mxu)  |
| `.rq` | `flags[1]`| requantize int32→int8 on store, using the `{m0,n}` word at `cfg[SCALAR]`    |

Layout consumed:
- **Activations** `A[t][i]`: int8, row-major, `act + t·ROWS + i`.
- **Weights** `W[i][j]`: ternary, **column-major, 2-bit packed** at `weight + j·(ROWS·2/8)`.
  Trit codes: `00 → 0`, `01 → +1`, `11 → −1` (bit0 = nonzero, bit1 = sign). See
  [scratchpad.md](scratchpad.md) §2.
- **Output** `C[t][j]`: int32 result stride `COLS·4` (no `.rq`), or int8 stride `COLS`
  when `.rq` narrows on store. Under `.acc`, the readback of the running partial always
  uses the int32 stride.

The requant `{m0,n}` word is read from the address in `cfg[SCALAR]` (`m0` in the low
`M0_W=12` bits, `n` in the next `N_W=4`). Requant arithmetic is §5.

### 4.2 VPU

All VPU ops take their vector length from `VLEN` (`cfg[1]`, ≤ 1023). Unless noted, they
read **int8** operands, accumulate in **int32**, and write an **int32** result (4-byte
stride) — the VPU does *not* narrow on writeback; `requant` is the only narrowing op. For
the microarchitecture and the full `vpu_op` table see [vpu.md](vpu.md).

`src1` doubles as the **scalar/param address** for the scalar-argument ops: the op reads
one int32 word from `scratch[src1]` and broadcasts it.

| Mnemonic  | Form | Fields → operands            | Result                                                        |
| --------- | ---- | ---------------------------- | ------------------------------------------------------------- |
| `vecdot`  | RRR  | dst, src0, src1              | `dst = Σᵢ src0[i]·src1[i]` → single int32 (reduction)         |
| `vecadd`  | RRR  | dst, src0, src1              | `dst[i] = src0[i] + src1[i]`                                  |
| `vecemul` | RRR  | dst, src0, src1              | `dst[i] = src0[i]·src1[i]`                                    |
| `vecmul`  | RRR  | dst, src0, scalar=src1       | `dst[i] = src0[i] × scalar` (int32 scalar, broadcast)         |
| `sadd`    | RRR  | dst, src0, scalar=src1       | `dst[i] = src0[i] + scalar` (broadcast; `x−max`, `x−mean`)    |
| `sdiv`    | RRR  | dst, src0, scalar=src1       | `dst[i] = round(src0[i]·2¹⁵ / scalar)` in Q15 (§5)            |
| `relu`    | RR   | dst, src0                    | `dst[i] = max(src0[i], 0)`                                    |
| `square`  | RR   | dst, src0                    | `dst[i] = src0[i]²` (LayerNorm variance)                      |
| `gelu`    | RR   | dst, src0                    | `dst[i] = gelu_lut[src0[i]]` (256-entry int8→int8 LUT)        |
| `exp`     | RR   | dst, src0                    | `dst[i] = exp_lut[src0[i]]` (softmax)                         |
| `redmax`  | RR   | dst, src0                    | `dst = maxᵢ src0[i]` → single int32                           |
| `redsum`  | RR   | dst, src0                    | `dst = Σᵢ src0[i]` → single int32                             |
| `requant` | RRR  | dst(int8), src0(int32), param=src1 | `dst[i] = clip((src0[i]·m0 + rnd) >> n)`, `{m0,n}` at `src1` |

Reductions (`vecdot`, `redmax`, `redsum`) collapse the whole vector to one int32 at
`dst`. `gelu`/`exp` LUTs are fixed hardware tables loaded once at init (the ISS needs
`gelu_lut`/`exp_lut`, 256×int8, or those ops raise). LUT tables and scales:
[vpu.md §Activation LUTs](vpu.md#activation-luts).

### 4.3 DMA and LINK

Byte length for all three comes from `LEN` (`cfg[2]`, 16 bits).

| Mnemonic  | Form  | Fields → operands            | Semantics                                                    |
| --------- | ----- | ---------------------------- | ----------------------------------------------------------- |
| `wrmem`   | SS    | src0 = scratch, src1 = DRAM  | DMA scratchpad → DRAM (spill)                                |
| `rdmem`   | RS    | dst = scratch, src0 = DRAM   | DMA DRAM → scratchpad (fill)                                 |
| `wrneigh` | NEIGH | src0 = local, src1 = neighbor, `flags` = dir | push local DRAM → neighbor's DRAM over the link |

`wrneigh` direction is `n/e/s/w = 0/1/2/3` (or a numeric 0..3), carried in `flags` — see
[comms.md](comms.md).

> **The DUT ties DMA/LINK `done` high with no engine attached** (`tpu_top.sv`), and the
> ISS models all three as no-ops. A program that issues them completes without moving
> any data. Programs stay correct across this by using **identical DRAM and scratchpad
> addresses** for each tensor, so `rdmem a, a` / `wrmem a, a` are the identity in
> simulation while byte-copying on real hardware — see the DRAM-staging note in the
> [tpulang memory model](../../tpulang/README.md).

### 4.4 Scalar arithmetic and moves

| Mnemonic | Form | Fields → operands | Semantics                                              |
| -------- | ---- | ----------------- | ------------------------------------------------------ |
| `adds`   | RRR  | dst, src0, src1   | `dst = src0 + src1` (int32, wraps)                      |
| `subs`   | RRR  | dst, src0, src1   | `dst = src0 − src1`                                     |
| `muls`   | RRR  | dst, src0, src1   | `dst = src0 × src1` (low 32 bits)                       |
| `li`     | RIMM | dst, imm16        | `dst = sign_extend(imm16)` — load an address/constant  |
| `loads`  | RS   | dst, src0         | `dst = scratch[src0]` (int32 read)                     |
| `stores` | SS   | src0, src1        | `scratch[src0] = src1` (int32 write)                   |
| `setcfg` | CFG  | cfgname, imm16    | `cfg[name] = zero_extend(imm16)`; name ∈ {tlen,vlen,len,scalar} (§6) |

`loads`/`stores` are the register↔scratchpad bridge that lets scalar code read a computed
value out of a tensor (e.g. a reduction result) and write one back (e.g. the negated max
for softmax's `x − max`).

### 4.5 Control flow

| Mnemonic | Form   | Encoding                          | Semantics                                              |
| -------- | ------ | --------------------------------- | ------------------------------------------------------ |
| `cmps`   | SS     | src0, src1                        | set `eq = (src0==src1)`, `lt = (src0<src1)` (signed)   |
| `branch cond, tgt` | BRANCH | `imm16 = tgt`, `flags = cond` | if `cond` holds, `pc = tgt`                          |
| `beq/bne/blt/bge tgt` | BRANCH | aliases with `cond` baked in | conditional jump; same opcode as `branch`         |
| `jmp tgt` | JMP   | `imm16 = tgt`                     | `pc = tgt` (unconditional)                             |
| `wait unit` | WAIT | `flags = unit`                   | block until the unit's `done` (unit ∈ {mxu,vpu,dma,link}) |
| `halt`   | NONE   | —                                 | stop; raise `done`; re-runnable on the next `host_run` |

Branch/jump targets are **word addresses** in instruction memory (labels in tpulang
resolve to these). Conditions read the flags most recently set by `cmps`. `cond` codes:
`eq=0b00`, `ne=0b01`, `lt=0b10`, `ge=0b11`.

---

## 5. Numeric conventions

These are shared verbatim by `mxu.sv`, `vpu.sv`/`requant.sv`, and the ISS, and are what
makes hardware and golden simulator agree byte-for-byte.

**Ternary weight packing.** Column-major, 2 bits per weight: code `00 → 0`, `01 → +1`,
`11 → −1`. Bit 0 is the nonzero flag, bit 1 the sign. `ROWS·2/8` bytes per output column.

**Requant (`matmul.rq`, `requant`).** BitNet fixed-point rescale of an int32 accumulator
to int8:

```
rnd     = (n == 0) ? 0 : (1 << (n - 1))
shifted = (acc·m0 + rnd) >> n          # arithmetic (floor) shift
dst     = clip(shifted, -128, 127)
```

`m0` is a positive `M0_W=12`-bit multiplier, `n` an `N_W=4`-bit shift, packed into one
int32 word (`m0` low, `n` above) at the `cfg[SCALAR]` address (matmul) or the `src1`
address (`requant`). `{m0=1, n=0}` is the identity+clip (a plain integer matmul).

**Scalar divide (`sdiv`).** A runtime divisor is reciprocated once, then reused per lane
(Q15 result), matching `vpu.sv`:

```
R = (d == 0) ? all-ones(2^32−1) : floor(2^31 / |d|)     # RECIP_Q = 31
q = round( (src0[i]·R) >> (31 − 15) )                    # DIV_Q = 15
dst[i] = (d < 0) ? −q : q
```

The result is always Q15, so its scale is a compile-time constant even though `d` is
runtime. A zero divisor saturates `R` (the compiler guarantees nonzero: `Σexp ≥ 1`).

**VPU datatypes.** Compute ops read int8, accumulate int32, write int32 (4-byte stride).
`requant` is the sole narrowing op (int32 in, int8 out, 1-byte stride). Reductions write
one int32. `int8` byte-stride buffers and `int32` byte-stride buffers therefore differ by
4× — the compiler/programmer must place them accordingly.

---

## 6. Config registers

Host-presettable while idle (`cfg_we`) and runtime-writable with `setcfg`. Named indices:

| Index | Name     | Drives                    | Used by                                   |
| ----- | -------- | ------------------------- | ----------------------------------------- |
| `0`   | `tlen`   | `mxu_t_len` (6 bits)      | `matmul` token-row count `T`              |
| `1`   | `vlen`   | `vpu_vlen` (10 bits)      | every VPU op's vector length              |
| `2`   | `len`    | `dma_len` / `nb_len` (16 bits) | `rdmem`/`wrmem`/`wrneigh` byte count |
| `3`   | `scalar` | `mxu_scalar_addr`         | `matmul.rq` requant `{m0,n}` word address |

Indices `4..15` exist (`CFG_AW=4`) and can be named `cfg4`..`cfg15` in tpulang, but are
unassigned. Config makes the same bitstream run different problem sizes without
resynthesis; a full host-visible config list (T, d, f, array dims, per-tensor scales,
neighbor bitmap) is in [scalar_unit.md §6](scalar_unit.md#6-config-registers).

---

## 7. Encoding quick-reference

**Instruction forms** (which register each field names; unused fields are 0):

| Form     | `dst`     | `src0`      | `src1`         | `flags`      | Mnemonics                                    |
| -------- | --------- | ----------- | -------------- | ------------ | -------------------------------------------- |
| `RRR`    | dst reg   | src0 reg    | src1 reg       | op modifier  | matmul, vecdot/mul/add/emul, sadd, sdiv, requant, adds/subs/muls |
| `RR`     | dst reg   | src0 reg    | —              | —            | relu, gelu, square, exp, redmax, redsum      |
| `RS`     | dst reg   | src0 reg    | —              | —            | rdmem, loads                                 |
| `SS`     | —         | src0 reg    | src1 reg       | —            | wrmem, stores, cmps                          |
| `RIMM`   | dst reg   | ⟵ imm16 ⟶   |                | —            | li                                           |
| `CFG`    | cfg idx   | ⟵ imm16 ⟶   |                | —            | setcfg                                       |
| `BRANCH` | —         | ⟵ imm16 (target) ⟶ |         | cond         | branch, beq/bne/blt/bge                       |
| `JMP`    | —         | ⟵ imm16 (target) ⟶ |         | —            | jmp                                          |
| `WAIT`   | —         | —           | —              | unit         | wait                                         |
| `NEIGH`  | —         | src0 reg    | src1 reg       | direction    | wrneigh                                      |
| `NONE`   | —         | —           | —              | —            | halt                                         |

**Flag / selector fields:**

| Field         | Values                                                        |
| ------------- | ------------------------------------------------------------ |
| matmul flags  | `flags[0]=.acc` (accumulate), `flags[1]=.rq` (requant)       |
| branch cond   | `eq=0b00`, `ne=0b01`, `lt=0b10`, `ge=0b11`                    |
| wait unit     | `mxu=0b00`, `vpu=0b01`, `dma=0b10`, `link=0b11`              |
| neighbor dir  | `n=0`, `e=1`, `s=2`, `w=3`                                    |
| config index  | `tlen=0`, `vlen=1`, `len=2`, `scalar=3`                       |

---

## 8. Execution & synchronization model

The scalar unit issues one instruction per decode, **in order**, under the v1
**issue-and-wait** model: a compute/comms dispatch asserts the unit's `start`, then the
scalar unit blocks in `S_WAIT` until that unit's `done` before retiring and advancing the
PC. Independent scalar/address work does *not* currently overlap a dispatch (a documented
future optimization is run-ahead scoreboarding — see [scalar_unit.md §2](scalar_unit.md#2-execution-model)).

FSM sketch (`scalar_unit.sv`): `FETCH → DECODE → EXEC → {LOAD | WAIT | FETCH}`.
- Single-cycle ops (scalar arith, `li`, `setcfg`, `cmps`, `branch`, `jmp`, `stores`)
  retire in `EXEC`.
- `loads` takes an extra cycle in `S_LOAD` for the synchronous scratchpad read.
- Dispatches (`matmul`, all VPU ops, `rdmem`/`wrmem`, `wrneigh`) and `wait` sit in
  `S_WAIT` until the selected unit's `done`.
- `halt` enters `S_HALT`, drives `done`, and re-enters on the next `host_run`.

Because the ISS runs every dispatch **atomically** (read operands → compute → write
back), the final scratchpad image it produces is exactly what the cycle-accurate hardware
must reproduce — this is what `gen_vectors.py` exports as the golden test vectors.

---

## 9. Worked encoding

Assembling `matmul.rq  r3, r1, r2` (out=r3, act=r1, weight=r2, requant):

```
opcode = 0x00 (matmul)          → 000000
dst    = 3                       → 00000011   (bits 25..18)
src0   = 1                       → 00000001   (bits 17..10)
src1   = 2                       → 00000010   (bits  9.. 2)
flags  = 0b10 (.rq)              → 10         (bits  1.. 0)

word = (0<<26) | (3<<18) | (1<<10) | (2<<2) | 2
     = 0x000C_0000 | 0x0000_0400 | 0x0000_0008 | 0x2
     = 0x000C_040A
```

And `li r1, 0x1000` (imm16 form):

```
opcode = 0x14 (li)  → 010100
dst    = 1
imm16  = 0x1000
word   = (0x14<<26) | (1<<18) | (0x1000<<2) | 0
       = 0x5000_0000 | 0x0004_0000 | 0x0000_4000
       = 0x5004_4000
```

Cross-check any program with `python assembler.py prog.tpu --listing`, which prints the
`addr / word / flags / source` for every instruction.
