#!/usr/bin/env python3
"""tpuasm — assembler for **tpulang**, the TPU's assembly language.

tpulang has a one-to-one mapping onto the scalar unit's instruction set
(`accel/tpu/rtl/scalar_unit.sv`). This tool turns a `.tpu` source file into the
32-bit machine words that get loaded into the TPU's instruction BRAM (the
`imem_*` host port, or a `$readmemh` preload). The encoding produced here is the
exact one the RTL decodes — hand-verified against the `enc_rrr`/`enc_imm`
helpers in `accel/tpu/tb/tpu_top_tb.sv`.

--------------------------------------------------------------------------------
Instruction word (scalar_unit.md §4)
--------------------------------------------------------------------------------
    bits  31..26  25..18  17..10   9..2    1..0
          opcode   dst     src0    src1   flags
    imm16 ops:     dst    <----- imm16 ---->  flags     (imm16 = bits 17..2)

The three 8-bit operand fields name **scalar registers** (r0..r31, r0 == 0). A
compute op reads the *contents* of those registers to get the scratchpad / DRAM
byte addresses it hands to its unit — so you first `li` an address into a
register, then pass the register to the op.

--------------------------------------------------------------------------------
Syntax
--------------------------------------------------------------------------------
    ; comments start with ';' or '#'         (rest of line ignored)
    label:                                    ; label -> instruction word address
    mnemonic[.flag...]  op, op, ...           ; one instruction per line

    .equ NAME expr        define an integer constant (used in immediates)
    .reg NAME rN          define a register alias  (rN, N in 0..31)

Operands:
    rN or a .reg alias    a register
    a number / .equ name  an immediate or an expression ('ACT + 0x40', 'T - 1')
    a label               a branch / jump target (resolved to a word address)

Numbers: decimal (`42`, `-3`) or hex (`0x1000`).

--------------------------------------------------------------------------------
Instruction reference (mnemonic -> opcode; see docs/scalar_unit.md §3, vpu.md)
--------------------------------------------------------------------------------
Compute / dispatch — operands are registers holding scratchpad byte addresses:

    matmul[.acc][.rq] rout, ract, rweight   MXU: out = act @ ternary-weight
                                             .acc = accumulate into C (tiling)
                                             .rq  = requant int32->int8 on store
                                             (requant {m0,n} word: cfg 'scalar')
    vecdot   rdst, rsrc0, rsrc1              out = Σ src0·src1   (scalar result)
    vecadd   rdst, rsrc0, rsrc1              dst[i] = src0[i] + src1[i]
    vecmul   rdst, rsrc0, rscalar            dst[i] = src0[i] * scalar
    vecemul  rdst, rsrc0, rsrc1              dst[i] = src0[i] * src1[i]
    relu     rdst, rsrc0                     dst[i] = max(src0[i], 0)
    gelu     rdst, rsrc0                     dst[i] = gelu_lut[src0[i]]
    square   rdst, rsrc0                     dst[i] = src0[i]^2
    exp      rdst, rsrc0                     dst[i] = exp_lut[src0[i]]
    redmax   rdst, rsrc0                     dst = max_i src0[i]  (scalar result)
    redsum   rdst, rsrc0                     dst = Σ_i src0[i]    (scalar result)
    sadd     rdst, rsrc0, rscalar            dst[i] = src0[i] + scalar (broadcast)
    sdiv     rdst, rsrc0, rscalar            dst[i] = round(src0[i]*2^15 / scalar)
    requant  rdst, rsrc0, rparam             dst[i] = clip((src0*m0+rnd) >> n)

Vector length for the VPU ops comes from cfg 'vlen'; MXU token count from cfg
'tlen'.

Memory / comms — operands are registers holding DRAM/scratchpad byte addresses:

    wrmem    rscratch, rdram                 DMA scratchpad -> DRAM  (spill)
    rdmem    rscratch, rdram                 DMA DRAM -> scratchpad  (fill)
    wrneigh  rmy, rnb, dir                   push local DRAM -> neighbor DRAM
                                             dir: 0..3 or n/e/s/w
Byte length for these comes from cfg 'len'.

Scalar / control:

    adds     rdst, ra, rb                    rdst = ra + rb
    subs     rdst, ra, rb                    rdst = ra - rb
    muls     rdst, ra, rb                    rdst = ra * rb
    cmps     ra, rb                          set flags from cmp(ra, rb)
    li       rdst, imm16                     rdst = sign_extend(imm16)
    setcfg   cfgname, imm16                  cfg[name] = zero_extend(imm16)
                                             cfgname: tlen|vlen|len|scalar
    loads    rdst, raddr                     rdst = scratch[raddr]
    stores   raddr, rval                     scratch[raddr] = rval
    branch   cond, label                     if cond (from a prior cmps): pc=label
    beq/bne/blt/bge label                    same, cond baked in
                                             cond: eq|ne|lt|ge
    jmp      label                           pc = label
    wait     unit                            block until unit done
                                             unit: mxu|vpu|dma|link
    halt                                     stop, raise `done`

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python assembler.py prog.tpu                 # hex words -> stdout
    python assembler.py prog.tpu -o prog.hex     # $readmemh-loadable image
    python assembler.py prog.tpu --format bin -o prog.bin   # $readmemb text
    python assembler.py prog.tpu --listing       # annotated listing -> stderr

    # programmatic:
    from assembler import assemble
    words = assemble(open('prog.tpu').read())    # -> list[int] (32-bit each)
"""

from __future__ import annotations

import argparse
import ast
import operator
import re
import sys
from dataclasses import dataclass

# --- register file / config file sizing (mirrors scalar_unit.sv defaults) -----
NUM_REGS = 32  # 2**REG_AW, REG_AW = 5
NUM_CFG = 32  # 2**CFG_AW, CFG_AW = 5

# --- config register names (scalar_unit.sv CFG_*) -----------------------------
CFG_NAMES = {
    # original four
    "tlen": 0,      # MXU token count T
    "vlen": 1,      # VPU vector length (elements)
    "len": 2,       # DMA / wrneigh byte count
    "scalar": 3,    # MXU requant {m0,n} word address
    # MXU hardware tiling geometry (docs/macro_ops.md §3/§4)
    "ktiles": 4,    # contraction tiles = K / ROWS
    "ntiles": 5,    # output tiles      = N / COLS
    "arow": 6,      # activation row stride, bytes (= K)
    "crow": 7,      # result row stride, bytes (= N*4, or N when requantizing)
    "wcol": 8,      # weight column stride, bytes (= K*2/8)
    # VPU macro-op geometry (docs/macro_ops.md §5)
    "vscalar": 9,   # macro-op {m0,n} word address (distinct from `scalar`:
                    # the MXU's requant word is live at the same time)
    "vrows": 10,    # row count (softmax/layernorm rows; vecmatmul query rows)
    "vcols": 11,    # vecmatmul key rows — independent of vrows, so decode
                    # (1 x t+1) and prefill (T x T) are the same instruction
    "vrow0": 12,    # vecmatmul src0 row stride, bytes
    "vrow1": 13,    # vecmatmul src1 row stride, bytes
    "vcrow": 14,    # vecmatmul dst row stride, bytes (int32). Distinct from
                    # `crow`, which is the MXU's — a layer has both live at once.
    # DMA transpose geometry (docs/dma.md §5), read only by `rdmem.t`/`wrmem.t`.
    # The source is walked row-major and the destination transposed, so these
    # three describe the *source* shape and where its columns land.
    "tcols": 15,    # source row length in elements  (0 => len: one row)
    "tsrow": 16,    # source row stride, bytes       (0 => tcols: dense)
    "tdrow": 17,    # destination row stride, bytes  (0 => 1)
}   # 18..31 unassigned

# --- branch condition codes (scalar_unit.sv C_*) ------------------------------
COND_CODES = {"eq": 0b00, "ne": 0b01, "lt": 0b10, "ge": 0b11}

# --- wait/neighbor unit selector (scalar_unit.sv U_*) -------------------------
UNIT_CODES = {"mxu": 0b00, "vpu": 0b01, "dma": 0b10, "link": 0b11}

# --- neighbor direction (comms) -----------------------------------------------
DIR_CODES = {
    "n": 0,
    "e": 1,
    "s": 2,
    "w": 3,
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}


class AsmError(Exception):
    """A user-facing assembly error, tagged with a source line number."""

    def __init__(self, lineno: int, msg: str):
        super().__init__(f"line {lineno}: {msg}")
        self.lineno = lineno
        self.msg = msg


# ------------------------------------------------------------------------------
# Instruction word packing. Field layout (scalar_unit.sv):
#   word = opcode<<26 | dst<<18 | src0<<10 | src1<<2 | flags
#   imm16 ops: word = opcode<<26 | dst<<18 | (imm16 & 0xFFFF)<<2 | flags
# ------------------------------------------------------------------------------
def _pack(
    opcode: int, dst: int = 0, src0: int = 0, src1: int = 0, flags: int = 0
) -> int:
    return (
        (opcode & 0x3F) << 26
        | (dst & 0xFF) << 18
        | (src0 & 0xFF) << 10
        | (src1 & 0xFF) << 2
        | (flags & 0x3)
    )


def _pack_imm(opcode: int, dst: int, imm16: int, flags: int = 0) -> int:
    return (
        (opcode & 0x3F) << 26
        | (dst & 0xFF) << 18
        | (imm16 & 0xFFFF) << 2
        | (flags & 0x3)
    )


@dataclass
class Spec:
    opcode: int
    form: str
    flag_bits: dict  # flag-suffix name -> OR'd flag value (matmul only)


# mnemonic -> Spec. Forms are handled in Assembler._encode().
SPECS: dict[str, Spec] = {
    # compute / dispatch
    "matmul": Spec(0x00, "RRR", {"acc": 0b01, "rq": 0b10}),
    # Same operands and flags as `matmul`, but the operands are tiles of a larger
    # matrix and the strides come from cfg arow/crow/wcol. A distinct opcode
    # rather than a third flag: `matmul`'s flags field is 2 bits and both are
    # taken, and gating on "the stride registers happen to be nonzero" would let
    # one program's leftover config corrupt the next program's plain matmul.
    "matmul_t": Spec(0x1D, "RRR", {"acc": 0b01, "rq": 0b10}),
    "vecdot": Spec(0x01, "RRR", {}),
    "vecmul": Spec(0x02, "RRR", {}),
    "vecadd": Spec(0x03, "RRR", {}),
    "relu": Spec(0x04, "RR", {}),
    "gelu": Spec(0x05, "RR", {}),
    "requant": Spec(0x09, "RRR", {}),
    "vecemul": Spec(0x0A, "RRR", {}),
    "square": Spec(0x0B, "RR", {}),
    "exp": Spec(0x0C, "RR", {}),
    "redmax": Spec(0x0D, "RR", {}),
    "redsum": Spec(0x0E, "RR", {}),
    "sadd": Spec(0x0F, "RRR", {}),
    "sdiv": Spec(0x1B, "RRR", {}),
    # VPU macro op: S[t][s] = sum_d src0[t][d]*src1[s][d] over cfg vrows x vcols,
    # contracting cfg vlen. Attention's Q@K^T and P@V — both operands are int8
    # activations, which the ternary-weight MXU cannot multiply at all.
    "vecmatmul": Spec(0x1E, "RRR", {}),
    # Row-wise softmax: `softmax dst, src, tmp`. Rows from cfg vrows, row length
    # from cfg vlen, requant {m0,n} from cfg vscalar. The tmp row must be
    # vlen+4 bytes -- the trailing int32 is where the hardware parks the
    # denominator between its internal passes.
    "softmax": Spec(0x20, "RRR", {}),
    # memory / comms
    # `.t` transposes: the source is read row-major (cfg tcols/tsrow) and the
    # destination written transposed (cfg tdrow). Both ops had a spare flags
    # field, so the mode costs no opcode — and carrying it in the instruction
    # rather than inferring it from nonzero strides keeps a stale cfg from
    # rearranging an unrelated plain copy (docs/macro_ops.md §4.0).
    "wrmem": Spec(0x06, "SS", {"t": 0b01}),  # scratch(src0) -> DRAM(src1)
    "rdmem": Spec(0x07, "RS", {"t": 0b01}),  # DRAM(src0) -> scratch(dst)
    "wrneigh": Spec(0x08, "NEIGH", {}),  # local(src0) -> neighbor(src1), dir=flags
    # scalar
    "adds": Spec(0x10, "RRR", {}),
    "subs": Spec(0x11, "RRR", {}),
    "muls": Spec(0x12, "RRR", {}),
    "cmps": Spec(0x13, "SS", {}),
    "li": Spec(0x14, "RIMM", {}),
    "setcfg": Spec(0x15, "CFG", {}),
    # Register -> config. The immediate form freezes every macro-op geometry at
    # assembly time; this is what lets one vary at runtime (docs/macro_ops.md
    # §9.2 — decode's key count grows by one per step).
    "setcfgr": Spec(0x1C, "CFGR", {}),
    "loads": Spec(0x16, "RS", {}),  # dst = scratch[src0]
    "stores": Spec(0x17, "SS", {}),  # scratch[src0] = src1
    # control
    "branch": Spec(0x18, "BRANCH", {}),
    "jmp": Spec(0x19, "JMP", {}),
    "wait": Spec(0x1A, "WAIT", {}),
    "halt": Spec(0x1F, "NONE", {}),
}

# conditional-branch aliases: same opcode, condition baked into the flags field.
BRANCH_ALIASES = {f"b{c}": code for c, code in COND_CODES.items()}


@dataclass
class SrcInstr:
    lineno: int
    mnemonic: str  # base mnemonic, lower-case, flags stripped
    flags: int  # flags decoded from the '.suffix' part
    operands: list[str]  # raw operand tokens (still symbolic)
    address: int  # word address in instruction memory


class Assembler:
    def __init__(self):
        self.consts: dict[str, int] = {}  # .equ names + labels
        self.regs: dict[str, int] = {}  # .reg aliases -> index
        self.instrs: list[SrcInstr] = []

    # --- expression / operand resolution -------------------------------------
    # A small, safe integer-expression evaluator: numbers (dec/hex), previously
    # defined .equ constants and labels, the arithmetic/shift operators below,
    # and parentheses. Evaluated over a restricted `ast` — no general eval().
    _BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
    }

    def eval_expr(self, expr: str, lineno: int) -> int:
        expr = expr.strip()
        if not expr:
            raise AsmError(lineno, "empty expression")
        try:
            tree = ast.parse(expr, mode="eval").body
        except SyntaxError:
            raise AsmError(lineno, f"malformed expression '{expr}'")
        return self._eval_node(tree, expr, lineno)

    def _eval_node(self, node, expr: str, lineno: int) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.consts:
                return self.consts[node.id]
            raise AsmError(lineno, f"undefined symbol '{node.id}'")
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINOPS:
            lhs = self._eval_node(node.left, expr, lineno)
            rhs = self._eval_node(node.right, expr, lineno)
            return self._BINOPS[type(node.op)](lhs, rhs)
        if isinstance(node, ast.UnaryOp):
            val = self._eval_node(node.operand, expr, lineno)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.Invert):
                return ~val
        raise AsmError(lineno, f"unsupported expression '{expr}'")

    def parse_reg(self, tok: str, lineno: int) -> int:
        if tok in self.regs:
            return self.regs[tok]
        m = re.fullmatch(r"[rR](\d+)", tok)
        if not m:
            raise AsmError(lineno, f"expected a register, got '{tok}'")
        idx = int(m.group(1))
        if idx >= NUM_REGS:
            raise AsmError(lineno, f"register r{idx} out of range (0..{NUM_REGS - 1})")
        return idx

    # --- pass 1: scan lines, collect labels/directives, assign addresses -----
    def scan(self, text: str) -> None:
        addr = 0
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.split(";", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue

            # peel off any leading labels ("a: b: op ...")
            while True:
                m = re.match(r"([A-Za-z_.][A-Za-z0-9_.]*)\s*:\s*(.*)", line)
                if not m:
                    break
                label = m.group(1)
                if label in self.consts or label in self.regs:
                    raise AsmError(lineno, f"duplicate symbol '{label}'")
                self.consts[label] = addr
                line = m.group(2).strip()
            if not line:
                continue

            parts = line.split(None, 1)
            head = parts[0]
            rest = parts[1].strip() if len(parts) > 1 else ""

            if head.lower() in (".equ", ".reg"):
                self._directive(head.lower(), rest, lineno)
                continue

            operands = [o.strip() for o in rest.split(",")] if rest else []
            operands = [o for o in operands if o != ""]
            base, flags = self._split_flags(head.lower(), lineno)
            self.instrs.append(SrcInstr(lineno, base, flags, operands, addr))
            addr += 1

    def _directive(self, name: str, rest: str, lineno: int) -> None:
        toks = rest.split(None, 1)
        if len(toks) != 2:
            raise AsmError(lineno, f"{name} needs a name and a value")
        sym, val = toks[0], toks[1].strip()
        if sym in self.consts or sym in self.regs:
            raise AsmError(lineno, f"duplicate symbol '{sym}'")
        if name == ".reg":
            self.regs[sym] = self.parse_reg(val, lineno)
        else:  # .equ
            self.consts[sym] = self.eval_expr(val, lineno)

    def _split_flags(self, mnem: str, lineno: int) -> tuple[str, int]:
        base, *suffixes = mnem.split(".")
        if base in BRANCH_ALIASES:
            if suffixes:
                raise AsmError(lineno, f"'{base}' takes no flag suffix")
            return base, 0
        spec = SPECS.get(base)
        if spec is None:
            raise AsmError(lineno, f"unknown instruction '{mnem}'")
        flags = 0
        for suf in suffixes:
            if suf not in spec.flag_bits:
                raise AsmError(lineno, f"'{base}' has no flag '.{suf}'")
            flags |= spec.flag_bits[suf]
        return base, flags

    # --- pass 2: encode ------------------------------------------------------
    def encode_all(self) -> list[int]:
        return [self._encode(ins) for ins in self.instrs]

    def _need(self, ins: SrcInstr, n: int) -> None:
        if len(ins.operands) != n:
            raise AsmError(
                ins.lineno,
                f"'{ins.mnemonic}' expects {n} operand(s), got {len(ins.operands)}",
            )

    def _imm16(self, expr: str, lineno: int, signed_ok: bool) -> int:
        v = self.eval_expr(expr, lineno)
        lo = -32768 if signed_ok else 0
        if v < 0 and not signed_ok:
            # `li` and `setcfg` both zero-extend, so a negative immediate cannot
            # mean what it says. Fail rather than silently delivering v & 0xFFFF.
            raise AsmError(
                lineno,
                f"immediate {v} is negative, but this instruction zero-extends "
                f"(a register would receive {v & 0xFFFF}). For a negative "
                f"constant use `li rN, {abs(v)}` then `subs rN, r0, rN`"
            )
        if not (lo <= v <= 65535):
            raise AsmError(lineno, f"immediate {v} out of 16-bit range")
        return v & 0xFFFF

    def _encode(self, ins: SrcInstr) -> int:
        ln = ins.lineno
        ops = ins.operands
        reg = lambda t: self.parse_reg(t, ln)

        # branch condition aliases (beq/bne/blt/bge) -> real branch encoding
        if ins.mnemonic in BRANCH_ALIASES:
            self._need(ins, 1)
            target = self.eval_expr(ops[0], ln)
            return _pack_imm(
                SPECS["branch"].opcode, 0, target, BRANCH_ALIASES[ins.mnemonic]
            )

        spec = SPECS[ins.mnemonic]
        form = spec.form

        if form == "RRR":
            self._need(ins, 3)
            return _pack(spec.opcode, reg(ops[0]), reg(ops[1]), reg(ops[2]), ins.flags)
        if form == "RR":  # dst, src0  (src1 unused)
            self._need(ins, 2)
            return _pack(spec.opcode, reg(ops[0]), reg(ops[1]), 0, ins.flags)
        if form == "RS":  # dst, src0  (src1 unused; loads/rdmem)
            self._need(ins, 2)
            return _pack(spec.opcode, reg(ops[0]), reg(ops[1]), 0, ins.flags)
        if form == "SS":  # src0, src1 (dst unused; wrmem/stores/cmps)
            self._need(ins, 2)
            return _pack(spec.opcode, 0, reg(ops[0]), reg(ops[1]), ins.flags)
        if form == "RIMM":  # li rdst, imm16 (zero-extended by hw)
            self._need(ins, 2)
            return _pack_imm(
                spec.opcode, reg(ops[0]), self._imm16(ops[1], ln, signed_ok=False)
            )
        if form == "CFG":  # setcfg cfgname, imm16 (zero-extended by hw)
            self._need(ins, 2)
            return _pack_imm(
                spec.opcode,
                self._cfg(ops[0], ln),
                self._imm16(ops[1], ln, signed_ok=False),
            )
        if form == "CFGR":  # setcfgr cfgname, rN  (full XLEN, no extension)
            self._need(ins, 2)
            return _pack(spec.opcode, self._cfg(ops[0], ln), reg(ops[1]))
        if form == "BRANCH":  # branch cond, label
            self._need(ins, 2)
            cond = ops[0].lower()
            if cond not in COND_CODES:
                raise AsmError(
                    ln, f"unknown condition '{ops[0]}' (use {'/'.join(COND_CODES)})"
                )
            target = self.eval_expr(ops[1], ln)
            return _pack_imm(spec.opcode, 0, target, COND_CODES[cond])
        if form == "JMP":  # jmp label
            self._need(ins, 1)
            return _pack_imm(spec.opcode, 0, self.eval_expr(ops[0], ln))
        if form == "WAIT":  # wait unit
            self._need(ins, 1)
            unit = ops[0].lower()
            if unit not in UNIT_CODES:
                raise AsmError(
                    ln, f"unknown unit '{ops[0]}' (use {'/'.join(UNIT_CODES)})"
                )
            # unit selector carried in the flags field (scalar_unit.sv U_*).
            return _pack(spec.opcode, 0, 0, 0, UNIT_CODES[unit])
        if form == "NEIGH":  # wrneigh rmy, rnb, dir
            self._need(ins, 3)
            direction = DIR_CODES.get(ops[2].lower())
            if direction is None:
                direction = self.eval_expr(ops[2], ln)
            if not (0 <= direction <= 3):
                raise AsmError(ln, f"neighbor direction {direction} out of range 0..3")
            return _pack(spec.opcode, 0, reg(ops[0]), reg(ops[1]), direction)
        if form == "NONE":  # halt
            self._need(ins, 0)
            return _pack(spec.opcode)

        raise AsmError(ln, f"internal: unhandled form '{form}'")

    def _cfg(self, tok: str, lineno: int) -> int:
        key = tok.lower()
        if key in CFG_NAMES:
            return CFG_NAMES[key]
        m = re.fullmatch(r"cfg(\d+)", key)
        idx = int(m.group(1)) if m else self.eval_expr(tok, lineno)
        if not (0 <= idx < NUM_CFG):
            raise AsmError(
                lineno, f"config register {idx} out of range (0..{NUM_CFG - 1})"
            )
        return idx


def assemble(text: str) -> list[int]:
    """Assemble tpulang source into a list of 32-bit instruction words."""
    asm = Assembler()
    asm.scan(text)
    return asm.encode_all()


def assemble_verbose(text: str) -> tuple[list[int], list[SrcInstr]]:
    asm = Assembler()
    asm.scan(text)
    return asm.encode_all(), asm.instrs


# ------------------------------------------------------------------------------
# Output formatting.
# ------------------------------------------------------------------------------
def format_words(words: list[int], fmt: str) -> str:
    if fmt == "hex":  # $readmemh
        return "".join(f"{w & 0xFFFFFFFF:08x}\n" for w in words)
    if fmt == "bin":  # $readmemb
        return "".join(f"{w & 0xFFFFFFFF:032b}\n" for w in words)
    raise ValueError(f"unknown output format '{fmt}'")


def format_listing(words: list[int], instrs: list[SrcInstr]) -> str:
    lines = ["addr  word      flags  source"]
    for w, ins in zip(words, instrs):
        src = ins.mnemonic + (" " + ", ".join(ins.operands) if ins.operands else "")
        lines.append(
            f"{ins.address:04x}  {w & 0xFFFFFFFF:08x}  {ins.flags:02b}     {src}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble tpulang into TPU machine code.")
    ap.add_argument("source", help="tpulang (.tpu) source file")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument(
        "--format",
        choices=("hex", "bin"),
        default="hex",
        help="hex ($readmemh) or bin ($readmemb) text image",
    )
    ap.add_argument(
        "--listing",
        action="store_true",
        help="print an annotated addr/word/source listing to stderr",
    )
    args = ap.parse_args(argv)

    try:
        # Explicit UTF-8, not the locale's default — the example sources carry
        # math notation in their comments that cp1252 cannot decode.
        with open(args.source, encoding="utf-8") as f:
            text = f.read()
        words, instrs = assemble_verbose(text)
    except AsmError as e:
        print(f"{args.source}: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.listing:
        sys.stderr.write(format_listing(words, instrs))

    out = format_words(words, args.format)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
