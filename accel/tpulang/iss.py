#!/usr/bin/env python3
"""iss.py — instruction-set simulator for the TPU scalar ISA.

Executes the 32-bit machine words produced by ``assembler.py`` against a model
of the scratchpad plus the MXU/VPU compute units, matching the RTL in
``accel/tpu/rtl/*.sv`` under the issue-and-wait execution model: every dispatch
op is *atomic* — read its operands out of the scratchpad, compute, write the
result back — so no cycle-level modelling is needed to reproduce the final
memory state a real run leaves behind.

It exists to produce the golden result vectors that
``accel/tpu/tb/tpu_top_tb.sv`` checks the DUT against: load the same program and
input tensors, run this simulator, and the final scratchpad image is what the
hardware must reproduce byte-for-byte.

Numerics are kept bit-exact with the RTL:
  * MXU matmul: int8 activations x ternary weights, int32 accumulate, optional
    ``clip((acc*m0 + round) >> n)`` requantize on store (mxu.sv requant8).
  * VPU ops: int8 operands, int32 accumulate, int32 writeback; VOP_REQUANT
    narrows int32->int8 with the same fixed-point rescale (vpu.sv requant8).

DMA (rdmem/wrmem) and inter-TPU LINK (wrneigh) are modelled as no-ops, matching
``tpu_top.sv``, which ties their ``done`` high with no engine attached — a
program that issues them completes without moving any data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- opcode map (must match scalar_unit.sv OP_* and assembler.py SPECS) --------
OP_MATMUL, OP_VECDOT, OP_VECMUL, OP_VECADD = 0x00, 0x01, 0x02, 0x03
OP_RELU, OP_GELU, OP_WRMEM, OP_RDMEM = 0x04, 0x05, 0x06, 0x07
OP_WRNEIGH, OP_REQUANT, OP_VECEMUL, OP_SQUARE = 0x08, 0x09, 0x0A, 0x0B
OP_EXP, OP_REDMAX, OP_REDSUM, OP_SADD = 0x0C, 0x0D, 0x0E, 0x0F
OP_ADDS, OP_SUBS, OP_MULS, OP_CMPS = 0x10, 0x11, 0x12, 0x13
OP_LIS, OP_SETCFG, OP_LOADS, OP_STORES = 0x14, 0x15, 0x16, 0x17
OP_BRANCH, OP_JMP, OP_WAIT, OP_SDIV, OP_HALT = 0x18, 0x19, 0x1A, 0x1B, 0x1F

# --- named config registers (scalar_unit.sv CFG_*) ----------------------------
CFG_TLEN, CFG_VLEN, CFG_LEN, CFG_SCALAR = 0, 1, 2, 3

# --- branch condition codes (scalar_unit.sv C_*) ------------------------------
C_EQ, C_NE, C_LT, C_GE = 0b00, 0b01, 0b10, 0b11


def s8(b: int) -> int:
    """Interpret an 8-bit value as signed int8."""
    b &= 0xFF
    return b - 0x100 if b >= 0x80 else b


def s32(v: int) -> int:
    """Wrap to 32-bit two's complement (signed)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class ISSError(Exception):
    pass


@dataclass
class TPU:
    """A byte-addressed scratchpad + register/config file + ISA interpreter.

    Geometry mirrors the DUT parameters in ``tpu_top_tb.sv`` and must match the
    testbench that consumes the vectors this simulator produces.
    """

    rows: int = 8          # MXU contraction dim d (activation vector length)
    cols: int = 8          # MXU output features
    addr_w: int = 16       # scratchpad byte-address width (2**addr_w bytes)
    m0_w: int = 12         # requant fixed-point multiplier width
    n_w: int = 4           # requant shift width
    recip_q: int = 31      # VOP_SCALAR_DIV reciprocal exponent (vpu.sv)
    div_q: int = 15        # VOP_SCALAR_DIV quotient fractional bits (Q15)
    gelu_lut: list | None = None   # 256 x int8, indexed by unsigned input byte
    exp_lut: list | None = None    # 256 x int8

    mem: bytearray = field(init=False)
    regs: list = field(init=False)          # 32 x int32 (r0 hardwired 0)
    cfg: list = field(init=False)           # 16 x int32
    written: set = field(init=False)        # byte addrs a compute/store op wrote

    def __post_init__(self):
        self.depth = 1 << self.addr_w
        self.mem = bytearray(self.depth)
        self.regs = [0] * 32
        self.cfg = [0] * 16
        self.written = set()
        self.wcol_bytes = (self.rows * 2) // 8   # packed ternary column bytes

    # ---- scratchpad access (addresses wrap mod depth, like the RTL) ----------
    def _a(self, addr: int) -> int:
        return addr & (self.depth - 1)

    def rd_u32(self, addr: int) -> int:
        return sum(self.mem[self._a(addr + k)] << (8 * k) for k in range(4))

    def rd_i32(self, addr: int) -> int:
        return s32(self.rd_u32(addr))

    def rd_i8(self, addr: int) -> int:
        return s8(self.mem[self._a(addr)])

    def wr_i8(self, addr: int, val: int, *, track: bool = True) -> None:
        a = self._a(addr)
        self.mem[a] = val & 0xFF
        if track:
            self.written.add(a)

    def wr_i32(self, addr: int, val: int, *, track: bool = True) -> None:
        val &= 0xFFFFFFFF
        for k in range(4):
            a = self._a(addr + k)
            self.mem[a] = (val >> (8 * k)) & 0xFF
            if track:
                self.written.add(a)

    # ---- register file (r0 == 0) ---------------------------------------------
    def rreg(self, field8: int) -> int:
        idx = field8 & 0x1F
        return 0 if idx == 0 else self.regs[idx]

    def wreg(self, field8: int, val: int) -> None:
        idx = field8 & 0x1F
        if idx != 0:
            self.regs[idx] = s32(val)

    # ---- requant: clip((acc*m0 + round) >> n) to int8 (mxu.sv/vpu.sv) ---------
    def requant8(self, acc: int, m0: int, n: int) -> int:
        prod = acc * m0                         # m0 is a positive scale
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (prod + rnd) >> n             # Python >> floors == Verilog >>>
        if shifted > 127:
            return 127
        if shifted < -128:
            return -128
        return shifted

    # ---- ternary weight decode (scratchpad.md §2: 00=0, 01=+1, 11=-1) --------
    @staticmethod
    def _trit(colint: int, i: int) -> int:
        code = (colint >> (2 * i)) & 0b11
        if not (code & 0b01):        # bit0 = nonzero flag
            return 0
        return -1 if (code >> 1) & 1 else 1   # bit1 = sign

    # =========================================================================
    # Execution.
    # =========================================================================
    def run(self, words: list, boot_pc: int = 0, max_steps: int = 100_000) -> None:
        pc = boot_pc
        flag_eq = flag_lt = False
        addr_mask = self.depth - 1

        for _ in range(max_steps):
            if pc >= len(words):
                raise ISSError(f"pc={pc} ran off the end of the program (no HALT?)")
            w = words[pc] & 0xFFFFFFFF

            opc = (w >> 26) & 0x3F
            f_dst = (w >> 18) & 0xFF
            f_src0 = (w >> 10) & 0xFF
            f_src1 = (w >> 2) & 0xFF
            flags = w & 0x3
            imm16 = (w >> 2) & 0xFFFF

            r_src0 = self.rreg(f_src0)
            r_src1 = self.rreg(f_src1)
            r_dst = self.rreg(f_dst)

            next_pc = pc + 1

            if opc == OP_HALT:
                return

            elif opc == OP_MATMUL:
                self._matmul(r_dst & addr_mask, r_src0 & addr_mask,
                             r_src1 & addr_mask, accumulate=bool(flags & 0b01),
                             requant=bool(flags & 0b10))

            elif opc in (OP_VECDOT, OP_VECMUL, OP_VECADD, OP_RELU, OP_GELU,
                         OP_REQUANT, OP_VECEMUL, OP_SQUARE, OP_EXP, OP_REDMAX,
                         OP_REDSUM, OP_SADD, OP_SDIV):
                self._vpu(opc, r_dst & addr_mask, r_src0 & addr_mask,
                          r_src1 & addr_mask)

            elif opc in (OP_WRMEM, OP_RDMEM, OP_WRNEIGH):
                pass  # no DMA/LINK engine in tpu_top.sv — completing no-op

            elif opc == OP_ADDS:
                self.wreg(f_dst, r_src0 + r_src1)
            elif opc == OP_SUBS:
                self.wreg(f_dst, r_src0 - r_src1)
            elif opc == OP_MULS:
                self.wreg(f_dst, r_src0 * r_src1)
            elif opc == OP_LIS:
                self.wreg(f_dst, imm16 - 0x10000 if imm16 & 0x8000 else imm16)
            elif opc == OP_SETCFG:
                self.cfg[f_dst & 0xF] = imm16           # zero-extended
            elif opc == OP_LOADS:
                self.wreg(f_dst, self.rd_i32(r_src0 & addr_mask))
            elif opc == OP_STORES:
                self.wr_i32(r_src0 & addr_mask, r_src1)
            elif opc == OP_CMPS:
                flag_eq = (s32(r_src0) == s32(r_src1))
                flag_lt = (s32(r_src0) < s32(r_src1))
            elif opc == OP_BRANCH:
                taken = {C_EQ: flag_eq, C_NE: not flag_eq,
                         C_LT: flag_lt, C_GE: not flag_lt}[flags]
                if taken:
                    next_pc = imm16
            elif opc == OP_JMP:
                next_pc = imm16
            elif opc == OP_WAIT:
                pass  # issue-and-wait: dispatch already ran atomically above
            else:
                raise ISSError(f"pc={pc}: unknown opcode 0x{opc:02x} (word 0x{w:08x})")

            pc = next_pc

        raise ISSError(f"exceeded {max_steps} steps without HALT")

    # ---- MXU matmul (mxu.sv) -------------------------------------------------
    def _matmul(self, out_a: int, act_a: int, wgt_a: int, *,
                accumulate: bool, requant: bool) -> None:
        t_len = self.cfg[CFG_TLEN] & 0x3F
        if t_len == 0:
            return
        rq_m0 = rq_n = 0
        if requant:
            word = self.rd_u32(self.cfg[CFG_SCALAR] & (self.depth - 1))
            rq_m0 = word & ((1 << self.m0_w) - 1)
            rq_n = (word >> self.m0_w) & ((1 << self.n_w) - 1)

        # Pre-decode the ternary weight columns once (col-major, 2-bit packed).
        wcol = []
        for j in range(self.cols):
            base = wgt_a + j * self.wcol_bytes
            colint = sum(self.mem[self._a(base + b)] << (8 * b)
                         for b in range(self.wcol_bytes))
            wcol.append([self._trit(colint, i) for i in range(self.rows)])

        res_stride = self.cols * 4    # int32 result row bytes
        for t in range(t_len):
            arow = [self.rd_i8(act_a + t * self.rows + i) for i in range(self.rows)]
            for j in range(self.cols):
                acc = sum(arow[i] * wcol[j][i] for i in range(self.rows))
                if accumulate:      # readback stride is always the int32 stride
                    acc += self.rd_i32(out_a + t * res_stride + j * 4)
                if requant:
                    self.wr_i8(out_a + t * self.cols + j,
                               self.requant8(acc, rq_m0, rq_n))
                else:
                    self.wr_i32(out_a + t * res_stride + j * 4, acc)

    # ---- VPU (vpu.sv) --------------------------------------------------------
    def _vpu(self, opc: int, dst: int, src0: int, src1: int) -> None:
        vlen = self.cfg[CFG_VLEN] & 0x3FF
        if vlen == 0:
            return
        # src1 doubles as the scalar/param address for scalar & requant ops.
        scalar = self.rd_i32(src1)

        if opc == OP_REQUANT:
            m0 = scalar & ((1 << self.m0_w) - 1)
            n = (scalar >> self.m0_w) & ((1 << self.n_w) - 1)
            for i in range(vlen):
                a32 = self.rd_i32(src0 + i * 4)
                self.wr_i8(dst + i, self.requant8(a32, m0, n))
            return

        if opc in (OP_VECDOT, OP_REDSUM, OP_REDMAX):   # reductions -> one int32
            acc = None
            for i in range(vlen):
                a = self.rd_i8(src0 + i)
                if opc == OP_VECDOT:
                    v = a * self.rd_i8(src1 + i)
                elif opc == OP_REDSUM:
                    v = a
                else:  # REDMAX (over int32-widened int8 operands)
                    v = a
                if opc == OP_REDMAX:
                    acc = v if acc is None else max(acc, v)
                else:
                    acc = v if acc is None else acc + v
            self.wr_i32(dst, s32(acc if acc is not None else 0))
            return

        # elementwise: int8 source(s) -> int32 dst
        for i in range(vlen):
            a = self.rd_i8(src0 + i)
            if opc == OP_VECADD:
                r = a + self.rd_i8(src1 + i)
            elif opc == OP_VECEMUL:
                r = a * self.rd_i8(src1 + i)
            elif opc == OP_VECMUL:
                r = a * s32(scalar)
            elif opc == OP_SADD:
                r = a + s32(scalar)
            elif opc == OP_SDIV:
                r = self._sdiv(a, s32(scalar))
            elif opc == OP_RELU:
                r = a if a > 0 else 0
            elif opc == OP_SQUARE:
                r = a * a
            elif opc == OP_GELU:
                r = self._lut(self.gelu_lut, "gelu", src0 + i)
            elif opc == OP_EXP:
                r = self._lut(self.exp_lut, "exp", src0 + i)
            else:
                raise ISSError(f"unhandled VPU opcode 0x{opc:02x}")
            self.wr_i32(dst + i * 4, s32(r))

    def _lut(self, lut, name: str, addr: int) -> int:
        if lut is None:
            raise ISSError(f"{name} op needs a {name}_lut (256 x int8)")
        return s8(lut[self.mem[self._a(addr)]])   # indexed by unsigned byte

    def _sdiv(self, a8: int, d: int) -> int:
        """round(a8 * 2**DIV_Q / d), matching vpu.sv's reciprocal datapath."""
        if d == 0:
            R = (1 << (self.recip_q + 1)) - 1      # div-by-zero saturates R
        else:
            R = (1 << self.recip_q) // abs(d)      # floor(2**RECIP_Q / |d|)
        prod = a8 * R
        shift = self.recip_q - self.div_q
        q = prod if shift == 0 else (prod + (1 << (shift - 1))) >> shift
        return -q if d < 0 else q
