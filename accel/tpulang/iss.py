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
    narrows int32->int8 with the same fixed-point rescale (vpu.sv requant8),
    and VOP_DYT does the same with a symmetric +-127 clip (vpu.sv dyt8) — that
    clip is DyT's hardtanh, not an approximation of it.

DMA (rdmem/wrmem) is modelled as a real byte copy between a separate external
**DRAM** space and the on-chip scratchpad, over ``cfg 'len'`` bytes — matching the
``dma.sv``/``sram.sv`` engine. The two memories are **different sizes**: the
scratchpad is ``2**addr_w`` (16 bits) and DRAM is ``2**mem_addr_w`` (19 bits, the
Cmod A7's 512K x 8 SRAM), so each side of a transfer is masked in its own space
(``_a`` vs ``_d``). Sizing DRAM off ``addr_w`` — as this did until
``scalar_unit.sv`` stopped truncating its DRAM address — silently aliases
anything above 64 KB back into the low window.

The engine is the one wired into ``tpu_top.sv`` (and is how ``tpu_top_tb.sv``
seeds/reads DRAM). This lets a kernel stream tiles: fill a small
fixed scratchpad buffer from an advancing DRAM address, compute, spill the result
back to a distinct DRAM address — so the operands need not all fit in scratchpad.
Inputs live in DRAM, ``rdmem`` fills scratchpad, and ``wrmem`` is what makes a byte
host-visible again, so the golden outputs are exactly the DRAM bytes ``wrmem`` wrote
(tracked in ``dram_written``). ``rdmem.t``/``wrmem.t`` move the same bytes with a
transposed destination order (docs/dma.md §5). Inter-TPU LINK (wrneigh) is still a
no-op — no link engine is attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- opcode map (must match scalar_unit.sv OP_* and assembler.py SPECS) --------
# 0x02, 0x05, 0x0A-0x0F, 0x1B and 0x20 are retired holes: vecmul, gelu,
# vecemul, square, exp, redmax, redsum, sadd, sdiv and softmax went with the VPU
# datapath that implemented them (rtl/vpu.sv header). They are not reused, so an
# old binary decodes to an unknown opcode rather than to a different
# instruction. `vecdot` (0x01) stayed: it is vecmatmul's inner primitive, so its
# datapath is not optional.
OP_MATMUL, OP_VECDOT, OP_VECADD = 0x00, 0x01, 0x03
OP_RELU, OP_WRMEM, OP_RDMEM = 0x04, 0x06, 0x07
OP_WRNEIGH, OP_REQUANT = 0x08, 0x09
OP_ADDS, OP_SUBS, OP_MULS, OP_CMPS = 0x10, 0x11, 0x12, 0x13
OP_LIS, OP_SETCFG, OP_LOADS, OP_STORES = 0x14, 0x15, 0x16, 0x17
OP_BRANCH, OP_JMP, OP_WAIT, OP_HALT = 0x18, 0x19, 0x1A, 0x1F
OP_SETCFGR = 0x1C
OP_MATMUL_T = 0x1D
OP_VECMM = 0x1E
OP_DYT = 0x21

# --- named config registers (scalar_unit.sv CFG_*) ----------------------------
CFG_TLEN, CFG_VLEN, CFG_LEN, CFG_SCALAR = 0, 1, 2, 3
# MXU hardware-tiling geometry (docs/macro_ops.md §3)
CFG_KTILES, CFG_NTILES, CFG_AROW, CFG_CROW, CFG_WCOL = 4, 5, 6, 7, 8
# VPU macro-op geometry
# CFG_VSCALAR (9) is retired with OP_SOFTMAX, the only op that read it; the
# index is kept so 10..17 do not shift under every existing program.
CFG_VROWS, CFG_VCOLS, CFG_VROW0, CFG_VROW1 = 10, 11, 12, 13
CFG_VCROW = 14   # vecmatmul dst row stride (the MXU owns CFG_CROW)
# DMA transpose geometry (docs/dma.md §5); read only by rdmem.t / wrmem.t.
CFG_TCOLS, CFG_TSROW, CFG_TDROW = 15, 16, 17

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
    # External DRAM is a *wider, separate* space: the async SRAM chip is 2**19
    # bytes (tpu_top.sv MEM_ADDR_W). This used to be sized off addr_w, which
    # modelled the scalar unit's old ADDR_W-truncated rdmem/wrmem address and so
    # made a 64 KB program window look like a hardware ceiling. scalar_unit.sv
    # now emits the full width; the two spaces are masked separately below.
    mem_addr_w: int = 19   # external DRAM byte-address width (2**mem_addr_w)
    m0_w: int = 12         # requant fixed-point multiplier width
    n_w: int = 4           # requant shift width

    mem: bytearray = field(init=False)      # on-chip scratchpad
    dram: bytearray = field(init=False)     # external DRAM (DMA source/destination)
    regs: list = field(init=False)          # 32 x int32 (r0 hardwired 0)
    cfg: list = field(init=False)           # 32 x int32 (CFG_AW = 5)
    written: set = field(init=False)        # scratchpad byte addrs a compute/store wrote
    dram_written: set = field(init=False)   # DRAM byte addrs a wrmem spilled (host output)

    def __post_init__(self):
        self.depth = 1 << self.addr_w
        self.dram_depth = 1 << self.mem_addr_w
        self.mem = bytearray(self.depth)
        self.dram = bytearray(self.dram_depth)
        self.regs = [0] * 32
        self.cfg = [0] * 32
        self.written = set()
        self.dram_written = set()
        self.wcol_bytes = (self.rows * 2) // 8   # packed ternary column bytes

    # ---- scratchpad access (addresses wrap mod depth, like the RTL) ----------
    def _a(self, addr: int) -> int:
        return addr & (self.depth - 1)

    def _d(self, addr: int) -> int:
        """Mask a **DRAM** byte address (mod 2**mem_addr_w).

        Distinct from :meth:`_a` because the two memories are different sizes.
        Every DRAM-side address goes through here and every scratchpad-side one
        through ``_a``; mixing them is the one way this model can silently
        disagree with the RTL, since a DRAM address masked to 16 bits aliases
        back into the low window instead of reaching the rest of the chip.
        """
        return addr & (self.dram_depth - 1)

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

    # ---- DMA: byte copy DRAM<->scratchpad over cfg 'len' (dma.sv/sram.sv) -----
    def _dma(self, *, dst: int, src: int, to_scratch: bool,
             transpose: bool = False) -> None:
        """Copy ``cfg['len']`` bytes. Fill (to_scratch) reads DRAM into the
        scratchpad; spill writes the scratchpad back to DRAM and records the
        touched DRAM bytes as host-visible output.

        With ``transpose`` (the ``.t`` instruction flag) the byte *count* is
        unchanged but the two address generators run in opposite orders: the
        source is read row-major over ``cfg tcols`` columns at ``cfg tsrow``
        stride, and the destination is written transposed at ``cfg tdrow``.
        Mirrors dma.sv's counters exactly, including the zero-means-unset
        fallbacks and the 16-bit wrap of both running offsets — those are the
        parts the RTL and this model have to agree on byte-for-byte.
        """
        n = self.cfg[CFG_LEN] & 0xFFFF

        if not transpose:
            for k in range(n):
                if to_scratch:              # fill: DRAM -> scratchpad
                    self.mem[self._a(dst + k)] = self.dram[self._d(src + k)]
                else:                       # spill: scratchpad -> DRAM
                    a = self._d(dst + k)
                    self.dram[a] = self.mem[self._a(src + k)]
                    self.dram_written.add(a)
            return

        cols = (self.cfg[CFG_TCOLS] & 0xFFFF) or n
        srow = (self.cfg[CFG_TSROW] & 0xFFFF) or cols
        drow = (self.cfg[CFG_TDROW] & 0xFFFF) or 1

        # The source always gets the row-major offset and the destination the
        # transposed one (dma.sv's lin_off/tr_off); which *memory* that is
        # follows from the direction, so each side is masked in its own space.
        col = row = src_row_off = dst_col_off = 0
        for _ in range(n):
            if to_scratch:                  # fill: source is DRAM
                s = self._d(src + src_row_off + col)
                d = self._a(dst + dst_col_off + row)
                self.mem[d] = self.dram[s]
            else:                           # spill: source is the scratchpad
                s = self._a(src + src_row_off + col)
                d = self._d(dst + dst_col_off + row)
                self.dram[d] = self.mem[s]
                self.dram_written.add(d)
            # Only the byte count ends the transfer, so a `len` that is not a
            # whole number of rows stops part-way through the last one.
            if col + 1 >= cols:
                col = 0
                row += 1
                src_row_off = (src_row_off + srow) & 0xFFFF
                dst_col_off = 0
            else:
                col += 1
                dst_col_off = (dst_col_off + drow) & 0xFFFF

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

    # ---- DyT: the same rescale, clipped symmetrically (vpu.sv dyt8) ----------
    def dyt8(self, acc: int, m0: int, n: int) -> int:
        """``clip_pm127((acc*m0 + round) >> n)`` — DyT / hardtanh.

        Identical to :meth:`requant8` except for the lower bound. ``hardtanh``
        is an odd function, so its floor has to be the negative of its ceiling;
        int8's -128 would put the saturated end at -128/127 = -1.0079. The
        multiplier carries ``alpha * s_in * 127``, which is what makes the clip
        coincide with the hardtanh rather than merely resemble it.
        """
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (acc * m0 + rnd) >> n         # Python >> floors, like >>>
        if shifted > 127:
            return 127
        if shifted < -127:
            return -127
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
        addr_mask = self.depth - 1              # scratchpad: ADDR_W bits
        dram_mask = self.dram_depth - 1         # DRAM: MEM_ADDR_W bits

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

            elif opc in (OP_MATMUL, OP_MATMUL_T):
                self._matmul(r_dst & addr_mask, r_src0 & addr_mask,
                             r_src1 & addr_mask, accumulate=bool(flags & 0b01),
                             requant=bool(flags & 0b10),
                             tiled=(opc == OP_MATMUL_T))

            elif opc == OP_VECMM:
                self._vecmatmul(r_dst & addr_mask, r_src0 & addr_mask,
                                r_src1 & addr_mask)

            elif opc in (OP_VECDOT, OP_VECADD, OP_RELU, OP_REQUANT, OP_DYT):
                self._vpu(opc, r_dst & addr_mask, r_src0 & addr_mask,
                          r_src1 & addr_mask)

            # Each operand is truncated in its own space, matching
            # scalar_unit.sv's split of dma_scratch_addr (ADDR_W) from
            # dma_dram_addr (MEM_ADDR_W).
            elif opc == OP_RDMEM:    # fill: DRAM(src0) -> scratchpad(dst)
                self._dma(dst=r_dst & addr_mask, src=r_src0 & dram_mask,
                          to_scratch=True, transpose=bool(flags & 0b01))
            elif opc == OP_WRMEM:    # spill: scratchpad(src0) -> DRAM(src1)
                self._dma(dst=r_src1 & dram_mask, src=r_src0 & addr_mask,
                          to_scratch=False, transpose=bool(flags & 0b01))
            elif opc == OP_WRNEIGH:
                pass  # no LINK engine attached in tpu_top.sv — completing no-op

            elif opc == OP_ADDS:
                self.wreg(f_dst, r_src0 + r_src1)
            elif opc == OP_SUBS:
                self.wreg(f_dst, r_src0 - r_src1)
            elif opc == OP_MULS:
                self.wreg(f_dst, r_src0 * r_src1)
            elif opc == OP_LIS:
                # Zero-extended (scalar_unit.sv OP_LIS). Sign extension made
                # `li r, 0x8000` alias to 0x78000 once DRAM addresses stopped
                # being truncated to 16 bits; see that file for the argument.
                self.wreg(f_dst, imm16)
            elif opc == OP_SETCFG:
                self.cfg[f_dst & 0x1F] = imm16          # zero-extended
            elif opc == OP_SETCFGR:
                # Register -> config, full 32 bits with no extension. Consumers
                # mask to their own field width, matching scalar_unit.sv.
                self.cfg[f_dst & 0x1F] = r_src0 & 0xFFFFFFFF
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
                accumulate: bool, requant: bool, tiled: bool = False) -> None:
        t_len = self.cfg[CFG_TLEN] & 0x3F
        if t_len == 0:
            return
        rq_m0 = rq_n = 0
        if requant:
            word = self.rd_u32(self.cfg[CFG_SCALAR] & (self.depth - 1))
            rq_m0 = word & ((1 << self.m0_w) - 1)
            rq_n = (word >> self.m0_w) & ((1 << self.n_w) - 1)

        # Operand strides. A plain `matmul` (tiled=False) ignores the config
        # registers entirely and uses the single-tile constants, so it cannot
        # inherit a stride an earlier program left behind — config survives
        # across runs. `matmul_t` reads them, falling back to the same constants
        # if one is unset. Matches how mxu.sv resolves them at dispatch.
        # `c_row` is the int32 row stride; a requantized store writes int8 and
        # therefore steps c_row/4, so `.acc.rq` uses both in one op.
        a_row = (tiled and (self.cfg[CFG_AROW] & 0xFFFF)) or self.rows
        c_row = (tiled and (self.cfg[CFG_CROW] & 0xFFFF)) or (self.cols * 4)
        w_col = (tiled and (self.cfg[CFG_WCOL] & 0xFFFF)) or self.wcol_bytes
        c_row_rq = c_row >> 2

        # Tile counts. Zero reads as 1, matching mxu.sv, so an unset config
        # register is a plain single-tile matmul rather than a no-op.
        kt = (tiled and (self.cfg[CFG_KTILES] & 0xFF)) or 1
        nt = (tiled and (self.cfg[CFG_NTILES] & 0xFF)) or 1

        # n outer, k inner — the hardware's loop order. The order is not
        # observable in the result (int32 accumulation is associative, including
        # under two's-complement wraparound), but matching it keeps this readable
        # against mxu.sv. Partials live in `res` across the k loop exactly as
        # they live in the hardware's resbuf, so the scratchpad is untouched
        # until the output tile is finished.
        for n in range(nt):
            out_n = out_a + n * self.cols * (1 if requant else 4)
            res = [[0] * self.cols for _ in range(t_len)]

            for k in range(kt):
                act_k = act_a + k * self.rows
                wgt_k = wgt_a + n * self.cols * w_col + k * self.wcol_bytes

                # Ternary weight columns for this tile (col-major, 2-bit packed).
                # Only wcol_bytes of each column are consumed — one tile's worth
                # of rows — even when w_col strides past a taller column.
                wcol = []
                for j in range(self.cols):
                    base = wgt_k + j * w_col
                    colint = sum(self.mem[self._a(base + b)] << (8 * b)
                                 for b in range(self.wcol_bytes))
                    wcol.append([self._trit(colint, i) for i in range(self.rows)])

                for t in range(t_len):
                    arow = [self.rd_i8(act_k + t * a_row + i)
                            for i in range(self.rows)]
                    for j in range(self.cols):
                        res[t][j] += sum(arow[i] * wcol[j][i]
                                         for i in range(self.rows))

            for t in range(t_len):
                for j in range(self.cols):
                    acc = res[t][j]
                    if accumulate:  # readback stride is always the int32 stride
                        acc += self.rd_i32(out_n + t * c_row + j * 4)
                    if requant:
                        self.wr_i8(out_n + t * c_row_rq + j,
                                   self.requant8(acc, rq_m0, rq_n))
                    else:
                        self.wr_i32(out_n + t * c_row + j * 4, acc)

    # ---- VPU macro op: vecmatmul (vpu.sv VOP_VECMATMUL) ----------------------
    def _vecmatmul(self, dst: int, src0: int, src1: int) -> None:
        """S[t][s] = sum_d src0[t][d] * src1[s][d], int8 operands, int32 result.

        The hardware runs this as one VOP_DOT per (t, s) pair, so the numerics
        are exactly the dot product's: int8 reads, int32 accumulate, one int32
        store. Modelled the same way rather than as a matrix product, because
        that equivalence *is* the contract.

        `src1` is read row-major and contracted over its own rows, so the
        transpose in Q@K^T is implicit in the loop order — K is never
        materialized transposed.
        """
        rows = (self.cfg[CFG_VROWS] & 0xFFFF) or 1
        cols = (self.cfg[CFG_VCOLS] & 0xFFFF) or 1
        vlen = self.cfg[CFG_VLEN] & 0x3FF
        row0 = self.cfg[CFG_VROW0] & 0xFFFF
        row1 = self.cfg[CFG_VROW1] & 0xFFFF
        crow = self.cfg[CFG_VCROW] & 0xFFFF
        if vlen == 0:
            return
        for t in range(rows):
            for s in range(cols):
                a = src0 + t * row0
                b = src1 + s * row1
                acc = sum(self.rd_i8(a + d) * self.rd_i8(b + d)
                          for d in range(vlen))
                self.wr_i32(dst + t * crow + s * 4, s32(acc))

    # ---- VPU (vpu.sv) --------------------------------------------------------
    def _vpu(self, opc: int, dst: int, src0: int, src1: int) -> None:
        vlen = self.cfg[CFG_VLEN] & 0x3FF
        if vlen == 0:
            return
        # src1 doubles as the {m0,n} word address for the two narrowing ops.
        scalar = self.rd_i32(src1)

        # The two narrowing ops: int32 in at stride 4, int8 out at stride 1,
        # {m0,n} from the third operand. They differ only in the clip.
        if opc in (OP_REQUANT, OP_DYT):
            m0 = scalar & ((1 << self.m0_w) - 1)
            n = (scalar >> self.m0_w) & ((1 << self.n_w) - 1)
            narrow = self.requant8 if opc == OP_REQUANT else self.dyt8
            for i in range(vlen):
                a32 = self.rd_i32(src0 + i * 4)
                self.wr_i8(dst + i, narrow(a32, m0, n))
            return

        # The one reduction: whole vector -> a single int32 at `dst`.
        if opc == OP_VECDOT:
            acc = 0
            for i in range(vlen):
                acc += self.rd_i8(src0 + i) * self.rd_i8(src1 + i)
            self.wr_i32(dst, s32(acc))
            return

        # elementwise: int8 source(s) -> int32 dst
        for i in range(vlen):
            a = self.rd_i8(src0 + i)
            if opc == OP_VECADD:
                r = a + self.rd_i8(src1 + i)
            elif opc == OP_RELU:
                r = a if a > 0 else 0
            else:
                raise ISSError(f"unhandled VPU opcode 0x{opc:02x}")
            self.wr_i32(dst + i * 4, s32(r))

