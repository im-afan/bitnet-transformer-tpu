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
  * MXU matmul: int4 activations (in int8 containers) x **row-major int4**
    weights, int32 accumulate, optional ``clip((acc*m0 + round) >> n)``
    requantize on store (mxu.sv requant8, clipping to [-8, 7]).
  * VPU ops: int8 operands, int32 accumulate, int32 writeback; VOP_REQUANT
    narrows int32->int4 with the same fixed-point rescale (vpu.sv requant8),
    and VOP_DYT does the same with a symmetric +-7 clip (vpu.sv dyt8) — that
    clip is DyT's hardtanh, not an approximation of it. VOP_QUANT4 is the same
    rescale written as a 4-bit two's-complement nibble, two elements per byte
    (vpu.sv quant4), which is how an activation becomes a weight operand
    without a round trip through the host.

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
# The VPU op selectors the op bodies branch on. These used to be tpulang opcodes
# decoded from a 32-bit instruction word; with that front end gone they are just
# an internal enum, and `TPU._VOP_TO_OP` maps the hardware's VOP_* onto them.
OP_VECDOT, OP_VECADD = 0x01, 0x03
OP_RELU, OP_REQUANT = 0x04, 0x09
OP_VECMM = 0x1E
OP_DYT = 0x21
OP_QUANT4 = 0x22

CFG_TLEN, CFG_VLEN, CFG_LEN, CFG_SCALAR = 0, 1, 2, 3
# MXU hardware-tiling geometry (docs/macro_ops.md §3)
CFG_KTILES, CFG_NTILES, CFG_AROW, CFG_CROW, CFG_WROW = 4, 5, 6, 7, 8
# VPU macro-op geometry
# CFG_VSCALAR (9) is retired with OP_SOFTMAX, the only op that read it; the
# index is kept so 10..17 do not shift under every existing program.
CFG_VROWS, CFG_VCOLS, CFG_VROW0, CFG_VROW1 = 10, 11, 12, 13
CFG_VCROW = 14   # vecmatmul dst row stride (the MXU owns CFG_CROW)
# DMA transpose geometry (docs/dma.md §5); read only by rdmem.t / wrmem.t.
CFG_TCOLS, CFG_TSROW, CFG_TDROW = 15, 16, 17

# --- branch condition codes (scalar_unit.sv C_*) ------------------------------
C_EQ, C_NE, C_LT, C_GE = 0b00, 0b01, 0b10, 0b11


# The int4 grid: mxu.sv ACT_QMIN/ACT_QMAX, vpu.sv Q4_MIN/Q4_MAX, and
# model/transformer.py INT4_QMIN/INT4_QMAX are all this pair.
Q4_MIN, Q4_MAX = -8, 7


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
    cfg: list = field(init=False)           # 32 x int32 (CFG_AW = 5)
    written: set = field(init=False)        # scratchpad byte addrs a compute/store wrote
    dram_written: set = field(init=False)   # DRAM byte addrs a wrmem spilled (host output)

    def __post_init__(self):
        self.depth = 1 << self.addr_w
        self.dram_depth = 1 << self.mem_addr_w
        self.mem = bytearray(self.depth)
        self.dram = bytearray(self.dram_depth)
        self.cfg = [0] * 32
        self.written = set()
        self.dram_written = set()
        # Weights are row-major int4: one array row is COLS nibbles.
        self.wrow_bytes = (self.cols * 4) // 8   # packed int4 weight-row bytes

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

    # ---- requant: clip((acc*m0 + round) >> n) to int8 (mxu.sv/vpu.sv) ---------
    def requant8(self, acc: int, m0: int, n: int) -> int:
        prod = acc * m0                         # m0 is a positive scale
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (prod + rnd) >> n             # Python >> floors == Verilog >>>
        if shifted > Q4_MAX:
            return Q4_MAX
        if shifted < Q4_MIN:
            return Q4_MIN
        return shifted

    # ---- DyT: the same rescale, clipped symmetrically (vpu.sv dyt8) ----------
    def dyt8(self, acc: int, m0: int, n: int) -> int:
        """``clip_pm7((acc*m0 + round) >> n)`` — DyT / hardtanh.

        Identical to :meth:`requant8` except for the lower bound. ``hardtanh``
        is an odd function, so its floor has to be the negative of its ceiling;
        int4's -8 would put the saturated end at -8/7 = -1.143. The multiplier
        carries ``alpha * s_in * 7``, which is what makes the clip coincide with
        the hardtanh rather than merely resemble it.
        """
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (acc * m0 + rnd) >> n         # Python >> floors, like >>>
        if shifted > Q4_MAX:
            return Q4_MAX
        if shifted < -Q4_MAX:
            return -Q4_MAX
        return shifted

    # ---- quant4: the same rescale, clipped to int4 (vpu.sv quant4) ----------
    def quant4(self, acc: int, m0: int, n: int) -> int:
        """``clip_[-8,7]((acc*m0 + round) >> n)`` — int8 in, one int4 out.

        Identical to :meth:`requant8` bar nothing at all now that requant also
        lands int4: the two differ only in *destination width*, which is the
        caller's business. The caller packs the result with :meth:`_nib_code`;
        the two together are one VOP_QUANT4 element.
        """
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (acc * m0 + rnd) >> n         # Python >> floors, like >>>
        if shifted > Q4_MAX:
            return Q4_MAX
        if shifted < Q4_MIN:
            return Q4_MIN
        return shifted

    @staticmethod
    def _nib_code(v: int) -> int:
        """An int4 value as the 4-bit two's-complement nibble the MXU reads."""
        return v & 0xF

    # ---- int4 weight decode (scratchpad.md §2: 4-bit two's complement) -------
    @staticmethod
    def _nib(rowint: int, j: int) -> int:
        code = (rowint >> (4 * j)) & 0xF
        return code - 16 if code >= 8 else code

    # =========================================================================
    # Execution.
    # =========================================================================
    # ---- MXU matmul (mxu.sv) -------------------------------------------------
    def _matmul(self, out_a: int, act_a: int, wgt_a: int, *,
                accumulate: bool, requant: bool, tiled: bool = False,
                rq_word: int = None) -> None:
        t_len = self.cfg[CFG_TLEN] & 0x3F
        if t_len == 0:
            return
        rq_m0 = rq_n = 0
        if requant:
            # `rq_word` is the literal the *command* carries. The tpulang path
            # passes None because its ISA names a scratchpad address instead,
            # and the scalar unit reads it over the S port before packing the
            # command (scalar_unit.sv's S_RQRD shim). Both arrive here as the
            # same 16 bits; only who dereferenced them differs.
            word = (rq_word if rq_word is not None
                    else self.rd_u32(self.cfg[CFG_SCALAR] & (self.depth - 1)))
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
        w_row = (tiled and (self.cfg[CFG_WROW] & 0xFFFF)) or self.wrow_bytes
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
                # Row-major weights: an n-tile steps *along* a row by one array
                # width of nibbles (a constant), a k-tile steps *down* ROWS whole
                # rows (which is what needs the stride). Column-major had the two
                # the other way round — mxu.sv wgt_ntile_step / wgt_ktile_step.
                wgt_k = wgt_a + n * self.wrow_bytes + k * self.rows * w_row

                # int4 weight rows for this tile (row-major, 4-bit packed). Only
                # wrow_bytes of each row are consumed — one tile's worth of
                # columns — even when w_row strides past a wider row.
                wrow = []
                for i in range(self.rows):
                    base = wgt_k + i * w_row
                    rowint = sum(self.mem[self._a(base + b)] << (8 * b)
                                 for b in range(self.wrow_bytes))
                    wrow.append([self._nib(rowint, j) for j in range(self.cols)])

                for t in range(t_len):
                    arow = [self.rd_i8(act_k + t * a_row + i)
                            for i in range(self.rows)]
                    for j in range(self.cols):
                        res[t][j] += sum(arow[i] * wrow[i][j]
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
    def _vpu(self, opc: int, dst: int, src0: int, src1: int,
             rq_word: int = None) -> None:
        vlen = self.cfg[CFG_VLEN] & 0x3FF
        if vlen == 0:
            return
        # src1 doubles as the {m0,n} word address for the narrowing ops — in the
        # tpulang ISA. A command carries the literal instead (see _matmul).
        scalar = self.rd_i32(src1) if rq_word is None else rq_word

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

        # The int4 narrow: int8 in at stride 1, **4 bits** out, so one output
        # byte per two input elements. The packing order is the weight encoding
        # `_nib` decodes, which is what makes the result a legal `matmul_t`
        # weight operand with no rearrangement in between. A vlen that is not a
        # multiple of 2 leaves the unfilled slot of the last byte at 0 rather
        # than preserving it — the RTL's write strobe is per byte, so there is
        # nothing finer to preserve with.
        if opc == OP_QUANT4:
            m0 = scalar & ((1 << self.m0_w) - 1)
            n = (scalar >> self.m0_w) & ((1 << self.n_w) - 1)
            for b in range((vlen + 1) // 2):
                byte = 0
                for j in range(2):
                    i = 2 * b + j
                    if i >= vlen:
                        break
                    v = self.quant4(self.rd_i8(src0 + i), m0, n)
                    byte |= self._nib_code(v) << (4 * j)
                self.wr_i8(dst + b, byte)
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


    # =========================================================================
    # Command front end (docs/picorv32_migration.md §8, phase 3).
    #
    # The second way into this model. `run()` above decodes 32-bit tpulang
    # instructions; this decodes the 128-bit macro-ops that both producers push,
    # exactly as cmd_{mxu,vpu,dma}.sv decode them, and calls the *same* op bodies
    # underneath. That is the whole of the re-fronting: the numerics are not
    # duplicated, forked or re-derived, so a command trace and a .tpu program
    # that mean the same thing produce byte-identical images by construction.
    #
    # Sticky geometry (MXU_GEOM / VPU_GEOM) lands in the `cfg` file because that
    # is where the op bodies already read it from. It is not the old global
    # config file coming back: nothing else writes these, and a command stream
    # carries its geometry in its own order, which is the property cmd_mxu.sv's
    # header is about.
    # =========================================================================

    U_MXU, U_VPU, U_DMA = 0, 1, 2

    # Command opcodes, from cmd_{mxu,vpu,dma}.sv.
    MXU_GEOM, MXU_MM = 0x01, 0x02
    VPU_CMD_OP, VPU_CMD_GEOM = 0x01, 0x02
    DMA_MOVE = 0x01

    # vpu.sv VOP_* -> the tpulang opcode `_vpu` dispatches on. Two encodings for
    # one op set is a wart of keeping both producers alive through phase 4; the
    # map is here rather than in the caller so there is exactly one of it.
    _VOP_TO_OP = {0: OP_VECDOT, 1: OP_VECADD, 3: OP_RELU, 10: OP_REQUANT,
                  13: OP_VECMM, 16: OP_DYT, 17: OP_QUANT4}

    def exec_command(self, unit: int, w0: int, w1: int, w2: int, w3: int) -> None:
        """Execute one 128-bit macro-op. Mirrors the RTL decoders field for field.

        An unknown opcode is *discarded*, not an error — cmd_{mxu,vpu,dma}.sv
        pop it with a `$display` and carry on, and a model that raised instead
        would disagree with the hardware about a malformed stream.
        """
        op = w0 & 0xFF

        if unit == self.U_MXU:
            if op == self.MXU_GEOM:
                self.cfg[CFG_AROW] = (w0 >> 16) & 0xFFFF
                self.cfg[CFG_CROW] = w1 & 0xFFFF
                self.cfg[CFG_WROW] = (w1 >> 16) & 0xFFFF
                self.cfg[CFG_KTILES] = w2 & 0xFF
                self.cfg[CFG_NTILES] = (w2 >> 8) & 0xFF
                self.cfg[CFG_TLEN] = (w2 >> 16) & 0x3F
            elif op == self.MXU_MM:
                self._matmul((w0 >> 16) & 0xFFFF, w1 & 0xFFFF, (w1 >> 16) & 0xFFFF,
                             accumulate=bool(w0 & (1 << 8)),
                             requant=bool(w0 & (1 << 9)),
                             tiled=bool(w0 & (1 << 10)),
                             rq_word=w2 & ((1 << (self.m0_w + self.n_w)) - 1))

        elif unit == self.U_VPU:
            if op == self.VPU_CMD_GEOM:
                self.cfg[CFG_VROWS] = (w0 >> 16) & 0xFFFF
                self.cfg[CFG_VCOLS] = w1 & 0xFFFF
                self.cfg[CFG_VROW0] = (w1 >> 16) & 0xFFFF
                self.cfg[CFG_VROW1] = w2 & 0xFFFF
                self.cfg[CFG_VCROW] = (w2 >> 16) & 0xFFFF
            elif op == self.VPU_CMD_OP:
                vop = (w0 >> 8) & 0x1F
                if vop not in self._VOP_TO_OP:
                    return
                opc = self._VOP_TO_OP[vop]
                dst, src0 = (w0 >> 16) & 0xFFFF, w1 & 0xFFFF
                src1 = (w1 >> 16) & 0xFFFF
                self.cfg[CFG_VLEN] = w2 & 0x3FF
                rq = (w2 >> 16) & ((1 << (self.m0_w + self.n_w)) - 1)
                if opc == OP_VECMM:
                    self._vecmatmul(dst, src0, src1)
                else:
                    self._vpu(opc, dst, src0, src1, rq_word=rq)

        elif unit == self.U_DMA:
            if op == self.DMA_MOVE:
                spad = (w0 >> 16) & 0xFFFF
                write = bool(w0 & (1 << 8))       # 1 = spill (scratchpad -> DRAM)
                transpose = bool(w0 & (1 << 9))
                dram = w1 & (self.dram_depth - 1)
                self.cfg[CFG_LEN] = w2 & 0xFFFF
                self.cfg[CFG_TCOLS] = (w2 >> 16) & 0xFFFF
                self.cfg[CFG_TSROW] = w3 & 0xFFFF
                self.cfg[CFG_TDROW] = (w3 >> 16) & 0xFFFF
                if write:
                    self._dma(dst=dram, src=spad, to_scratch=False,
                              transpose=transpose)
                else:
                    self._dma(dst=spad, src=dram, to_scratch=True,
                              transpose=transpose)

    def run_trace(self, records) -> list:
        """Execute a producer's command trace; return the commands, in order.

        `records` is the parsed output of a `-DTPU_TRACE` firmware build (see
        `fw/mock/tpu_trace.c`): ``("CMD", unit, w0, w1, w2, w3)`` and
        ``("WAIT", unit)``.

        WAIT is a no-op here, and deliberately so. This model has no
        concurrency: it retires every command completely before looking at the
        next, which is the strongest ordering any barrier placement can produce.
        So a firmware that *omits* a needed cross-unit barrier still gets correct
        golden images out of this — and then diverges on the RTL, where the three
        queues really are independent. Keeping the images a statement about
        intent, and letting the RTL run be the thing that tests ordering, is what
        makes a mismatch mean something specific.
        """
        cmds = []
        for rec in records:
            if rec[0] != "CMD":
                continue
            _, unit, w0, w1, w2, w3 = rec
            self.exec_command(unit, w0, w1, w2, w3)
            cmds.append((unit, w0, w1, w2, w3))
        return cmds


def parse_trace(text: str) -> list:
    """Parse `fw/mock/tpu_trace.c` output into records for :meth:`TPU.run_trace`."""
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if f[0] == "CMD" and len(f) == 6:
            out.append(("CMD", int(f[1], 16) if f[1].startswith("0x") else int(f[1]),
                        *(int(x, 16) for x in f[2:6])))
        elif f[0] == "WAIT" and len(f) == 2:
            out.append(("WAIT", int(f[1])))
        else:
            raise ISSError(f"trace line {lineno}: cannot parse {line!r}")
    return out
