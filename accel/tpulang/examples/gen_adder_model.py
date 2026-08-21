#!/usr/bin/env python3
"""gen_adder_model.py — build `adder_model.tpu` with pytpu instead of by hand.

The whole `adder_ternary_vanilla` forward pass — four transformer layers and the
ternary output head — emitted as one tpulang program.  The hand-written
[`adder_model.tpu`](adder_model.tpu) is the reference; this generator is
written against the *same* memory maps and the same instruction order, so the
two can be diffed line for line.  Read `../adder_kernel.md` for the byte-level
contract and the hand-written file for why the kernel is shaped the way it is —
none of that is repeated here.  What *is* here is the part that only matters to
a generator: which values earn a permanent register, and where the register
budget goes.

    python examples/gen_adder_model.py             # -> examples/out/adder_model_gen.tpu
    python examples/gen_adder_model.py --print     # ...and dump the source

`.equ` names are a host ABI, not private labels: `gen_vectors.program_kind`
identifies the program by finding `LSTRIDE`, and `gen_vectors.build_image`,
`torch_ref.adder_model`, `adder_export.stage_dram` and `host/run_adder.py` index
the constant table by name.  Every name the hand-written kernel defines is
therefore emitted here with the same spelling, and the generated file drops
straight into the existing verification pipeline:

    python ../gen_vectors.py -p examples/out/adder_model_gen.tpu
    python ../torch_ref.py   -p examples/out/adder_model_gen.tpu

One cosmetic difference from the hand-written file: `Program.equ` stores the
*evaluated* integer, so the generated source reads `.equ VPAD 16` where the
hand-written one reads `.equ VPAD (VOCAB + COLS - 1) // COLS * COLS`.  The
values are identical and the arithmetic that produced them is right here in
Python; only the derivation is no longer visible downstream.

--------------------------------------------------------------------------------
The register budget — the one thing a generator gets wrong
--------------------------------------------------------------------------------
There are 31 allocatable registers, no spilling, and roughly 60 distinct
addresses in the program.  pytpu turns any bare `int` operand into a
`p.const()` — cached, hoisted, and **permanent** — so writing this the obvious
way (`p.matmul_t(Q8, x_res, wgt_win)`) would burn one register per address and
run out somewhere in layer 0.

So the split is explicit, and it is the same split the hand-written kernel makes
by hand:

* **18 permanent** (`p.const`): the five loop strides, eleven scratchpad bases
  that are read every layer, and the three loop bounds plus `1` that `p.loop`
  hoists for itself.
* **everything else through `Reg.set()`**: five long-lived scratch registers
  (`dram_p`, `rq_word`, `buf_a`, `buf_b`, `lay_base`) reloaded with `li` at each
  use, exactly as the hand-written kernel reloads r1..r3 and r21/r22.

Peak live set is the head loop: 18 permanent + 4 scratch + `lay_base` + the
layer counter + four per-head pointers + the head counter = **29 of 31**.  The
four head pointers sit in a `p.scope()` so they are reclaimed for the sections
that follow.  If this ever raises "out of scalar registers", that is the count
to redo — not the kernel.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # accel/tpulang

from pytpu import IMEM_WORDS, Program                           # noqa: E402

OUT = os.path.join(HERE, "out", "adder_model_gen.tpu")


def build() -> Program:
    p = Program("adder_model — the whole adder_ternary_vanilla forward pass, "
                "four layers and the ternary head, in one program")

    # =======================================================================
    # Geometry — model/transformer.py adder_ternary_vanilla()
    # =======================================================================
    T      = p.equ("T", 32)               # tokens; <= 63 for cfg tlen
    D      = p.equ("D", 128)              # model dim
    HEADS  = p.equ("HEADS", 4)            # q_heads == kv_heads
    DH     = p.equ("DH", D // HEADS)      # head_dim
    D3     = p.equ("D3", D * 3)           # fused Q|K|V block: 3D columns
    p.equ("F", 128)                       # feed-forward hidden dim (== D today,
                                          # which is why §6 needs no new cfg)
    VOCAB  = p.equ("VOCAB", 13)
    LAYERS = p.equ("LAYERS", 4)           # the only thing that moves with depth

    ROWS   = p.equ("ROWS", 8)             # MXU array contraction dim
    COLS   = p.equ("COLS", 8)             # MXU array output cols
    # The array stores whole COLS-wide tiles, so a 13-column head would spill
    # three int32 words of each row onto the next row's first three.  Round up.
    VPAD   = p.equ("VPAD", (VOCAB + COLS - 1) // COLS * COLS)

    KTD    = p.equ("KTD", D // ROWS)      # contraction tiles, D-deep matmul = 16
    NTD    = p.equ("NTD", D // COLS)      # output tiles, 128 columns        = 16
    NTV    = p.equ("NTV", VPAD // COLS)   # output tiles, the padded head    =  2
    KTDH   = p.equ("KTDH", DH // ROWS)    # contraction tiles, Q @ K^T       =  4
    NTT    = p.equ("NTT", T // COLS)      # output tiles, T keys             =  4
    KTT    = p.equ("KTT", T // ROWS)      # contraction tiles, P @ V         =  4
    NTDH   = p.equ("NTDH", DH // COLS)    # output tiles, one head's DH cols =  4

    CHUNK  = p.equ("CHUNK", 512)          # cfg vlen is 10 bits: 1024 reads as 0
    PKCH   = p.equ("PKCH", CHUNK // 4)    # ...and what one `tquant` pass writes
    NCHUNK = p.equ("NCHUNK", T * D // CHUNK)   # passes over a [T][D] activation

    # ---- strides (bytes) --------------------------------------------------
    AROW   = p.equ("AROW", D)             # activation row: D int8          = 128
    WCOL   = p.equ("WCOL", D * 2 // 8)    # ternary column, D trits deep    =  32
    TCOL   = p.equ("TCOL", T * 2 // 8)    # ternary column, T trits deep    =   8
    KHOFF  = p.equ("KHOFF", DH * 2 // 8)  # head step inside a packed K row =   8
    VHOFF  = p.equ("VHOFF", DH * TCOL)    # head step inside packed V^T     = 256
    CROWD  = p.equ("CROWD", D * 4)        # int32 row of a [T][D] out       = 512
    SROW   = p.equ("SROW", T * 4)         # int32 row of a [T][T] block     = 128
    LROW   = p.equ("LROW", VPAD * 4)      # int32 row of the [T][VPAD] logits

    # =======================================================================
    # Scratchpad map (64 KB; highest byte used is 0xCC3F)
    # =======================================================================
    WBUF   = p.equ("WBUF",  0x0000)       # weight window, refilled 4x/layer
    Q8     = p.equ("Q8",    0x3000)       # [T][D] int8 Q
    K8     = p.equ("K8",    0x4000)       # [T][D] int8 K, before ternarizing
    V8     = p.equ("V8",    0x5000)       # [T][D] int8 V, before transposing
    VT     = p.equ("VT",    0x6000)       # [D][T] int8 V^T, all heads
    X      = p.equ("X",     0x7000)       # [T][D] int8 residual stream
    T1     = p.equ("T1",    0x8000)       # [T][D] int8 temp: A, then H, then F
    T2     = p.equ("T2",    0x9000)       # [T][D] int8 temp: O, XO, relu(H)
    S32    = p.equ("S32",   0xA000)       # [T][VPAD] int32 logits
    TMP32  = p.equ("TMP32", 0xB000)       # CHUNK int32 elementwise temp
    S8     = p.equ("S8",    0xB800)       # [T][T] int8 scores
    P8     = p.equ("P8",    0xBC00)       # [T][T] int8 attention weights
    KTP    = p.equ("KTP",   0xC000)       # [T][D] trits — K as an MXU weight
    VTP    = p.equ("VTP",   0xC400)       # [D][T] trits — V^T as an MXU weight
    MASK   = p.equ("MASK",  0xC800)       # [T][T] int8 causal mask, 0 / -128
    RQW    = p.equ("RQW",   0xCC00)       # this layer's 16 requant {m0,n} words

    # The 16 requant slots, in adder_kernel.md §4's block order.  Four are read
    # by an op other than `requant` — two `dyt`, two `tquant` — but the word
    # format is identical, so DRAM staging does not care which.
    RQ_NAMES = ("RQ_Q", "RQ_K", "RQ_KT", "RQ_V", "RQ_VT", "RQ_S", "RQ_ID",
                "RQ_P", "RQ_A", "RQ_O", "RQ_XO", "RQ_X1", "RQ_H", "RQ_HR",
                "RQ_F", "RQ_X2")
    RQ = {n: p.equ(n, RQW + 4 * k) for k, n in enumerate(RQ_NAMES)}

    # =======================================================================
    # DRAM map (512 KB; highest byte used is 0x21FFF).  Globals below 0x5000 so
    # each one fits a 16-bit `li`; the layer blocks are reached through
    # `lay_base` instead, since layer 3's starts at 0x1AC00.
    # =======================================================================
    D_XIN  = p.equ("D_XIN",  0x0000)      # [T][D] int8, the embedded input
    D_MASK = p.equ("D_MASK", 0x1000)      # [T][T] int8 causal mask
    D_A8   = p.equ("D_A8",   0x1400)      # [T][D] int8 attention out (checkpoint)
    D_VT   = p.equ("D_VT",   0x2400)      # [D][T] int8 V^T (scratch + checkpoint)
    D_WFC  = p.equ("D_WFC",  0x3400)      # [D][VPAD] trits, fc.weight
    D_LOG  = p.equ("D_LOG",  0x3C00)      # [T][VPAD] int32 logits — the result
    D_L0   = p.equ("D_L0",   0x5000)      # layer 0's block
    LSTRIDE = p.equ("LSTRIDE", 0x7400)    # per-layer block stride

    O_WQKV = p.equ("O_WQKV", 0x0000)      # [D][3D] trits, col-major 2b
    O_WO   = p.equ("O_WO",   0x3000)      # [D][D] trits
    O_W1   = p.equ("O_W1",   0x4000)      # [D][F] trits
    O_W2   = p.equ("O_W2",   0x5000)      # [F][D] trits
    O_RQW  = p.equ("O_RQW",  0x6000)      # this layer's 16 requant words
    O_XOUT = p.equ("O_XOUT", 0x6400)      # [T][D] int8 — this layer's X

    B_WQKV = p.equ("B_WQKV", D3 * WCOL)   # 12288
    B_WD   = p.equ("B_WD",   D * WCOL)    #  4096
    B_ACT  = p.equ("B_ACT",  T * D)       #  4096  a [T][D] int8 activation
    B_VT   = p.equ("B_VT",   D * T)       #  4096
    B_SQ   = p.equ("B_SQ",   T * T)       #  1024  a [T][T] int8 block
    p.equ("B_TRIT", T * D * 2 // 8)       #  1024  a [T][D] tensor as trits
    B_RQW  = p.equ("B_RQW",  64)
    B_WFC  = p.equ("B_WFC",  VPAD * WCOL) #   512  the head is ternary too
    B_LOG  = p.equ("B_LOG",  T * VPAD * 4)#  2048

    # =======================================================================
    # Registers
    # =======================================================================
    # Long-lived scratch: reloaded with `li` at each use rather than given one
    # meaning.  `dram_p` also carries the layer-block *offset* on its way to an
    # absolute address (`dram_p.set(O_WO); dram_p += lay_base`), which is the
    # hand-written kernel's blk_off/dram_p pair folded into one register.
    dram_p  = p.reg("dram_p")      # DRAM side of every rdmem/wrmem
    rq_word = p.reg("rq_word")     # the {m0,n} word a VPU narrow reads
    buf_a   = p.reg("buf_a")       # general scratchpad-buffer pointers; which
    buf_b   = p.reg("buf_b")       # is source and which destination is local

    # Permanent: the strides every loop steps by...
    chunk_b = p.const(CHUNK, "chunk_b")   # one int8 VPU pass, in bytes
    pack_b  = p.const(PKCH,  "pack_b")    # ...and what `tquant` writes: CHUNK/4
    head_i8 = p.const(DH,    "head_i8")   # a head's columns in an int8 [T][D]
    head_kt = p.const(KHOFF, "head_kt")   # ...within a packed K row
    head_vt = p.const(VHOFF, "head_vt")   # ...whole rows, in packed V^T
    # ...and the scratchpad bases read every layer.  S8/P8/MASK get two apiece
    # because a [T][T] block is 1024 elements and `cfg vlen` tops out at 1023.
    wgt_win = p.const(WBUF,  "wgt_win")
    x_res   = p.const(X,     "x_res")
    acc_i32 = p.const(TMP32, "acc_i32")
    scr_lo  = p.const(S8,    "scr_lo")
    scr_hi  = p.const("S8 + CHUNK",   "scr_hi")
    att_lo  = p.const(P8,    "att_lo")
    att_hi  = p.const("P8 + CHUNK",   "att_hi")
    msk_lo  = p.const(MASK,  "msk_lo")
    msk_hi  = p.const("MASK + CHUNK", "msk_hi")

    # =======================================================================
    # 0. The model-wide inputs
    # =======================================================================
    p.comment("=" * 71)
    p.comment("0. X and the causal mask, loaded once for the whole model.")
    p.comment("=" * 71)
    dram_p.set(D_XIN)
    p.cfg(len=B_ACT)
    p.rdmem(x_res, dram_p)                # DRAM D_XIN -> scratch X.  X then
                                          # stays resident: each layer updates
                                          # it in place, no DRAM round trip.
    dram_p.set(D_MASK)
    p.cfg(len=B_SQ)
    p.rdmem(msk_lo, dram_p)               # one fill covers both halves

    p.cfg(tlen=T)                         # the one MXU config every matmul shares

    lay_base = p.reg("lay_base", init=D_L0)

    # =======================================================================
    with p.loop(LAYERS, name="layer_i"):
        # ===================================================================
        p.comment("this layer's 16 requant words, into the fixed RQW slot")
        dram_p.set(O_RQW)
        dram_p += lay_base
        buf_a.set(RQW)
        p.cfg(len=B_RQW)
        p.rdmem(buf_a, dram_p)            # RQ_* are compile-time offsets into it

        # ---------------------------------------------------------------
        p.comment("1. Q, K, V — three ternary matmuls out of one fused block")
        # [Wq|Wk|Wv] is one rdmem but three matmuls: each projection carries
        # its own absmean, so each needs its own requant word.  A single
        # 48-tile dispatch would force one multiplier on all three.
        p.cfg(len=B_WQKV)
        p.rdmem(wgt_win, lay_base)        # O_WQKV is 0, so lay_base already *is*
                                          # the address; the hand-written kernel
                                          # spends an `adds ..., r0` to say so

        # The D-deep projection geometry.  Sticky: every matmul_t until §4
        # re-sets these reads exactly them.
        p.cfg(arow=AROW, wcol=WCOL, ktiles=KTD, ntiles=NTD, crow=CROWD)

        buf_a.set(Q8)
        p.cfg(scalar=RQ["RQ_Q"])
        p.matmul_t(buf_a, x_res, wgt_win, rq=True)          # Q = X @ Wq

        buf_a.set(K8)
        buf_b.set("WBUF + D * WCOL")                        # Wk: columns 128..255
        p.cfg(scalar=RQ["RQ_K"])
        p.matmul_t(buf_a, x_res, buf_b, rq=True)            # K = X @ Wk

        buf_a.set(V8)
        buf_b.set("WBUF + 2 * D * WCOL")                    # Wv: columns 256..383
        p.cfg(scalar=RQ["RQ_V"])
        p.matmul_t(buf_a, x_res, buf_b, rq=True)            # V = X @ Wv

        # ---------------------------------------------------------------
        p.comment("2. K -> trits.  A packed row of K *is* a weight column of K^T,")
        p.comment("   so Q @ K^T needs no transpose at all.")
        p.cfg(vlen=CHUNK)
        rq_word.set(RQ["RQ_KT"])
        buf_a.set(K8)                     # source: int8
        buf_b.set(KTP)                    # destination: packed trits
        with p.loop(NCHUNK, name="i"):
            p.tquant(buf_b, buf_a, rq_word)
            buf_a += chunk_b              # +512 read...
            buf_b += pack_b               # ...+128 written.  A trit is 2 bits.

        # ---------------------------------------------------------------
        p.comment("3. V^T in one transposing DMA, then V^T -> trits")
        # P @ V contracts over keys, which is V's row axis, so the weight
        # column is a column of V.  The DMA moves bytes and not trits, which
        # is why V is ternarized after the transpose rather than before.
        buf_a.set(V8)
        dram_p.set(D_VT)
        p.cfg(tcols=D, tsrow=AROW, tdrow=T, len=B_VT)
        p.wrmem(buf_a, dram_p, t=True)    # scratch V -> DRAM V^T
        buf_a.set(VT)
        p.rdmem(buf_a, dram_p)            # ...and back, untransposed: the mode
                                          # is in the mnemonic, not in the cfg

        p.cfg(vlen=CHUNK)
        rq_word.set(RQ["RQ_VT"])
        buf_b.set(VTP)                    # buf_a is already VT, the source
        with p.loop(NCHUNK, name="i"):
            p.tquant(buf_b, buf_a, rq_word)
            buf_a += chunk_b
            buf_b += pack_b

        # ---------------------------------------------------------------
        p.comment("4. Attention, one head at a time — both matmuls on the MXU")
        # Head h is a contiguous DH-column slice of Q and a byte offset into
        # each packed block, so there is no reshape anywhere.  The four
        # pointers are scoped: nothing after this section needs them.
        with p.scope():
            q_head  = p.reg("q_head",  init=Q8)    # step head_i8
            kt_head = p.reg("kt_head", init=KTP)   # step head_kt
            vt_head = p.reg("vt_head", init=VTP)   # step head_vt
            a_head  = p.reg("a_head",  init=T1)    # step head_i8
            with p.loop(HEADS, name="h"):
                # S8 = requant(Q_h @ K_h^T).  RQ_S carries 1/sqrt(head_dim),
                # and the narrow is the array's own store-side one.
                p.cfg(arow=AROW, wcol=WCOL, crow=SROW,
                      ktiles=KTDH, ntiles=NTT, scalar=RQ["RQ_S"])
                p.matmul_t(scr_lo, q_head, kt_head, rq=True)

                # P = relu(S8 + mask), in two CHUNK passes.  The mask is data,
                # not control flow: adding -128 makes a masked entry negative
                # for every score scale, so the relu takes it to exactly zero.
                p.cfg(vlen=CHUNK)
                rq_word.set(RQ["RQ_ID"])            # {m0=1, n=0}: narrow + clip
                p.vecadd(acc_i32, scr_lo, msk_lo)
                p.requant(scr_lo, acc_i32, rq_word)
                p.vecadd(acc_i32, scr_hi, msk_hi)
                p.requant(scr_hi, acc_i32, rq_word)
                rq_word.set(RQ["RQ_P"])
                p.relu(acc_i32, scr_lo)
                p.requant(att_lo, acc_i32, rq_word)
                p.relu(acc_i32, scr_hi)
                p.requant(att_hi, acc_i32, rq_word)

                # A[:, h*DH:(h+1)*DH] = requant(P_h @ V_h).  crow = CROWD is a
                # stride of the whole A, so the tile lands directly in head h's
                # column block — no gather, no scatter.
                p.cfg(arow=T, wcol=TCOL, crow=CROWD,
                      ktiles=KTT, ntiles=NTDH, scalar=RQ["RQ_A"])
                p.matmul_t(a_head, att_lo, vt_head, rq=True)

                q_head  += head_i8
                kt_head += head_kt
                vt_head += head_vt
                a_head  += head_i8

        # ---------------------------------------------------------------
        p.comment("5. O = A @ Wo, then the residual X <- norm1(X + (O + X))")
        buf_a.set(T1)
        dram_p.set(D_A8)
        p.cfg(len=B_ACT)
        p.wrmem(buf_a, dram_p)            # checkpoint A; not a data path

        dram_p.set(O_WO)
        dram_p += lay_base
        p.cfg(len=B_WD)
        p.rdmem(wgt_win, dram_p)          # Wo over the QKV block

        # §4 left the MXU set up for attention, and cfg is sticky, so all five
        # have to be restored even where the value is unchanged.
        p.cfg(arow=AROW, wcol=WCOL, ktiles=KTD, ntiles=NTD, crow=CROWD)
        buf_a.set(T1)                     # activation: A
        buf_b.set(T2)                     # destination: O
        p.cfg(scalar=RQ["RQ_O"])
        p.matmul_t(buf_b, buf_a, wgt_win, rq=True)

        # The attention residual is 2X + O, not X + O: MultiHeadAttention
        # returns `O + X` and Transformer adds `X` again.  vecadd takes two
        # operands, so that is two adds with a narrow between them — and that
        # narrow (RQ_XO) has to be {1,0}, because the second add needs both
        # sides still on X's scale.
        p.cfg(vlen=CHUNK)
        rq_word.set(RQ["RQ_XO"])
        buf_a.set(X)                      # src0
        buf_b.set(T2)                     # src1 and dst: O, then XO over it
        with p.loop(NCHUNK, name="i"):
            p.vecadd(acc_i32, buf_a, buf_b)
            p.requant(buf_b, acc_i32, rq_word)
            buf_a += chunk_b
            buf_b += chunk_b

        # X1 = norm1(XO + X).  `dyt` is the same narrow with a symmetric clip:
        # RQ_X1 carries norm1's alpha and lands on 1/127, at which point
        # saturating at +-127 *is* hardtanh saturating at +-1.  No extra pass.
        rq_word.set(RQ["RQ_X1"])
        buf_a.set(X)                      # src0 and dst: X, then X1
        buf_b.set(T2)
        with p.loop(NCHUNK, name="i"):
            p.vecadd(acc_i32, buf_a, buf_b)
            p.dyt(buf_a, acc_i32, rq_word)
            buf_a += chunk_b
            buf_b += chunk_b

        # ---------------------------------------------------------------
        p.comment("6. Feed-forward: X <- norm2(X1 + W2(relu(W1(X1))))")
        # x_res still points at X — §5's dyt wrote X1 over it in place — and
        # §5's MXU geometry is still correct here, since F == D.
        dram_p.set(O_W1)
        dram_p += lay_base
        p.cfg(len=B_WD)
        p.rdmem(wgt_win, dram_p)
        buf_a.set(T1)                     # destination: H
        p.cfg(scalar=RQ["RQ_H"])
        p.matmul_t(buf_a, x_res, wgt_win, rq=True)

        p.cfg(vlen=CHUNK)
        rq_word.set(RQ["RQ_HR"])
        buf_a.set(T1)                     # source: H
        buf_b.set(T2)                     # destination: relu(H)
        with p.loop(NCHUNK, name="i"):
            p.relu(acc_i32, buf_a)
            p.requant(buf_b, acc_i32, rq_word)
            buf_a += chunk_b
            buf_b += chunk_b

        dram_p.set(O_W2)
        dram_p += lay_base
        p.cfg(len=B_WD)
        p.rdmem(wgt_win, dram_p)
        buf_a.set(T1)                     # destination: F (H is dead now)
        buf_b.set(T2)                     # activation:  relu(H)
        p.cfg(scalar=RQ["RQ_F"])
        p.matmul_t(buf_a, buf_b, wgt_win, rq=True)

        # X2 = norm2(X1 + F).  RQ_X2 targets 1/127 too, which is what makes the
        # next layer's input scale analytic rather than calibrated.
        p.cfg(vlen=CHUNK)
        rq_word.set(RQ["RQ_X2"])
        buf_a.set(X)
        buf_b.set(T1)
        with p.loop(NCHUNK, name="i"):
            p.vecadd(acc_i32, buf_a, buf_b)
            p.dyt(buf_a, acc_i32, rq_word)
            buf_a += chunk_b
            buf_b += chunk_b

        # ---------------------------------------------------------------
        p.comment("7. Checkpoint this layer's X, then advance the block base")
        dram_p.set(O_XOUT)
        dram_p += lay_base
        p.cfg(len=B_ACT)
        p.wrmem(x_res, dram_p)            # verification only; X stays resident

        dram_p.set(LSTRIDE)               # dram_p is free here
        lay_base += dram_p

    # =======================================================================
    p.comment("=" * 71)
    p.comment("8. Output head: logits = X @ fc.weight")
    p.comment("=" * 71)
    # Model.fc is a TernaryLinear, so this is an ordinary matmul_t against the
    # same D-deep geometry the projections use, and `TernaryLinear.w` is
    # already [in, out] — the orientation the array packs a weight in.
    # VPAD, not VOCAB: see the .equ above.  Spilled as raw int32 with no
    # requant, because an argmax over 13 logits does not care about scale.
    dram_p.set(D_WFC)
    p.cfg(len=B_WFC)
    p.rdmem(wgt_win, dram_p)

    p.cfg(arow=AROW, wcol=WCOL, ktiles=KTD, ntiles=NTV, crow=LROW)
    log_i32 = p.reg("log_i32", init=S32)
    p.matmul_t(log_i32, x_res, wgt_win)   # no .rq: LOG[t][v] stays int32

    dram_p.set(D_LOG)
    p.cfg(len=B_LOG)
    p.wrmem(log_i32, dram_p)              # the result the host reads back
    p.halt()
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=OUT, help="output .tpu path")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="dump the generated source to stdout")
    args = ap.parse_args()

    p = build()
    words = p.assemble()                  # raises if it does not fit in IMEM
    path = p.save(args.out)
    if args.show:
        print(p.render())
    print(f"{path}: {len(words)} words of {IMEM_WORDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
