/* adder.c — the whole int4 adder model, in firmware.
 *
 * `model/transformer.py::adder_int4_vanilla` — d=64, f=256, layers=4,
 * q_heads=kv_heads=4 (head_dim=16), vocab=13, int4 weights *and* int4
 * activations, no bias, no LayerNorm, no positional encoding — as one program,
 * one run. It is the successor to `accel/tpulang/examples/adder_model.tpu`, which
 * went with the rest of the tpulang toolchain (git history if you want it), moved
 * from ternary/column-major weights to int4/row-major:
 *
 *   for L in 0..3:
 *       Q, K, V = X@Wq, X@Wk, X@Wv          int4 weights, int4 activations
 *       S       = Q @ K^T / sqrt(head_dim)  per head, on the array
 *       P       = relu(S + causal_mask)     ReLU attention, NOT softmax
 *       A       = P @ V                     per head, on the array
 *       X       = dyt(X + (Wo(A) + X))      DOUBLE residual, then DyT
 *       X       = dyt(X + W2(relu(W1(X))))
 *   logits = X @ fc.w                       int32, never requantized
 *
 * The host owns exactly two things, both structural: the token embedding (the
 * ISA has no gather) and the final argmax (nothing returns an index). X0 and
 * the causal mask arrive in DRAM; the logits leave in DRAM.
 *
 * ---- why K is transposed and V is not ---------------------------------------
 *
 * The array reads weights row-major, so for C = A @ W the operand W[k][n] has
 * row k contiguous over n.
 *
 *   Q @ K^T contracts over the head dim: its weight is K^T[h][s], row h
 *           contiguous over s — K *column*-major. K leaves its projection
 *           row-major, so this one needs the transpose.
 *   P @ V   contracts over keys: its weight is V[s][h], row s contiguous over
 *           h, which is exactly how V left its projection. Free.
 *
 * That is the opposite of the ternary column-major kernel, where K was free and
 * V needed the transpose; row-major swapped the two. The transpose goes through
 * int8 because the DMA moves *bytes* and a packed int4 nibble is half of one,
 * so K is transposed first (SP_KT) and packed second (SP_KTP).
 *
 * ---- the requant table ------------------------------------------------------
 *
 * Every tensor carries a compile-time scale; integer v means real v*s. The 16
 * {m0,n} words per layer are the *only* thing in this kernel that depends on
 * the checkpoint, and they arrive as ADDER_RQ_INIT from a generated header:
 *
 *   adder_rq.h                the checked-in default, tuned for the synthetic
 *                             operands ../../tpulang/fw_vectors.py stages, so
 *                             `make fw FWPROG=adder` needs no checkpoint
 *   -DADDER_RQ_H='"..."'      a header derived from a real checkpoint by
 *                             ../../tpulang/adder_export.py
 *
 * They are literals in the command, so unlike the scalar unit's ISA there is no
 * path by which the device could read them out of memory — the table has to be
 * in the image. See adder_export.py for where each multiplier comes from.
 */
#include "tpu.h"

#ifdef ADDER_RQ_H
#include ADDER_RQ_H
#else
#include "adder_rq.h"
#endif

#define ROWS 8                  /* MXU array geometry — fixed by the bitstream */
#define COLS 8

#define T      32               /* tokens (train.py --max_tokens) */
#define D      64               /* model width */
#define DFF    256              /* feed-forward width */
#define NH     4                /* heads (q_heads == kv_heads) */
#define DH     (D / NH)         /* head dim, 16 */
#define LAYERS 4
#define VOCAB  13
/* The array stores a whole COLS-wide tile, so a 13-column head is two tiles and
 * a 13-word row stride would let the second tile land on the next token's row.
 * Round the head's output up to a whole tile and read 13 of every 16 words
 * back; the padding columns are staged as zero weights. */
#define VPAD   16

/* Requant sites, in block order — the index into one layer's ADDER_RQ row.
 * Six of the sixteen are the {1,0} identity by construction and are kept in the
 * table anyway so it matches model/transformer.py's site list one for one. */
enum {
    RQ_Q, RQ_K, RQ_V,           /* the three projections            */
    RQ_KP, RQ_VP,               /* quant4 packs — {1,0}, both int4  */
    RQ_S,                       /* Q@K^T, with 1/sqrt(head_dim)     */
    RQ_ID,                      /* mask clamp — {1,0}               */
    RQ_P,                       /* relu(S) -> P — {1,0}, s_p = s_s  */
    RQ_A,                       /* P@V                              */
    RQ_O,                       /* A@Wo -> O, pinned to s_x         */
    RQ_XO,                      /* X + O — {1,0}, stays on s_x      */
    RQ_X1,                      /* dyt(XO + X)                      */
    RQ_H,                       /* X1@W1                            */
    RQ_HR,                      /* relu(H) — {1,0}, s_hr = s_h      */
    RQ_F,                       /* HR@W2 -> F, pinned to s_x1       */
    RQ_X2,                      /* dyt(X1 + F)                      */
    RQ_N
};

static const uint16_t rq_tab[LAYERS][RQ_N] = ADDER_RQ_INIT;

/* ---- weight row strides, bytes (row-major int4: a row is N nibbles) ------- */
#define WROW_QKV ((3 * D) * 4 / 8)   /* the fused [D][3D] projection block */
#define WROW_D   (D * 4 / 8)         /* [.][D]  — Wo, W2                   */
#define WROW_F   (DFF * 4 / 8)       /* [D][F]  — W1                       */
#define WROW_KT  (T * 4 / 8)         /* [D][T]  — packed K^T               */
#define WROW_FC  (VPAD * 4 / 8)      /* [D][16] — the output head          */

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_X     0x00000u       /* [T][D]  int8  — embedded input, host     */
#define DR_MASK  0x00800u       /* [T][T]  int8  — causal mask, host        */
#define DR_KT    0x00C00u       /* [D][T]  int8  — K^T scratch, device      */
#define DR_WFC   0x01400u       /* [D][16] int4  — output head, host        */
#define DR_LOG   0x01800u       /* [T][16] int32 — the result, device       */
#define DR_LAYER 0x02000u       /* layer 0's weight block                   */
#define DR_LSTEP 0x06000u       /* ...and the stride between layers         */
#define LW_QKV   0x0000u        /*   [D][3D] int4, 6144 B                   */
#define LW_O     0x1800u        /*   [D][D]  int4, 2048 B                   */
#define LW_1     0x2000u        /*   [D][F]  int4, 8192 B                   */
#define LW_2     0x4000u        /*   [F][D]  int4, 8192 B                   */

/* ---- scratchpad (64 KB; top byte used is 0xDFFF) -------------------------- */
/* One 8 KB weight window, refilled four times per layer, sized by the largest
 * single fill (W1/W2). Everything else is resident for the whole run. */
#define SP_W    0x0000u         /* 8192 — the weight window          */
#define SP_X    0x2000u         /* [T][D]  int8 — the residual stream */
#define SP_Q    0x2800u         /* [T][D]  int8                       */
#define SP_K    0x3000u
#define SP_V    0x3800u
#define SP_KT   0x4000u         /* [D][T]  int8  — K transposed       */
#define SP_KTP  0x4800u         /* [D][T]  int4  — the Q@K^T weight   */
#define SP_VP   0x4C00u         /* [T][D]  int4  — the P@V weight     */
#define SP_MASK 0x5000u         /* [T][T]  int8                       */
#define SP_S    0x5400u         /* [T][T]  int8                       */
#define SP_SM   0x5800u         /* [T][T]  int8  — S after the mask   */
#define SP_P    0x5C00u         /* [T][T]  int8                       */
#define SP_A    0x6000u         /* [T][D]  int8                       */
#define SP_O    0x6800u         /* [T][D]  int8                       */
#define SP_XO   0x7000u         /* [T][D]  int8                       */
#define SP_X1   0x7800u         /* [T][D]  int8                       */
#define SP_H    0x8000u         /* [T][F]  int8                       */
#define SP_HR   0xA000u         /* [T][F]  int8                       */
#define SP_T32  0xC000u         /* CHUNK int32 — the one wide temp    */
#define SP_FF   0xD000u         /* [T][D]  int8 — the FFN output      */
#define SP_LOG  0xD800u         /* [T][16] int32                      */

/* `vpu_vlen` is 10 bits, so no pass may exceed 1023 elements; every elementwise
 * tensor here is a multiple of 512, which also keeps the int32 temp at 2 KB. */
#define CHUNK 512u

static uint32_t chunk_len(uint32_t n, uint32_t i)
{
    return (n - i < CHUNK) ? (n - i) : CHUNK;
}

/* dst = narrow(src0 + src1) over n elements. `vecadd` reads int8 and writes
 * int32, and only an explicit narrow comes back down (vpu.md), so a residual
 * add is always this pair. `narrow_op` is REQUANT or DYT — the two differ only
 * in the clip, and choosing DYT here is how a kernel says "this is norm1". */
static void add_narrow(unsigned narrow_op, uint32_t dst, uint32_t src0,
                       uint32_t src1, uint32_t n, uint32_t rq)
{
    for (uint32_t i = 0; i < n; i += CHUNK) {
        uint32_t len = chunk_len(n, i);
        tpu_vpu(TPU_V_ADD, SP_T32, src0 + i, src1 + i, len, 0u);
        tpu_vpu(narrow_op, dst + i, SP_T32, 0u, len, rq);
    }
}

/* dst = requant(relu(src)) over n elements — the same widening/narrowing pair. */
static void relu_narrow(uint32_t dst, uint32_t src, uint32_t n, uint32_t rq)
{
    for (uint32_t i = 0; i < n; i += CHUNK) {
        uint32_t len = chunk_len(n, i);
        tpu_vpu(TPU_V_RELU, SP_T32, src + i, 0u, len, 0u);
        tpu_vpu(TPU_V_REQUANT, dst + i, SP_T32, 0u, len, rq);
    }
}

/* dst = quant4(src): int8 in, 4 bits out, so the destination advances half as
 * fast as the source. This is what turns an activation into a legal weight
 * operand without a repacking pass or a host round trip. */
static void pack4(uint32_t dst, uint32_t src, uint32_t n, uint32_t rq)
{
    for (uint32_t i = 0; i < n; i += CHUNK) {
        uint32_t len = chunk_len(n, i);
        tpu_vpu(TPU_V_QUANT4, dst + (i >> 1), src + i, 0u, len, rq);
    }
}

int main(void)
{
    tpu_dma(SP_X,    DR_X,    T * D, TPU_DMA_FILL);
    tpu_dma(SP_MASK, DR_MASK, T * T, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    for (unsigned l = 0; l < LAYERS; l++) {
        const uint16_t *rq = rq_tab[l];
        uint32_t wb = DR_LAYER + l * DR_LSTEP;

        /* ---- Q, K, V: one fused [D][3D] weight fill, three dispatches ----
         * Fusing the *fill* is free; fusing the matmul is not, because each
         * projection has its own weight scale and so its own {m0,n}. Column c
         * of the fused block is nibble c, i.e. byte c/2. */
        tpu_dma(SP_W, wb + LW_QKV, D * WROW_QKV, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);

        tpu_mxu_geom(D, D * 4, WROW_QKV, D / ROWS, D / COLS, T);
        tpu_mxu_mm(SP_Q, SP_X, SP_W,             TPU_MM_TILED | TPU_MM_RQ, rq[RQ_Q]);
        tpu_mxu_mm(SP_K, SP_X, SP_W + D / 2,     TPU_MM_TILED | TPU_MM_RQ, rq[RQ_K]);
        tpu_mxu_mm(SP_V, SP_X, SP_W + D,         TPU_MM_TILED | TPU_MM_RQ, rq[RQ_V]);
        tpu_wait(TPU_U_MXU);

        /* ---- K -> K^T, through DRAM, as bytes ----
         * dest[col*tdrow + row] = src[row*tsrow + col] (dma.sv §5), so a [T][D]
         * block transposed wants tcols = D, tsrow = D, tdrow = T. */
        tpu_dma_t(SP_K, DR_KT, T * D, TPU_DMA_SPILL, D, D, T);
        tpu_wait(TPU_U_DMA);
        tpu_dma(SP_KT, DR_KT, T * D, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);

        /* Both attention weight operands. K and V are already int4 — whatever
         * requant produced them clipped to [-8, 7] — so both packs are the
         * {1,0} identity and lose nothing. */
        pack4(SP_KTP, SP_KT, D * T, rq[RQ_KP]);
        pack4(SP_VP,  SP_V,  T * D, rq[RQ_VP]);
        tpu_wait(TPU_U_VPU);

        for (unsigned h = 0; h < NH; h++) {
            /* S = requant(Q_h @ K_h^T). Head h is a column slice of Q (row
             * stride D) and a row slice of K^T (rows h*DH..h*DH+DH-1). */
            tpu_mxu_geom(D, T * 4, WROW_KT, DH / ROWS, T / COLS, T);
            tpu_mxu_mm(SP_S, SP_Q + h * DH, SP_KTP + h * DH * WROW_KT,
                       TPU_MM_TILED | TPU_MM_RQ, rq[RQ_S]);
            tpu_wait(TPU_U_MXU);

            /* P = requant(relu(requant(S + mask))).
             *
             * The mask is 0 or -8 and S is already int4, so a masked entry is
             * at most -1 whatever s_s is and ReLU takes it to exactly zero —
             * exact, not a tolerance. It costs the RQ_ID narrow because `relu`
             * reads int8 while `vecadd` writes int32. */
            add_narrow(TPU_V_REQUANT, SP_SM, SP_S, SP_MASK, T * T, rq[RQ_ID]);
            relu_narrow(SP_P, SP_SM, T * T, rq[RQ_P]);
            tpu_wait(TPU_U_VPU);

            /* A_h = requant(P @ V_h), written straight into A's column block.
             * The wait below is what keeps the next head's P from overwriting
             * the operand this dispatch is still reading. */
            tpu_mxu_geom(T, D * 4, WROW_D, T / ROWS, DH / COLS, T);
            tpu_mxu_mm(SP_A + h * DH, SP_P, SP_VP + h * DH / 2,
                       TPU_MM_TILED | TPU_MM_RQ, rq[RQ_A]);
            tpu_wait(TPU_U_MXU);
        }

        /* ---- O = requant(A @ Wo) ----
         * RQ_O has to land on s_x: `vecadd` takes two int8 operands at one
         * scale, so the residual pins O's output scale to the stream's. */
        tpu_dma(SP_W, wb + LW_O, D * WROW_D, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);
        tpu_mxu_geom(D, D * 4, WROW_D, D / ROWS, D / COLS, T);
        tpu_mxu_mm(SP_O, SP_A, SP_W, TPU_MM_TILED | TPU_MM_RQ, rq[RQ_O]);
        tpu_wait(TPU_U_MXU);

        /* ---- the double residual, then DyT ----
         * MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so the attention residual is 2X + O. `vecadd` takes two
         * operands, hence two adds with the {1,0} identity between them. */
        add_narrow(TPU_V_REQUANT, SP_XO, SP_X,  SP_O, T * D, rq[RQ_XO]);
        add_narrow(TPU_V_DYT,     SP_X1, SP_XO, SP_X, T * D, rq[RQ_X1]);
        tpu_wait(TPU_U_VPU);

        /* ---- H = requant(X1 @ W1) ---- */
        tpu_dma(SP_W, wb + LW_1, D * WROW_F, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);
        tpu_mxu_geom(D, DFF * 4, WROW_F, D / ROWS, DFF / COLS, T);
        tpu_mxu_mm(SP_H, SP_X1, SP_W, TPU_MM_TILED | TPU_MM_RQ, rq[RQ_H]);
        tpu_wait(TPU_U_MXU);

        relu_narrow(SP_HR, SP_H, T * DFF, rq[RQ_HR]);
        tpu_wait(TPU_U_VPU);

        /* ---- F = requant(HR @ W2), pinned to s_x1 by the second residual --- */
        tpu_dma(SP_W, wb + LW_2, DFF * WROW_D, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);
        tpu_mxu_geom(DFF, D * 4, WROW_D, DFF / ROWS, D / COLS, T);
        tpu_mxu_mm(SP_FF, SP_HR, SP_W, TPU_MM_TILED | TPU_MM_RQ, rq[RQ_F]);
        tpu_wait(TPU_U_MXU);

        /* X2 lands back in X: this layer's input is dead by now. */
        add_narrow(TPU_V_DYT, SP_X, SP_X1, SP_FF, T * D, rq[RQ_X2]);
        tpu_wait(TPU_U_VPU);
    }

    /* ---- logits = X @ fc.w ----
     * Never requantized: the head's output scale is irrelevant to an argmax, so
     * the device spills the raw int32 accumulator. */
    tpu_dma(SP_W, DR_WFC, D * WROW_FC, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);
    tpu_mxu_geom(D, VPAD * 4, WROW_FC, D / ROWS, VPAD / COLS, T);
    tpu_mxu_mm(SP_LOG, SP_X, SP_W, TPU_MM_TILED, 0u);
    tpu_wait(TPU_U_MXU);

    tpu_dma(SP_LOG, DR_LOG, T * VPAD * 4, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
