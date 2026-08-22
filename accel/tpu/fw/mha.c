/* mha.c — one head of ReLU attention, in firmware.
 *
 *   Q = requant(X @ Wq)    K = requant(X @ Wk)    V = requant(X @ Wv)
 *   S = requant(Q @ K^T)
 *   P = requant(relu(S))                          identity: P shares S's scale
 *   A = requant(P @ V)
 *
 * The kernel that exercises everything at once: both units, all three DMA modes
 * (fill, spill, **transposing** spill), and the `quant4` narrow that turns an
 * activation into a weight operand. No causal mask — this is a datapath and ISA
 * test, not the model.
 *
 * ---- why K is transposed and V is not ---------------------------------------
 *
 * The array reads weights **row-major**: for C = A @ W the operand W[k][n] is
 * stored with row k contiguous over n.
 *
 *   Q @ K^T contracts over the head dim. Its weight is K^T[h][s], so row h must
 *           be contiguous over s — that is K *column*-major, and K comes out of
 *           its projection row-major. So this one needs the transpose.
 *   P @ V   contracts over keys. Its weight is V[s][h], row s contiguous over h,
 *           which is exactly how V left its projection. Free.
 *
 * That is the **opposite** of the ternary column-major kernel
 * (the deleted adder_model.tpu; see adder.c's header), where K was free and V needed
 * the transpose. Row-major swapped the two, and the `.t` moved with it.
 *
 * The transpose goes through int8 because the DMA moves *bytes* and a packed
 * int4 nibble is half of one — so K is transposed first and packed second,
 * which is why there is a KT8 buffer between the two.
 */
#include "tpu.h"

#define ROWS 8                  /* MXU array geometry — fixed by the bitstream */
#define COLS 8

#define T  8                    /* tokens (= keys) */
#define D  8                    /* model width */
#define DH 8                    /* head dim */

/* {m0, n}: m0 in the low 12 bits, n above. The shifts are *tuned*, not guessed —
 * the projection accumulators only reach ~56 here, and the first draft used
 * n=5/6 which drove every score to zero. A test whose golden answer is all
 * zeros passes against any datapath, so these were chosen to spread the result
 * across the int4 grid: 12 / 7 / 12 distinct values in Q / S / A. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_QKV RQ(1u, 3u)       /* X @ Wq|Wk|Wv -> Q/K/V */
#define RQ_PK  RQ(1u, 0u)       /* the two packs: values are already int4 */
#define RQ_S   RQ(1u, 4u)       /* Q @ K^T -> S */
#define RQ_P   RQ(1u, 0u)       /* relu(S) -> P : identity, P shares S's scale */
#define RQ_A   RQ(1u, 3u)       /* P @ V   -> A */

/* Same address in DRAM and scratchpad. */
#define X_ADDR   0x0000u        /* [T][D]  int8   */
#define WQ_ADDR  0x0400u        /* [D][DH] int4, row-major */
#define WK_ADDR  0x0500u
#define WV_ADDR  0x0600u
#define Q_ADDR   0x1000u        /* [T][DH] int8   */
#define K_ADDR   0x1100u
#define V_ADDR   0x1200u
#define KT_ADDR  0x1300u        /* [DH][T] int8 — K transposed */
#define KTP_ADDR 0x1400u        /* [DH][T] int4 packed — the Q@K^T weight */
#define VP_ADDR  0x1500u        /* [T][DH] int4 packed — the P@V weight   */
#define S_ADDR   0x1600u        /* [T][T]  int8   */
#define P32_ADDR 0x1800u        /* [T][T]  int32 — relu output */
#define P_ADDR   0x1C00u        /* [T][T]  int8   */
#define A_ADDR   0x2000u        /* [T][DH] int8 — the result */

#define WROW_D  ((DH * 4) / 8)  /* a projection weight row is DH nibbles */
#define WROW_T  ((T  * 4) / 8)  /* K^T's rows are T nibbles              */
#define X_BYTES (T * D)
#define W_BYTES (D * WROW_D)
#define A_BYTES (T * DH)

int main(void)
{
    /* operands in */
    tpu_dma(X_ADDR,  X_ADDR,  X_BYTES, TPU_DMA_FILL);
    tpu_dma(WQ_ADDR, WQ_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_dma(WK_ADDR, WK_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_dma(WV_ADDR, WV_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- projections: all three share one geometry ---- */
    tpu_mxu_geom(D, DH * 4, WROW_D, 1u, 1u, T);
    tpu_mxu_mm(Q_ADDR, X_ADDR, WQ_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_mxu_mm(K_ADDR, X_ADDR, WK_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_mxu_mm(V_ADDR, X_ADDR, WV_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_wait(TPU_U_MXU);

    /* ---- K -> K^T, through DRAM, as bytes ----
     * dest[col*tdrow + row] = src[row*tsrow + col] (dma.sv §5), so a [T][DH]
     * block transposed wants tcols = DH, tsrow = DH, tdrow = T. */
    tpu_dma_t(K_ADDR, KT_ADDR, T * DH, TPU_DMA_SPILL, DH, DH, T);
    tpu_wait(TPU_U_DMA);
    tpu_dma(KT_ADDR, KT_ADDR, T * DH, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- pack both weight operands ----
     * quant4 writes 4 bits per element, so each destination advances half as
     * fast as its source and vlen must be even (vpu.md). */
    tpu_vpu(TPU_V_QUANT4, KTP_ADDR, KT_ADDR, 0u, DH * T, RQ_PK);
    tpu_vpu(TPU_V_QUANT4, VP_ADDR,  V_ADDR,  0u, T * DH, RQ_PK);
    tpu_wait(TPU_U_VPU);

    /* ---- S = requant(Q @ K^T) ---- */
    tpu_mxu_geom(DH, T * 4, WROW_T, 1u, 1u, T);
    tpu_mxu_mm(S_ADDR, Q_ADDR, KTP_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_S);
    tpu_wait(TPU_U_MXU);

    /* ---- P = requant(relu(S)) ---- */
    tpu_vpu(TPU_V_RELU,    P32_ADDR, S_ADDR,   0u, T * T, 0u);
    tpu_vpu(TPU_V_REQUANT, P_ADDR,   P32_ADDR, 0u, T * T, RQ_P);
    tpu_wait(TPU_U_VPU);

    /* ---- A = requant(P @ V) ---- */
    tpu_mxu_geom(T, DH * 4, WROW_D, 1u, 1u, T);
    tpu_mxu_mm(A_ADDR, P_ADDR, VP_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_A);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
