/* ffn.c — the transformer's feed-forward block, in firmware.
 *
 *   H  = requant( X  @ W1 )        [T][D] @ [D][F] -> [T][F]
 *   HR = requant( relu(H) )        identity requant: HR shares H's scale
 *   Y  = requant( HR @ W2 )        [T][F] @ [F][D] -> [T][D]
 *
 * The first kernel to issue a VPU command. matmul.c proved the CPU could drive
 * the array; this proves it can drive the vector unit too, and that the two
 * queues interleave correctly — the `relu` reads what the matmul wrote, so the
 * MXU must be waited for before the VPU is pushed.
 *
 * Shapes are one array tile wide except F, which is two, so the hardware tile
 * loop runs in both directions across the pair (ntiles=2 then ktiles=2).
 *
 *   X  : T x D int8, row-major             arow = D
 *   W1 : D x F int4, row-major 4-bit       wrow = F*4/8
 *   W2 : F x D int4, row-major 4-bit       wrow = D*4/8
 *
 * Operands and the golden result come from ../../tpulang/fw_vectors.py, which
 * runs this kernel's own command trace through iss.py.
 */
#include "tpu.h"

#define ROWS 8                  /* MXU array geometry — fixed by the bitstream */
#define COLS 8

#define T 8                     /* tokens */
#define D 8                     /* model width   (one tile) */
#define F 16                    /* hidden width  (two tiles) */

#define AROW_X  D
#define AROW_HR F
#define WROW_1  ((F * 4) / 8)   /* W1 is D x F: a row is F nibbles */
#define WROW_2  ((D * 4) / 8)   /* W2 is F x D: a row is D nibbles */
#define CROW_H  (F * 4)
#define CROW_Y  (D * 4)

/* {m0, n} literals: m0 in the low 12 bits, n above. The shifts keep results
 * inside the int4 grid rather than pinned at the clip. Tuned, not guessed: the
 * accumulators here only reach ~56, and a larger shift collapses the whole
 * result to zero — which would pass against any datapath at all. These give 13
 * distinct values in H and 13 in Y, spanning [-8, 7]. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_H  RQ(1u, 3u)        /* X @ W1  -> H  */
#define RQ_HR RQ(1u, 0u)        /* relu(H) -> HR : identity, HR shares H's scale */
#define RQ_Y  RQ(1u, 3u)        /* HR @ W2 -> Y  */

/* Same address in DRAM and scratchpad, as in every kernel here. */
#define X_ADDR   0x0000u
#define W1_ADDR  0x0400u
#define W2_ADDR  0x0800u
#define H_ADDR   0x1000u        /* [T][F] int8  — matmul.rq output          */
#define HR32_ADDR 0x1400u       /* [T][F] int32 — relu output (VPU is wide) */
#define HR_ADDR  0x1C00u        /* [T][F] int8  — requantized back down     */
#define Y_ADDR   0x2000u        /* [T][D] int8  — the result                */

#define X_BYTES  (T * D)
#define W1_BYTES (D * WROW_1)
#define W2_BYTES (F * WROW_2)
#define Y_BYTES  (T * D)

int main(void)
{
    /* operands in */
    tpu_dma(X_ADDR,  X_ADDR,  X_BYTES,  TPU_DMA_FILL);
    tpu_dma(W1_ADDR, W1_ADDR, W1_BYTES, TPU_DMA_FILL);
    tpu_dma(W2_ADDR, W2_ADDR, W2_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* H = requant(X @ W1). ntiles=2: F is two array widths. */
    tpu_mxu_geom(AROW_X, CROW_H, WROW_1, 1u, F / COLS, T);
    tpu_mxu_mm(H_ADDR, X_ADDR, W1_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_H);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* HR = requant(relu(H)). Two ops because relu widens int8 -> int32 and only
     * an explicit requant narrows back (vpu.md) — the same two-step the .tpu
     * kernel used, and the reason RQ_HR is the {1,0} identity. */
    tpu_vpu(TPU_V_RELU,    HR32_ADDR, H_ADDR,    0u, T * F, 0u);
    tpu_vpu(TPU_V_REQUANT, HR_ADDR,   HR32_ADDR, 0u, T * F, RQ_HR);
    tpu_wait(TPU_U_VPU);

    /* Y = requant(HR @ W2). ktiles=2 now: the contraction is the wide axis. */
    tpu_mxu_geom(AROW_HR, CROW_Y, WROW_2, F / ROWS, 1u, T);
    tpu_mxu_mm(Y_ADDR, HR_ADDR, W2_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_Y);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(Y_ADDR, Y_ADDR, Y_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
