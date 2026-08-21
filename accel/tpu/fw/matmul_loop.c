/* matmul_loop.c — the same C[M][N] = A[M][K] @ W[K][N] as matmul.c, with the
 * tile grid walked in firmware instead of by the MXU.
 *
 * matmul.c sets KTILES/NTILES and issues ONE TPU_MM_TILED matmul: the array
 * walks the 4x2 grid itself, so the int32 partials stay in its result buffer
 * for a whole contraction and reach the scratchpad once per output tile. Here
 * the grid is a C `for` pair and each of the 8 tiles is its own MXU_MM, so the
 * partials round-trip through the scratchpad between contraction tiles —
 * k == 0 initialises the int32 tile, k > 0 adds into it with TPU_MM_ACC. That
 * is the same flag dance ../../tpulang/examples/tiled_matmul.tpu runs on the
 * scalar unit, and the C traffic the hardware loop was built to delete
 * (docs/macro_ops.md §4.2).
 *
 * A requantized result would put TPU_MM_RQ on the k == KTILES-1 dispatch only,
 * since the narrow has to see the finished sum. Not done here: matmul.c leaves
 * C int32 and this must match it byte for byte.
 *
 * Layout, addresses and result are identical to matmul.c, so
 * host/run_fw_matmul.py checks this one unchanged:
 *
 *     make -C accel/tpu/fw PROG=matmul_loop
 *     make -C accel/tpu/fw run PROG=matmul_loop PORT=COM5
 *
 * TPU_MM_TILED is still set on every dispatch, with both tile counts at 1. The
 * flag selects the configured strides over the single-tile constants
 * (rtl/mxu.sv, `act_row_stride_sel`); at a count of 1 the hardware's tile loop
 * runs exactly one pass, so nothing here is hardware-managed. Dropping the flag
 * would force arow = ROWS and crow = COLS*4 — a dense, tile-shaped operand,
 * which is the other way to write this: DMA each tile into a small fixed buffer
 * as tiled_matmul.tpu does, at one fill per tile. Keeping A, W and C resident
 * and flat is what makes this comparable to matmul.c.
 *
 *   A : M x K int8, row-major            arow = K
 *   W : K x N trits, col-major 2-bit      wcol = K*2/8
 *   C : M x N int32                      crow = N*4
 */
#include "tpu.h"

#define ROWS 8                  /* MXU array geometry — fixed by the bitstream */
#define COLS 8

/* Shape, overridable exactly as in matmul.c — the two must be built with the
 * same three numbers to be comparable. M <= 32 (mxu.sv MAX_TOKENS). */
#ifndef M
#define M      8                /* token rows */
#endif
#ifndef KTILES
#define KTILES 4
#endif
#ifndef NTILES
#define NTILES 2
#endif

#define K (KTILES * ROWS)
#define N (NTILES * COLS)

#define AROW K                  /* A row stride, bytes */
#define WCOL ((K * 2) / 8)      /* W column stride, bytes */
#define CROW (N * 4)            /* C row stride, bytes (int32) */

/* Per-tile steps. The hardware derives these from the tile indices; with the
 * counts pinned at 1 they are the loop's own address arithmetic. */
#define AKSTEP ROWS             /* 8  — one k tile along A's row            */
#define WKSTEP ((ROWS * 2) / 8) /* 2  — one k tile down a weight column     */
#define WNSTEP (COLS * WCOL)    /* one n tile of whole weight columns       */
#define CNSTEP (COLS * 4)       /* 32 — one n tile of int32 C               */

#define A_BYTES (M * K)
#define W_BYTES (N * WCOL)
#define C_BYTES (M * N * 4)

/* Same value in DRAM and in the scratchpad, as in the .tpu examples. */
#define A_ADDR 0x0000u
#define W_ADDR 0x2000u
#define C_ADDR 0x4000u

int main(void)
{
    unsigned n, k;

    /* operands in */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_FILL);
    tpu_dma(W_ADDR, W_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* the MXU queue is not ordered against the DMA's */

    /* Strides describe the whole matrix; the counts describe one array pass. */
    tpu_mxu_geom(AROW, CROW, WCOL, 1u, 1u, M);

    /* n outer, k inner — the order the hardware loop uses, so the two paths
     * differ only in who walks the grid. No fence inside: one unit's queue is
     * in-order, so the k > 0 read-modify-write of C cannot pass the k == 0
     * store. Past the queue's 8 entries the ninth push stalls the CPU inside
     * the store, which is why there is no software flow control here either. */
    for (n = 0; n < NTILES; n++)
        for (k = 0; k < KTILES; k++)
            tpu_mxu_mm(C_ADDR + n * CNSTEP,
                       A_ADDR + k * AKSTEP,
                       W_ADDR + n * WNSTEP + k * WKSTEP,
                       TPU_MM_TILED | (k ? TPU_MM_ACC : 0u),
                       0u);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(C_ADDR, C_ADDR, C_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
