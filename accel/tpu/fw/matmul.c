/* matmul.c — C[M][N] = A[M][K] @ W[K][N], the whole contraction in one dispatch.
 *
 * The C counterpart of ../../tpulang/examples/tiled_matmul_hw.tpu: same problem,
 * same layout, same addresses, but the commands come from PicoRV32 firmware
 * instead of the scalar unit. Operands are staged in and the result out by the
 * DMA, so the host only ever touches DRAM.
 *
 *   A : M x K int8, row-major            arow = K
 *   W : K x N int4, ROW-major 4-bit        wrow = N*4/8
 *   C : M x N int32                      crow = N*4
 *
 * The MXU walks the 4x2 tile grid itself (n outer, k inner), so the int32
 * partials stay in its result buffer and reach the scratchpad once per output
 * tile. Geometry must match host/run_fw_matmul.py, which builds the operands
 * and checks the result.
 */
#include "tpu.h"

#define ROWS 8                  /* MXU array geometry — fixed by the bitstream */
#define COLS 8

/* Shape. Overridable from the Makefile (`make M=8 KTILES=4 NTILES=2`) so the
 * sweep in ../tb/run_fw_sweep.sh can walk it; the testbench and the host script
 * take the same three numbers. M <= 32 (mxu.sv MAX_TOKENS, and tlen is 6 bits). */
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
#define WROW ((N * 4) / 8)      /* W row stride, bytes (row-major int4) */
#define CROW (N * 4)            /* C row stride, bytes (int32) */

#define A_BYTES (M * K)
#define W_BYTES (K * WROW)
#define C_BYTES (M * N * 4)

/* Same value in DRAM and in the scratchpad, as in the .tpu examples. The bases
 * are spaced for shapes well past the default: A <= 8 KB, W <= 8 KB, C <= 48 KB
 * of the 64 KB scratchpad. */
#define A_ADDR 0x0000u
#define W_ADDR 0x2000u
#define C_ADDR 0x4000u

int main(void)
{
    /* operands in */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_FILL);
    tpu_dma(W_ADDR, W_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* the MXU queue is not ordered against the DMA's */

    /* the contraction */
    tpu_mxu_geom(AROW, CROW, WROW, KTILES, NTILES, M);
    tpu_mxu_mm(C_ADDR, A_ADDR, W_ADDR, TPU_MM_TILED, 0u);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(C_ADDR, C_ADDR, C_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
