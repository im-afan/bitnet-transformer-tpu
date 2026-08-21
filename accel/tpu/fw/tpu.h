/* tpu.h — the MMIO command plane, as seen from firmware.
 *
 * Mirrors rtl/cpu_subsys.sv (the aperture) and rtl/cmd_{mxu,dma}.sv (the
 * command fields). Nothing here is a driver abstraction: each builder packs one
 * 128-bit macro-op exactly as the decoder reads it, and the constants are
 * duplicated from the RTL by house convention (see cmd_mxu.sv's note).
 *
 * Address map (cpu_subsys.sv):
 *   0x8000_0000 + 0x10*unit   4-word command aperture, COMMIT ON WORD 3
 *   0x8000_0040               status block: issued/retired/level per unit
 *   0x8000_0070               write = done; read = unit_idle
 *
 * A full queue withholds the write response on word 3, so the CPU stalls inside
 * the store and flow control needs no software. What software *does* own is
 * cross-unit ordering: the three units have independent queues, so anything the
 * MXU reads must be waited for with tpu_wait(TPU_U_DMA) first.
 */
#ifndef TPU_H
#define TPU_H

#include <stdint.h>

#define TPU_MMIO 0x80000000u

#define TPU_U_MXU 0u
#define TPU_U_VPU 1u
#define TPU_U_DMA 2u

#define TPU_CMD(unit)  ((volatile uint32_t *)(TPU_MMIO + 0x10u * (unit)))
#define TPU_STAT(i)    (*(volatile uint32_t *)(TPU_MMIO + 0x40u + 4u * (i)))

/* Per unit: issued, retired, queue level. Four words apart, in unit order. */
#define TPU_ISSUED(unit)  TPU_STAT(4u * (unit) + 0u)
#define TPU_RETIRED(unit) TPU_STAT(4u * (unit) + 1u)
#define TPU_LEVEL(unit)   TPU_STAT(4u * (unit) + 2u)

#define TPU_DONE (*(volatile uint32_t *)(TPU_MMIO + 0x70u))

/* Push one command. The store to word 3 assembles the 128 bits and enqueues, so
 * a torn command cannot be seen by the unit. */
static inline void tpu_push(unsigned unit, uint32_t w0, uint32_t w1,
                            uint32_t w2, uint32_t w3)
{
    volatile uint32_t *port = TPU_CMD(unit);
    port[0] = w0;
    port[1] = w1;
    port[2] = w2;
    port[3] = w3;
}

/* Block until everything pushed to `unit` has retired. The counters free-run
 * across runs (only rst_n clears them), so compare as a signed difference
 * rather than against a fixed sequence number. */
static inline void tpu_wait(unsigned unit)
{
    uint32_t issued = TPU_ISSUED(unit);

    while ((int32_t)(issued - TPU_RETIRED(unit)) > 0)
        ;
}

/* ---- DMA (cmd_dma.sv) --------------------------------------------------- */

#define TPU_DMA_MOVE  0x01u
#define TPU_DMA_FILL  0u        /* DRAM -> scratchpad */
#define TPU_DMA_SPILL 1u        /* scratchpad -> DRAM */

/* `len` bytes between scratchpad `spad` and 19-bit DRAM `dram`. Linear only:
 * the transpose geometry words are left zero. */
static inline void tpu_dma(uint32_t spad, uint32_t dram, uint32_t len,
                           unsigned dir)
{
    tpu_push(TPU_U_DMA,
             TPU_DMA_MOVE | ((uint32_t)dir << 8) | (spad << 16),
             dram,
             len,
             0u);
}

/* ---- MXU (cmd_mxu.sv) --------------------------------------------------- */

#define TPU_MXU_GEOM 0x01u
#define TPU_MXU_MM   0x02u

#define TPU_MM_ACC   (1u << 8)   /* add into the existing int32 C */
#define TPU_MM_RQ    (1u << 9)   /* narrow the store to int8 via {m0,n} */
#define TPU_MM_TILED (1u << 10)  /* use the strides and tile counts below */

/* Operand geometry for the matmuls that follow it in this unit's queue.
 * Strides are bytes: arow = K, crow = N*4 (int32; a requantized store derives
 * N from crow/4), wcol = K*2/8. tlen is the token-row count M (6 bits). */
static inline void tpu_mxu_geom(uint32_t arow, uint32_t crow, uint32_t wcol,
                                uint32_t ktiles, uint32_t ntiles, uint32_t tlen)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_GEOM | (arow << 16),
             crow | (wcol << 16),
             ktiles | (ntiles << 8) | (tlen << 16),
             0u);
}

/* One matmul: C = A @ W over the geometry above. `rq_word` is the {m0,n}
 * literal, used only with TPU_MM_RQ. */
static inline void tpu_mxu_mm(uint32_t out, uint32_t act, uint32_t wgt,
                              uint32_t flags, uint32_t rq_word)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_MM | flags | (out << 16),
             act | (wgt << 16),
             rq_word,
             0u);
}

#endif /* TPU_H */
