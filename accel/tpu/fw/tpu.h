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

/* ---- The two primitives, and the only target-specific code in the firmware --
 *
 * Everything below this point is plain uint32_t arithmetic. That is what lets
 * -DTPU_TRACE compile the *same* kernel source with the host compiler and have
 * it emit its command trace instead of executing it — see
 * docs/picorv32_migration.md §8.1. Because only these two are swapped, no
 * builder is ever duplicated, so the trace producer cannot drift from the
 * firmware. Add new builders below in terms of tpu_push and they are traced for
 * free.
 */
#ifdef TPU_TRACE

/* Implemented by mock/tpu_trace.c. Declared, not defined, so a builder that
 * forgets to go through tpu_push fails to link rather than silently escaping
 * the trace. */
void tpu_trace_push(unsigned unit, uint32_t w0, uint32_t w1,
                    uint32_t w2, uint32_t w3);
void tpu_trace_wait(unsigned unit);

static inline void tpu_push(unsigned unit, uint32_t w0, uint32_t w1,
                            uint32_t w2, uint32_t w3)
{
    tpu_trace_push(unit, w0, w1, w2, w3);
}

static inline void tpu_wait(unsigned unit)
{
    tpu_trace_wait(unit);
}

#else

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

#endif /* TPU_TRACE */

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
 * N from crow/4), wrow = N*4/8 (weights are row-major int4). tlen is the
 * token-row count M (6 bits). */
static inline void tpu_mxu_geom(uint32_t arow, uint32_t crow, uint32_t wrow,
                                uint32_t ktiles, uint32_t ntiles, uint32_t tlen)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_GEOM | (arow << 16),
             crow | (wrow << 16),
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

/* ---- DMA, transposing (dma.sv §5) --------------------------------------- */

/* The `.t` mode: the byte *count* is unchanged, but the source is read
 * row-major over `tcols` columns at `tsrow` stride while the destination is
 * written transposed at `tdrow`. `tcols == 0` degenerates to one row, which is
 * a strided scatter/gather rather than a fault (dma.sv's zero-means-unset
 * fallbacks). This is what builds a weight operand out of an activation the
 * array produced column-wise. */
static inline void tpu_dma_t(uint32_t spad, uint32_t dram, uint32_t len,
                             unsigned dir, uint32_t tcols, uint32_t tsrow,
                             uint32_t tdrow)
{
    tpu_push(TPU_U_DMA,
             TPU_DMA_MOVE | ((uint32_t)dir << 8) | (1u << 9) | (spad << 16),
             dram,
             len | (tcols << 16),
             tsrow | (tdrow << 16));
}

/* ---- VPU (cmd_vpu.sv) --------------------------------------------------- */

#define TPU_VPU_OP   0x01u
#define TPU_VPU_GEOM 0x02u

/* vpu.sv VOP_*. The opcode the *unit* decodes, not the tpulang mnemonic. */
#define TPU_V_DOT     0u
#define TPU_V_ADD     1u
#define TPU_V_RELU    3u
#define TPU_V_REQUANT 10u
#define TPU_V_VECMM   13u
#define TPU_V_DYT     16u
#define TPU_V_QUANT4  17u

/* One vector op over `vlen` elements. `rq_word` is the {m0,n} literal and is
 * read only by REQUANT / DYT / QUANT4; the rest ignore it.
 *
 * Element widths are per-op and are the caller's business (vpu.md): ADD/RELU
 * read int8 and write int32, REQUANT/DYT read int32 and write int8, and QUANT4
 * reads int8 and writes 4 bits — so its destination advances **half** as fast
 * as its source and `vlen` must be even. */
static inline void tpu_vpu(unsigned vop, uint32_t dst, uint32_t src0,
                           uint32_t src1, uint32_t vlen, uint32_t rq_word)
{
    tpu_push(TPU_U_VPU,
             TPU_VPU_OP | (vop << 8) | (dst << 16),
             src0 | (src1 << 16),
             vlen | (rq_word << 16),
             0u);
}

/* Geometry for the vecmatmul macro op: rows x cols pairs, with per-operand row
 * strides. Sticky in this unit's queue exactly as MXU_GEOM is. */
static inline void tpu_vpu_geom(uint32_t rows, uint32_t cols, uint32_t row0,
                                uint32_t row1, uint32_t crow)
{
    tpu_push(TPU_U_VPU,
             TPU_VPU_GEOM | (rows << 16),
             cols | (row0 << 16),
             row1 | (crow << 16),
             0u);
}

#endif /* TPU_H */
