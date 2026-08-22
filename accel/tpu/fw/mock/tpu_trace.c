/* tpu_trace.c — the host-side implementation of tpu.h's two primitives.
 *
 * Link this against any firmware kernel with -DTPU_TRACE and the host compiler,
 * and running the result prints the kernel's command trace instead of executing
 * it. The kernel source is unmodified and unconditionally compiled: this is the
 * whole of what replaces the RV32IM interpreter phase 3 originally called for
 * (docs/picorv32_migration.md §8.1).
 *
 *   cc -DTPU_TRACE -I.. mock/tpu_trace.c matmul.c -o matmul.trace
 *   ./matmul.trace > trace.txt
 *
 * Format, one record per line, all hex, consumed by
 * accel/tpulang/fw_vectors.py:
 *
 *   CMD  <unit> <w0> <w1> <w2> <w3>     one 128-bit macro-op
 *   WAIT <unit>                          a producer barrier on that unit
 *
 * WAIT carries no hardware effect — it is a spin on the retired counter — but it
 * is recorded because cross-unit ordering is software's job (tpu.h header), so a
 * missing barrier is a real firmware bug and the trace is where it is visible.
 * The ISS executes commands in trace order, which is the *strongest* ordering
 * any correct barrier placement can produce, so a trace that needs a barrier it
 * does not have still yields correct golden images here and diverges on the RTL.
 * That asymmetry is deliberate: the images stay a statement about intent, and
 * the RTL run is what tests the ordering.
 */
#include <stdint.h>
#include <stdio.h>

void tpu_trace_push(unsigned unit, uint32_t w0, uint32_t w1,
                    uint32_t w2, uint32_t w3)
{
    printf("CMD %u %08x %08x %08x %08x\n", unit, w0, w1, w2, w3);
}

void tpu_trace_wait(unsigned unit)
{
    printf("WAIT %u\n", unit);
}
