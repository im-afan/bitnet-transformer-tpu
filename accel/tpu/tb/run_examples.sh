#!/bin/sh
# run_examples.sh — every example kernel through the whole core, checked against
# its own golden vectors.
#
# This is the ISA regression. Between them the examples cover every path a
# dispatch can take through the command plane (docs/picorv32_migration.md):
# single-tile `matmul`, software tiling, the hardware tile loop (`matmul_t`) and
# its config strides, the VPU macro op `vecmatmul` and its geometry command,
# runtime config via `setcfgr`, the transposing DMA, and DRAM above 64 KB.
#
# Vectors are regenerated on every run, so this also checks that assembler.py,
# iss.py and the RTL still agree — a kernel that passes here is byte-identical
# to what the golden simulator produced.
#
#   ./run_examples.sh                       all of them
#   ./run_examples.sh relu_layer vecmatmul  just these
#
# Driven by `make examples` from this directory.

set -u

TPULANG=../../tpulang
RTL="../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv
     ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv
     ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
     ../rtl/perf_counters.sv
     ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv
     ../rtl/cpu_subsys.sv ../rtl/vendor/picorv32.v"

WATCHDOG=${EX_WATCHDOG:-60000000}

if [ $# -gt 0 ]; then
    PROGS="$*"
else
    PROGS="vector_add relu_layer tiled_matmul tiled_matmul_hw strided_matmul
           vecmatmul setcfgr transpose_dma highmem_dma vpu_matmul"
fi

pass=0
fail=0

for p in $PROGS; do
    if ! python "$TPULANG/gen_vectors.py" -p "$TPULANG/examples/$p.tpu" \
                -o "vec_$p" >/dev/null 2>&1; then
        printf '%-18s VECTOR GENERATION FAILED\n' "$p"
        fail=$((fail + 1))
        continue
    fi

    iverilog -g2012 -DWATCHDOG_NS="$WATCHDOG" -DNO_VCD \
        -DPROG_FILE="\"vec_$p/tpu_prog.hex\"" \
        -DSPAD_IN_FILE="\"vec_$p/tpu_spad_in.hex\"" \
        -DSPAD_EXP_FILE="\"vec_$p/tpu_spad_exp.hex\"" \
        -o "ex_$p.vvp" $RTL tpu_top_tb.sv 2>/dev/null

    r=$(vvp "ex_$p.vvp" 2>&1 | grep -E "ALL TESTS PASSED|FAILURES|TIMEOUT" | tail -1)
    printf '%-18s %s\n' "$p" "$r"

    case "$r" in
        *"ALL TESTS PASSED"*) pass=$((pass + 1)) ;;
        *)                    fail=$((fail + 1)) ;;
    esac
done

echo "--- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
