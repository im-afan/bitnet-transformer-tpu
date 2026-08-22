#!/bin/sh
# run_fw_sweep.sh — how CPU issue overhead scales with the problem, for both
# tiling strategies.
#
# ../fw/matmul.c issues a constant 2 MXU commands whatever the shape (the array
# walks the tile grid). ../fw/matmul_loop.c issues 1 + KTILES*NTILES. This walks
# a range of shapes, builds both kernels at each, runs both through
# fw_matmul_tb.sv and tabulates what the CPU cost was.
#
# The metric is `idlec` (docs/picorv32_migration.md §9.5): clocks in which NO
# unit was busy, i.e. the part of the run that is nothing but the CPU. It is the
# EXPOSED overhead, not the issued overhead -- anything the CPU does while a unit
# is busy is hidden, so `idlec + mxu` is what a dispatch really costs the CPU and
# `idlec` alone is what it costs the *run*.
#
# `qfull` is deliberately not tabulated: both producers gate their own `cmd_we`
# with `!cmd_full` (scalar_unit.sv:522, cpu_subsys.sv:177), and tpu_top samples
# `p_cmd_we & p_cmd_full`, so the counter is structurally 0 and says nothing
# about whether the queue backed up.
#
#   ./run_fw_sweep.sh                    the default shape list
#   ./run_fw_sweep.sh "8 4 4" "16 4 4"   explicit "M KTILES NTILES" triples
#
# Driven by `make fwsweep`. Needs the RISC-V cross gcc, like `make fw`.

set -u

RTL="../rtl/tpu_top.sv ../rtl/mxu.sv ../rtl/vpu.sv
     ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv
     ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
     ../rtl/perf_counters.sv
     ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv
     ../rtl/cpu_subsys.sv ../rtl/vendor/picorv32.v"

WATCHDOG=${FW_WATCHDOG:-100000000}

if [ $# -gt 0 ]; then
    SHAPES="$*"
else
    # Axis 1: tile count at M=8 (1, 4, 16, 64, 256 tiles).
    # Axis 2: token rows at 4x4 tiles -- MXU work grows, dispatch count does not.
    SHAPES="8_1_1 8_2_2 8_4_4 8_8_8 8_16_16 16_4_4 32_4_4"
fi

fail=0

printf '%-12s %-12s %8s %8s %8s %8s %6s %8s\n' \
       shape kernel run mxu dma idlec cmds mxu/cmd

for s in $SHAPES; do
    m=$(echo "$s" | cut -d_ -f1)
    kt=$(echo "$s" | cut -d_ -f2)
    nt=$(echo "$s" | cut -d_ -f3)

    # Make cannot see a changed -D, so rebuild both kernels from scratch.
    make -C ../fw clean >/dev/null 2>&1
    if ! make -C ../fw M="$m" KTILES="$kt" NTILES="$nt" >/dev/null 2>&1 ||
       ! make -C ../fw PROG=matmul_loop M="$m" KTILES="$kt" NTILES="$nt" \
              >/dev/null 2>&1 ||
       ! make -C ../fw matmul.trace matmul_loop.trace \
              M="$m" KTILES="$kt" NTILES="$nt" >/dev/null 2>&1; then
        printf '%-12s BUILD FAILED\n' "${m}x$((kt*8))x$((nt*8))"
        fail=$((fail + 1))
        continue
    fi

    for p in matmul matmul_loop; do
        # Golden vectors are per shape *and* per kernel now: they come from that
        # kernel's own command trace (docs/picorv32_migration.md §8.1), and the
        # trace differs between matmul and matmul_loop even though the answer
        # does not. Regenerate before every run or the trace check compares a
        # kernel against the other one's expectations.
        mkdir -p vectors_fw
        if ! ../fw/"$p".trace > vectors_fw/"$p".trace.txt 2>/dev/null ||
           ! python ../../tpulang/fw_vectors.py -t vectors_fw/"$p".trace.txt \
                 -o vectors_fw -M "$m" --ktiles "$kt" --ntiles "$nt" \
                 >/dev/null 2>&1; then
            printf '%-12s %-12s VECTORS FAILED\n' "${m}x$((kt*8))x$((nt*8))" "$p"
            fail=$((fail + 1)); continue
        fi

        iverilog -g2012 -DWATCHDOG_NS="$WATCHDOG" \
            -DFW_HEX="\"../fw/$p.hex\"" -DFW_VEC_DIR='"vectors_fw"' \
            -DFW_M="$m" -DFW_KT="$kt" -DFW_NT="$nt" \
            -o fw_sweep.vvp $RTL fw_matmul_tb.sv 2>/dev/null

        out=$(vvp fw_sweep.vvp 2>&1)
        line=$(echo "$out" | grep '^SWEEP' | tail -1)
        case "$out" in
            *"ALL TESTS PASSED"*) ;;
            *) printf '%-12s %-12s FAILED\n' "${m}x$((kt*8))x$((nt*8))" "$p"
               fail=$((fail + 1)); continue ;;
        esac

        # SWEEP m= kt= nt= run= mxu= dma= idlec= qfull= mxucmd= dmacmd=
        eval "$(echo "$line" | sed 's/^SWEEP //; s/\([a-z]*\)=\([0-9]*\)/v_\1=\2/g')"
        # mxu/cmd is the array work one dispatch buys -- the budget the CPU's own
        # per-dispatch cost hides behind.
        mm=$((v_mxucmd - 1))                    # MXU_MM commands (one is GEOM)
        printf '%-12s %-12s %8s %8s %8s %8s %6s %8s\n' \
               "${m}x$((kt*8))x$((nt*8))" "$p" \
               "$v_run" "$v_mxu" "$v_dma" "$v_idlec" \
               "$((v_mxucmd + v_dmacmd))" "$((v_mxu / mm))"
    done
done

rm -f fw_sweep.vvp
echo "--- $fail failure(s)"
[ "$fail" -eq 0 ]
