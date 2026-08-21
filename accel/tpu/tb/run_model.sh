#!/bin/sh
# run_model.sh — the whole adder model kernel through the core, at a chosen depth.
#
# `adder_model.tpu` is the shipped model, and `.equ LAYERS` is the only thing
# that moves with depth (gen_vectors.py reads it out of the program's own table),
# so a one-layer copy exercises every code path the four-layer one does at a
# quarter of the clocks. That makes it the edit-run loop for anything touching
# the datapath or the dispatch plane:
#
#   ./run_model.sh 1     ~182 k clocks   (a few minutes)
#   ./run_model.sh 4      ~690 k clocks   (the shipped kernel; long)
#
# Driven by `make model` / `make model LAYERS=4`. Pass +HEARTBEAT through
# HB=1 to watch a long run make progress — a frozen PC with a unit stuck busy is
# a hang, a moving one is just slow.

set -u

LAYERS=${1:-1}
HB=${HB:-0}
TPULANG=../../tpulang
SRC="$TPULANG/examples/adder_model.tpu"
GEN="$TPULANG/examples/out/adder_${LAYERS}layer.tpu"
VEC="vec_model$LAYERS"

RTL="../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv
     ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv
     ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
     ../rtl/perf_counters.sv
     ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv
     ../rtl/cpu_subsys.sv ../rtl/vendor/picorv32.v"

mkdir -p "$TPULANG/examples/out"
sed "s/^\.equ LAYERS   [0-9]*/.equ LAYERS   $LAYERS/" "$SRC" > "$GEN"

echo "== adder_model at layers=$LAYERS =="
python "$TPULANG/gen_vectors.py" -p "$GEN" -o "$VEC" | tail -3

iverilog -g2012 -DWATCHDOG_NS=400000000 -DNO_VCD \
    -DPROG_FILE="\"$VEC/tpu_prog.hex\"" \
    -DSPAD_IN_FILE="\"$VEC/tpu_spad_in.hex\"" \
    -DSPAD_EXP_FILE="\"$VEC/tpu_spad_exp.hex\"" \
    -o "model$LAYERS.vvp" $RTL tpu_top_tb.sv 2>/dev/null

if [ "$HB" = "0" ]; then
    vvp "model$LAYERS.vvp"
else
    vvp "model$LAYERS.vvp" +HEARTBEAT "+HEARTBEAT_CLK=${HB}"
fi
