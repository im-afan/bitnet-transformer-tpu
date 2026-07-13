# VPU testbench (accel/tpu/tb/vpu_tb.sv against ../rtl/vpu.sv)
#
#   make sim    compile + run with Icarus Verilog (iverilog -g2012)
#   make wave   same, dumping vpu_tb.vcd for GTKWave
#   make clean

TOP := vpu_tb
RTL := ../rtl/vpu.sv
TB  := vpu_tb.sv

.PHONY: sim wave clean

sim:
	iverilog -g2012 -Wall -o $(TOP).vvp $(RTL) $(TB)
	vvp $(TOP).vvp

wave: sim
	@echo "open $(TOP).vcd in gtkwave"

clean:
	rm -f $(TOP).vvp $(TOP).vcd
