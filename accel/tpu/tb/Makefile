# Generic testbench driver for accel/tpu (Icarus Verilog, -g2012).
#
# Each test <name> pairs tb/<name>_tb.sv with rtl/<name>.sv. Pick one with TEST:
#
#   make TEST=mxu           compile + run mxu_tb
#   make TEST=vpu sim       same (sim is the default target)
#   make TEST=mxu wave      run and remind you where the VCD is (tb/mxu_tb.vcd)
#   make list               show every test that has a tb/<name>_tb.sv
#   make all                run every discovered test in turn
#   make clean               remove all *_tb.vvp / *_tb.vcd
#
# Multi-file tests: override RTL to pass extra sources, e.g.
#   make TEST=mxu RTL="../rtl/mxu.sv ../rtl/requant.sv"
#
# UART end-to-end (one command: assemble a tpulang program, generate its golden
# vectors, then drive the whole thing into the DUT over the UART pins and check):
#   make uart                          default program (examples/relu_layer_dma.tpu)
#   make uart PROG=examples/foo.tpu    your own program (path relative to tpulang/)
# The program must be self-contained over DRAM (rdmem inputs in, wrmem outputs out)
# and match gen_vectors.py's geometry/address map — relu_layer_dma.tpu is the template.

TEST ?= mxu

TOP := $(TEST)_tb
TB  := $(TEST)_tb.sv
RTL ?= ../rtl/$(TEST).sv

IVERILOG := iverilog -g2012 -Wall
VVP      := vvp

# --- UART end-to-end wiring ---------------------------------------------------
PROG     ?= examples/relu_layer_dma.tpu
TPULANG  := ../../tpulang
UART_TOP := tpu_top_uart_tb
UART_RTL := ../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv \
            ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv \
            ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv

# Every test discoverable from the *_tb.sv files present.
TESTS := $(patsubst %_tb.sv,%,$(wildcard *_tb.sv))

.PHONY: sim wave list all clean uart

sim: $(TB) $(RTL)
	$(IVERILOG) -o $(TOP).vvp $(RTL) $(TB)
	$(VVP) $(TOP).vvp

# Assemble PROG + generate golden vectors (into vectors_uart/, kept separate from
# the backdoor tpu_top_tb's vectors/), then run the UART-driven end-to-end TB.
uart: $(UART_TOP).sv $(UART_RTL)
	python $(TPULANG)/gen_vectors.py -p $(TPULANG)/$(PROG) -o vectors_uart
	$(IVERILOG) -o $(UART_TOP).vvp $(UART_RTL) $(UART_TOP).sv
	$(VVP) $(UART_TOP).vvp

wave: sim
	@echo "open $(TOP).vcd in gtkwave"

list:
	@echo "available tests:"
	@for t in $(TESTS); do echo "  make TEST=$$t"; done

all:
	@for t in $(TESTS); do \
		echo "=== running $$t ==="; \
		$(MAKE) --no-print-directory TEST=$$t sim || exit $$?; \
	done

clean:
	rm -f *_tb.vvp *_tb.vcd
