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
#   make uart                          default program (examples/relu_layer.tpu)
#   make uart PROG=examples/foo.tpu    your own program (path relative to tpulang/)
# All examples are self-contained over DRAM (rdmem inputs in, wrmem outputs out);
# a custom program must also match gen_vectors.py's geometry/address map.
#
# UART echo self-test (docs/uart_selftest.md) — the loopback core that goes into
# the cmod_a7_echo bitstream, driven full duplex:
#   make echo
#
# UART + external SRAM, no TPU core — the cmod_a7_mem bitstream's core, driven
# with real 'W'/'R' command frames against a behavioral SRAM chip model:
#   make mem

TEST ?= mxu

TOP := $(TEST)_tb
TB  := $(TEST)_tb.sv

# Tests needing more than rtl/<name>.sv declare it here as RTL_<name>, so
# `make TEST=<name>` works without spelling the list out on the command line.
# An explicit RTL="..." on the command line still wins over both.
RTL_uart_echo := ../rtl/uart_echo.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
RTL_uart_memory := ../rtl/uart_memory.sv ../rtl/uart_interface.sv \
                   ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/sram.sv

RTL ?= $(if $(RTL_$(TEST)),$(RTL_$(TEST)),../rtl/$(TEST).sv)

IVERILOG := iverilog -g2012 -Wall
VVP      := vvp

# --- UART end-to-end wiring ---------------------------------------------------
PROG     ?= examples/relu_layer.tpu
TPULANG  := ../../tpulang
UART_TOP := tpu_top_uart_tb
UART_RTL := ../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv \
            ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv \
            ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv

# Every test discoverable from the *_tb.sv files present.
TESTS := $(patsubst %_tb.sv,%,$(wildcard *_tb.sv))

.PHONY: sim wave list all clean uart echo mem

sim: $(TB) $(RTL)
	$(IVERILOG) -o $(TOP).vvp $(RTL) $(TB)
	$(VVP) $(TOP).vvp

# Assemble PROG + generate golden vectors (into vectors_uart/, kept separate from
# the backdoor tpu_top_tb's vectors/), then run the UART-driven end-to-end TB.
uart: $(UART_TOP).sv $(UART_RTL)
	python $(TPULANG)/gen_vectors.py -p $(TPULANG)/$(PROG) -o vectors_uart
	$(IVERILOG) -o $(UART_TOP).vvp $(UART_RTL) $(UART_TOP).sv
	$(VVP) $(UART_TOP).vvp

# Alias for `make TEST=uart_echo sim` (RTL_uart_echo above supplies the sources).
echo:
	@$(MAKE) --no-print-directory TEST=uart_echo sim

# Alias for `make TEST=uart_memory sim`.
mem:
	@$(MAKE) --no-print-directory TEST=uart_memory sim

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
