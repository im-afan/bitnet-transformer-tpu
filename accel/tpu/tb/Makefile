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
#   make uart                          default pair (relu_layer then vector_add)
#   make uart PROG=examples/foo.tpu PROG2=examples/bar.tpu
# Two programs, because the TB runs both back to back over one link with no
# reset between them -- the case that used to leave the second one running the
# first one's instructions (see the note in scalar_unit.sv's imem/cfg gating).
# All examples are self-contained over DRAM (rdmem inputs in, wrmem outputs out);
# a custom program must also match gen_vectors.py's geometry/address map.
#
# UART echo self-test (docs/uart_selftest.md) — the block-loopback core that goes
# into the cmod_a7_echo bitstream: 64 bytes in, the same 64 bytes back:
#   make echo
#
# UART + external SRAM, no TPU core — the cmod_a7_mem bitstream's core, driven
# with real 'W'/'R' command frames against a behavioral SRAM chip model:
#   make mem
#
# The same protocol with the external memory removed as well — the cmod_a7_bram
# bitstream's core, backed by on-chip block RAM (rtl/bram.sv):
#   make bram
#
# The cmod_a7_mem image again, but driven by host/test_uart_link.py itself
# instead of by hand-written stimulus — the real host driver, the real RTL, no
# board (tb/uart_memory_cosim_tb.sv + host/sim_link.py):
#   make cosim
#   make cosim COSIM_ARGS="--only sram_roundtrip --sim-fsm"

TEST ?= mxu

TOP := $(TEST)_tb
TB  := $(TEST)_tb.sv

# Tests needing more than rtl/<name>.sv declare it here as RTL_<name>, so
# `make TEST=<name>` works without spelling the list out on the command line.
# An explicit RTL="..." on the command line still wins over both.
RTL_uart_echo := ../rtl/uart_echo.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
RTL_uart_interface := ../rtl/uart_interface.sv ../rtl/uart_receiver.sv \
                      ../rtl/uart_transmitter.sv ../rtl/sram.sv
RTL_uart_memory := ../rtl/uart_memory.sv ../rtl/uart_interface.sv \
                   ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/sram.sv
RTL_uart_bram := ../rtl/uart_bram.sv ../rtl/uart_interface.sv \
                 ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/bram.sv
# dma_tb instantiates the DUT against the real scratchpad and a behavioral SRAM
# chip, so it needs all three (without this `make all` cannot elaborate it).
RTL_dma := ../rtl/dma.sv ../rtl/scratchpad.sv ../rtl/sram.sv

RTL ?= $(if $(RTL_$(TEST)),$(RTL_$(TEST)),../rtl/$(TEST).sv)

IVERILOG := iverilog -g2012 -Wall
VVP      := vvp

# --- UART end-to-end wiring ---------------------------------------------------
PROG     ?= examples/relu_layer.tpu
PROG2    ?= examples/vector_add.tpu
TPULANG  := ../../tpulang
UART_TOP := tpu_top_uart_tb
UART_RTL := ../rtl/tpu_top.sv ../rtl/scalar_unit.sv ../rtl/mxu.sv ../rtl/vpu.sv \
            ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv \
            ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv \
            ../rtl/perf_counters.sv

# Both tpu_top testbenches need the whole core. Declared here (rather than up with
# the other RTL_<name> lines) so the file list is written once: the backdoor TB
# does not drive the UART, but tpu_top instantiates the UART blocks either way and
# will not elaborate without them.
RTL_tpu_top      := $(UART_RTL)
RTL_tpu_top_uart := $(UART_RTL)

# Every test discoverable from the *_tb.sv files present. uart_memory_cosim is
# excluded: it has no rtl/<name>.sv and cannot run without host/sim_link.py
# driving it, so `make all` would only fail on it. Use `make cosim`.
TESTS := $(filter-out uart_memory_cosim,$(patsubst %_tb.sv,%,$(wildcard *_tb.sv)))

.PHONY: sim wave list all clean uart echo mem bram cosim

sim: $(TB) $(RTL)
	$(IVERILOG) -o $(TOP).vvp $(RTL) $(TB)
	$(VVP) $(TOP).vvp

# Assemble PROG and PROG2 + generate golden vectors (into vectors_uart/ and
# vectors_uart2/, kept separate from the backdoor tpu_top_tb's vectors/), then
# run the UART-driven end-to-end TB. It loads and runs *both* programs over one
# link with no reset in between, which is what run_program.py does when invoked
# twice with different --program arguments -- PROG2 must be a different program
# from PROG for that to mean anything.
uart: $(UART_TOP).sv $(UART_RTL)
	python $(TPULANG)/gen_vectors.py -p $(TPULANG)/$(PROG) -o vectors_uart
	python $(TPULANG)/gen_vectors.py -p $(TPULANG)/$(PROG2) -o vectors_uart2
	$(IVERILOG) -o $(UART_TOP).vvp $(UART_RTL) $(UART_TOP).sv
	$(VVP) $(UART_TOP).vvp

# Alias for `make TEST=uart_echo sim` (RTL_uart_echo above supplies the sources).
echo:
	@$(MAKE) --no-print-directory TEST=uart_echo sim

# Alias for `make TEST=uart_memory sim`.
mem:
	@$(MAKE) --no-print-directory TEST=uart_memory sim

# Alias for `make TEST=uart_bram sim`.
bram:
	@$(MAKE) --no-print-directory TEST=uart_bram sim

# The Python host driving the RTL directly. sim_link.py compiles the testbench
# itself (it has to pass +plusargs and own the process), so there is no iverilog
# line here -- this target only exists so the entry point is where the others are.
COSIM_ARGS ?=
cosim:
	python ../host/test_uart_link.py --sim $(COSIM_ARGS)

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
