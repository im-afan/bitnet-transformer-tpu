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
# The command front ends share one FIFO module.
RTL_cmd_queue := ../rtl/cmd_queue.sv
RTL_cmd_plane := ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv

RTL ?= $(if $(RTL_$(TEST)),$(RTL_$(TEST)),../rtl/$(TEST).sv)

# Extra iverilog flags, for tests that take a -D switch:
#   make TEST=sram IVFLAGS=-DCPA=2      the SRAM controller's slower beat
#   make TEST=tpu_top IVFLAGS="-DWATCHDOG_NS=400000000 -DNO_VCD"
IVFLAGS  ?=
IVERILOG := iverilog -g2012 -Wall $(IVFLAGS)
VVP      := vvp

# --- UART end-to-end wiring ---------------------------------------------------
TPULANG  := ../../tpulang
UART_TOP := tpu_top_uart_tb
UART_RTL := ../rtl/tpu_top.sv ../rtl/mxu.sv ../rtl/vpu.sv \
            ../rtl/scratchpad.sv ../rtl/dma.sv ../rtl/sram.sv \
            ../rtl/uart_interface.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv \
            ../rtl/perf_counters.sv \
            ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv \
            ../rtl/cpu_subsys.sv ../rtl/vendor/picorv32.v

# Both tpu_top testbenches need the whole core. Declared here (rather than up with
# the other RTL_<name> lines) so the file list is written once: the backdoor TB
# does not drive the UART, but tpu_top instantiates the UART blocks either way and
# will not elaborate without them.
RTL_cpu_smoke    := $(UART_RTL)
RTL_tpu_top      := $(UART_RTL)
RTL_tpu_top_uart := $(UART_RTL)
RTL_fw_matmul    := $(UART_RTL)

# Every test discoverable from the *_tb.sv files present. Two are excluded:
# uart_memory_cosim has no rtl/<name>.sv and cannot run without host/sim_link.py
# driving it (use `make cosim`), and fw_matmul needs a firmware image built by a
# RISC-V cross gcc (use `make fw`). `make all` would only fail on either.
TESTS := $(filter-out uart_memory_cosim fw_matmul,$(patsubst %_tb.sv,%,$(wildcard *_tb.sv)))

.PHONY: sim wave list all clean uart echo mem bram cosim fw fwsweep

sim: $(TB) $(RTL)
	$(IVERILOG) -o $(TOP).vvp $(RTL) $(TB)
	$(VVP) $(TOP).vvp


# `examples` and `model` are gone. They drove the scalar unit through
# tpu_top_tb.sv over assembled .tpu programs, all of which were deleted with the
# tpulang toolchain; `make fw FWPROG=<kernel>` is what covers the dispatch plane
# now, and it checks strictly more (the command stream as well as the DRAM
# image). `make fw FWPROG=adder` is the successor to `make model`.


# The C firmware through the whole core: cpu_subsys.sv loads ../fw/$(FWPROG).hex
# through FW_INIT and the PicoRV32 pushes the commands itself. Build the image
# first (`make -C ../fw [PROG=...]`) -- that needs a RISC-V cross gcc, which is
# why this is not in `make all`.
#
#   make fw                     ../fw/matmul.hex
#   make fw FWPROG=matmul_loop  the software tile loop, same expectations
#   make fw FWPROG=ffn          the feed-forward block (first VPU commands)
#   make fw FWPROG=mha          one head of ReLU attention
#   make fw FWPROG=adder        the whole int4 adder model, ~440 k clocks / ~3 min
#   make fw M=8 KTILES=16 NTILES=16   a different shape (rebuild ../fw with the
#                                     SAME three numbers or the TB will mismatch)
FWPROG ?= matmul
M      ?=
KTILES ?=
NTILES ?=
FWSHAPE := $(if $(M),-DFW_M=$(M)) $(if $(KTILES),-DFW_KT=$(KTILES)) \
           $(if $(NTILES),-DFW_NT=$(NTILES))

# The testbench's 2 ms watchdog is sized for the small kernels; the adder model
# runs ~440 k clocks (4.4 ms) and would trip it while working perfectly. Per
# kernel rather than globally, because a watchdog that never fires on a hang is
# worth nothing -- and skipped entirely if IVFLAGS already sets one.
FW_WATCHDOG_adder := 60000000
FW_WATCHDOG := $(if $(findstring WATCHDOG,$(IVFLAGS)),,\
                 $(if $(FW_WATCHDOG_$(FWPROG)),-DWATCHDOG_NS=$(FW_WATCHDOG_$(FWPROG))))

# Golden vectors now come from the kernel's own command trace: the same .c is
# compiled natively (-DTPU_TRACE, no cross toolchain needed), its trace is run
# through iss.py, and the testbench checks both the DRAM image and the command
# stream the real CPU issued. docs/picorv32_migration.md §8.1.
FWVEC := vectors_fw

# The shape now has to agree in THREE places — the RISC-V image, the native
# trace, and the testbench's -D defines — and ../fw/Makefile cannot see a changed
# -D, so a stale .hex from an earlier shape silently issues a differently sized
# problem. (That is not hypothetical: it is what the sweep leaves behind, and the
# only thing that catches it is the "nothing past C" guard.) So `fw` rebuilds
# both firmware artifacts from clean at the requested shape every time. The
# images are a few hundred bytes; correctness is worth the second.
FWSHAPEARGS = $(if $(FW_M),M=$(FW_M)) $(if $(FW_KT),KTILES=$(FW_KT)) \
              $(if $(FW_NT),NTILES=$(FW_NT))

.PHONY: fwvec
fwvec: | $(FWVEC)
	@$(MAKE) --no-print-directory -C ../fw clean
	@$(MAKE) --no-print-directory -C ../fw $(FWPROG).hex $(FWPROG).trace \
	         PROG=$(FWPROG) $(FWSHAPEARGS)
	../fw/$(FWPROG).trace > $(FWVEC)/$(FWPROG).trace.txt
	python $(TPULANG)/fw_vectors.py -t $(FWVEC)/$(FWPROG).trace.txt -o $(FWVEC) \
	       -k $(FWPROG) \
	       -M $(or $(FW_M),8) --ktiles $(or $(FW_KT),4) --ntiles $(or $(FW_NT),2)

fw: $(UART_RTL) fw_matmul_tb.sv
	@$(MAKE) --no-print-directory fwvec
	$(IVERILOG) -DFW_HEX='"../fw/$(FWPROG).hex"' -DFW_VEC_DIR='"$(FWVEC)"' \
	            $(FWSHAPE) $(FW_WATCHDOG) -o fw_matmul_tb.vvp $(UART_RTL) fw_matmul_tb.sv
	$(VVP) fw_matmul_tb.vvp

$(FWVEC):
	mkdir -p $@

# Both kernels over a range of shapes, tabulating what each costs the CPU.
# Rebuilds the firmware per shape, so it needs the cross gcc like `make fw`.
fwsweep: $(UART_RTL) fw_matmul_tb.sv run_fw_sweep.sh
	sh ./run_fw_sweep.sh

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
	rm -f *_tb.vvp *_tb.vcd ex_*.vvp
	rm -rf vec_*
	rm -f model*.vvp
