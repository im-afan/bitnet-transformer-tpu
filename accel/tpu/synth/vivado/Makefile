# Convenience wrapper around build.tcl (accel/tpu/docs/synth.md is the full story).
#
# build.tcl is still the recipe — this file only spells its invocation so the
# common flows are one word. Anything here is exactly equivalent to running
# vivado by hand; `make print-cmd` shows the command a target would issue.
#
#   make rtl                     elaborate only (seconds) — do this first
#   make synth                   synthesis + utilization/timing reports
#   make impl                    + place & route, no bitstream
#   make bit                     + bitstream                       [default]
#   make mcs                     + QSPI flash image
#   make program                 download the existing .bit (no build)
#   make deploy                  build then program, in one command
#
#   make BOARD=cmod_a7_echo deploy      any target, any board
#   make echo / mem / bram / prod       shorthands that deploy one board
#
#   make ooc MODULE=mxu          out-of-context area for one block
#   make reports                 list the reports from the last build
#   make clean                   remove synth/build/<board>
#   make distclean               remove synth/build and the vivado journals
#
# Overrides are passed straight through: any build.tcl key=value works as a
# make variable, plus EXTRA for the rest.
#
#   make synth ROWS=16 COLS=16
#   make bit INCR=1 THREADS=16
#   make deploy EXTRA="clk_mhz 24"
#   make program BIT=../build/cmod_a7/cmod_a7_top.bit
#
# On Windows `vivado` is vivado.bat and cmd.exe splits arguments on '=', so
# every pair below is passed in build.tcl's space-separated form instead.

VIVADO ?= vivado
BOARD  ?= cmod_a7

VIVADO_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TPU_DIR    := $(abspath $(VIVADO_DIR)/../..)
OUTDIR     ?= $(TPU_DIR)/synth/build/$(BOARD)

# Optional build.tcl keys. Each is forwarded only when the make variable is
# actually set — the space-separated spelling has no way to express an empty
# value. The make variable is the upper-case spelling of build.tcl's key.
OPTS = $(if $(MODULE),module $(MODULE)) $(if $(THREADS),threads $(THREADS)) \
       $(if $(INCR),incr $(INCR)) $(if $(BIT),bit $(BIT)) \
       $(if $(PART),part $(PART)) $(if $(TOP),top $(TOP)) \
       $(if $(ROWS),rows $(ROWS)) $(if $(COLS),cols $(COLS)) \
       $(if $(VPU_BYTES),vpu_bytes $(VPU_BYTES)) $(if $(ADDR_W),addr_w $(ADDR_W)) \
       $(if $(XLEN),xlen $(XLEN)) $(if $(IMEM_AW),imem_aw $(IMEM_AW)) \
       $(if $(CFG_AW),cfg_aw $(CFG_AW)) $(if $(REG_AW),reg_aw $(REG_AW)) \
       $(if $(M0_W),m0_w $(M0_W)) $(if $(N_W),n_w $(N_W)) \
       $(if $(MEM_ADDR_W),mem_addr_w $(MEM_ADDR_W)) \
       $(if $(MEM_DATA_W),mem_data_w $(MEM_DATA_W)) \
       $(if $(SRAM_CPA),sram_cpa $(SRAM_CPA)) $(if $(BRAM_AW),bram_aw $(BRAM_AW)) \
       $(if $(CLK_MHZ),clk_mhz $(CLK_MHZ)) $(if $(BAUD),baud $(BAUD)) \
       $(if $(CPB),cpb $(CPB)) $(EXTRA)

TCLARGS = board $(BOARD) outdir $(OUTDIR) $(OPTS)
RUN     = cd $(VIVADO_DIR) && $(VIVADO) -mode batch -source build.tcl -tclargs

.PHONY: help rtl ooc synth impl bit mcs program deploy \
        prod echo mem bram boards reports print-cmd clean distclean

bit:
	$(RUN) mode bit $(TCLARGS)

rtl synth impl mcs program deploy:
	$(RUN) mode $@ $(TCLARGS)

# mode=ooc needs module=<name>; fail here rather than after Vivado has started.
ooc:
	@test -n "$(MODULE)" || { \
	  echo "make ooc needs MODULE=<rtl module name>, e.g. make ooc MODULE=mxu"; \
	  exit 1; }
	$(RUN) mode ooc $(TCLARGS)

# One-word deploys for the four boards. Each is `make deploy` with BOARD set.
prod: ; @$(MAKE) --no-print-directory deploy BOARD=cmod_a7
echo: ; @$(MAKE) --no-print-directory deploy BOARD=cmod_a7_echo
mem:  ; @$(MAKE) --no-print-directory deploy BOARD=cmod_a7_mem
bram: ; @$(MAKE) --no-print-directory deploy BOARD=cmod_a7_bram

help:
	$(RUN) mode help

boards:
	@ls -1 $(VIVADO_DIR)/boards

reports:
	@ls -lt $(OUTDIR)/reports 2>/dev/null || echo "no reports — nothing built for $(BOARD) yet"

# What the equivalent hand-run would be, for pasting into a shell or a bug report.
print-cmd:
	@echo "$(VIVADO) -mode batch -source build.tcl -tclargs mode <mode> $(TCLARGS)"

clean:
	rm -rf $(OUTDIR)

distclean:
	rm -rf $(TPU_DIR)/synth/build
	rm -f $(VIVADO_DIR)/vivado*.jou $(VIVADO_DIR)/vivado*.log \
	      $(VIVADO_DIR)/clockInfo.txt
