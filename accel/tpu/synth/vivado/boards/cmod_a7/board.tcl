# -----------------------------------------------------------------------------
# board.tcl — Digilent Cmod A7-35T target definition.
#
# Sourced by ../../build.tcl. Everything board- or target-specific lives here;
# build.tcl itself is board-agnostic. To add a second board, copy this directory
# and run `build.tcl board=<name>`.
#
# Sets:
#   PART        Vivado part string
#   TOP         top-level module name (the board wrapper, not tpu_top)
#   BOARD_SRCS  extra sources beyond RTL_SRCS (the wrapper)
#   XDC_FILES   constraint files, applied in order
#   GENERICS    {name value ...} passed to synth_design as -generic
#
# Any of these may be overridden from the command line, e.g.
#   vivado -mode batch -source build.tcl -tclargs rows=16 cols=16
# -----------------------------------------------------------------------------

set BOARD_DIR [file normalize [file dirname [info script]]]

# TPU_DIR (= accel/tpu) is set by build.tcl before it sources this file. Do not
# recompute it here — the relative depth differs between the two files and it is
# an easy way to end up with constraints resolving under synth/ instead.
if {![info exists TPU_DIR]} {
    error "board.tcl: TPU_DIR unset — this file is sourced by build.tcl, not run directly"
}

# ---- Part ---------------------------------------------------------------------
# Cmod A7-35T. The -15T variant is xc7a15t-cpg236-1; override with part=...
set PART xc7a35t-cpg236-1

# ---- Top level ----------------------------------------------------------------
set TOP        cmod_a7_top
set BOARD_SRCS [list [file join $BOARD_DIR cmod_a7_top.sv]]
set XDC_FILES  [list [file join $TPU_DIR constraints cmod_a7.xdc]]

# ---- Core clock ---------------------------------------------------------------
# The board's only oscillator is 12 MHz on L17 and the wrapper feeds it straight
# to the core -- no MMCM, no clocking wizard, no IP in the repo. UART_CPB is
# derived from it below; if you ever add an MMCM, change CLK_MHZ and CPB follows.
set CLK_MHZ  12
set BAUD     115200
set UART_CPB [expr {int(round(double($CLK_MHZ) * 1.0e6 / $BAUD))}]   ;# 104

# ---- Core geometry ------------------------------------------------------------
# These are NOT the tpu_top defaults (128x128). They are the geometry that
# tb/tpu_top_uart_tb.sv and tpulang/gen_vectors.py are both written against, so a
# bitstream built with them accepts the same programs and golden vectors the
# end-to-end simulation already passes. 128x128 does not remotely fit this part
# (see docs/synth.md "Sizing").
set ROWS       8
set COLS       8
set VPU_BYTES  64
set ADDR_W     16
set XLEN       32
set IMEM_AW    10
set CFG_AW     5
set REG_AW     5
set M0_W       12
set N_W        4
set MEM_ADDR_W 19    ;# Cmod A7 cellular SRAM: 512K x 8
set MEM_DATA_W 8
# sram_controller's *extra* clocks per byte, on top of the one the beat always
# takes. 0 is full rate: one byte per 83.3 ns clock on reads, one per two on
# writes, against a part rated for a 10 ns access. See rtl/sram.sv's header for
# the per-parameter timing budget and constraints/cmod_a7.xdc for why these pins
# are false-pathed.
set SRAM_CPA   0

# ---- Generics passed to the top level ----------------------------------------
# Assembled after the argv overrides in build.tcl have been applied, so this
# proc is called (not evaluated) at that point.
proc board_generics {} {
    uplevel #0 {
        set GENERICS [list \
            ROWS            $ROWS \
            COLS            $COLS \
            VPU_BYTES       $VPU_BYTES \
            ADDR_W          $ADDR_W \
            XLEN            $XLEN \
            IMEM_AW         $IMEM_AW \
            CFG_AW          $CFG_AW \
            REG_AW          $REG_AW \
            M0_W            $M0_W \
            N_W             $N_W \
            MEM_ADDR_W      $MEM_ADDR_W \
            MEM_DATA_W      $MEM_DATA_W \
            SRAM_CPA        $SRAM_CPA \
            UART_CPB        $UART_CPB \
            UART_RX_TIMEOUT 0 \
        ]
    }
}

# ---- Flash (mode=mcs) ---------------------------------------------------------
# Cmod A7 boots from a 4 MB Spansion S25FL032 QSPI part.
set CFGMEM_PART s25fl032p-spi-x1_x2_x4
set CFGMEM_SIZE 4
