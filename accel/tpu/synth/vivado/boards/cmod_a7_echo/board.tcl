# -----------------------------------------------------------------------------
# board.tcl — Cmod A7-35T, UART echo self-test image.
#
# A second target on the same physical board as boards/cmod_a7, not a variant of
# it: different top level, different constraints, different output directory.
# build.tcl needs no changes -- it already dispatches on board=.
#
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_echo mode=rtl
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_echo mode=bit
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_echo mode=program
#
# Output lands in synth/build/cmod_a7_echo/, so the self-test and production
# bitstreams cannot be mistaken for one another. See docs/uart_selftest.md.
# -----------------------------------------------------------------------------

set BOARD_DIR [file normalize [file dirname [info script]]]

# TPU_DIR (= accel/tpu) is set by build.tcl before it sources this file. Do not
# recompute it here — the relative depth differs between the two files.
if {![info exists TPU_DIR]} {
    error "board.tcl: TPU_DIR unset — this file is sourced by build.tcl, not run directly"
}

# ---- Part ---------------------------------------------------------------------
# Cmod A7-35T. The -15T variant is xc7a15t-cpg236-1; override with part=...
set PART xc7a35t-cpg236-1

# ---- Top level ----------------------------------------------------------------
set TOP        cmod_a7_echo_top
set BOARD_SRCS [list [file join $BOARD_DIR cmod_a7_echo_top.sv]]

# Its own .xdc. constraints/cmod_a7.xdc cannot be reused: it constrains 31 SRAM
# ports that this top level does not have, and set_property against an empty
# get_ports result is an error.
set XDC_FILES  [list [file join $TPU_DIR constraints cmod_a7_echo.xdc]]

# ---- Core clock ---------------------------------------------------------------
# Same 12 MHz oscillator straight into the core as boards/cmod_a7 — the echo
# image must run the receiver at the identical divisor, or it is not measuring
# the same thing. build.tcl re-derives UART_CPB if clk_mhz= or baud= is passed.
set CLK_MHZ  12
set BAUD     115200
set UART_CPB [expr {int(round(double($CLK_MHZ) * 1.0e6 / $BAUD))}]   ;# 104

# ---- Echo block length --------------------------------------------------------
# Bytes buffered per exchange: the device receives this many, then sends them
# back. host/uart_echo.py's --block must match — nothing on the wire negotiates
# it. See rtl/uart_echo.sv.
set BLOCK_LEN 64

# ---- Geometry placeholders ----------------------------------------------------
# There is no TPU core in this image — no MXU, no scratchpad, no external SRAM.
# These exist only because build.tcl's banner prints them unconditionally; they
# are not passed to the top level (board_generics below does not list them), so
# the zeros in the banner are accurate: this image has none of it.
set ROWS      0
set COLS      0
set ADDR_W    0
set VPU_BYTES 0

# ---- Generics passed to the top level ----------------------------------------
# Only what cmod_a7_echo_top actually declares. build.tcl's generic_args would
# filter anything else out with a note, but keeping the list honest means the
# note stays empty and a real typo is visible.
proc board_generics {} {
    uplevel #0 {
        set GENERICS [list \
            UART_CPB  $UART_CPB \
            BLOCK_LEN $BLOCK_LEN \
        ]
    }
}

# ---- Flash (mode=mcs) ---------------------------------------------------------
# Cmod A7 boots from a 4 MB Spansion S25FL032 QSPI part. Writing the *echo* image
# to flash would make the board come up as a self-test rig every power-on, which
# is almost never what you want — build mode=bit and program volatile instead.
set CFGMEM_PART s25fl032p-spi-x1_x2_x4
set CFGMEM_SIZE 4
