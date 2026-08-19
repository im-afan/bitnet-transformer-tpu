# -----------------------------------------------------------------------------
# board.tcl — Cmod A7-35T, UART + external SRAM bring-up image (no TPU core).
#
# A third target on the same physical board as boards/cmod_a7 and
# boards/cmod_a7_echo. build.tcl needs no changes -- it already dispatches on
# board=.
#
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=rtl
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=deploy
#
# Output lands in synth/build/cmod_a7_mem/, so this and the production bitstream
# cannot be mistaken for one another. See rtl/uart_memory.sv for what it is for.
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
set TOP        cmod_a7_mem_top
set BOARD_SRCS [list [file join $BOARD_DIR cmod_a7_mem_top.sv]]

# Shares constraints/cmod_a7.xdc with the production image rather than taking a
# copy. cmod_a7_mem_top declares exactly the same ports as cmod_a7_top -- same
# names, same widths -- so every set_property in that file resolves, and the two
# images are guaranteed the identical pinout. That matters: the SRAM pins
# switching next to the J17/J18 UART pair is part of what is under test, so a
# divergent pin assignment would quietly invalidate the comparison.
#
# (boards/cmod_a7_echo cannot do this -- it declares none of the 31 SRAM ports,
# and set_property against an empty get_ports result is an error.)
set XDC_FILES  [list [file join $TPU_DIR constraints cmod_a7.xdc]]

# ---- Core clock ---------------------------------------------------------------
# Same 12 MHz oscillator straight into the core as boards/cmod_a7 -- this image
# must run the receiver at the identical divisor, or it is not measuring the same
# thing. build.tcl re-derives UART_CPB if clk_mhz= or baud= is passed.
set CLK_MHZ  12
set BAUD     115200
set UART_CPB [expr {int(round(double($CLK_MHZ) * 1.0e6 / $BAUD))}]   ;# 104

# ---- External SRAM ------------------------------------------------------------
# Cmod A7 cellular SRAM: 512K x 8. Must match the 19 address / 8 data lines in
# constraints/cmod_a7.xdc, and matches boards/cmod_a7 so the controller behaves
# identically in both images.
set MEM_ADDR_W 19
set MEM_DATA_W 8
set SRAM_CPA   0

# ---- Geometry placeholders ----------------------------------------------------
# There is no TPU core in this image -- no MXU, no VPU, no scratchpad. These
# exist only because build.tcl's banner prints them unconditionally; they are not
# passed to the top level (board_generics below does not list them), so the zeros
# in the banner are accurate: this image has none of it.
set ROWS      0
set COLS      0
set ADDR_W    0
set VPU_BYTES 0

# ---- Generics passed to the top level ----------------------------------------
# Only what cmod_a7_mem_top actually declares. build.tcl's generic_args would
# filter anything else out with a note, but keeping the list honest means the
# note stays empty and a real typo is visible.
#
# UART_RX_TIMEOUT is 0 to match boards/cmod_a7 exactly -- that is what compiles
# uart_interface's mid-frame abort out of the design, and this image exists to
# reproduce the production behaviour, not to improve on it. Overriding it is a
# one-off experiment, not a default:
#
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=deploy
# then edit the 0 below, or add rx_timeout to build.tcl's ARGMAP if it becomes
# something you sweep.
proc board_generics {} {
    uplevel #0 {
        set GENERICS [list \
            UART_CPB        $UART_CPB \
            UART_RX_TIMEOUT 0 \
            MEM_ADDR_W      $MEM_ADDR_W \
            MEM_DATA_W      $MEM_DATA_W \
            SRAM_CPA        $SRAM_CPA \
        ]
    }
}

# ---- Flash (mode=mcs) ---------------------------------------------------------
# Cmod A7 boots from a 4 MB Spansion S25FL032 QSPI part. Writing a *debug* image
# to flash would make the board come up as a bring-up rig every power-on, which
# is almost never what you want -- build mode=deploy and program volatile instead.
set CFGMEM_PART s25fl032p-spi-x1_x2_x4
set CFGMEM_SIZE 4
