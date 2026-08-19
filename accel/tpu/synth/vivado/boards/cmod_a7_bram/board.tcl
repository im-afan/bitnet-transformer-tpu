# -----------------------------------------------------------------------------
# board.tcl — Cmod A7-35T, UART + on-chip block RAM bring-up image (no TPU core,
# no external memory).
#
# A fourth target on the same physical board as boards/cmod_a7, boards/cmod_a7_mem
# and boards/cmod_a7_echo. build.tcl needs no changes -- it already dispatches on
# board=.
#
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_bram mode=rtl
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_bram mode=deploy
#
# Output lands in synth/build/cmod_a7_bram/, so this and the production bitstream
# cannot be mistaken for one another. See rtl/uart_bram.sv for what it is for:
# it is boards/cmod_a7_mem with the SRAM controller and the memory bus removed,
# which is the one cut that separates uart_interface from sram_controller.
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
set TOP        cmod_a7_bram_top
set BOARD_SRCS [list [file join $BOARD_DIR cmod_a7_bram_top.sv]]

# Shares constraints/cmod_a7_echo.xdc rather than taking a copy of it.
# cmod_a7_bram_top declares exactly the ports cmod_a7_echo_top does -- sysclk,
# btn[1:0], led[1:0], uart_txd_in, uart_rxd_out, and none of the 31 SRAM ports --
# so every set_property in that file resolves, and the two no-memory images are
# guaranteed the identical clock and UART pinout. Same reasoning as
# boards/cmod_a7_mem sharing constraints/cmod_a7.xdc: a copy could drift, and
# the UART reaching the receiver through the same package pins at the same I/O
# standard is a precondition for comparing runs across images at all.
#
# constraints/cmod_a7.xdc is *not* usable here for the same reason it is not
# usable for the echo image: set_property against an empty get_ports result is
# an error, and this top declares none of the SRAM ports.
set XDC_FILES  [list [file join $TPU_DIR constraints cmod_a7_echo.xdc]]

# ---- Core clock ---------------------------------------------------------------
# Same 12 MHz oscillator straight into the core as boards/cmod_a7 -- this image
# must run the receiver at the identical divisor, or it is not measuring the same
# thing. build.tcl re-derives UART_CPB if clk_mhz= or baud= is passed.
set CLK_MHZ  12
set BAUD     115200
set UART_CPB [expr {int(round(double($CLK_MHZ) * 1.0e6 / $BAUD))}]   ;# 104

# ---- Memory -------------------------------------------------------------------
# MEM_ADDR_W is the *protocol* address width and must stay 19: it is what
# uart_interface range-checks incoming frames against, and a rig that accepts or
# rejects different frames than the production image is not a control. It also
# fixes the 3-byte address field on the wire ((19+7)/8 = 3), so lowering it would
# change the protocol itself.
set MEM_ADDR_W 19
set MEM_DATA_W 8

# What is actually built. 2**16 bytes = 64 KiB = 16 RAMB36 tiles of the 35T's 50
# (and it still fits the -15T's 25). Addresses above this alias down onto the
# window -- deterministic, flagged on `aliased`, and documented in rtl/bram.sv.
#
# 64 KiB covers every host test except the two that deliberately probe the top of
# the 19-bit space (`sram_isolation`, `sram_address_bus` -- they are testing
# address lines that do not exist in this image) and `sram_long_transfer`, whose
# 69631-byte split transfer needs:
#
#   vivado ... -tclargs board=cmod_a7_bram mode=deploy bram_aw=17
#
# which is 128 KiB / 32 tiles: fine on a -35T, over budget on a -15T.
set BRAM_AW    16

# Extra clocks per access. Held equal to boards/cmod_a7_mem's SRAM_CPA on
# purpose -- bram_controller does not need any of the SRAM's beat timing, but if
# it completed a transfer faster than sram_controller does, every turnaround in
# uart_interface would shift and the two images would stop being comparable.
# Named SRAM_CPA here (rather than MEM_CPA) so build.tcl's existing sram_cpa=
# override reaches it; board_generics below maps it onto the port's actual name.
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
# Only what cmod_a7_bram_top actually declares. build.tcl's generic_args would
# filter anything else out with a note, but keeping the list honest means the
# note stays empty and a real typo is visible.
#
# UART_RX_TIMEOUT is 0 to match boards/cmod_a7 and boards/cmod_a7_mem exactly --
# that is what compiles uart_interface's mid-frame abort out of the design, and
# this image exists to reproduce the production behaviour, not to improve on it.
proc board_generics {} {
    uplevel #0 {
        set GENERICS [list \
            UART_CPB        $UART_CPB \
            UART_RX_TIMEOUT 0 \
            MEM_ADDR_W      $MEM_ADDR_W \
            MEM_DATA_W      $MEM_DATA_W \
            BRAM_AW         $BRAM_AW \
            MEM_CPA         $SRAM_CPA \
        ]
    }
}

# ---- Flash (mode=mcs) ---------------------------------------------------------
# Cmod A7 boots from a 4 MB Spansion S25FL032 QSPI part. Writing a *debug* image
# to flash would make the board come up as a bring-up rig every power-on, which
# is almost never what you want -- build mode=deploy and program volatile instead.
set CFGMEM_PART s25fl032p-spi-x1_x2_x4
set CFGMEM_SIZE 4
