# -----------------------------------------------------------------------------
# sources.tcl — single source of truth for the synthesizable RTL file list.
#
# Sourced by build.tcl. Kept separate so the file list is declared once and can
# also be consumed by other flows (a future Vivado xsim run, a lint pass, etc.)
# without re-enumerating the RTL. Sets:
#
#   RTL_DIR    absolute path to accel/tpu/rtl
#   RTL_SRCS   ordered list of synthesizable SystemVerilog sources
#
# Order does not matter to Vivado (it elaborates from -top), but is kept
# matching tb/Makefile's UART_RTL list so the two stay visibly in sync.
# -----------------------------------------------------------------------------

set RTL_DIR [file normalize [file join [file dirname [info script]] .. .. rtl]]

set RTL_SRCS [list \
    [file join $RTL_DIR tpu_top.sv] \
    [file join $RTL_DIR scalar_unit.sv] \
    [file join $RTL_DIR mxu.sv] \
    [file join $RTL_DIR vpu.sv] \
    [file join $RTL_DIR scratchpad.sv] \
    [file join $RTL_DIR dma.sv] \
    [file join $RTL_DIR sram.sv] \
    [file join $RTL_DIR uart_interface.sv] \
    [file join $RTL_DIR uart_receiver.sv] \
    [file join $RTL_DIR uart_transmitter.sv] \
]

# Leaf blocks that nothing instantiates today. Read anyway so they stay
# compile-checked by every synthesis run; Vivado prunes unreferenced modules
# once -top is elaborated. Move a file up into RTL_SRCS when it gets wired in.
set RTL_UNUSED [list \
    [file join $RTL_DIR requant.sv] \
]

foreach f [concat $RTL_SRCS $RTL_UNUSED] {
    if {![file exists $f]} {
        error "sources.tcl: missing RTL source '$f'"
    }
}
