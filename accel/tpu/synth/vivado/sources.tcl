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
    [file join $RTL_DIR perf_counters.sv] \
    [file join $RTL_DIR cmd_queue.sv] \
    [file join $RTL_DIR cmd_mxu.sv] \
    [file join $RTL_DIR cmd_vpu.sv] \
    [file join $RTL_DIR cmd_dma.sv] \
    [file join $RTL_DIR cpu_subsys.sv] \
    [file join $RTL_DIR vendor picorv32.v] \
]

# picorv32.v is vendored, not written here: YosysHQ/picorv32, ISC licensed.
# It is plain Verilog-2005 and reads fine alongside the SystemVerilog above.
# Only picorv32_axi and what it instantiates is referenced; Vivado prunes the
# rest (picorv32_wb, the PCPI divider) once -top is elaborated.

# Sources outside the tpu_top hierarchy. Read anyway so they stay compile-checked
# by every synthesis run; Vivado prunes unreferenced modules once -top is
# elaborated, so carrying them costs nothing in the production build.
#
#   requant.sv     a leaf block nothing instantiates yet — move it up into
#                  RTL_SRCS when it gets wired in.
#   uart_echo.sv   the UART loopback self-test core (docs/uart_selftest.md). Not
#                  part of tpu_top; it is the child of boards/cmod_a7_echo's top
#                  level, which is why it lives here rather than in RTL_SRCS.
#   uart_memory.sv the UART + external SRAM bring-up core: tpu_top with the core
#                  removed. Child of boards/cmod_a7_mem's top level. Its own
#                  children (uart_interface/receiver/transmitter, sram.sv) are
#                  already in RTL_SRCS above and are shared, not copied.
#   uart_bram.sv   the same core with the external memory removed as well:
#                  bram_controller in place of sram_controller, everything else
#                  identical. Child of boards/cmod_a7_bram's top level.
#   bram.sv        bram_controller — on-chip block RAM behind sram_controller's
#                  user-side handshake. Only uart_bram.sv instantiates it.
set RTL_UNUSED [list \
    [file join $RTL_DIR requant.sv] \
    [file join $RTL_DIR uart_echo.sv] \
    [file join $RTL_DIR uart_memory.sv] \
    [file join $RTL_DIR uart_bram.sv] \
    [file join $RTL_DIR bram.sv] \
]

foreach f [concat $RTL_SRCS $RTL_UNUSED] {
    if {![file exists $f]} {
        error "sources.tcl: missing RTL source '$f'"
    }
}
