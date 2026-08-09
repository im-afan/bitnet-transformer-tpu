# =============================================================================
# build.tcl — Vivado non-project (batch) build for accel/tpu.
#
# Nothing about a Vivado *project* is version controlled: no .xpr, no .runs/,
# no .cache/. This script is the recipe, and everything it produces lands in
# synth/build/<board>/ which is gitignored. Two consequences worth knowing:
#
#   * The build is reproducible from a clean checkout — `git clone` then run
#     this. There is no "open the project and hope it's configured right".
#   * You never merge a project file. The reviewable surface is this script,
#     sources.tcl, boards/<board>/board.tcl and the .xdc.
#
# Usage (see docs/synth.md for the full walkthrough):
#
#   vivado -mode batch -source build.tcl
#   vivado -mode batch -source build.tcl -tclargs mode=synth
#   vivado -mode batch -source build.tcl -tclargs mode=ooc module=mxu rows=8 cols=8
#   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=deploy
#
# Arguments are key=value, order independent. Run with `mode=help` for a list.
# =============================================================================

set BUILD_START [clock seconds]

set VIVADO_DIR [file normalize [file dirname [info script]]]
set TPU_DIR    [file normalize [file join $VIVADO_DIR .. ..]]

# -----------------------------------------------------------------------------
# Flow-level defaults. Everything else (part, top, geometry) comes from the
# board file and may be overridden by argv below.
# -----------------------------------------------------------------------------
set MODE    bit          ;# help | rtl | ooc | synth | impl | bit | mcs | program | deploy
set BOARD   cmod_a7
set MODULE  ""           ;# mode=ooc: which RTL module to synthesize standalone
set OUTDIR  ""           ;# defaults to synth/build/<board>
set THREADS 8
set BITFILE ""           ;# mode=program/mcs: override which bitstream to use
set INCR    0            ;# 1 = reuse post_synth.dcp if it is newer than the RTL

# Command-line key -> Tcl variable. Anything not listed here is rejected, so a
# typo fails loudly instead of being silently ignored.
array set ARGMAP {
    mode        MODE
    board       BOARD
    module      MODULE
    outdir      OUTDIR
    threads     THREADS
    bit         BITFILE
    incr        INCR

    part        PART
    top         TOP

    rows        ROWS
    cols        COLS
    vpu_bytes   VPU_BYTES
    addr_w      ADDR_W
    xlen        XLEN
    imem_aw     IMEM_AW
    cfg_aw      CFG_AW
    reg_aw      REG_AW
    m0_w        M0_W
    n_w         N_W
    mem_addr_w  MEM_ADDR_W
    mem_data_w  MEM_DATA_W
    sram_cpa    SRAM_CPA
    bram_aw     BRAM_AW
    clk_mhz     CLK_MHZ
    baud        BAUD
    cpb         UART_CPB
}

# Every directory under boards/ that actually defines a target. Enumerated
# rather than hard-coded so adding a board makes it show up in `mode=help`
# without anyone remembering to edit this file.
proc board_list {} {
    global VIVADO_DIR
    set names {}
    foreach d [lsort [glob -nocomplain -directory [file join $VIVADO_DIR boards] *]] {
        if {[file exists [file join $d board.tcl]]} { lappend names [file tail $d] }
    }
    return $names
}

proc usage {} {
    puts ""
    puts "build.tcl — Vivado batch build for accel/tpu"
    puts ""
    puts "  vivado -mode batch -source build.tcl -tclargs \[key=value ...\]"
    puts ""
    puts "Modes (mode=...):"
    puts "  rtl       elaborate only — fast syntax/elaboration check, no netlist"
    puts "  ooc       out-of-context synth of one module (needs module=<name>);"
    puts "            the way to get real utilization for a block in isolation"
    puts "  synth     full synthesis + utilization/timing reports, then stop"
    puts "  impl      synth + opt/place/route + reports, no bitstream"
    puts "  bit       everything, ending in a .bit           \[default\]"
    puts "  mcs       bit, then write a .mcs image for the QSPI flash"
    puts "  program   download an existing .bit to a connected board (no build)"
    puts "  deploy    bit, then program the connected board — build and flash in"
    puts "            one command, so what is on the board is what was just built"
    puts "  help      this message"
    puts ""
    puts "Boards (board=...), from boards/<name>/board.tcl:"
    foreach b [board_list] {
        switch -- $b {
            cmod_a7      { set what "the full TPU (production image)" }
            cmod_a7_mem  { set what "UART link + external SRAM, no core (rtl/uart_memory.sv)" }
            cmod_a7_bram { set what "UART link + on-chip block RAM, no core (rtl/uart_bram.sv)" }
            cmod_a7_echo { set what "UART loopback self-test (docs/uart_selftest.md)" }
            default      { set what "" }
        }
        puts [format "  %-14s %s" $b $what]
    }
    puts ""
    puts "Common overrides:"
    puts "  board=<name>      target under boards/            \[cmod_a7\]"
    puts "  part=<part>       Vivado part string              \[from board.tcl\]"
    puts "  rows=N cols=N     MXU systolic array size         \[8 8\]"
    puts "  addr_w=N          scratchpad address width        \[16\]"
    puts "  bram_aw=N         cmod_a7_bram: built memory      \[16\] (2**N bytes)"
    puts "  clk_mhz=N         core clock; UART_CPB follows it \[12\]"
    puts "  outdir=<path>     build directory                 \[synth/build/<board>\]"
    puts "  threads=N         Vivado max threads              \[8\]"
    puts "  incr=1            reuse post_synth.dcp if current"
    puts ""
    puts "On Windows, cmd.exe splits on '=' inside vivado.bat, so `mode rtl`"
    puts "(space separated) is accepted as well as `mode=rtl`."
    puts ""
}

# -----------------------------------------------------------------------------
# Parse argv. Deliberately strict: unknown keys are errors.
#
# Two accepted spellings for a pair, because on Windows `vivado` is vivado.bat
# and cmd.exe's batch parser treats '=' as an argument delimiter — `-tclargs
# mode=rtl` arrives here as the two words {mode rtl}, not as one. So:
#
#   mode=rtl        the documented form (Linux, and Windows when quoted)
#   mode rtl        a bare known key followed by its value (Windows fallback)
#
# Unambiguous either way: values are only consumed straight after a key.
# -----------------------------------------------------------------------------
set CLI_KEYS {}
set ARGV_N [llength $argv]
for {set i 0} {$i < $ARGV_N} {incr i} {
    # cmd may also hand the quotes through verbatim.
    set arg [string trim [lindex $argv $i] "\""]
    if {$arg eq ""} continue

    if {[regexp {^([A-Za-z_][A-Za-z0-9_]*)=(.*)$} $arg -> key val]} {
        set key [string tolower $key]
    } elseif {[info exists ARGMAP([string tolower $arg])] && $i + 1 < $ARGV_N} {
        set key [string tolower $arg]
        set val [string trim [lindex $argv [incr i]] "\""]
    } else {
        usage
        error "build.tcl: expected key=value, got '$arg'"
    }

    if {![info exists ARGMAP($key)]} {
        usage
        error "build.tcl: unknown argument '$key'"
    }
    lappend CLI_KEYS $key
    set PENDING($key) $val
}

if {[info exists PENDING(mode)]}  { set MODE  $PENDING(mode) }
if {[info exists PENDING(board)]} { set BOARD $PENDING(board) }

if {$MODE eq "help"} { usage; return }

# -----------------------------------------------------------------------------
# Board definition, then apply the remaining overrides on top of it.
# -----------------------------------------------------------------------------
set BOARD_FILE [file join $VIVADO_DIR boards $BOARD board.tcl]
if {![file exists $BOARD_FILE]} {
    error "build.tcl: no such board '$BOARD' (expected $BOARD_FILE)"
}
source $BOARD_FILE

foreach key $CLI_KEYS {
    if {$key in {mode board}} continue
    set [set ARGMAP($key)] $PENDING($key)
}

# UART_CPB is derived from the core clock, so a clk_mhz/baud override has to
# repropagate — unless the user pinned cpb= explicitly, which wins.
if {"cpb" ni $CLI_KEYS} {
    set UART_CPB [expr {int(round(double($CLK_MHZ) * 1.0e6 / $BAUD))}]
}

board_generics

if {$OUTDIR eq ""} { set OUTDIR [file join $TPU_DIR synth build $BOARD] }
set OUTDIR  [file normalize $OUTDIR]
set REPDIR  [file join $OUTDIR reports]
file mkdir $REPDIR

source [file join $VIVADO_DIR sources.tcl]

# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------

proc banner {text} {
    puts ""
    puts "=============================================================================="
    puts "  $text"
    puts "=============================================================================="
}

# Parameters a module actually declares, so we never pass a -generic the top
# level does not have (Vivado treats that as an error, and it is a real trap
# when synthesizing a single block out of context).
proc module_params {path} {
    set fh [open $path r]
    set txt [read $fh]
    close $fh
    set names {}
    set pat {\yparameter\y\s+(?:int\s+|integer\s+|bit\s+|logic\s+)?\s*(?:\[[^\]]*\]\s*)?([A-Za-z_]\w*)}
    foreach {full name} [regexp -all -inline $pat $txt] {
        lappend names $name
    }
    return [lsort -unique $names]
}

# Locate the file declaring `module <name>`.
proc find_module {name} {
    global RTL_SRCS RTL_UNUSED BOARD_SRCS
    foreach path [concat $RTL_SRCS $RTL_UNUSED $BOARD_SRCS] {
        set fh [open $path r]
        set txt [read $fh]
        close $fh
        if {[regexp "\\ymodule\\s+$name\\y" $txt]} { return $path }
    }
    return ""
}

# GENERICS filtered down to what `top` declares, formatted for synth_design.
proc generic_args {top} {
    global GENERICS
    set path [find_module $top]
    if {$path eq ""} { error "build.tcl: cannot find module '$top' in any source" }
    set declared [module_params $path]
    set args {}
    set skipped {}
    foreach {name value} $GENERICS {
        if {$name in $declared} {
            lappend args -generic "$name=$value"
        } else {
            lappend skipped $name
        }
    }
    if {[llength $skipped]} {
        puts "  note: '$top' does not declare [join $skipped {, }] — not passed"
    }
    return $args
}

# Pull the headline numbers out of report_utilization so the log carries a
# summary without anyone opening the report.
#
# Parsed row-wise rather than with one big regexp because the table gained a
# "Prohibited" column around 2020.1 and the 7-series rows are named "Slice ..."
# where UltraScale uses "CLB ...". Splitting on '|' and taking "first numeric
# column = Used, last two = Available and Util%" survives both. Best-effort: a
# miss prints nothing rather than failing the build.
proc utilization_summary {} {
    if {[catch {report_utilization -return_string} txt]} { return }

    array set want {
        "Slice LUTs"        "LUTs"
        "CLB LUTs"          "LUTs"
        "Slice Registers"   "Registers"
        "CLB Registers"     "Registers"
        "Block RAM Tile"    "BRAM tiles"
        "DSPs"              "DSPs"
        "Bonded IOB"        "I/O"
    }

    set rows {}
    foreach line [split $txt "\n"] {
        if {![string match "|*" [string trim $line]]} continue
        set fields {}
        foreach f [split [string trim $line] "|"] { lappend fields [string trim $f] }
        # split on a leading+trailing '|' yields empty first/last elements
        set fields [lrange $fields 1 end-1]
        if {[llength $fields] < 3} continue
        set name [lindex $fields 0]
        if {![info exists want($name)]} continue
        set used  [lindex $fields 1]
        set avail [lindex $fields end-1]
        set pct   [lindex $fields end]
        if {![string is integer -strict $used]} continue
        lappend rows [list $want($name) $used $avail $pct]
    }
    if {![llength $rows]} { return }

    puts ""
    puts "  resource                     used    avail       %"
    puts "  ------------------------------------------------"
    foreach r $rows {
        puts [format "  %-20s %10s %8s %7s" {*}$r]
    }
    puts ""
}

# WNS/WHS after routing, with an explicit verdict. A routed design that misses
# timing still writes a bitstream, so this is the only thing standing between
# you and flashing a board that behaves nondeterministically.
proc timing_verdict {} {
    global REPDIR
    set wns "n/a"
    set whs "n/a"
    catch {
        set p [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
        if {[llength $p]} { set wns [get_property SLACK [lindex $p 0]] }
    }
    catch {
        set p [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
        if {[llength $p]} { set whs [get_property SLACK [lindex $p 0]] }
    }
    puts ""
    puts "  WNS (setup worst slack): $wns ns"
    puts "  WHS (hold  worst slack): $whs ns"
    set bad 0
    foreach s [list $wns $whs] {
        if {[string is double -strict $s] && $s < 0} { set bad 1 }
    }
    if {$bad} {
        puts "  *** TIMING NOT MET — do not trust this bitstream ***"
        puts "  See [file join $REPDIR post_route_timing.rpt]"
    } else {
        puts "  timing met"
    }
    puts ""
    return $bad
}

# Download a bitstream to the attached board. Factored out of `mode=program` so
# `mode=deploy` runs the identical sequence rather than a second copy of it —
# the two must not be able to drift.
proc program_device {bitfile} {
    if {![file exists $bitfile]} {
        error "build.tcl: no bitstream at '$bitfile' (build one first, or pass bit=<path>)"
    }
    banner "program — $bitfile"
    open_hw_manager
    connect_hw_server
    open_hw_target
    set dev [lindex [get_hw_devices] 0]
    current_hw_device $dev
    refresh_hw_device -update_hw_probes false $dev
    set_property PROGRAM.FILE $bitfile $dev
    program_hw_devices $dev
    # Read the part before closing the manager: the device handle is not valid
    # afterwards.
    set part [get_property PART $dev]
    close_hw_manager
    puts "  programmed $part."
}

# -----------------------------------------------------------------------------
# Read the design.
# -----------------------------------------------------------------------------
proc read_design {{with_constraints 1}} {
    global PART RTL_SRCS RTL_UNUSED BOARD_SRCS XDC_FILES THREADS TPU_DIR
    set_param general.maxThreads $THREADS
    # $readmemh paths baked into the RTL (the board wrapper's GELU_INIT /
    # EXP_INIT, SPAD_INIT) are resolved by Vivado against its *working*
    # directory, which is wherever the user launched it — synth/ and synth/vivado/
    # have both been used. Pin it to accel/tpu so those paths mean one thing.
    # Everything else this script touches is already an absolute path (OUTDIR and
    # REPDIR are normalized, sources.tcl and board.tcl build theirs from
    # [info script]), so nothing else moves.
    cd $TPU_DIR
    create_project -in_memory -part $PART
    set_property target_language Verilog [current_project]
    read_verilog -sv [concat $RTL_SRCS $RTL_UNUSED $BOARD_SRCS]
    if {$with_constraints} {
        foreach x $XDC_FILES { read_xdc $x }
    }
}

# =============================================================================
# Flow.
# =============================================================================

banner "accel/tpu — board=$BOARD  part=$PART  mode=$MODE"
puts "  top       : $TOP"
puts "  geometry  : ${ROWS}x${COLS} array, ADDR_W=$ADDR_W, VPU_BYTES=$VPU_BYTES"
puts "  clock     : ${CLK_MHZ} MHz  ->  UART_CPB=$UART_CPB @ ${BAUD} baud"
puts "  outdir    : $OUTDIR"

set BIT  [file join $OUTDIR ${TOP}.bit]
set MCS  [file join $OUTDIR ${TOP}.mcs]
set DCP_SYNTH [file join $OUTDIR post_synth.dcp]
set DCP_ROUTE [file join $OUTDIR post_route.dcp]

switch -- $MODE {

    rtl {
        # Elaboration only. Catches port/width/parameter mistakes in seconds
        # instead of after a multi-minute synthesis run. Use this first, always.
        banner "elaborate (RTL) — $TOP"
        read_design 0
        synth_design -rtl -name rtl_1 -top $TOP -part $PART {*}[generic_args $TOP]
        puts "\n  elaboration clean.\n"
    }

    ooc {
        # Out-of-context synthesis of one block. No I/O buffers, no constraints,
        # no top-level context — just "how big is this module, really". The
        # fastest way to find out which block is blowing the area budget.
        if {$MODULE eq ""} { error "build.tcl: mode=ooc needs module=<name>" }
        banner "out-of-context synthesis — $MODULE"
        read_design 0
        synth_design -mode out_of_context -top $MODULE -part $PART \
                     {*}[generic_args $MODULE]
        report_utilization -hierarchical \
            -file [file join $REPDIR ooc_${MODULE}_utilization.rpt]
        utilization_summary
        write_checkpoint -force [file join $OUTDIR ooc_${MODULE}.dcp]
    }

    synth - impl - bit - mcs - deploy {
        set reuse [expr {$INCR && [file exists $DCP_SYNTH]}]
        if {$reuse} {
            foreach f [concat $RTL_SRCS $RTL_UNUSED $BOARD_SRCS $XDC_FILES] {
                if {[file mtime $f] > [file mtime $DCP_SYNTH]} { set reuse 0; break }
            }
        }

        if {$reuse} {
            banner "synthesis — reusing $DCP_SYNTH (incr=1, sources unchanged)"
            set_param general.maxThreads $THREADS
            open_checkpoint $DCP_SYNTH
        } else {
            banner "synthesis — $TOP"
            read_design 1
            synth_design -top $TOP -part $PART {*}[generic_args $TOP]
            write_checkpoint -force $DCP_SYNTH
        }

        report_utilization -hierarchical -file [file join $REPDIR post_synth_utilization.rpt]
        report_timing_summary            -file [file join $REPDIR post_synth_timing.rpt]
        utilization_summary

        if {$MODE eq "synth"} {
            puts "  stopping after synthesis (mode=synth)."
            puts "  reports: $REPDIR\n"
            return
        }

        banner "implementation — $TOP"
        opt_design
        place_design
        phys_opt_design
        route_design
        write_checkpoint -force $DCP_ROUTE

        report_utilization -hierarchical -file [file join $REPDIR post_route_utilization.rpt]
        report_timing_summary -max_paths 10 -file [file join $REPDIR post_route_timing.rpt]
        report_drc                       -file [file join $REPDIR post_route_drc.rpt]
        report_io                        -file [file join $REPDIR post_route_io.rpt]
        report_power                     -file [file join $REPDIR post_route_power.rpt]

        utilization_summary
        set timing_failed [timing_verdict]

        if {$MODE eq "impl"} {
            puts "  stopping after routing (mode=impl).\n"
            return
        }

        banner "bitstream"
        write_bitstream -force $BIT
        puts "  wrote $BIT"

        if {$MODE eq "mcs"} {
            banner "flash image"
            write_cfgmem -force -format MCS -interface SPIx4 -size $CFGMEM_SIZE \
                         -loadbit "up 0x0 $BIT" -file $MCS
            puts "  wrote $MCS"
        }

        if {$timing_failed} {
            puts "\n  NOTE: bitstream written despite a timing violation.\n"
        }

        if {$MODE eq "deploy"} {
            # Build then flash, in one command, so the board is running exactly
            # what was just built with exactly these arguments. Deliberately
            # placed after the timing verdict: a design that missed timing still
            # writes a .bit, and this is where you get told before it lands on
            # the board.
            if {$timing_failed} {
                puts "  programming anyway (mode=deploy) — the board will be"
                puts "  running a bitstream that does not meet timing.\n"
            }
            program_device $BIT
        }
    }

    program {
        # No build — just download. Separated so re-flashing does not risk
        # silently rebuilding with different arguments than the .bit was made
        # with. Use mode=deploy when you *do* want both.
        if {$BITFILE eq ""} { set BITFILE $BIT }
        program_device $BITFILE
    }

    default {
        usage
        error "build.tcl: unknown mode '$MODE'"
    }
}

banner [format "done in %d s" [expr {[clock seconds] - $BUILD_START}]]
