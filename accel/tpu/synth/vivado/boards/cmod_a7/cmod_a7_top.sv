`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmod_a7_top.sv — Digilent Cmod A7-35T board wrapper for tpu_top.
//
// tpu_top is board-agnostic and exposes two host paths (docs/uart_host.md):
//
//   1. a wide parallel port (host_run/boot_pc, imem_we/waddr/wdata, cfg_*), and
//   2. the UART serial link (uart_rx/uart_tx).
//
// The Cmod A7 has 48 usable I/O and no room for (1) -- imem_wdata alone is 32
// pins. So this wrapper ties the parallel port off and brings out only the UART,
// which already covers everything: load external SRAM, load instruction memory,
// start the program, read results back. That is exactly the configuration
// tb/tpu_top_uart_tb.sv validates (it also holds host_run/imem_we/cfg_we at 0
// and drives the DUT purely over the serial pins), so the hardware and the
// passing end-to-end simulation are the same design.
//
// Clocking. The board's only oscillator is 12 MHz (L17) and it feeds the core
// directly -- no MMCM, no Clocking Wizard, hence no IP or .xci in the repo. The
// consequence is UART_CPB: 12e6 / 115200 = 104, not the 868 that docs/uart_host.md
// quotes for a 100 MHz core. The host side is unaffected (tpu_uart.py still talks
// 115200); only the divisor changes. build.tcl derives it from board.tcl's
// CLK_MHZ so the two cannot drift.
//
// Reset. rst_n = power-on reset AND (not btn[0]). The POR is required because
// btn[0] is released at power-up, so without it the core would never see a
// reset after configuration.
//
// String parameters (MEM_STYLE, SPAD_INIT) are fixed here rather than exposed as
// build-time generics -- Vivado's `synth_design -generic` does not pass Verilog
// string parameters cleanly. Edit them here if you need a preloaded scratchpad.
//
// There used to be two more: GELU_INIT / EXP_INIT, pointing at rtl/luts/*.hex.
// Those tables were the numerics of the `gelu` / `exp` instructions, and both
// instructions were removed along with the rest of the VPU ops the current
// model does not issue (vpu.sv header). The design now reads no $readmemh file
// except an optional scratchpad preload, so nothing here depends on build.tcl's
// working directory any more.
// -----------------------------------------------------------------------------

module cmod_a7_top #(
    // Geometry — overridden from synth/vivado/boards/cmod_a7/board.tcl.
    parameter int ROWS       = 8,
    parameter int COLS       = 8,
    parameter int VPU_BYTES  = 64,
    parameter int ADDR_W     = 16,
    parameter int XLEN       = 32,
    parameter int IMEM_AW    = 10,
    parameter int CFG_AW     = 5,
    parameter int REG_AW     = 5,
    parameter int M0_W       = 12,
    parameter int N_W        = 4,

    // External SRAM (the "DRAM" behind the DMA engine): 512K x 8.
    parameter int MEM_ADDR_W = 19,
    parameter int MEM_DATA_W = 8,
    parameter int SRAM_CPA   = 0,

    // UART: 115200 8N1 off a 12 MHz core clock. 12e6/115200 = 104.17 -> 104,
    // a 0.16% bit-period error, well inside 8N1's ~5% tolerance.
    parameter int UART_CPB        = 104,
    parameter int UART_RX_TIMEOUT = 0,

    // Power-on reset length, in core clocks (2**POR_W). 2**8 @ 12 MHz ~= 21 us.
    parameter int POR_W = 8
) (
    // ---- 12 MHz system clock (L17) ------------------------------------------
    input  logic                   sysclk,

    // ---- Buttons: btn[0] = active-high manual reset; btn[1] reserved --------
    input  logic [1:0]             btn,

    // ---- LEDs: led[0] = core busy, led[1] = program halted ------------------
    output logic [1:0]             led,

    // ---- USB-serial bridge --------------------------------------------------
    //   Digilent's names are from the *host* end of the wire: uart_txd_in is
    //   the host's transmit line (an FPGA input) and uart_rxd_out is the host's
    //   receive line (an FPGA output). They cross over relative to tpu_top's
    //   uart_rx / uart_tx, which are named from the FPGA's point of view.
    input  logic                   uart_txd_in,
    output logic                   uart_rxd_out,

    // ---- Cellular SRAM ------------------------------------------------------
    output logic [MEM_ADDR_W-1:0]  sram_addr,
    inout  wire  [MEM_DATA_W-1:0]  sram_data,
    output logic                   sram_we,
    output logic                   sram_ce,
    output logic                   sram_oen
);

    // =========================================================================
    // Reset generation.
    //
    // 7-series flops come out of configuration at their declared initial value,
    // so por_cnt starts at 0 and holds por_n low for 2**POR_W clocks before
    // saturating. btn[0] is asynchronous to sysclk and gets a two-flop
    // synchronizer (ASYNC_REG keeps the pair in one slice for MTBF).
    // =========================================================================
    logic [POR_W-1:0] por_cnt = '0;
    logic             por_n   = 1'b0;

    localparam logic [POR_W-1:0] POR_MAX = {POR_W{1'b1}};

    always_ff @(posedge sysclk) begin
        if (por_cnt != POR_MAX) por_cnt <= por_cnt + 1'b1;
        por_n <= (por_cnt == POR_MAX);
    end

    (* ASYNC_REG = "TRUE" *) logic btn0_meta = 1'b0;
    (* ASYNC_REG = "TRUE" *) logic btn0_sync = 1'b0;

    always_ff @(posedge sysclk) begin
        btn0_meta <= btn[0];
        btn0_sync <= btn0_meta;
    end

    // Synchronous, glitch-free: both terms are registered off sysclk. Not
    // debounced -- a bouncing press just asserts reset a few extra times, which
    // is harmless.
    logic rst_n;
    assign rst_n = por_n & ~btn0_sync;

    // =========================================================================
    // Status.
    //   `done` is a level (scalar_unit: done = (state == S_HALT)), not a pulse,
    //   so it drives the LED directly and stays lit until the next run.
    // =========================================================================
    logic busy, done;
    assign led[0] = busy;
    assign led[1] = done;

    // btn[1] is brought out to a pin but intentionally unused; it is reserved
    // for a future manual run/step trigger. Referencing it here keeps the port
    // from reading as an oversight.
    wire unused_btn1 = btn[1];

    // =========================================================================
    // The TPU core.
    // =========================================================================
    tpu_top #(
        .ROWS            (ROWS),
        .COLS            (COLS),
        .VPU_BYTES       (VPU_BYTES),
        .ADDR_W          (ADDR_W),
        .XLEN            (XLEN),
        .M0_W            (M0_W),
        .N_W             (N_W),
        .REG_AW          (REG_AW),
        .IMEM_AW         (IMEM_AW),
        .CFG_AW          (CFG_AW),
        .MEM_ADDR_W      (MEM_ADDR_W),
        .MEM_DATA_W      (MEM_DATA_W),
        .SRAM_CPA        (SRAM_CPA),
        .UART_CPB        (UART_CPB),
        .UART_RX_TIMEOUT (UART_RX_TIMEOUT),
        .MEM_STYLE       ("BRAM"),
        .SPAD_INIT       ("")
    ) u_tpu (
        .clk   (sysclk),
        .rst_n (rst_n),

        // Parallel host port — unused on this board; the UART link below does
        // program load, config and run. Held inactive, matching tpu_top_uart_tb.
        // Widths are written out rather than using '0: Vivado has been
        // inconsistent about unsized literals in port connections.
        .host_run   (1'b0),
        .boot_pc    ({IMEM_AW{1'b0}}),
        .busy       (busy),
        .done       (done),
        .pc_dbg     (),                     // no pins to spare; left open
        .imem_we    (1'b0),
        .imem_waddr ({IMEM_AW{1'b0}}),
        .imem_wdata (32'd0),
        .cfg_we     (1'b0),
        .cfg_waddr  ({CFG_AW{1'b0}}),
        .cfg_wdata  ({XLEN{1'b0}}),

        // External SRAM pins
        .sram_addr (sram_addr),
        .sram_data (sram_data),
        .sram_we   (sram_we),
        .sram_ce   (sram_ce),
        .sram_oen  (sram_oen),

        // Serial link (note the crossover described in the port comment above)
        .uart_rx (uart_txd_in),
        .uart_tx (uart_rxd_out)
    );

endmodule
