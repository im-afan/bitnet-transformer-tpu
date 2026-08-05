`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmod_a7_mem_top.sv — Cmod A7-35T wrapper for the UART + SRAM bring-up image.
//
// Third board target on the same physical board, sitting between the other two:
//
//   cmod_a7        the full TPU. The image test_uart_link.py corrupts against.
//   cmod_a7_mem    this one: uart_receiver/interface/transmitter + the real
//                  sram_controller driving the real chip. No core.
//   cmod_a7_echo   receiver and transmitter only, wired back to back.
//
// The port list is *deliberately identical* to cmod_a7_top's — same pins, same
// widths, same names — so it reuses constraints/cmod_a7.xdc verbatim rather than
// getting an .xdc of its own. That is not laziness: the 30 bank-14 SRAM pins
// switching on the same package pins next to the same J17/J18 UART pair is a
// condition of the experiment. An .xdc copy could drift; a shared one cannot.
//
// Clocking and reset are copied verbatim from cmod_a7_top, for the same reason
// cmod_a7_echo_top copies them: the receiver must see the same clock and the
// same reset release it sees in the real image.
//
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=rtl
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_mem mode=deploy
//
// Output lands in synth/build/cmod_a7_mem/, so the three bitstreams cannot be
// mistaken for one another. host/test_uart_link.py drives this image unchanged
// as long as it is restricted to the SRAM tests — 'I' and 'G' are decoded and
// ACK'd here but do nothing, there being no instruction memory and no core:
//
//   python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
//
// LEDs:
//   led[0]  ~1.4 Hz heartbeat normally; a fast blink if a host byte ever arrived
//           while the device was mid-transmit (sticky, cleared by btn[0]). On a
//           link where the host waits for every reply that must never happen, so
//           a fast blink is a finding — see rtl/uart_memory.sv.
//   led[1]  lit for ~44 ms per received byte: solid while a block streams.
// -----------------------------------------------------------------------------

module cmod_a7_mem_top #(
    // UART: 115200 8N1 off the 12 MHz core clock. 12e6/115200 = 104.17 -> 104,
    // a 0.16% bit-period error, well inside 8N1's ~5% tolerance. Derived in
    // board.tcl from CLK_MHZ so the two cannot drift.
    parameter int UART_CPB        = 104,
    parameter int UART_RX_TIMEOUT = 0,

    // External SRAM (the "DRAM" behind the DMA engine in the full image): 512K x 8.
    parameter int MEM_ADDR_W = 19,
    parameter int MEM_DATA_W = 8,
    parameter int SRAM_CPA   = 2,

    // Power-on reset length, in core clocks (2**POR_W). 2**8 @ 12 MHz ~= 21 us.
    parameter int POR_W = 8
) (
    // ---- 12 MHz system clock (L17) ------------------------------------------
    input  logic                   sysclk,

    // ---- Buttons: btn[0] = active-high manual reset; btn[1] reserved --------
    input  logic [1:0]             btn,

    // ---- LEDs: see the header comment ---------------------------------------
    output logic [1:0]             led,

    // ---- USB-serial bridge --------------------------------------------------
    //   Digilent's names are from the *host* end of the wire: uart_txd_in is the
    //   host's transmit line (an FPGA input) and uart_rxd_out is the host's
    //   receive line (an FPGA output). They cross over relative to uart_memory's
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
    // Reset generation — verbatim from cmod_a7_top.sv.
    //
    // 7-series flops come out of configuration at their declared initial value,
    // so por_cnt starts at 0 and holds por_n low for 2**POR_W clocks before
    // saturating. The POR is required because btn[0] is released at power-up, so
    // without it the core would never see a reset after configuration.
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
    // debounced -- a bouncing press just asserts reset a few extra times.
    logic rst_n;
    assign rst_n = por_n & ~btn0_sync;

    // btn[1] is brought out to a pin but intentionally unused, matching
    // cmod_a7_top. Referencing it keeps the port from reading as an oversight.
    wire unused_btn1 = btn[1];

    // =========================================================================
    // The UART + SRAM core.
    // =========================================================================
    logic blink_slow, blink_fast, activity, collision;

    uart_memory #(
        .UART_CPB        (UART_CPB),
        .UART_RX_TIMEOUT (UART_RX_TIMEOUT),
        .MEM_ADDR_W      (MEM_ADDR_W),
        .MEM_DATA_W      (MEM_DATA_W),
        .SRAM_CPA        (SRAM_CPA)
    ) u_mem (
        .clk        (sysclk),
        .rst_n      (rst_n),

        // Note the crossover described in the port comment above.
        .uart_rx    (uart_txd_in),
        .uart_tx    (uart_rxd_out),

        .sram_addr  (sram_addr),
        .sram_data  (sram_data),
        .sram_we    (sram_we),
        .sram_ce    (sram_ce),
        .sram_oen   (sram_oen),

        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .collision  (collision)
    );

    assign led[0] = collision ? blink_fast : blink_slow;
    assign led[1] = activity;

endmodule
