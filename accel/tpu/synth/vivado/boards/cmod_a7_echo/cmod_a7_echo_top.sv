`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmod_a7_echo_top.sv — Cmod A7-35T wrapper for the UART echo self-test.
//
// Sibling of cmod_a7_top.sv, and deliberately a separate board target rather
// than a mode of it: this image has no TPU core, no DMA and no external SRAM, so
// it brings out four pins instead of thirty-five. Anything that goes wrong here
// is in the UART blocks, the cable or the host, with nothing else left to blame.
//
// It builds to synth/build/cmod_a7_echo/cmod_a7_echo_top.bit, so the self-test
// and production bitstreams can never be confused for one another:
//
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_echo mode=rtl
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_echo mode=bit
//
// Clocking and reset are copied verbatim from cmod_a7_top so the receiver sees
// the same clock and the same reset release it sees in the real image: the
// board's only oscillator is 12 MHz on L17 feeding the core directly (no MMCM),
// hence UART_CPB = 12e6/115200 = 104. Copied rather than factored out on
// purpose -- a shared wrapper would be one more thing the two images have in
// common, and the whole point is to have as few of those as possible.
//
// LEDs:
//   led[0]  ~1.4 Hz heartbeat normally; a fast blink if a byte ever arrived
//           while the device was sending a block back (sticky, cleared by
//           btn[0]). Proves the clock is running and the FPGA is configured,
//           independently of the serial link.
//   led[1]  lit for ~44 ms per received byte, so it is solid while a block is
//           streaming in and dark during the reply.
// -----------------------------------------------------------------------------

module cmod_a7_echo_top #(
    // 115200 8N1 off the 12 MHz core clock. 12e6/115200 = 104.17 -> 104, a 0.16%
    // bit-period error, well inside 8N1's ~5% tolerance. Derived in board.tcl
    // from CLK_MHZ so the two cannot drift.
    parameter int UART_CPB = 104,

    // Bytes per exchange: the device buffers this many, then sends them back.
    // The host (host/uart_echo.py --block) must use the same number — there is
    // no framing on the wire to negotiate it with.
    parameter int BLOCK_LEN = 64,

    // Power-on reset length, in core clocks (2**POR_W). 2**8 @ 12 MHz ~= 21 us.
    parameter int POR_W = 8
) (
    // ---- 12 MHz system clock (L17) ------------------------------------------
    input  logic       sysclk,

    // ---- Buttons: btn[0] = active-high manual reset; btn[1] reserved --------
    input  logic [1:0] btn,

    // ---- LEDs: see the header comment ---------------------------------------
    output logic [1:0] led,

    // ---- USB-serial bridge --------------------------------------------------
    //   Digilent's names are from the *host* end of the wire: uart_txd_in is the
    //   host's transmit line (an FPGA input) and uart_rxd_out is the host's
    //   receive line (an FPGA output). They cross over relative to uart_echo's
    //   uart_rx / uart_tx, which are named from the FPGA's point of view.
    input  logic       uart_txd_in,
    output logic       uart_rxd_out
);

    // =========================================================================
    // Reset generation.
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
    // The echo core.
    // =========================================================================
    logic blink_slow, blink_fast, activity, overrun;

    uart_echo #(
        .CLK_PER_BIT (UART_CPB),
        .BLOCK_LEN   (BLOCK_LEN)
    ) u_echo (
        .clk        (sysclk),
        .rst_n      (rst_n),

        // Note the crossover described in the port comment above.
        .uart_rx    (uart_txd_in),
        .uart_tx    (uart_rxd_out),

        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .overrun    (overrun)
    );

    assign led[0] = overrun ? blink_fast : blink_slow;
    assign led[1] = activity;

endmodule
