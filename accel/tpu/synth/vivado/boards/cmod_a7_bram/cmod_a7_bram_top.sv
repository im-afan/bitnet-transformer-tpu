`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmod_a7_bram_top.sv — Cmod A7-35T wrapper for the UART + block-RAM image.
//
// Fourth board target on the same physical board, one rung below cmod_a7_mem:
//
//   cmod_a7        the full TPU. The image test_uart_link.py corrupts against.
//   cmod_a7_mem    uart_receiver/interface/transmitter + the real sram_controller
//                  driving the real chip. No core.
//   cmod_a7_bram   this one: the same protocol, backed by on-chip block RAM.
//                  No external memory, no bidirectional bus, six pins.
//   cmod_a7_echo   receiver and transmitter only, wired back to back.
//
// The command protocol on the wire is byte-for-byte what cmod_a7_mem speaks, so
// host/test_uart_link.py drives this image unchanged (see below for the two
// tests that do not apply). What is gone is the memory: run the same soak
// against both images and the difference is `sram_controller` and the 30 bank-14
// pins it switches, and nothing else. See rtl/uart_bram.sv.
//
// Clocking and reset are copied verbatim from cmod_a7_top, for the same reason
// the other two wrappers copy them: the receiver must see the same clock and the
// same reset release it sees in the real image.
//
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_bram mode=rtl
//   vivado -mode batch -source build.tcl -tclargs board=cmod_a7_bram mode=deploy
//
// Output lands in synth/build/cmod_a7_bram/, so the four bitstreams cannot be
// mistaken for one another.
//
//   python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
//   python accel/tpu/host/test_uart_link.py -p COM5 --only sram_write
//
// `sram_isolation` and `sram_address_bus` are expected to FAIL here and are not
// evidence of anything: both probe the top of the 19-bit space to check the
// physical address lines of a chip this image does not have, and only the low
// BRAM_AW bits exist (led[1], below, says so). `sram_long_transfer` (--slow)
// writes 69631 bytes from the base and needs bram_aw=17.
//
// LEDs:
//   led[0]  ~1.4 Hz heartbeat normally; a fast blink if a host byte ever arrived
//           while the device was mid-transmit, or if uart_interface ever lost
//           one to a full holding register (sticky, cleared by btn[0]). On a link
//           where the host waits for every reply that must never happen, so a
//           fast blink is a finding — see rtl/uart_bram.sv.
//   led[1]  lit for ~44 ms per received byte: solid while a block streams. A
//           fast blink while otherwise idle means an access aliased — the host
//           addressed a byte above 2**BRAM_AW and got the folded-down one.
// -----------------------------------------------------------------------------

module cmod_a7_bram_top #(
    // UART: 115200 8N1 off the 12 MHz core clock. 12e6/115200 = 104.17 -> 104,
    // a 0.16% bit-period error, well inside 8N1's ~5% tolerance. Derived in
    // board.tcl from CLK_MHZ so the two cannot drift.
    parameter int UART_CPB        = 104,
    parameter int UART_RX_TIMEOUT = 0,

    // Protocol address space (19 bits, as on cmod_a7) vs. what is built.
    parameter int MEM_ADDR_W = 19,
    parameter int MEM_DATA_W = 8,
    parameter int BRAM_AW    = 16,   // 2**16 = 64 KiB = 16 RAMB36 of the 35T's 50
    parameter int MEM_CPA    = 0,    // = boards/cmod_a7_mem's SRAM_CPA

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
    //   receive line (an FPGA output). They cross over relative to uart_bram's
    //   uart_rx / uart_tx, which are named from the FPGA's point of view.
    input  logic       uart_txd_in,
    output logic       uart_rxd_out
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
    // The UART + block-RAM core.
    // =========================================================================
    logic blink_slow, blink_fast, activity, collision, aliased;

    uart_bram #(
        .UART_CPB        (UART_CPB),
        .UART_RX_TIMEOUT (UART_RX_TIMEOUT),
        .MEM_ADDR_W      (MEM_ADDR_W),
        .MEM_DATA_W      (MEM_DATA_W),
        .BRAM_AW         (BRAM_AW),
        .MEM_CPA         (MEM_CPA)
    ) u_bram (
        .clk        (sysclk),
        .rst_n      (rst_n),

        // Note the crossover described in the port comment above.
        .uart_rx    (uart_txd_in),
        .uart_tx    (uart_rxd_out),

        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .collision  (collision),
        .aliased    (aliased)
    );

    // Two flags on two LEDs. `aliased` is OR'd into the activity LED rather than
    // replacing it: an aliased access says the host is addressing memory this
    // image does not have, which is worth seeing, but not at the cost of the
    // activity indication for the rest of a 30-minute soak. During traffic
    // led[1] is solid either way; the blink is what shows up once the link goes
    // quiet.
    assign led[0] = collision ? blink_fast : blink_slow;
    assign led[1] = activity | (aliased & blink_fast);

endmodule
