// -----------------------------------------------------------------------------
// uart_bram.sv — UART command link + on-chip block RAM. rtl/uart_memory.sv with
// the external memory taken out.
//
// One substitution and nothing else: sram_controller becomes bram_controller
// (rtl/bram.sv), which presents the identical user-side handshake with the same
// cycle counts, so uart_interface is driven exactly as it is in the other two
// images and is again instantiated **unmodified**. What leaves the design with
// the SRAM is the 30 switching bank-14 pins, the bidirectional data bus, the
// output-enable/chip-select timing and the chip itself.
//
// That makes it the next rung down the ladder in docs/uart_selftest.md:
//
//   cmod_a7        everything. The image that fails.
//   cmod_a7_mem    the protocol and the external memory, no core.
//   cmod_a7_bram   this one: the protocol only. Memory is on-chip block RAM.
//   cmod_a7_echo   neither. Just the wire framing.
//
// cmod_a7_mem already narrowed the fault to `uart_interface` or
// `sram_controller` (CLAUDE.md). This image is the cut between those two, and it
// is a cut the echo image cannot make: the echo has no command FSM, no address
// phase and no ACK, so a clean echo run says nothing about `uart_interface`.
// Here the protocol is intact byte for byte on the wire — same command set, same
// 3-byte addresses, same 19-bit range checks, same turnaround — and only the
// storage behind it has changed. So:
//
//   * still corrupts  =>  the fault is in uart_interface (or the host), and
//                         sram_controller and the memory bus are cleared.
//   * runs clean      =>  the fault needs the external memory present:
//                         sram_controller, the bus, or the pins it switches.
//
// Differences a driver can observe, all of them consequences of on-chip storage:
//
//   * Capacity. The protocol address space is still the full 19 bits, because
//     the range checks in uart_interface are part of what is under test, but
//     only 2**BRAM_AW bytes exist. Addresses above that alias down onto the
//     window (see rtl/bram.sv). With the default 64 KiB, `host/test_uart_link.py`
//     passes everything except the two tests that deliberately probe the top of
//     the 19-bit space — `sram_isolation` and `sram_address_bus`, both of which
//     are testing the physical address lines of a chip that is not in this
//     image. `sram_long_transfer` (--slow) needs bram_aw=17. Every access that
//     aliases sets the sticky `aliased` output, so this is never silent.
//   * Memory comes up zeroed at configuration rather than holding whatever the
//     SRAM was left with, so a read before the first write returns 0x00 instead
//     of stale data.
//
// Everything else is uart_memory verbatim, including the two deliberate
// departures from tpu_top:
//
//   * 'I' (write IMEM) and 'G' (go) are decoded, range-checked and ACK'd,
//     because uart_interface is unmodified — but there is no instruction memory
//     and no core, so the write lands nowhere and the run never starts. Use
//     'R'/'W' only.
//   * `core_busy` is tied low, so nothing is ever NAK'd for arbitration.
// -----------------------------------------------------------------------------

module uart_bram #(
    // ---- UART host link ------------------------------------------------------
    parameter int UART_CPB        = 104,  // 12 MHz / 115200 baud
    parameter int UART_RX_TIMEOUT = 0,    // inter-byte abort (clocks; 0 = off)

    // ---- Memory --------------------------------------------------------------
    //   MEM_ADDR_W is the protocol address width and must stay 19 to match
    //   boards/cmod_a7 — it is what uart_interface range-checks against, and a
    //   rig that rejects different frames than the real image is not a control.
    //   BRAM_AW is how much of that space is actually built.
    parameter int MEM_ADDR_W      = 19,
    parameter int MEM_DATA_W      = 8,
    parameter int BRAM_AW         = 16,   // 2**16 = 64 KiB = 16 RAMB36 of 50
    parameter int MEM_CPA         = 0,    // = boards/cmod_a7_mem's SRAM_CPA

    // ---- Instruction memory address width -----------------------------------
    //   No IMEM exists in this image; this only has to match what the host's
    //   'I' frames would be range-checked against.
    parameter int IMEM_AW         = 10,

    // ---- Status LED timing ---------------------------------------------------
    parameter int ACT_W           = 19,   // activity stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W            = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    // ---- USB-serial link -----------------------------------------------------
    input  logic uart_rx,
    output logic uart_tx,

    // ---- Status, for the board wrapper's LEDs -------------------------------
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag the two below
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic collision,    // sticky: see below
    output logic aliased       // sticky: an access landed outside the built window
);

    // =========================================================================
    // Memory controller user side. One requester, so no mux — same as
    // uart_memory: a mux with one input is a thing that can be wrong.
    // =========================================================================
    logic                   mem_start, mem_we, mem_busy, mem_done;
    logic [MEM_ADDR_W-1:0]  mem_addr;
    logic [MEM_DATA_W-1:0]  mem_din, mem_dout;

    // ---- UART host interface -------------------------------------------------
    logic [7:0]             uart_rx_data;
    logic                   uart_rx_valid;
    logic                   uart_tx_start, uart_tx_busy;
    logic [7:0]             uart_tx_data;
    logic                   uart_host_busy;

    // IMEM write port and run trigger. Decoded and driven by uart_interface, and
    // then dropped: there is no scalar unit in this image. Declared rather than
    // left as unnamed open ports so the dead end is explicit in the netlist.
    logic                   uart_imem_we;
    logic [IMEM_AW-1:0]     uart_imem_waddr;
    logic [31:0]            uart_imem_wdata;
    logic                   uart_run_start;
    logic [IMEM_AW-1:0]     uart_run_pc;
    logic                   uart_rx_overrun;

    // =========================================================================
    // Block RAM — the only line that differs from uart_memory.sv.
    // =========================================================================
    bram_controller #(
        .CLOCKS_PER_ACCESS (MEM_CPA),
        .ADDR_W            (MEM_ADDR_W),
        .DATA_W            (MEM_DATA_W),
        .DEPTH_W           (BRAM_AW)
    ) u_bram (
        .clk   (clk),
        .rst_n (rst_n),

        // user side ← UART host (sole owner), one byte per transaction: the
        // range interface at len = 1 (see uart_memory.sv for the same wiring).
        .start  (mem_start),
        .we     (mem_we),
        .addr   (mem_addr),
        .len    (16'd1),
        .stride (16'd0),

        .din       (mem_din),
        .din_valid (1'b1),
        .din_ready (),

        .dout       (mem_dout),
        .dout_valid (),
        .dout_ready (1'b1),   // host path takes every byte

        .busy (mem_busy),
        .done (mem_done),

        .aliased (aliased)
    );

    // =========================================================================
    // UART host link — the production blocks, untouched.
    // =========================================================================
    uart_receiver #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_rx (
        .clk     (clk),
        .rst_n   (rst_n),
        .uart_rx (uart_rx),
        .data    (uart_rx_data),
        .valid   (uart_rx_valid)
    );

    uart_transmitter #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_tx (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (uart_tx_start),
        .data    (uart_tx_data),
        .uart_tx (uart_tx),
        .busy    (uart_tx_busy)
    );

    uart_interface #(
        .ADDR_W     (MEM_ADDR_W),
        .LENGTH_W   (16),
        .IMEM_AW    (IMEM_AW),
        .RX_TIMEOUT (UART_RX_TIMEOUT)
    ) u_uart (
        .clk   (clk),
        .rst_n (rst_n),

        // No core to arbitrate against: nothing is ever NAK'd for being busy.
        .core_busy (1'b0),

        // No scalar unit, so no run to time: 'T' is still decoded and still
        // answers, and it answers 0 forever.
        .cycle_count (32'd0),

        // receiver / transmitter
        .data_in           (uart_rx_data),
        .receiver_valid    (uart_rx_valid),
        .transmitter_start (uart_tx_start),
        .data_out          (uart_tx_data),
        .transmitter_busy  (uart_tx_busy),

        // memory controller user side (sole owner here). The port names are
        // uart_interface's and still say `sram_` — that block is unmodified on
        // purpose, and what it is talking to is not its business.
        .sram_start (mem_start),
        .sram_we    (mem_we),
        .sram_addr  (mem_addr),
        .sram_din   (mem_din),
        .sram_dout  (mem_dout),
        .sram_busy  (mem_busy),
        .sram_done  (mem_done),

        // instruction-memory write port → nowhere (no scalar unit)
        .imem_we    (uart_imem_we),
        .imem_waddr (uart_imem_waddr),
        .imem_wdata (uart_imem_wdata),

        // run trigger → nowhere (no scalar unit)
        .run_start  (uart_run_start),
        .run_pc     (uart_run_pc),

        .host_busy  (uart_host_busy),
        .rx_overrun (uart_rx_overrun)
    );

    // =========================================================================
    // Instrumentation — identical to uart_memory.sv, so the two images report
    // the same events on the same LEDs and a soak run can be compared directly.
    //
    // `rx_byte` is derived exactly as uart_interface.sv derives it: the rising
    // edge of the receiver's `valid`, a level held from STOP until half way
    // through the next start bit. Counting it here means the status logic sees
    // precisely the bytes the command FSM sees, no more.
    //
    // `collision` sets on either of two things, both meaning "the host got ahead
    // of the device":
    //
    //   * a byte landed while the device was mid-transmit. The host is the sole
    //     master and waits for each reply, so it should never happen. Survivable
    //     since uart_interface grew its RX holding register — the byte is kept —
    //     so it is information about the *host*, not a device fault on its own.
    //   * uart_interface's `rx_overrun`: a byte arrived with the holding register
    //     still full, so one really was lost. That is the fault.
    //
    // Sticky until reset: the interesting question is "did this ever happen",
    // and a one-clock pulse 40 minutes into a soak is not something anyone is
    // watching for.
    // =========================================================================
    logic uart_rx_valid_prev;
    wire  rx_byte = uart_rx_valid & ~uart_rx_valid_prev;

    // The transmitter asserts `busy` the cycle *after* it accepts `start`, so
    // the strobe has to be OR'd in or the first cycle of every frame would look
    // idle.
    wire  tx_active = uart_tx_busy | uart_tx_start;

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_rx_valid_prev <= 1'b0;
            act_cnt            <= '0;
            hb_cnt             <= '0;
            collision          <= 1'b0;
        end else begin
            uart_rx_valid_prev <= uart_rx_valid;
            hb_cnt             <= hb_cnt + 1'b1;

            if (uart_rx_overrun) collision <= 1'b1;   // sticky until reset

            if (rx_byte) begin
                act_cnt <= '1;
                if (tx_active) collision <= 1'b1;   // sticky until reset
            end else if (act_cnt != 0) begin
                act_cnt <= act_cnt - 1'b1;
            end
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
