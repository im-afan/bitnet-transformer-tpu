// -----------------------------------------------------------------------------
// uart_echo.sv — UART loopback self-test core.
//
// Every byte that arrives on `uart_rx` goes straight back out on `uart_tx`. No
// commands, no protocol, no modes: it echoes from reset until power-off.
//
// This exists to shrink the search space behind the intermittent corruption on
// the real link (docs/uart_selftest.md). A bad byte there could come from
// uart_receiver, uart_transmitter, uart_interface, the SRAM controller, the
// arbitration mux, the cable or the host; built into cmod_a7_echo_top this
// deletes four of those. It deliberately instantiates the *same* receiver and
// transmitter the production image uses, unmodified — an instrument that alters
// the thing it measures is worthless.
//
// The consequence, stated up front: an echo cannot separate "the receiver read
// the byte wrong" from "the transmitter sent it back wrong". A mismatch here
// implicates both blocks, which is still a large narrowing.
// -----------------------------------------------------------------------------

module uart_echo #(
    parameter int CLK_PER_BIT = 104,  // 12 MHz / 115200 baud
    parameter int FIFO_AW     = 8,    // echo buffer depth = 2**FIFO_AW (see below)
    parameter int ACT_W       = 19,   // activity LED stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W        = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    input  logic uart_rx,
    output logic uart_tx,

    // Status, for the board wrapper's LEDs.
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag `overflow`
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic overflow      // sticky: a byte arrived with the FIFO full
);

    localparam int DEPTH = 1 << FIFO_AW;

    // =========================================================================
    // Receiver / transmitter — the production blocks, untouched.
    // =========================================================================
    logic [7:0] rx_data;
    logic       rx_valid;

    uart_receiver #(
        .CLK_PER_BIT (CLK_PER_BIT)
    ) u_rx (
        .clk     (clk),
        .rst_n   (rst_n),
        .uart_rx (uart_rx),
        .data    (rx_data),
        .valid   (rx_valid)
    );

    logic [7:0] tx_data;
    logic       tx_start, tx_busy;

    uart_transmitter #(
        .CLK_PER_BIT (CLK_PER_BIT)
    ) u_tx (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (tx_start),
        .data    (tx_data),
        .uart_tx (uart_tx),
        .busy    (tx_busy)
    );

    // `rx_valid` is a level, not a pulse: uart_receiver raises it entering STOP
    // and only clears it half way through the *next* start bit. So one new byte
    // is one rising edge — the same thing uart_interface.sv derives as `rx_byte`.
    // Matching it matters: this core should see exactly what the real consumer
    // sees, including any byte the receiver merges or invents.
    logic rx_valid_prev;
    wire  rx_byte = rx_valid & ~rx_valid_prev;

    // =========================================================================
    // Echo buffer.
    //
    // Why a FIFO rather than a holding register. The transmitter can only start
    // once a byte has fully arrived, so at the same baud it permanently lags by
    // about one byte. Worse, it never quite catches up: a transmitted byte costs
    // 10*CLK_PER_BIT clocks on the wire plus 2 for the start handshake below,
    // against a received byte arriving every 10 host bit periods. At 12 MHz the
    // FPGA's bit period is 0.16% short of the host's, which claws back most of
    // it, but the net is still ~0.03 us of lag per byte -- occupancy creeps up
    // over a long unbroken stream.
    //
    // 256 entries is ~1.3 bytes of headroom per 4096-byte block, and the host
    // (host/uart_echo.py) waits for each block to return before sending the
    // next, which resets the drift. So `overflow` should never set; if it does,
    // that is a finding rather than a capacity problem, and the board flags it
    // on led[0] without needing the host at all.
    //
    // Pointers carry one extra bit so full and empty are distinguishable.
    // =========================================================================
    logic [7:0]       fifo [0:DEPTH-1];
    logic [FIFO_AW:0] wptr, rptr;

    // `fill` is one bit wider than the index, so it counts 0..DEPTH honestly
    // instead of wrapping DEPTH back onto 0 the way an FIFO_AW-bit difference
    // would. Pushes are gated on !full, so it never exceeds DEPTH.
    wire [FIFO_AW:0] fill  = wptr - rptr;
    wire             full  = (fill == DEPTH);
    wire             empty = (wptr == rptr);

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_valid_prev <= 1'b0;
            wptr          <= '0;
            rptr          <= '0;
            tx_start      <= 1'b0;
            tx_data       <= 8'b0;
            overflow      <= 1'b0;
            act_cnt       <= '0;
            hb_cnt        <= '0;
        end else begin
            rx_valid_prev <= rx_valid;
            hb_cnt        <= hb_cnt + 1'b1;

            // ---- push ------------------------------------------------------
            if (rx_byte) begin
                if (full) begin
                    overflow <= 1'b1;          // sticky until reset
                end else begin
                    fifo[wptr[FIFO_AW-1:0]] <= rx_data;
                    wptr <= wptr + 1'b1;
                end
                act_cnt <= '1;
            end else if (act_cnt != 0) begin
                act_cnt <= act_cnt - 1'b1;
            end

            // ---- pop -------------------------------------------------------
            // Same shape as uart_interface's RD_TX ("if not busy, strobe start
            // and hand over the byte"), so the transmitter is driven exactly as
            // it is in the production design. `tx_start` is a one-cycle strobe;
            // the !tx_start guard covers the cycle after it is raised, before
            // the transmitter has had time to assert `busy`.
            tx_start <= 1'b0;
            if (!tx_start && !tx_busy && !empty) begin
                tx_data  <= fifo[rptr[FIFO_AW-1:0]];
                tx_start <= 1'b1;
                rptr     <= rptr + 1'b1;
            end
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
