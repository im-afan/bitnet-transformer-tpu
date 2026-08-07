// -----------------------------------------------------------------------------
// uart_echo.sv — UART block-loopback self-test core.
//
// Receives BLOCK_LEN bytes into a register file, then sends those BLOCK_LEN
// bytes back, then repeats. No commands, no addresses, no modes: the block
// length is the whole protocol, fixed at synthesis, and the core runs from reset
// until power-off.
//
// This exists to shrink the search space behind the intermittent corruption on
// the real link (docs/uart_selftest.md). A bad byte there could come from
// uart_receiver, uart_transmitter, uart_interface, the SRAM controller, the
// arbitration mux, the cable or the host; built into cmod_a7_echo_top this
// deletes four of those. It deliberately instantiates the *same* receiver and
// transmitter the production image uses, unmodified — an instrument that alters
// the thing it measures is worthless.
//
// Store-and-forward rather than a streaming echo, and that is a real change of
// what is measured, stated up front:
//
//   * The link is now half duplex by construction. The device never transmits
//     while it is receiving, so TX→RX crosstalk and the receiver's behaviour
//     under simultaneous transmit are no longer exercised. A streaming echo
//     covered those; this does not.
//   * In exchange it reproduces the *turnaround* the production protocol has:
//     a burst in, a gap, a burst out, then the host's next byte arriving right
//     after the reply ends. That gap is where uart_interface's blind
//     SEND_STATUS window sat, so this is the shape of the real traffic.
//   * A byte that arrives while the reply is going out has nowhere to go. It is
//     dropped and flagged on `overrun` (sticky), so the host outrunning the
//     turnaround is a visible event rather than a silent corruption.
//
// And the limitation an echo of any shape carries: it cannot separate "the
// receiver read the byte wrong" from "the transmitter sent it back wrong". A
// mismatch here implicates both blocks, which is still a large narrowing.
// -----------------------------------------------------------------------------

module uart_echo #(
    parameter int CLK_PER_BIT = 104,  // 12 MHz / 115200 baud
    parameter int BLOCK_LEN   = 64,   // bytes buffered per exchange
    parameter int ACT_W       = 19,   // activity LED stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W        = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    input  logic uart_rx,
    output logic uart_tx,

    // Status, for the board wrapper's LEDs.
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag `overrun`
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic overrun       // sticky: a byte arrived while the reply was going out
);

    // Index width for 0..BLOCK_LEN-1. BLOCK_LEN need not be a power of two; the
    // wrap is an explicit compare against BLOCK_LEN-1, not a counter rollover.
    localparam int CNT_W = (BLOCK_LEN <= 1) ? 1 : $clog2(BLOCK_LEN);

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
    // Block buffer and sequencer.
    //
    // One counter serves both phases — it can, because they never overlap: RECV
    // fills 0..BLOCK_LEN-1 and hands over to SEND, which drains the same indices
    // and hands back. That is the whole state: `state` plus `cnt`.
    //
    // SEND returns to RECV when the *last* byte is handed to the transmitter,
    // not when it has finished leaving the pin, so the final frame is still on
    // the wire for ~10 bit periods afterwards. That is deliberate: a host which
    // waits for all BLOCK_LEN bytes before sending again is unaffected, and one
    // which starts early gets its byte accepted instead of counted as an
    // overrun.
    // =========================================================================
    logic [7:0]       block_mem [0:BLOCK_LEN-1];
    logic [CNT_W-1:0] cnt;

    // Encoded as localparams rather than an enum, matching uart_receiver.sv and
    // uart_interface.sv.
    localparam RECV = 1'b0,   // filling block_mem, silent
               SEND = 1'b1;   // draining block_mem, deaf

    logic state;

    // `cnt` is unsigned and zero-extends to the integer compare, so this is a
    // plain equality against the last index in either phase.
    wire last = (cnt == BLOCK_LEN - 1);

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_valid_prev <= 1'b0;
            state         <= RECV;
            cnt           <= '0;
            tx_start      <= 1'b0;
            tx_data       <= 8'b0;
            overrun       <= 1'b0;
            act_cnt       <= '0;
            hb_cnt        <= '0;
        end else begin
            rx_valid_prev <= rx_valid;
            hb_cnt        <= hb_cnt + 1'b1;

            // `tx_start` is a one-cycle strobe; SEND re-raises it below.
            tx_start <= 1'b0;

            case (state)
                RECV: begin
                    if (rx_byte) begin
                        block_mem[cnt] <= rx_data;
                        if (last) begin
                            cnt   <= '0;
                            state <= SEND;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                SEND: begin
                    // Nothing is listening this phase. Record it rather than
                    // dropping it quietly — the host outrunning the turnaround
                    // and the link corrupting a byte look identical from the
                    // other end otherwise.
                    if (rx_byte) overrun <= 1'b1;

                    // Same shape as uart_interface's RD_TX ("if not busy, strobe
                    // start and hand over the byte"), so the transmitter is
                    // driven exactly as it is in the production design. The
                    // !tx_start guard covers the cycle after the strobe, before
                    // the transmitter has had time to assert `busy`.
                    if (!tx_start && !tx_busy) begin
                        tx_data  <= block_mem[cnt];
                        tx_start <= 1'b1;
                        if (last) begin
                            cnt   <= '0;
                            state <= RECV;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end
            endcase

            // ---- activity LED, independent of phase -------------------------
            if (rx_byte) act_cnt <= '1;
            else if (act_cnt != 0) act_cnt <= act_cnt - 1'b1;
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
