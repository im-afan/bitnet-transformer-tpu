`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_echo_tb.sv — testbench for rtl/uart_echo.sv.
//
//   make echo
//
// Drives bytes into uart_rx and decodes uart_tx concurrently, so the DUT is
// receiving and transmitting at the same time — the condition the real link runs
// under, and the one a "send everything, then read everything" testbench never
// reaches.
//
// The headline case is `back-to-back`: 256 bytes with zero inter-byte gap. That
// is where an echo has to actually keep up, and where any re-arm timing problem
// in uart_receiver's return to IDLE would show. See docs/uart_selftest.md.
//
// CLK_PER_BIT is 104, matching the hardware image (12 MHz / 115200), rather than
// the 868 the other UART testbenches use for a 100 MHz core. The sampling margin
// is what is under test here, and it is proportionally tighter at 104.
// -----------------------------------------------------------------------------

module uart_echo_tb;
    localparam int CLK_PER_BIT = 104;      // 12 MHz / 115200, as on the board
    localparam int FIFO_AW     = 8;
    localparam int MAX_BYTES   = 256;

    logic clk = 0, rst_n = 0;
    logic uart_rx = 1, uart_tx;
    logic blink_slow, blink_fast, activity, overflow;

    int errors = 0;

    logic [7:0] sent [0:MAX_BYTES-1];
    logic [7:0] got  [0:MAX_BYTES-1];

    uart_echo #(
        .CLK_PER_BIT (CLK_PER_BIT),
        .FIFO_AW     (FIFO_AW),
        .ACT_W       (6),               // tiny, so the LED logic still toggles in sim
        .HB_W        (8)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .uart_rx    (uart_rx),
        .uart_tx    (uart_tx),
        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .overflow   (overflow)
    );

    always #5 clk = ~clk;

    // =========================================================================
    // Frame counters, taken from inside the DUT rather than inferred from the
    // pins. `rx_byte` is the rising edge of the receiver's valid — exactly what
    // uart_interface counts — and `tx_start` is one strobe per transmitted
    // frame. Counting the wire instead would mean telling a start bit from a
    // data bit that happens to be low, which is the very thing under test.
    // =========================================================================
    int rx_frames = 0, tx_frames = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.rx_byte)  rx_frames++;
            if (dut.tx_start) tx_frames++;
        end
    end

    // =========================================================================
    // Wire tasks.
    // =========================================================================

    // 8N1 out of the host: start low, 8 data bits LSB first, stop high, then
    // `gap_bits` extra idle bit periods (0 = fully back to back).
    task automatic tx_byte(input [7:0] b, input int gap_bits);
        uart_rx = 1'b0;
        repeat (CLK_PER_BIT) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            uart_rx = b[i];
            repeat (CLK_PER_BIT) @(posedge clk);
        end
        uart_rx = 1'b1;
        repeat (CLK_PER_BIT) @(posedge clk);
        if (gap_bits > 0) repeat (gap_bits * CLK_PER_BIT) @(posedge clk);
    endtask

    // Decode one frame off uart_tx the way the host's FTDI does: find the
    // falling edge, wait a bit and a half to land in the middle of data bit 0,
    // then step a bit period at a time. Returns at the centre of the stop bit,
    // half a bit before the next start edge can occur, so back-to-back frames
    // are decoded without a race.
    task automatic rx_frame(output logic [7:0] b, output logic framing_ok);
        @(negedge uart_tx);
        repeat (CLK_PER_BIT + CLK_PER_BIT/2) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            b[i] = uart_tx;
            repeat (CLK_PER_BIT) @(posedge clk);
        end
        framing_ok = (uart_tx === 1'b1);
    endtask

    // =========================================================================
    // One full-duplex pass: stream `n` bytes in while decoding `n` back out.
    // =========================================================================
    task automatic run_stream(input int n, input int gap_bits, input string what);
        logic       ok;
        logic [7:0] b;
        int         rx0, tx0, err0, quiet_tx;
        int         first_bad;

        rx0 = rx_frames;
        tx0 = tx_frames;
        err0 = errors;
        first_bad = -1;

        fork
            begin : drive_host_tx
                for (int i = 0; i < n; i++) tx_byte(sent[i], gap_bits);
            end
            begin : decode_device_tx
                for (int i = 0; i < n; i++) begin
                    // Decoded into a scalar first: passing an array element as a
                    // task output argument is legal SystemVerilog but not
                    // uniformly supported.
                    rx_frame(b, ok);
                    got[i] = b;
                    if (!ok) begin
                        $display("  FAIL %s: byte %0d came back with a low stop bit",
                                 what, i);
                        errors++;
                    end
                end
            end
        join

        for (int i = 0; i < n; i++) begin
            if (got[i] !== sent[i]) begin
                if (first_bad < 0) begin
                    first_bad = i;
                    $display("  FAIL %s: byte %0d echoed as %02h, sent %02h (xor %08b)",
                             what, i, got[i], sent[i], got[i] ^ sent[i]);
                end
                errors++;
            end
        end

        if (rx_frames - rx0 != n) begin
            $display("  FAIL %s: receiver saw %0d bytes, host sent %0d",
                     what, rx_frames - rx0, n);
            errors++;
        end
        if (tx_frames - tx0 != n) begin
            $display("  FAIL %s: transmitter sent %0d frames, expected %0d",
                     what, tx_frames - tx0, n);
            errors++;
        end

        // Nothing more may come out. A phantom frame here is the hardware
        // symptom that started all this (the device over-running a reply).
        quiet_tx = tx_frames;
        repeat (20 * CLK_PER_BIT) @(posedge clk);
        if (tx_frames != quiet_tx) begin
            $display("  FAIL %s: %0d extra frame(s) after the stream ended",
                     what, tx_frames - quiet_tx);
            errors++;
        end

        if (errors == err0)
            $display("  ok   %s: %0d bytes echoed intact, %0d bit gap",
                     what, n, gap_bits);
    endtask

    // =========================================================================
    // Stimulus.
    // =========================================================================
    logic [15:0] lfsr;

    initial begin
        // Deterministic pseudo-random payload. An LFSR rather than $random so
        // the vector is identical under any simulator.
        lfsr = 16'hACE1;
        for (int i = 0; i < MAX_BYTES; i++) begin
            for (int k = 0; k < 8; k++)
                lfsr = {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            sent[i] = lfsr[7:0];
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // The case that matters: a completely gapless stream, long enough that
        // any per-byte drift in the echo path would have accumulated.
        run_stream(MAX_BYTES, 0, "back-to-back");

        // Same payload with room to breathe. If this passes and the one above
        // fails, the fault is in the re-arm/turnaround timing, not in sampling.
        run_stream(64, 3, "3-bit gaps");

        // Single bytes, well separated — the trivial case, checked last so a
        // failure here means something very basic is wrong.
        run_stream(8, 20, "isolated bytes");

        if (overflow) begin
            $display("  FAIL echo FIFO overflowed (depth %0d)", 1 << FIFO_AW);
            errors++;
        end

        if (errors == 0)
            $display("ALL TESTS PASSED (%0d bytes in, %0d frames out)",
                     rx_frames, tx_frames);
        else
            $display("FAIL %0d ERRORS", errors);

        $finish;
    end

    // Watchdog. The full run is about 3.4 ms of simulated time; anything past
    // 20 ms means a task is blocked waiting for a frame that never came.
    initial begin
        #20_000_000;
        $display("FAIL timeout — %0d bytes in, %0d frames out", rx_frames, tx_frames);
        $finish;
    end

    initial begin
        $dumpfile("uart_echo_tb.vcd");
        $dumpvars(0, uart_echo_tb);
    end

endmodule
