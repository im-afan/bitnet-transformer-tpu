`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_echo_tb.sv — testbench for rtl/uart_echo.sv.
//
//   make echo
//
// The DUT is store-and-forward: it takes BLOCK_LEN bytes in, then sends the same
// BLOCK_LEN bytes back. So the checks are about the *exchange*, not about
// keeping up with a stream:
//
//   * nothing comes out until the block is complete — the device must not reply
//     to the first BLOCK_LEN-1 bytes,
//   * all BLOCK_LEN come back, in order, and then it goes quiet,
//   * consecutive blocks work, i.e. the sequencer returns to RECV cleanly and
//     the second block is not off by a byte,
//   * a byte arriving during the reply is dropped, flagged on `overrun`, and
//     does not corrupt the reply or desync the next block.
//
// The reply decoder still runs in a `fork` alongside the driver even though the
// protocol is half duplex. It has to: uart_receiver raises `valid` entering the
// stop bit, so the device starts transmitting roughly a bit period *before* the
// last byte's stop bit finishes on the wire. A decoder started after the drive
// loop returns would miss the first reply frame.
//
// CLK_PER_BIT is 104, matching the hardware image (12 MHz / 115200), rather than
// the 868 the other UART testbenches use for a 100 MHz core. The sampling margin
// is what is under test here, and it is proportionally tighter at 104.
// -----------------------------------------------------------------------------

module uart_echo_tb;
    localparam int CLK_PER_BIT = 104;      // 12 MHz / 115200, as on the board
    localparam int BLOCK_LEN   = 64;       // as on the board (board.tcl BLOCK_LEN)

    logic clk = 0, rst_n = 0;
    logic uart_rx = 1, uart_tx;
    logic blink_slow, blink_fast, activity, overrun;

    int errors = 0;

    logic [7:0] sent [0:BLOCK_LEN-1];
    logic [7:0] got  [0:BLOCK_LEN-1];

    uart_echo #(
        .CLK_PER_BIT (CLK_PER_BIT),
        .BLOCK_LEN   (BLOCK_LEN),
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
        .overrun    (overrun)
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
    // One exchange: BLOCK_LEN bytes in, BLOCK_LEN bytes back.
    //
    // `extra_byte` (>= 0) injects one more byte immediately after the block, so
    // it lands while the reply is going out. It must be dropped and flagged, not
    // stored — that is the overrun case.
    // =========================================================================
    task automatic run_block(input int gap_bits, input int extra_byte,
                             input string what);
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
                for (int i = 0; i < BLOCK_LEN; i++) begin
                    tx_byte(sent[i], gap_bits);

                    // With BLOCK_LEN-1 bytes delivered the device must still be
                    // silent. A streaming echo would already have replied to
                    // most of them; this one owes nothing until the block is
                    // whole.
                    if (i == BLOCK_LEN - 2 && tx_frames != tx0) begin
                        $display("  FAIL %s: %0d frame(s) came back before the block was complete",
                                 what, tx_frames - tx0);
                        errors++;
                    end
                end
                // Truncated to 8 bits by the task's port width.
                if (extra_byte >= 0) tx_byte(extra_byte, 0);
            end
            begin : decode_device_tx
                for (int i = 0; i < BLOCK_LEN; i++) begin
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

        for (int i = 0; i < BLOCK_LEN; i++) begin
            if (got[i] !== sent[i]) begin
                if (first_bad < 0) begin
                    first_bad = i;
                    $display("  FAIL %s: byte %0d echoed as %02h, sent %02h (xor %08b)",
                             what, i, got[i], sent[i], got[i] ^ sent[i]);
                end
                errors++;
            end
        end

        if (rx_frames - rx0 != BLOCK_LEN + (extra_byte >= 0 ? 1 : 0)) begin
            $display("  FAIL %s: receiver saw %0d bytes, host sent %0d",
                     what, rx_frames - rx0,
                     BLOCK_LEN + (extra_byte >= 0 ? 1 : 0));
            errors++;
        end
        if (tx_frames - tx0 != BLOCK_LEN) begin
            $display("  FAIL %s: transmitter sent %0d frames, expected %0d",
                     what, tx_frames - tx0, BLOCK_LEN);
            errors++;
        end

        // Nothing more may come out. A phantom frame here is the hardware
        // symptom that started all this (the device over-running a reply), and
        // in the overrun case it would mean the dropped byte was not dropped.
        quiet_tx = tx_frames;
        repeat (20 * CLK_PER_BIT) @(posedge clk);
        if (tx_frames != quiet_tx) begin
            $display("  FAIL %s: %0d extra frame(s) after the reply ended",
                     what, tx_frames - quiet_tx);
            errors++;
        end

        if (errors == err0)
            $display("  ok   %s: %0d bytes in and back, %0d bit gap",
                     what, BLOCK_LEN, gap_bits);
    endtask

    // Refill `sent` from the LFSR, so consecutive blocks carry different data
    // and a stale buffer cannot pass as a correct reply.
    logic [15:0] lfsr;

    task automatic fill_payload();
        for (int i = 0; i < BLOCK_LEN; i++) begin
            for (int k = 0; k < 8; k++)
                lfsr = {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            sent[i] = lfsr[7:0];
        end
    endtask

    // =========================================================================
    // Stimulus.
    // =========================================================================
    initial begin
        // Deterministic pseudo-random payload. An LFSR rather than $random so
        // the vector is identical under any simulator.
        lfsr = 16'hACE1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // The case that matters: a completely gapless block, so the receiver
        // gets no slack between frames and any per-byte drift accumulates.
        fill_payload();
        run_block(0, -1, "back-to-back");

        // A second gapless block with fresh data, immediately after the first.
        // This is what catches a sequencer that returns to RECV with a stale
        // count: the payload would come back shifted, not corrupted.
        fill_payload();
        run_block(0, -1, "second block");

        // Same shape with room to breathe. If this passes and the ones above
        // fail, the fault is in the re-arm/turnaround timing, not in sampling.
        fill_payload();
        run_block(3, -1, "3-bit gaps");

        if (overrun) begin
            $display("  FAIL overrun set during normal exchanges");
            errors++;
        end

        // Last, because `overrun` is sticky: a byte sent while the reply is
        // going out must be dropped (reply intact, no extra frame, and the next
        // block still lines up) and must be visible on the status output.
        fill_payload();
        run_block(0, 8'h5A, "byte during reply");
        if (!overrun) begin
            $display("  FAIL a byte arrived during the reply but overrun stayed low");
            errors++;
        end

        // And the dropped byte must not have left the sequencer half a block
        // out of step.
        fill_payload();
        run_block(0, -1, "block after overrun");

        if (errors == 0)
            $display("ALL TESTS PASSED (%0d bytes in, %0d frames out)",
                     rx_frames, tx_frames);
        else
            $display("FAIL %0d ERRORS", errors);

        $finish;
    end

    // Watchdog. Each exchange is 128 frames — 1280 bit periods of 104 clocks at
    // 10 ns, so ~1.4 ms of simulated time; the six here plus gaps land around
    // 9 ms. Anything past this means a task is blocked waiting for a frame that
    // never came.
    initial begin
        #40_000_000;
        $display("FAIL timeout — %0d bytes in, %0d frames out", rx_frames, tx_frames);
        $finish;
    end

    initial begin
        $dumpfile("uart_echo_tb.vcd");
        $dumpvars(0, uart_echo_tb);
    end

endmodule
