`timescale 1ns / 1ps

module uart_receiver_tb;
    localparam int CLK_PER_BIT = 868;

    logic clk, rst_n, uart_rx;
    logic [7:0] uart_data;
    logic valid;
    int errors = 0;

    // High while the host is driving a frame (start bit + 8 data bits).
    // valid must not newly assert during this window.
    logic transmitting = 0;

    uart_receiver #(
        .CLK_PER_BIT(CLK_PER_BIT)
    ) receiver (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx(uart_rx), .data(uart_data),
        .valid(valid)
    );

    always #5 clk = ~clk;

    // Every byte the testbench puts on the wire. Compared against the number of
    // times `valid` rose, so a glitch that manufactures a phantom byte is caught
    // even if it happens to decode to the value we expected anyway.
    int bytes_sent = 0;

    task automatic uart_write_byte(input [7:0] data, input int idle_cycles);
        bytes_sent++;
        transmitting = 1;
        uart_rx = 0;
        repeat(CLK_PER_BIT) @(posedge clk);
        for(int i = 0; i < 8; i++) begin
            uart_rx = data[i];
            repeat(CLK_PER_BIT) @(posedge clk);
        end
        // Data transfer is done; the stop bit is where valid is allowed to rise.
        transmitting = 0;
        uart_rx = 1;
        repeat(CLK_PER_BIT) @(posedge clk);
        repeat(idle_cycles * CLK_PER_BIT) @(posedge clk);
    endtask

    // As above, but with a `glitch_clks`-long low pulse punched into the stop bit
    // `glitch_at` clocks in. A receiver that arms on a single low sample reads
    // that pulse as a start bit and frames the *next* byte up to a bit period
    // early, which rewrites it silently (0x40 arrives as 0x80).
    task automatic uart_write_byte_glitch(input [7:0] data,
                                          input int glitch_at,
                                          input int glitch_clks);
        bytes_sent++;
        transmitting = 1;
        uart_rx = 0;
        repeat(CLK_PER_BIT) @(posedge clk);
        for(int i = 0; i < 8; i++) begin
            uart_rx = data[i];
            repeat(CLK_PER_BIT) @(posedge clk);
        end
        transmitting = 0;
        uart_rx = 1;
        repeat(glitch_at) @(posedge clk);
        uart_rx = 0;
        repeat(glitch_clks) @(posedge clk);
        uart_rx = 1;
        repeat(CLK_PER_BIT - glitch_at - glitch_clks) @(posedge clk);
    endtask

    task automatic test_uart_byte(input [7:0] data, input int idle_cycles);
        uart_write_byte(data, idle_cycles);

        if(!valid) begin
            $display("  FAIL valid got: %b  exp: %b", valid, 1'b1);
            errors++;
        end
        if(data != uart_data) begin
            $display("  FAIL data got: %b  exp: %b", uart_data, data);
            errors++;
        end
    endtask

    task automatic check_byte(input [7:0] exp, input string what);
        if(!valid || uart_data !== exp) begin
            $display("  FAIL %s: got %02h (valid=%b)  exp: %02h",
                     what, uart_data, valid, exp);
            errors++;
        end
    endtask

    // A short low in the stop bit of `first` must leave both `first` and the
    // byte that follows it intact.
    task automatic test_glitch(input [7:0] first, input [7:0] second,
                               input int glitch_at, input int glitch_clks,
                               input string what);
        uart_write_byte_glitch(first, glitch_at, glitch_clks);
        check_byte(first, what);
        uart_write_byte(second, 0);
        check_byte(second, what);
    endtask

    // A low pulse on a line that has been idle for a while, with no byte around
    // it at all: must not produce anything, and must not disturb the next byte.
    task automatic test_idle_glitch(input [7:0] data, input int glitch_clks,
                                    input string what);
        uart_rx = 1;
        repeat(2 * CLK_PER_BIT) @(posedge clk);
        uart_rx = 0;
        repeat(glitch_clks) @(posedge clk);
        uart_rx = 1;
        repeat(2 * CLK_PER_BIT) @(posedge clk);
        uart_write_byte(data, 0);
        check_byte(data, what);
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        uart_rx = 1'b1;

        repeat(4) @(posedge clk);
        rst_n = 1'b1;

        test_uart_byte(8'h53, 0);
        test_uart_byte(8'h31, 1);
        test_uart_byte(8'haf, 0);
        test_uart_byte(8'hf3, 2);
        test_uart_byte(8'h00, 100);
        test_uart_byte(8'hff, 0);
        test_uart_byte(8'hd1, 0);

        // ---- start-bit glitch rejection --------------------------------------
        //
        // Reproduces the hardware failure: a `R` command header is 52 00 00 00 00
        // 40, so the byte that follows a run of zeros is the length field. A false
        // start in the stop bit before it shifts the frame and 0x40 is received as
        // 0x80 -- the interface then streams back twice the bytes the host asked
        // for. The four glitch positions cover the whole stop bit, because how far
        // the frame shifts depends on where in the bit the glitch lands:
        //   4    -> a full bit period of shift  (0x40 -> 0x80)
        //   440  -> about half a bit of shift   (0x40 -> 0xc0)
        //   860  -> a few clocks, only caught by re-arming immediately rather
        //           than waiting out the half bit before rejecting
        test_glitch(8'h00, 8'h40,   4,   8, "glitch at stop-bit start");
        test_glitch(8'h00, 8'h40, 200,   8, "glitch mid stop bit");
        test_glitch(8'h00, 8'h40, 440,   8, "glitch at half a bit of shift");
        test_glitch(8'h00, 8'h40, 860,   4, "glitch just before the real start bit");

        // Longest pulse the receiver is still required to reject: anything up to
        // the mid-bit resample point. Past that it is indistinguishable from a
        // start bit and only a filter on the pin can help.
        test_glitch(8'h55, 8'hff,  50, CLK_PER_BIT/2 - 1, "longest rejectable glitch");

        test_idle_glitch(8'h40, 8, "glitch on an idle line");

        if(valid_rises != bytes_sent) begin
            $display("  FAIL valid rose %0d times for %0d bytes sent (phantom byte)",
                     valid_rises, bytes_sent);
            errors++;
        end

        if(errors == 0) begin
            $display("ALL TESTS PASSED (%0d bytes sent, %0d valid rises)",
                     bytes_sent, valid_rises);
        end else begin
            $display("FAIL %d ERRORS", errors); 
        end

        $finish;
    end

    // Monitor: valid must not assert while the host is still transmitting a
    // frame. Watch for a rising edge of valid during the transmit window; a
    // valid held over from the previous byte (stale data) is not a violation.
    logic valid_d = 0;
    int   valid_rises = 0;
    always @(posedge clk) begin
        if(rst_n) begin
            if(valid && !valid_d) begin
                valid_rises++;
                if(transmitting) begin
                    $display("  FAIL valid asserted mid-transmission at time %0t", $time);
                    errors++;
                end
            end
            valid_d <= valid;
        end else begin
            valid_d <= 0;
        end
    end

    initial begin
        $dumpfile("uart_receiver_tb.vcd");
        $dumpvars(0, uart_receiver_tb);
    end

endmodule