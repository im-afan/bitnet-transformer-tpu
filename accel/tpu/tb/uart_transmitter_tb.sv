`timescale 1ns / 1ps

module uart_transmitter_tb;
    localparam int CLK_PER_BIT = 868;

    logic clk, rst_n, start;
    logic [7:0] tx_data;
    logic uart_tx;
    logic busy;
    int errors = 0;

    uart_transmitter #(
        .CLK_PER_BIT(CLK_PER_BIT)
    ) transmitter (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .data(tx_data),
        .uart_tx(uart_tx),
        .busy(busy)
    );

    always #5 clk = ~clk;

    // Kick off a transmission: one-cycle start pulse with payload latched.
    task automatic uart_send(input [7:0] data);
        @(posedge clk);
        tx_data = data;
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;
    endtask

    // Independently decode the tx line, sampling each bit at its center.
    // Checks framing (start/stop bits) and that busy is asserted mid-frame.
    task automatic uart_read_byte(output logic [7:0] data);
        // wait for the start bit edge, then align to the middle of it
        @(negedge uart_tx);
        repeat(CLK_PER_BIT / 2) @(posedge clk);
        if(uart_tx !== 1'b0) begin
            $display("  FAIL start bit not low: got %b", uart_tx);
            errors++;
        end
        if(!busy) begin
            $display("  FAIL busy low during transmission");
            errors++;
        end
        // 8 data bits, LSB first, sampled at their centers
        for(int i = 0; i < 8; i++) begin
            repeat(CLK_PER_BIT) @(posedge clk);
            data[i] = uart_tx;
        end
        // stop bit
        repeat(CLK_PER_BIT) @(posedge clk);
        if(uart_tx !== 1'b1) begin
            $display("  FAIL stop bit not high: got %b", uart_tx);
            errors++;
        end
    endtask

    task automatic test_uart_byte(input [7:0] data, input int idle_cycles);
        logic [7:0] rx;
        rx = 8'bx;

        fork
            uart_send(data);
            uart_read_byte(rx);
        join

        if(rx !== data) begin
            $display("  FAIL data got: %b  exp: %b", rx, data);
            errors++;
        end

        // after the frame the line must idle high and the core return to !busy
        repeat(2) @(posedge clk);
        if(uart_tx !== 1'b1) begin
            $display("  FAIL line not idle-high after frame: got %b", uart_tx);
            errors++;
        end
        if(busy) begin
            $display("  FAIL busy still high after frame");
            errors++;
        end

        repeat(idle_cycles * CLK_PER_BIT) @(posedge clk);
    endtask

    // While the core is idle (not busy) the line must sit high, and a start
    // pulse asserted while busy must be ignored (checked in the sequence below).
    always @(posedge clk) begin
        if(rst_n && !busy && uart_tx !== 1'b1) begin
            $display("  FAIL line not high while idle at time %0t", $time);
            errors++;
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        tx_data = 8'b0;

        repeat(4) @(posedge clk);
        rst_n = 1'b1;

        test_uart_byte(8'h53, 0);
        test_uart_byte(8'h31, 1);
        test_uart_byte(8'haf, 0);
        test_uart_byte(8'hf3, 2);
        test_uart_byte(8'h00, 3);
        test_uart_byte(8'hff, 0);
        test_uart_byte(8'hd1, 0);

        // start pulses raised while busy must not disturb the in-flight frame
        fork
            begin
                uart_send(8'h5a);
                // hammer start with junk mid-transmission; should be ignored
                repeat(3) begin
                    repeat(2 * CLK_PER_BIT) @(posedge clk);
                    tx_data = 8'hff;
                    start = 1'b1;
                    @(posedge clk);
                    start = 1'b0;
                end
            end
            begin
                logic [7:0] rx;
                uart_read_byte(rx);
                if(rx !== 8'h5a) begin
                    $display("  FAIL in-flight frame corrupted: got %b exp %b", rx, 8'h5a);
                    errors++;
                end
            end
        join

        repeat(4) @(posedge clk);

        if(errors == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("FAIL %d ERRORS", errors);
        end

        $finish;
    end

    initial begin
        $dumpfile("uart_transmitter_tb.vcd");
        $dumpvars(0, uart_transmitter_tb);
    end

endmodule
