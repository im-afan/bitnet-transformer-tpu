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

    task automatic uart_write_byte(input [7:0] data, input int idle_cycles);
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

        if(errors == 0) begin
            $display("ALL TESTS PASSED"); 
        end else begin
            $display("FAIL %d ERRORS", errors); 
        end

        $finish;
    end

    // Monitor: valid must not assert while the host is still transmitting a
    // frame. Watch for a rising edge of valid during the transmit window; a
    // valid held over from the previous byte (stale data) is not a violation.
    logic valid_d = 0;
    always @(posedge clk) begin
        if(rst_n) begin
            if(transmitting && valid && !valid_d) begin
                $display("  FAIL valid asserted mid-transmission at time %0t", $time);
                errors++;
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