module uart_receiver #(
    parameter CLK_PER_BIT = 868 // e.g., 100MHz clock / 115200 baud rate
)(
    input            clk,         // System clock
    input            rst_n,       // Active-low asynchronous reset
    input            uart_rx,          // Incoming serial RX line
    output reg       valid,     // High for 1 cycle when byte is ready
    output reg [7:0] data      // 8-bit parallel output data
);

    // Define state encoding
    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    // Internal registers
    reg [1:0]  state;
    reg [15:0] clk_count; // Tracks clock cycles within a single bit time
    reg [2:0]  bit_idx;   // Tracks current data bit being received (0 to 7)
    reg [7:0]  rx_shift;  // Shifts in incoming data bits
    
    // Double-register the RX input to prevent metastability
    reg rx_reg1;
    reg rx_reg2;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_reg1 <= 1'b1;
            rx_reg2 <= 1'b1;
        end else begin
            rx_reg1 <= uart_rx;
            rx_reg2 <= rx_reg1;
        end
    end

    // Receiver Finite State Machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            clk_count <= 0;
            bit_idx   <= 0;
            rx_shift  <= 8'h00;
            data   <= 8'h00;
            valid   <= 1'b0;
        end else begin
            valid <= 1'b0; // Default state for single-cycle pulse

            case (state)
                IDLE: begin
                    clk_count <= 0;
                    bit_idx   <= 0;
                    // Detect falling edge (Start bit) from synchronized rx line
                    if (rx_reg2 == 1'b0) begin
                        state <= START;
                    end
                end

                START: begin
                    // Wait until the middle of the start bit to verify it is valid
                    if (clk_count == (CLK_PER_BIT / 2) - 1) begin
                        if (rx_reg2 == 1'b0) begin
                            clk_count <= 0; // Reset counter for next bit
                            state     <= DATA;
                        end else begin
                            state     <= IDLE; // Spurious glitch, return to IDLE
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DATA: begin
                    // Sample every bit window in the dead center
                    if (clk_count == CLK_PER_BIT - 1) begin
                        clk_count          <= 0;
                        rx_shift[bit_idx]  <= rx_reg2; // UART is LSB-first
                        
                        if (bit_idx == 7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                STOP: begin
                    // Wait for the full length of the stop bit (must be high)
                    if (clk_count == CLK_PER_BIT - 1) begin
                        if (rx_reg2 == 1'b1) begin
                            data <= rx_shift; // Output valid byte
                            valid <= 1'b1;     // Pulse completion flag
                        end
                        state     <= IDLE;
                        clk_count <= 0;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule