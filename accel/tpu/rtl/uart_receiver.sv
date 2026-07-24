// 115200 baud, 8N1
module uart_receiver #(
    parameter CLK_PER_BIT = 868 // 100 MHz, 115200 baud
) (
    input logic clk,
    input logic rst_n,

    input logic uart_rx, 
    output logic [7:0] data,
    output logic valid 
);

    localparam [3:0] IDLE = 0, START = 1, DATA = 2, STOP = 3;

    reg [3:0] state, state_n;

    reg [15:0] clk_cnt;
    reg [3:0] bit_cnt;

    always_comb begin
        state_n = state;
        case (state)
            IDLE: if(!uart_rx) state_n = START;
            START: if(bit_cnt == 1) state_n = DATA;
            DATA: if(bit_cnt == 9) state_n = STOP;
            STOP: if(uart_rx) state_n = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= IDLE;
            clk_cnt <= 16'b0;
            bit_cnt <= 4'b0;
            valid <= 0;
            data <= 8'b0;
        end else begin
            state <= state_n;

            if(state == IDLE) begin
                clk_cnt <= 16'b0;
                bit_cnt <= 4'b0;
            end else begin
                if(clk_cnt + 1 == CLK_PER_BIT) begin
                    clk_cnt <= 16'b0;
                    bit_cnt <= bit_cnt + 1;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            case (state) 
                IDLE: begin 
                     
                end
                START: begin
                    valid <= 0;
                end
                DATA: begin
                    if(clk_cnt == CLK_PER_BIT / 2) begin
                        // $display("data[%d] <= %d\n", bit_cnt-1, uart_rx);
                        data[bit_cnt-1] <= uart_rx;
                    end
                end
                STOP: begin
                    valid <= 1;
                end
            endcase
        end
    end
endmodule