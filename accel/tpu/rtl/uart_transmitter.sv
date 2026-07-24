// 115200 baud, 8N1
module uart_transmitter #(
    parameter CLK_PER_BIT = 868 // 100 MHz, 115200 baud
) (
    input logic clk,
    input logic rst_n,

    input logic start,         // pulse (while !busy) to begin sending `data`
    input logic [7:0] data,
    output logic uart_tx,
    output logic busy          // high from acceptance of `start` until stop bit ends
);

    localparam [3:0] IDLE = 0, START = 1, DATA = 2, STOP = 3;

    reg [3:0] state, state_n;

    reg [15:0] clk_cnt;
    reg [3:0] bit_cnt;
    reg [7:0] shift;

    // one full bit period has elapsed
    wire bit_done = (clk_cnt + 1 == CLK_PER_BIT);

    always_comb begin
        state_n = state;
        case (state)
            IDLE:  if(start) state_n = START;
            START: if(bit_done) state_n = DATA;
            DATA:  if(bit_done && bit_cnt == 7) state_n = STOP;
            STOP:  if(bit_done) state_n = IDLE;
        endcase
    end

    // line idles high; start bit low, data LSB-first, stop bit high
    always_comb begin
        case (state)
            START:   uart_tx = 1'b0;
            DATA:    uart_tx = shift[bit_cnt];
            default: uart_tx = 1'b1; // IDLE, STOP
        endcase
    end

    always_comb busy = (state != IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= IDLE;
            clk_cnt <= 16'b0;
            bit_cnt <= 4'b0;
            shift <= 8'b0;
        end else begin
            state <= state_n;

            if(state == IDLE) begin
                clk_cnt <= 16'b0;
                bit_cnt <= 4'b0;
                if(start) shift <= data; // latch payload on acceptance
            end else begin
                if(bit_done) begin
                    clk_cnt <= 16'b0;
                    if(state == DATA) bit_cnt <= bit_cnt + 1;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
        end
    end
endmodule
