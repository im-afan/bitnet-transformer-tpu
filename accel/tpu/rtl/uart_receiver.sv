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
    reg rx_reg1, rx_reg2;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rx_reg1 <= 1;
            rx_reg2 <= 1;
        end else begin
            rx_reg1 <= uart_rx;
            rx_reg2 <= rx_reg1; 
        end
    end

    reg [15:0] clk_cnt;
    reg [3:0] bit_cnt;

    // Start-bit re-validation.
    //
    // IDLE arms on a single sample of uart_rx, which is an asynchronous pin with
    // no glitch filter in front of it, so any brief low during a stop bit or the
    // idle gap looks exactly like the start of a byte. Taken at face value it
    // shifts the whole frame by up to one bit period, which rewrites the byte
    // silently -- a one-bit shift turns 0x40 into 0x80 -- and on a command length
    // field that makes uart_interface read or write far past the end of the frame.
    //
    // A real start bit stays low for a full bit period; a glitch does not. So the
    // line must hold low all the way to the bit centre -- the first sample that
    // comes back high drops us to IDLE to re-arm, rather than waiting out the half
    // bit, so the real start bit that may be a few clocks behind the glitch is
    // still caught. `data` and `valid` are left untouched, so a rejected glitch
    // has no effect the rest of the design can observe.
    //
    // Aborting cannot lose a byte: IDLE re-arms on the same condition, so a bounce
    // on a genuine falling edge just re-enters START a clock or two later, well
    // inside the sampling margin.
    // bit_cnt == 0 keeps this to the start bit itself: clk_cnt wraps back to 0 at
    // the end of the bit and the FSM sits in START one more cycle before moving to
    // DATA, so without the guard the window would reopen on top of data bit 0.
    wire in_start_bit = (state == START) && (bit_cnt == 4'd0);
    wire start_glitch = in_start_bit && (clk_cnt <= CLK_PER_BIT / 2) &&  rx_reg2;
    wire start_ok     = in_start_bit && (clk_cnt == CLK_PER_BIT / 2) && !rx_reg2;

    always_comb begin
        state_n = state;
        case (state)
            IDLE: if(!rx_reg2) state_n = START;
            START: begin
                if(start_glitch)      state_n = IDLE;   // not a start bit after all
                else if(bit_cnt == 1) state_n = DATA;
            end
            DATA: if(bit_cnt == 9) state_n = STOP;
            STOP: if(rx_reg2) state_n = IDLE;
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
                    // Only once the start bit has held, so a rejected glitch
                    // cannot pull `valid` down under the consumer's edge detect.
                    if(start_ok) valid <= 0;
                end
                DATA: begin
                    if(clk_cnt == CLK_PER_BIT / 2) begin
                        // $display("data[%d] <= %d\n", bit_cnt-1, uart_rx);
                        data[bit_cnt-1] <= rx_reg2;
                    end
                end
                STOP: begin
                    valid <= 1;
                end
            endcase
        end
    end
endmodule