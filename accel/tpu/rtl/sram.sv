/*
SRAM controller for CMOD A7 (ISSI IS61/64WV5128, 512K x 8 async SRAM)
SRAM datasheet: https://www.issi.com/WW/pdf/61-64WV5128Axx-Bxx.pdf

Single-word read/write engine with a simple start/busy/done handshake.
The request (addr/din/we) is latched on `start`, so the caller only has to
hold the inputs valid for the cycle `start` is asserted.

FSM: IDLE -> (WRITE|READ) -> WAIT -> IDLE
  WAIT holds for CLOCKS_PER_ACCESS cycles so the asynchronous access time is
  met before we sample read data / end the write pulse.
*/

module sram_controller #(
    parameter integer CLOCKS_PER_ACCESS = 2,
    parameter integer ADDR_W = 19,
    parameter integer DATA_W = 8
) (
    input  logic clk,
    input  logic rst_n,

    // user signals
    input  logic start,
    input  logic we,
    input  logic [ADDR_W-1:0] addr,
    input  logic [DATA_W-1:0] din,
    output logic [DATA_W-1:0] dout,
    output logic busy,
    output logic done,

    // chip signals
    output logic [ADDR_W-1:0] sram_addr,
    inout  wire  [DATA_W-1:0] sram_data,   // inout must be a net, not a var
    output logic sram_we,
    output logic sram_ce,
    output logic sram_oen
);
    localparam [1:0] IDLE = 2'b00, WRITE = 2'b01, READ = 2'b10, WAIT = 2'b11;

    reg [1:0] state, state_n;
    reg [7:0] counter;

    // latched request so the caller need only hold inputs during `start`
    reg [ADDR_W-1:0] addr_q;
    reg [DATA_W-1:0] din_q;
    reg              wr_q;      // 1 = latched op is a write
    reg              drive_en;  // 1 = FPGA drives the bidirectional bus

    // Only drive the bus while we own it (a write); otherwise release to Hi-Z
    // so the SRAM can drive it during reads.
    assign sram_data = drive_en ? din_q : {DATA_W{1'bz}};
    assign sram_addr = addr_q;

    // next-state logic (default: hold current state)
    always_comb begin
        state_n = state;
        case (state)
            IDLE:  if (start) state_n = we ? WRITE : READ;
            WRITE: state_n = WAIT;
            READ:  state_n = WAIT;
            WAIT:  if (counter >= CLOCKS_PER_ACCESS) state_n = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            counter  <= 8'b0;
            sram_we  <= 1'b1;
            sram_ce  <= 1'b1;
            sram_oen <= 1'b1;
            drive_en <= 1'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
            dout     <= '0;
            addr_q   <= '0;
            din_q    <= '0;
            wr_q     <= 1'b0;
        end else begin
            state <= state_n;
            done  <= 1'b0;  // one-cycle strobe, default low

            case (state)
                IDLE: begin
                    counter  <= 8'b0;
                    sram_ce  <= 1'b1;   // deselect while idle
                    sram_we  <= 1'b1;
                    sram_oen <= 1'b1;
                    drive_en <= 1'b0;
                    busy     <= start;  // accept the request this cycle
                    if (start) begin
                        addr_q <= addr;
                        din_q  <= din;
                        wr_q   <= we;
                    end
                end
                WRITE: begin
                    busy     <= 1'b1;
                    sram_ce  <= 1'b0;
                    sram_we  <= 1'b0;   // assert write pulse
                    sram_oen <= 1'b1;   // SRAM outputs off during a write
                    drive_en <= 1'b1;   // drive din onto the bus
                end
                READ: begin
                    busy     <= 1'b1;
                    sram_ce  <= 1'b0;
                    sram_we  <= 1'b1;
                    sram_oen <= 1'b0;   // let the SRAM drive the bus
                    drive_en <= 1'b0;
                end
                WAIT: begin
                    busy    <= 1'b1;
                    counter <= counter + 1'b1;
                    if (wr_q) begin
                        // End the write pulse while addr/data are still held,
                        // so the word is committed on the WE# rising edge.
                        sram_we <= 1'b1;
                    end else begin
                        // OE# has been low long enough; capture read data.
                        dout <= sram_data;
                    end
                    if (state_n == IDLE) begin
                        done     <= 1'b1;
                        busy     <= 1'b0;
                        sram_ce  <= 1'b1;
                        sram_oen <= 1'b1;
                        drive_en <= 1'b0;
                    end
                end
            endcase
        end
    end
endmodule
