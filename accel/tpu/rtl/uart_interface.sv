// UART host <-> external SRAM bridge (command FSM).
//
// Device side of docs/uart_host.md: the host PC is the sole master and reads or
// writes a byte range of the external SRAM over a single UART. A command frame is
//
//   CMD('R'/'W') | A2 A1 A0 (24-bit addr, big-endian) | L1 L0 (16-bit len, BE) | [data]
//
// Reads stream exactly `len` data bytes back with no framing; writes carry `len`
// payload bytes and get a single status byte in reply (ACK on success, NAK on a
// rejected command). A bad command byte or a failed address/length validation is
// answered with NAK and the FSM resynchronises to IDLE.
//
// The UART receiver/transmitter live *outside* this module (instantiated in
// tpu_top); this block only consumes decoded bytes (data_in/receiver_valid),
// emits bytes (transmitter_start/data_out/transmitter_busy) and drives an
// sram_controller user-side port. `host_busy` is the select for the SRAM
// arbitration mux (DMA vs. host), see docs/uart_host.md §6.
module uart_interface #(
    parameter integer ADDR_W     = 19,
    parameter integer LENGTH_W   = 16,
    parameter integer RX_TIMEOUT = 0    // clocks; 0 disables mid-frame abort
) (
    input  logic clk,
    input  logic rst_n,

    // receiver interface (from uart_receiver)
    input  logic [7:0] data_in,
    input  logic       receiver_valid,

    // transmitter interface (to uart_transmitter)
    output logic       transmitter_start,
    output logic [7:0] data_out,
    input  logic       transmitter_busy,

    // sram_controller user-side interface
    output logic              sram_start,
    output logic              sram_we,
    output logic [ADDR_W-1:0] sram_addr,
    output logic [7:0]        sram_din,
    input  logic [7:0]        sram_dout,
    input  logic              sram_busy,   // unused in v1 (done-driven handshake)
    input  logic              sram_done,

    // high while a command is in progress (SRAM arbitration mux select)
    output logic host_busy
);
    localparam integer ADDR_BYTES   = (ADDR_W + 7) / 8;     // 3
    localparam integer LENGTH_BYTES = (LENGTH_W + 7) / 8;   // 2
    localparam integer ADDR_BITS    = ADDR_BYTES * 8;       // 24
    localparam integer LEN_BITS     = LENGTH_BYTES * 8;     // 16

    // largest legal address+1 (2^ADDR_W); a frame with addr+len above this is NAK'd
    localparam [ADDR_BITS:0] MEM_LIMIT = (1 << ADDR_W);

    // protocol constants
    localparam [7:0] CMD_READ  = 8'h52; // 'R'
    localparam [7:0] CMD_WRITE = 8'h57; // 'W'
    localparam [7:0] STAT_ACK  = 8'h06;
    localparam [7:0] STAT_NAK  = 8'h15;

    localparam [3:0]
        IDLE             = 4'd0,
        RX_ADDR          = 4'd1,   // collect ADDR_BYTES address bytes (MSB first)
        RX_LEN           = 4'd2,   // collect LENGTH_BYTES length bytes (MSB first)
        VALIDATE         = 4'd3,   // range/len check; branch to read or write
        RD_ISSUE         = 4'd4,   // issue one SRAM read
        RD_WAIT          = 4'd5,   // wait for read data
        RD_TX            = 4'd6,   // hand the byte to the transmitter
        RD_TX_WAIT       = 4'd7,   // wait for the byte to finish on the wire
        WR_RX            = 4'd8,   // wait for a payload byte from the host
        WR_ISSUE         = 4'd9,   // issue one SRAM write
        WR_WAIT          = 4'd10,  // wait for write to commit
        SEND_STATUS      = 4'd11,  // send ACK/NAK
        SEND_STATUS_WAIT = 4'd12;  // wait for the status byte to finish

    logic [3:0] state;

    // receiver_valid and transmitter_busy are level-held across many cycles, so we
    // work off their edges: a rising valid = one new byte, a falling busy = one
    // completed transmit frame.
    logic receiver_valid_prev, transmitter_busy_prev;
    wire  rx_byte = receiver_valid & ~receiver_valid_prev;
    wire  tx_done = transmitter_busy_prev & ~transmitter_busy;

    logic [ADDR_BITS-1:0] addr;    // assembled base address (24-bit)
    logic [LEN_BITS-1:0]  len;     // byte count (16-bit)
    logic [LEN_BITS-1:0]  idx;     // payload position, 0 .. len-1
    logic [7:0]           cnt;     // header byte counter
    logic                 op;      // 0 = read, 1 = write
    logic [7:0]           rd_byte, wr_byte, status_byte;
    logic [31:0]          to_cnt;  // inter-byte timeout counter

    // frame validation over the (registered) addr/len
    wire addr_top_ok = (addr[ADDR_BITS-1:ADDR_W] == '0);
    wire len_ok      = (len != '0);
    wire [ADDR_BITS:0] end_addr = {1'b0, addr} + len;   // widened so it can't wrap
    wire range_ok    = (end_addr <= MEM_LIMIT);
    wire frame_ok    = addr_top_ok & len_ok & range_ok;

    // mid-frame states where we are stalled waiting on the host for the next byte
    wire rx_waiting  = (state == RX_ADDR) | (state == RX_LEN) | (state == WR_RX);

    assign host_busy = (state != IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                 <= IDLE;
            receiver_valid_prev   <= 1'b0;
            transmitter_busy_prev <= 1'b0;
            transmitter_start     <= 1'b0;
            data_out              <= 8'b0;
            sram_start            <= 1'b0;
            sram_we               <= 1'b0;
            sram_addr             <= '0;
            sram_din              <= 8'b0;
            addr                  <= '0;
            len                   <= '0;
            idx                   <= '0;
            cnt                   <= 8'b0;
            op                    <= 1'b0;
            rd_byte               <= 8'b0;
            wr_byte               <= 8'b0;
            status_byte           <= 8'b0;
            to_cnt                <= 32'b0;
        end else begin
            receiver_valid_prev   <= receiver_valid;
            transmitter_busy_prev <= transmitter_busy;

            // one-cycle strobe defaults; states below override as needed
            sram_start        <= 1'b0;
            transmitter_start <= 1'b0;

            case (state)
                IDLE: begin
                    idx  <= '0;
                    cnt  <= 8'b0;
                    addr <= '0;
                    len  <= '0;
                    if (rx_byte) begin
                        if (data_in == CMD_READ)  begin op <= 1'b0; state <= RX_ADDR; end
                        else if (data_in == CMD_WRITE) begin op <= 1'b1; state <= RX_ADDR; end
                        else begin status_byte <= STAT_NAK; state <= SEND_STATUS; end
                    end
                end

                RX_ADDR: begin
                    if (rx_byte) begin
                        addr <= {addr[ADDR_BITS-9:0], data_in};   // shift in, MSB first
                        cnt  <= cnt + 8'd1;
                        if (cnt == ADDR_BYTES - 1) begin
                            cnt   <= 8'b0;
                            state <= RX_LEN;
                        end
                    end
                end

                RX_LEN: begin
                    if (rx_byte) begin
                        len <= {len[LEN_BITS-9:0], data_in};
                        cnt <= cnt + 8'd1;
                        if (cnt == LENGTH_BYTES - 1) begin
                            cnt   <= 8'b0;
                            state <= VALIDATE;
                        end
                    end
                end

                VALIDATE: begin
                    idx <= '0;
                    if (frame_ok) state <= op ? WR_RX : RD_ISSUE;
                    else begin
                        status_byte <= STAT_NAK;
                        state       <= SEND_STATUS;
                    end
                end

                // ---------- read path ----------
                RD_ISSUE: begin
                    sram_start <= 1'b1;
                    sram_we    <= 1'b0;
                    sram_addr  <= addr[ADDR_W-1:0] + idx;
                    state      <= RD_WAIT;
                end
                RD_WAIT: begin
                    if (sram_done) begin
                        rd_byte <= sram_dout;
                        state   <= RD_TX;
                    end
                end
                RD_TX: begin
                    if (!transmitter_busy) begin
                        transmitter_start <= 1'b1;
                        data_out          <= rd_byte;
                        state             <= RD_TX_WAIT;
                    end
                end
                RD_TX_WAIT: begin
                    if (tx_done) begin
                        idx <= idx + 1'b1;
                        if (idx + 1'b1 == len) state <= IDLE;   // reads carry no status
                        else                   state <= RD_ISSUE;
                    end
                end

                // ---------- write path ----------
                WR_RX: begin
                    if (rx_byte) begin
                        wr_byte <= data_in;
                        state   <= WR_ISSUE;
                    end
                end
                WR_ISSUE: begin
                    sram_start <= 1'b1;
                    sram_we    <= 1'b1;
                    sram_addr  <= addr[ADDR_W-1:0] + idx;
                    sram_din   <= wr_byte;
                    state      <= WR_WAIT;
                end
                WR_WAIT: begin
                    if (sram_done) begin
                        idx <= idx + 1'b1;
                        if (idx + 1'b1 == len) begin
                            status_byte <= STAT_ACK;
                            state       <= SEND_STATUS;
                        end else state <= WR_RX;
                    end
                end

                // ---------- status reply ----------
                SEND_STATUS: begin
                    if (!transmitter_busy) begin
                        transmitter_start <= 1'b1;
                        data_out          <= status_byte;
                        state             <= SEND_STATUS_WAIT;
                    end
                end
                SEND_STATUS_WAIT: begin
                    if (tx_done) state <= IDLE;
                end

                default: state <= IDLE;
            endcase

            // Inter-byte timeout: if we sit mid-frame waiting on the host for
            // longer than RX_TIMEOUT clocks, abort silently back to IDLE so a
            // stalled/aborted frame can't wedge the link (docs/uart_host.md §7).
            // This runs after the case, so an abort overrides the state above.
            if (RX_TIMEOUT != 0) begin
                if (rx_waiting) begin
                    if (rx_byte)                     to_cnt <= 32'b0;
                    else if (to_cnt + 1 >= RX_TIMEOUT) begin
                        to_cnt <= 32'b0;
                        state  <= IDLE;
                    end else                         to_cnt <= to_cnt + 32'd1;
                end else to_cnt <= 32'b0;
            end
        end
    end
endmodule
