/*
On-chip block-RAM controller — a drop-in stand-in for sram_controller (rtl/sram.sv).

Same user-side contract, clock for clock: a range request (start/we/addr/len/
stride) latched on `start`, `busy` high from the accept cycle, a `dout_valid`
pulse per byte read, a ready/valid stream for the bytes written, and a one-cycle
`done` at the end. The FSM below is sram_controller's FSM with the chip pins
removed and the storage moved inside the FPGA; CLOCKS_PER_ACCESS is honoured even
though nothing here needs the wait, because the *point* is that the two behave
identically.

That is not decoration. This block exists so boards/cmod_a7_bram can run the UART
command protocol with the external memory taken out of the picture (see
rtl/uart_bram.sv). If it completed a transfer in one clock instead of the
controller's beat, every turnaround in uart_interface would shift and the image
would no longer be a controlled comparison against boards/cmod_a7_mem — it would
just be a different design that also happens to work. Keep the latency equal.

Two things genuinely differ from the SRAM, both unavoidable:

  * Capacity. ADDR_W is the *protocol* address width (19 bits, 512 KiB) and
    DEPTH_W is what is actually built. 2**19 bytes will not fit in an
    Artix-7 35T (1800 Kb of block RAM total), so addresses above 2**DEPTH_W
    alias down onto the implemented window — addr[DEPTH_W-1:0] is the index and
    the high bits are discarded. Deterministic, so a write and a read of the same
    address still agree, but two addresses 2**DEPTH_W apart are the same byte.
    `aliased` goes sticky-high the first time it happens so this is a visible
    event rather than a silent one; see rtl/uart_bram.sv for what it drives.
  * There is no bidirectional bus, no output enable and no chip select, so
    nothing here can be defeated by bus contention, drive strength, or the 30
    bank-14 pins switching. Removing exactly that is the reason the image exists.
    The write pulse goes with them: sram_controller's two-clock write beat exists
    to place a WE# edge safely inside it, and a synchronous BRAM has no such
    edge — but the beat length is kept anyway, for the reason above.

Storage comes up all-zero, matching a configured BRAM (the simulation-only
initial block below makes the two agree; hardware needs no help).
*/

module bram_controller #(
    parameter integer CLOCKS_PER_ACCESS = 0,   // held equal to sram_controller's
    parameter integer ADDR_W   = 19,           // protocol address width
    parameter integer DATA_W   = 8,
    parameter integer LEN_W    = 16,
    parameter integer STRIDE_W = 16,
    parameter integer DEPTH_W  = 16            // implemented: 2**DEPTH_W words
) (
    input  logic clk,
    input  logic rst_n,

    // user signals — identical to sram_controller's
    input  logic                start,
    input  logic                we,
    input  logic [ADDR_W-1:0]   addr,
    input  logic [LEN_W-1:0]    len,
    input  logic [STRIDE_W-1:0] stride,

    input  logic [DATA_W-1:0] din,
    input  logic              din_valid,
    output logic              din_ready,

    output logic [DATA_W-1:0] dout,
    output logic              dout_valid,

    output logic busy,
    output logic done,

    // sticky: an access addressed a byte above the implemented window and was
    // folded back into it. Cleared only by reset.
    output logic aliased
);
    localparam integer DEPTH = 1 << DEPTH_W;

    initial begin
        if (DEPTH_W > ADDR_W)
            $fatal(1, "bram_controller: DEPTH_W (%0d) > ADDR_W (%0d)",
                   DEPTH_W, ADDR_W);
    end

    // Beat lengths, copied from sram_controller so the two stay interchangeable.
    localparam integer RD_BEAT = CLOCKS_PER_ACCESS + 1;
    localparam integer WR_BEAT = (CLOCKS_PER_ACCESS + 1 > 2) ? CLOCKS_PER_ACCESS + 1 : 2;
    localparam integer BEAT_W  = (CLOCKS_PER_ACCESS < 2) ? 2 : $clog2(CLOCKS_PER_ACCESS + 2);

    localparam [1:0] IDLE = 2'b00, READ = 2'b01, WFETCH = 2'b10, WRITE = 2'b11;

    reg [1:0]          state;
    reg [BEAT_W-1:0]   beat;
    reg [LEN_W-1:0]    cnt;

    reg [ADDR_W-1:0]   addr_q;    // address of the beat in progress
    reg [STRIDE_W-1:0] stride_q;
    reg [DATA_W-1:0]   din_q;

    wire [ADDR_W-1:0] stride_eff = (stride_q == '0) ? ADDR_W'(1) : ADDR_W'(stride_q);
    wire              rd_last    = (beat == BEAT_W'(RD_BEAT - 1));
    wire              wr_last    = (beat == BEAT_W'(WR_BEAT - 1));
    wire              final_byte = (cnt == LEN_W'(1));

    assign din_ready = (state == WFETCH) || (state == WRITE && wr_last && !final_byte);

    // Is this beat outside the implemented window? Written as a generate rather
    // than a ternary because addr_q[ADDR_W-1:DEPTH_W] is an illegal empty range
    // when the two are equal, and an elaboration error does not care that the
    // branch is unreachable.
    wire addr_above_window;
    generate
        if (DEPTH_W < ADDR_W) begin : g_alias_check
            assign addr_above_window = |addr_q[ADDR_W-1:DEPTH_W];
        end else begin : g_alias_check
            assign addr_above_window = 1'b0;
        end
    endgenerate

    // =========================================================================
    // The memory. One address, mutually exclusive read/write enables — the
    // single-port no-change template Vivado infers as a RAMB. DEPTH_W = 16 is
    // 16 RAMB36 tiles (4 KiB each in x9 mode) of the 35T's 50.
    //
    // Both enables land on the beat's last clock, which is where sram_controller
    // samples the asynchronous chip bus and where its WE# pulse has just ended.
    // `q` is therefore loaded on the same edge as the SRAM's `dout` register,
    // and is driven out directly rather than re-registered.
    // =========================================================================
    (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [DATA_W-1:0] q;

    wire [DEPTH_W-1:0] index = addr_q[DEPTH_W-1:0];   // the alias, made explicit
    wire               en_wr = (state == WRITE) && wr_last;
    wire               en_rd = (state == READ)  && rd_last;

    always_ff @(posedge clk) begin
        if (en_wr) mem[index] <= din_q;
        if (en_rd) q          <= mem[index];
    end

    assign dout = q;

    // Configuration leaves block RAM zeroed; give the simulator the same start.
    // Synthesis-only initialization of a 2**16-entry array would otherwise be
    // carried through the netlist as an INIT string for no gain.
// synthesis translate_off
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
        q = '0;
    end
// synthesis translate_on

    // =========================================================================
    // FSM — sram_controller's, unchanged apart from the chip pins.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            beat       <= '0;
            cnt        <= '0;
            addr_q     <= '0;
            stride_q   <= '0;
            din_q      <= '0;
            busy       <= 1'b0;
            done       <= 1'b0;
            dout_valid <= 1'b0;
            aliased    <= 1'b0;
        end else begin
            done       <= 1'b0;   // one-clock strobes, default low
            dout_valid <= 1'b0;

            if (en_rd || en_wr)
                if (addr_above_window) aliased <= 1'b1;   // sticky

            case (state)
                IDLE: begin
                    beat <= '0;
                    busy <= 1'b0;
                    if (start) begin
                        addr_q   <= addr;
                        stride_q <= stride;
                        cnt      <= len;
                        if (len == '0) begin
                            done <= 1'b1;          // empty range: handshake only
                        end else begin
                            busy  <= 1'b1;
                            state <= we ? WFETCH : READ;
                        end
                    end
                end

                READ: begin
                    beat <= beat + BEAT_W'(1);
                    if (rd_last) begin
                        dout_valid <= 1'b1;        // `q` is loaded on this edge
                        beat       <= '0;
                        cnt        <= cnt - LEN_W'(1);
                        addr_q     <= addr_q + stride_eff;
                        if (final_byte) begin
                            done  <= 1'b1;
                            busy  <= 1'b0;
                            state <= IDLE;
                        end
                    end
                end

                WFETCH: begin
                    if (din_valid) begin
                        din_q <= din;
                        beat  <= '0;
                        state <= WRITE;
                    end
                end

                WRITE: begin
                    beat <= beat + BEAT_W'(1);
                    if (wr_last) begin
                        beat <= '0;
                        cnt  <= cnt - LEN_W'(1);
                        if (final_byte) begin
                            done  <= 1'b1;
                            busy  <= 1'b0;
                            state <= IDLE;
                        end else begin
                            addr_q <= addr_q + stride_eff;
                            if (din_valid) begin   // taken on din_ready above
                                din_q <= din;
                                state <= WRITE;    // straight into the next beat
                            end else begin
                                state <= WFETCH;   // producer stalled
                            end
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
