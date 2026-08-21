/*
SRAM controller for CMOD A7 (ISSI IS61/64WV5128, 512K x 8 async SRAM)
SRAM datasheet: https://www.issi.com/WW/pdf/61-64WV5128Axx-Bxx.pdf

**Range engine.** One request moves a whole range, not one byte: `addr` is the
first address, `len` the number of bytes, and `stride` the address step between
them. The old one-byte-per-`start` handshake is the `len = 1` case of this and
behaves identically from the caller's side, which is what lets `uart_interface`
stay as it was.

That is the point of the rewrite. Byte-at-a-time, every access paid the FSM's
IDLE/ACCESS/WAIT sequence *plus* the caller's own dispatch loop around it --
about 8 clocks per byte through the DMA, of which one was the access. A range
request pays the sequencing once and then runs at the memory's own rate.

---- Rate -------------------------------------------------------------------

The core clock is 12 MHz (83.3 ns) and the chip is rated for an 8-10 ns access,
so one whole clock per byte is already ~8x more time than the part needs. A read
therefore issues one address per clock and samples the returned byte on the next
edge. `CLOCKS_PER_ACCESS` (0 by default) adds that many *extra* clocks per beat
if a faster clock or a slower part ever needs them:

    read beat  = CLOCKS_PER_ACCESS + 1 clocks       (1 byte/clock at the default)
    write beat = max(2, CLOCKS_PER_ACCESS + 1)      (see WE# below)

---- Why a write beat is two clocks and a read beat is one ------------------

A read is address-in/data-out and commits nothing, so the address may change on
every rising edge. A write commits on the **WE# rising edge**, so that edge has
to land while the address and data of *that* byte are still on the pins. Driving
WE# from the rising edge would put its rise on the same edge the address
advances, leaving the chip's address-hold requirement (tHA, 0 ns) to be met by
output-path skew alone -- a coin flip in the direction of writing a byte to its
neighbour.

So WE# is driven from the **falling** edge. Both of its edges then sit in the
middle of a clock, half a clock away from every address/data change:

    clk       __|~~|__|~~|__|~~|__|~~|__      beat = 2 clocks (C0, C1)
    addr/din  X=====================X====     changes on rising edges only
    WE#       ~~~~~\_____________/~~~~~~~     low: negedge C0 -> negedge C1

    address setup to WE# low    41.6 ns   (tAS  needs 0)
    WE# pulse width             83.3 ns   (tPWE needs 8)
    address valid to write end  83.3 ns   (tAW  needs 8)
    data setup to write end    125.0 ns   (tSD  needs 6)
    address / data hold         41.6 ns   (tHA/tHD need 0)

Two clocks is the minimum for that shape: the pulse must start after the beat
begins and end before it ends, and one clock offers a single edge in between.
It also makes the behavioural chip models in tb/ race-free, because `posedge
sram_we` can no longer coincide with the address changing under it.

---- Streams ----------------------------------------------------------------

A beat is one or two clocks, so the caller cannot be asked to turn a request
around per byte. Each direction is a stream instead:

  read   `dout_valid` pulses for one clock per byte with `dout` holding it. The
         last pulse coincides with `done`, so a `len = 1` caller that samples
         `dout` on `done` (uart_interface) still sees its byte.
  write  ready/valid. `din_ready` marks the clock on which the controller will
         latch `din`; it is asserted during the *last* clock of the preceding
         beat, so a producer with a one-clock turnaround (the DMA, whose
         scratchpad read has exactly that) sustains the full write rate. While
         `din_valid` is low the controller parks with the bus held and nothing
         committed.

`stride` is 0-means-1, the "zero is not set" convention the MXU and DMA geometry
registers already use (docs/macro_ops.md 4.0). It exists for the transposing
DMA: a `wrmem.t` writes its destination column-major, so its DRAM addresses step
by `tdrow` and would otherwise be thousands of separate one-byte requests.

FSM:  IDLE -> READ -> IDLE                     (read range)
      IDLE -> WFETCH <-> WRITE -> IDLE         (write range)
*/

module sram_controller #(
    parameter integer CLOCKS_PER_ACCESS = 0,   // extra clocks per beat
    parameter integer ADDR_W   = 19,
    parameter integer DATA_W   = 8,
    parameter integer LEN_W    = 16,           // bytes per request
    parameter integer STRIDE_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    // ---- request: latched on `start`, so the caller need only hold it there --
    input  logic                start,
    input  logic                we,       // 1 = write range, 0 = read range
    input  logic [ADDR_W-1:0]   addr,     // address of the first byte
    input  logic [LEN_W-1:0]    len,      // bytes (0 = move nothing, still done)
    input  logic [STRIDE_W-1:0] stride,   // address step per byte (0 => 1)

    // ---- write data stream (ready/valid) ------------------------------------
    input  logic [DATA_W-1:0] din,
    input  logic              din_valid,
    output logic              din_ready,  // `din` is taken on this clock's edge

    // ---- read data stream (ready/valid) -------------------------------------
    // `dout_ready` low holds the beat: the address stays put, CE#/OE# stay
    // asserted, and the async SRAM keeps driving the same byte, so resuming
    // re-samples exactly the byte that was paused. This is what lets the DMA
    // stop the fill stream when the scratchpad write port is denied for longer
    // than its skid buffer can absorb (dma.sv), instead of dropping bytes.
    // A consumer that cannot stall ties it high and nothing changes.
    output logic [DATA_W-1:0] dout,
    output logic              dout_valid, // `dout` holds a fresh byte
    input  logic              dout_ready, // taken on this clock's edge

    output logic busy,
    output logic done,                    // one-clock pulse, end of range

    // chip signals
    output logic [ADDR_W-1:0] sram_addr,
    inout  wire  [DATA_W-1:0] sram_data,   // inout must be a net, not a var
    output logic sram_we,
    output logic sram_ce,
    output logic sram_oen
);
    // Beat lengths in clocks. A write cannot be shorter than two (see header).
    localparam integer RD_BEAT = CLOCKS_PER_ACCESS + 1;
    localparam integer WR_BEAT = (CLOCKS_PER_ACCESS + 1 > 2) ? CLOCKS_PER_ACCESS + 1 : 2;
    localparam integer BEAT_W  = (CLOCKS_PER_ACCESS < 2) ? 2 : $clog2(CLOCKS_PER_ACCESS + 2);

    localparam [1:0] IDLE = 2'b00, READ = 2'b01, WFETCH = 2'b10, WRITE = 2'b11;

    reg [1:0]          state;
    reg [BEAT_W-1:0]   beat;      // clock index inside the current beat
    reg [LEN_W-1:0]    cnt;       // bytes still to move, including the one in flight

    reg [ADDR_W-1:0]   addr_q;    // address on the pins == address of this beat
    reg [STRIDE_W-1:0] stride_q;
    reg [DATA_W-1:0]   din_q;
    reg                drive_en;  // 1 = FPGA drives the bidirectional bus
    reg                we_win;    // 1 = WE# is to be low across this clock's negedge

    // Only drive the bus while we own it (a write); otherwise release to Hi-Z
    // so the SRAM can drive it during reads.
    assign sram_data = drive_en ? din_q : {DATA_W{1'bz}};
    assign sram_addr = addr_q;

    wire [ADDR_W-1:0] stride_eff = (stride_q == '0) ? ADDR_W'(1) : ADDR_W'(stride_q);
    wire              rd_last    = (beat == BEAT_W'(RD_BEAT - 1));
    wire              wr_last    = (beat == BEAT_W'(WR_BEAT - 1));
    wire              final_byte = (cnt == LEN_W'(1));

    // The next byte is asked for on the last clock of the current beat, so a
    // producer with a one-clock turnaround never stalls the pipe.
    assign din_ready = (state == WFETCH) || (state == WRITE && wr_last && !final_byte);

    // -------------------------------------------------------------------------
    // WE#, driven from the falling edge (header). `we_win` is a rising-edge
    // signal, high for every clock of a write beat except the last, so WE# falls
    // in the middle of the beat's first clock and rises in the middle of its
    // last -- half a clock clear of the address/data change at either end.
    // -------------------------------------------------------------------------
    always_ff @(negedge clk or negedge rst_n) begin
        if (!rst_n) sram_we <= 1'b1;
        else        sram_we <= ~we_win;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            beat       <= '0;
            cnt        <= '0;
            addr_q     <= '0;
            stride_q   <= '0;
            din_q      <= '0;
            drive_en   <= 1'b0;
            we_win     <= 1'b0;
            sram_ce    <= 1'b1;
            sram_oen   <= 1'b1;
            busy       <= 1'b0;
            done       <= 1'b0;
            dout       <= '0;
            dout_valid <= 1'b0;
        end else begin
            done       <= 1'b0;   // one-clock strobes, default low
            dout_valid <= 1'b0;

            case (state)
                // ---- idle: accept a request ---------------------------------
                IDLE: begin
                    beat     <= '0;
                    sram_ce  <= 1'b1;    // deselect while idle
                    sram_oen <= 1'b1;
                    drive_en <= 1'b0;
                    we_win   <= 1'b0;
                    busy     <= 1'b0;
                    if (start) begin
                        addr_q   <= addr;
                        stride_q <= stride;
                        cnt      <= len;
                        if (len == '0) begin
                            done <= 1'b1;          // empty range: handshake only
                        end else begin
                            busy    <= 1'b1;
                            sram_ce <= 1'b0;
                            if (we) begin
                                drive_en <= 1'b1;  // held for the whole range
                                state    <= WFETCH;
                            end else begin
                                sram_oen <= 1'b0;
                                state    <= READ;
                            end
                        end
                    end
                end

                // ---- read: one address per beat, sampled on its last clock ---
                READ: begin
                    if (!rd_last) begin
                        beat <= beat + BEAT_W'(1);
                    end else if (dout_ready) begin
                        dout       <= sram_data;
                        dout_valid <= 1'b1;
                        beat       <= '0;
                        cnt        <= cnt - LEN_W'(1);
                        addr_q     <= addr_q + stride_eff;
                        if (final_byte) begin
                            done     <= 1'b1;
                            busy     <= 1'b0;
                            sram_ce  <= 1'b1;
                            sram_oen <= 1'b1;
                            state    <= IDLE;
                        end
                    end
                    // else: hold at the last beat with the address unchanged.
                end

                // ---- write: park until the producer has the next byte -------
                WFETCH: begin
                    if (din_valid) begin
                        din_q  <= din;
                        beat   <= '0;
                        we_win <= 1'b1;
                        state  <= WRITE;
                    end
                end

                WRITE: begin
                    beat <= beat + BEAT_W'(1);
                    // Close the window one clock early so WE# rises on this
                    // beat's last negedge, half a clock before addr/din move.
                    if (beat == BEAT_W'(WR_BEAT - 2)) we_win <= 1'b0;
                    if (wr_last) begin
                        beat <= '0;
                        cnt  <= cnt - LEN_W'(1);
                        if (final_byte) begin
                            done     <= 1'b1;
                            busy     <= 1'b0;
                            sram_ce  <= 1'b1;
                            drive_en <= 1'b0;
                            state    <= IDLE;
                        end else begin
                            addr_q <= addr_q + stride_eff;
                            if (din_valid) begin    // taken on din_ready above
                                din_q  <= din;
                                we_win <= 1'b1;
                                state  <= WRITE;    // straight into the next beat
                            end else begin
                                state  <= WFETCH;   // producer stalled
                            end
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
