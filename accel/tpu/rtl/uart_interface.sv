// UART host <-> TPU bridge (command FSM).
//
// Device side of docs/uart_host.md, extended past the v1 memory-only scope with
// two control commands. The host PC is the sole master over a single UART; the
// FSM decodes fixed-header command frames and drives the external SRAM, the
// scalar unit's instruction memory, and the scalar unit's run trigger.
//
// Commands (byte 0 of every frame):
//   'R' 0x52  read  SRAM : hdr = CMD + A2 A1 A0 + L1 L0            -> len data bytes
//   'W' 0x57  write SRAM : hdr = CMD + A2 A1 A0 + L1 L0 + data[len]-> ACK/NAK
//   'I' 0x49  write IMEM : hdr = CMD + A2 A1 A0 + L1 L0 + data[len]-> ACK/NAK
//   'G' 0x47  go / run   : hdr = CMD + A2 A1 A0                    -> ACK/NAK
//   'T' 0x54  read timers: hdr = CMD                       -> TIMER_WORDS*4 bytes
//
//   * Addresses are 3 bytes, big-endian. For 'R'/'W' this is a 19-bit SRAM byte
//     address; for 'I' it is an instruction-word index; for 'G' it is the boot PC.
//   * Length is 2 bytes, big-endian, in BYTES. For 'I' it must be a multiple of 4
//     (one 32-bit instruction word per 4 bytes, MSB first); words are written at
//     ascending instruction addresses.
//   * 'G' pulses `run_start` with `run_pc = addr` — no length, no data.
//   * 'T' is the whole frame: no address, no length, nothing to validate. It
//     replies with `cycle_count` as TIMER_WORDS 32-bit words, MSB first both
//     within and across words, and no status byte — header-less like a read,
//     because the length is fixed at build time and both ends know it.
//     TIMER_WORDS defaults to 1 (the run-length counter alone, the historical
//     reply); tpu_top raises it to expose the per-unit counters in
//     perf_counters.sv, keeping the run length in the first word so a short
//     reader still decodes it correctly.
//
// Arbitration (core priority): the scalar unit owns the machine. Any command that
// arrives while `core_busy` is high is rejected with NAK and touches nothing —
// the SRAM mux in tpu_top hands the controller to the DMA engine while a program
// runs, so the host and the core never contend. The host regains access once the
// program halts (`core_busy` low).
//
// **'T' is exempt from that rule**, and is the only command that is. It reads a
// counter and nothing else: no SRAM access, no IMEM write, no run trigger, so
// there is nothing for it to contend over and no state for it to corrupt
// mid-run. Answering it while the core runs is also the useful case — it is how
// a host watches a long program make progress, since a rising count is the one
// piece of live evidence this link can give that the core is still working
// rather than wedged. (The count freezing is not by itself proof the run
// finished: it also stops if the core never started.)
//
// The UART receiver/transmitter live outside this module (instantiated in
// tpu_top); this block consumes decoded bytes (data_in/receiver_valid), emits
// bytes (transmitter_start/data_out/transmitter_busy), and drives the SRAM
// user-side port, the IMEM write port, and the run trigger.
//
// Received bytes land in a one-byte holding register (docs/uart_host.md §5:
// "a single holding byte in each direction suffices"), captured unconditionally
// in every state. See the comment on `rx_hold` for why that is not optional.
module uart_interface #(
    parameter integer ADDR_W     = 19,   // external SRAM byte-address width
    parameter integer LENGTH_W   = 16,   // transfer length field width
    parameter integer IMEM_AW    = 10,   // instruction memory address width
    parameter integer RX_TIMEOUT = 0,    // clocks; 0 disables mid-frame abort
    // 32-bit words in the 'T' reply. 1 = just the run-length counter, which is
    // the historical behaviour and what an image with no perf_counters block
    // behind it should use. tpu_top sets this to perf_counters' N.
    parameter integer TIMER_WORDS = 1
) (
    input  logic clk,
    input  logic rst_n,

    // arbitration: high while the scalar unit is running (see file header)
    input  logic core_busy,

    // Run counters (perf_counters.sv), reported verbatim by the 'T' command.
    // Free-running relative to this FSM: sampled once, at the clock the command
    // byte is decoded, so the bytes that go out are one coherent reading and not
    // values that moved between them — which also means the whole block is a
    // consistent snapshot of a single run, so ratios between counters are
    // meaningful. Tie to '0 in an image with no scalar unit behind it.
    //
    // Wire order is high word first: bits [TIMER_WORDS*32-1 -: 32] are sent
    // before bits [31:0]. tpu_top places the run-length counter in the high word
    // so a 'T' reply stays prefix-compatible with the single-word version.
    input  logic [TIMER_WORDS*32-1:0] cycle_count,

    // receiver interface (from uart_receiver)
    input  logic [7:0] data_in,
    input  logic       receiver_valid,

    // transmitter interface (to uart_transmitter)
    output logic       transmitter_start,
    output logic [7:0] data_out,
    input  logic       transmitter_busy,

    // sram_controller user-side interface (muxed with the DMA engine in tpu_top)
    output logic              sram_start,
    output logic              sram_we,
    output logic [ADDR_W-1:0] sram_addr,
    output logic [7:0]        sram_din,
    input  logic [7:0]        sram_dout,
    input  logic              sram_busy,   // unused in v1 (done-driven handshake)
    input  logic              sram_done,

    // scalar-unit instruction-memory write port (host load, while idle)
    output logic               imem_we,
    output logic [IMEM_AW-1:0] imem_waddr,
    output logic [31:0]        imem_wdata,

    // scalar-unit run trigger (host_run / boot_pc)
    output logic               run_start,   // one-cycle pulse
    output logic [IMEM_AW-1:0] run_pc,

    // high while a command is in progress (informational / debug)
    output logic host_busy,

    // sticky: a byte arrived while the holding register was still full, i.e.
    // the FSM could not keep up with the host. Cleared only by reset. On a link
    // where the host waits for every reply this must never set, so if it does it
    // is a finding and not a capacity problem — see `rx_hold` below.
    output logic rx_overrun
);
    localparam integer ADDR_BYTES   = (ADDR_W + 7) / 8;     // 3
    localparam integer LENGTH_BYTES = (LENGTH_W + 7) / 8;   // 2
    localparam integer ADDR_BITS    = ADDR_BYTES * 8;       // 24
    localparam integer LEN_BITS     = LENGTH_BYTES * 8;     // 16

    // one-past-the-last legal byte / word for the two address spaces
    localparam [ADDR_BITS:0] MEM_LIMIT  = (1 << ADDR_W);    // 2^19 (SRAM bytes)
    localparam [ADDR_BITS:0] IMEM_LIMIT = (1 << IMEM_AW);   // 2^10 (IMEM words)

    // protocol constants
    localparam [7:0] CMD_READ  = 8'h52; // 'R'  read  SRAM
    localparam [7:0] CMD_WRITE = 8'h57; // 'W'  write SRAM
    localparam [7:0] CMD_IMEM  = 8'h49; // 'I'  write instruction memory
    localparam [7:0] CMD_GO    = 8'h47; // 'G'  start program at addr
    localparam [7:0] CMD_TIMER = 8'h54; // 'T'  read the run-length counter
    localparam [7:0] STAT_ACK  = 8'h06;
    localparam [7:0] STAT_NAK  = 8'h15;

    // bytes in a 'T' reply (the counter block, MSB first within and across words)
    localparam integer TIMER_BITS  = TIMER_WORDS * 32;
    localparam integer TIMER_BYTES = TIMER_WORDS * 4;

    // op selector, latched in IDLE from the command byte. 'T' needs no entry: it
    // has no header to collect and no validation to branch on, so IDLE sends it
    // straight to the reply states without ever consulting `op`.
    localparam [1:0] OP_RD = 2'd0, OP_WR = 2'd1, OP_IMEM = 2'd2, OP_GO = 2'd3;

    // 5 bits, not 4: the two timer states take the encoding past 15.
    localparam [4:0]
        IDLE             = 5'd0,
        RX_ADDR          = 5'd1,   // collect ADDR_BYTES address bytes (MSB first)
        RX_LEN           = 5'd2,   // collect LENGTH_BYTES length bytes (MSB first)
        VALIDATE         = 5'd3,   // arbitration + range check; branch per op
        RD_ISSUE         = 5'd4,   // issue one SRAM read
        RD_WAIT          = 5'd5,   // wait for read data
        RD_TX            = 5'd6,   // hand the byte to the transmitter
        RD_TX_WAIT       = 5'd7,   // wait for the byte to finish on the wire
        WR_RX            = 5'd8,   // wait for a payload byte from the host
        WR_ISSUE         = 5'd9,   // issue one SRAM write
        WR_WAIT          = 5'd10,  // wait for write to commit
        IMEM_RX          = 5'd11,  // collect 4 payload bytes into one instr word
        IMEM_WR          = 5'd12,  // write one instruction word
        SEND_STATUS      = 5'd13,  // send ACK/NAK
        SEND_STATUS_WAIT = 5'd14,  // wait for the status byte to finish
        TMR_TX           = 5'd15,  // hand one counter byte to the transmitter
        TMR_TX_WAIT      = 5'd16;  // wait for that byte to finish on the wire

    logic [4:0] state;

    // receiver_valid and transmitter_busy are level-held across many cycles, so we
    // work off their edges: a rising valid = one new byte, a falling busy = one
    // completed transmit frame.
    logic receiver_valid_prev, transmitter_busy_prev;
    wire  rx_strobe = receiver_valid & ~receiver_valid_prev;
    wire  tx_done   = transmitter_busy_prev & ~transmitter_busy;

    // ---- one-byte receive holding register ----------------------------------
    //
    // `rx_strobe` is a single clock wide. Consuming it directly inside the state
    // case — which is what this FSM used to do — makes every state that is not
    // IDLE / RX_ADDR / RX_LEN / WR_RX / IMEM_RX a window in which an arriving
    // byte is destroyed with no error, no retry and no trace.
    //
    // The windows are not theoretical:
    //
    //   SEND_STATUS + SEND_STATUS_WAIT   ~10*CLK_PER_BIT clocks — a full byte
    //       time, sitting exactly at the command turnaround. The device starts
    //       the ACK about one bit *before* the host has finished the last
    //       payload byte's stop bit, so this window opens straight into where
    //       the host's next command byte goes.
    //   WR_ISSUE + WR_WAIT               a few clocks between every pair of
    //       payload bytes, inside a streaming write.
    //   VALIDATE, IMEM_WR                one clock each.
    //   RD_ISSUE .. RD_TX_WAIT           the whole read reply.
    //
    // One dropped byte is not one bad command: the FSM re-enters IDLE one byte
    // out of phase and every subsequent payload byte decodes as an unknown
    // command and draws a NAK (IDLE's `default` branch below), so the host sees
    // a flood of bytes it never asked for. With RX_TIMEOUT = 0 — what both board
    // targets build (synth/vivado/boards/*/board.tcl) — the mid-frame abort is
    // compiled out and nothing ever recovers.
    //
    // So the capture below runs every cycle in every state, ahead of the FSM,
    // and the FSM reads the register instead of the wire. That is the "single
    // holding byte in each direction" docs/uart_host.md §5 specifies and the
    // implementation never had.
    logic [7:0] rx_hold;
    logic       rx_pending;   // rx_hold holds a byte the FSM has not taken yet

    logic [ADDR_BITS-1:0] addr;    // assembled base address / index (24-bit)
    logic [LEN_BITS-1:0]  len;     // byte count (16-bit)
    logic [LEN_BITS-1:0]  idx;     // payload position (bytes for SRAM, words for IMEM)
    logic [7:0]           cnt;     // header / word-assembly byte counter
    logic [1:0]           op;      // OP_RD / OP_WR / OP_IMEM / OP_GO
    logic [7:0]           rd_byte, wr_byte, status_byte;
    logic [31:0]          word;    // instruction word under assembly
    logic [31:0]          to_cnt;  // inter-byte timeout counter
    logic [TIMER_BITS-1:0] tmr;    // 'T': counter snapshot, shifted out MSB first

    // number of instruction words in an 'I' payload (len is bytes, 4 per word)
    wire [LEN_BITS-1:0] nwords = len >> 2;

    // ---- frame validation over the (registered) addr/len --------------------
    // SRAM ('R'/'W'): 19-bit space, len != 0, addr+len within 2^ADDR_W.
    wire sram_top_ok = (addr[ADDR_BITS-1:ADDR_W] == '0);
    wire [ADDR_BITS:0] sram_end = {1'b0, addr} + len;
    wire sram_frame_ok = sram_top_ok & (len != '0) & (sram_end <= MEM_LIMIT);

    // IMEM ('I'): word index in 2^IMEM_AW, len != 0 and a multiple of 4,
    // addr+nwords within 2^IMEM_AW.
    wire imem_top_ok = (addr[ADDR_BITS-1:IMEM_AW] == '0);
    wire [ADDR_BITS:0] imem_end = {1'b0, addr} + nwords;
    wire imem_frame_ok = imem_top_ok & (len != '0) & (len[1:0] == 2'b00) &
                         (imem_end <= IMEM_LIMIT);

    // GO ('G'): the boot PC must be a valid instruction address.
    wire go_frame_ok = (addr[ADDR_BITS-1:IMEM_AW] == '0);

    // states that take a byte off the holding register this cycle
    wire rx_ready   = (state == IDLE)  | (state == RX_ADDR) | (state == RX_LEN) |
                      (state == WR_RX) | (state == IMEM_RX);
    wire rx_consume = rx_pending & rx_ready;

    // mid-frame states where we are stalled waiting on the host for the next byte
    wire rx_waiting = (state == RX_ADDR) | (state == RX_LEN) |
                      (state == WR_RX)   | (state == IMEM_RX);

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
            imem_we               <= 1'b0;
            imem_waddr            <= '0;
            imem_wdata            <= 32'b0;
            run_start             <= 1'b0;
            run_pc                <= '0;
            addr                  <= '0;
            len                   <= '0;
            idx                   <= '0;
            cnt                   <= 8'b0;
            op                    <= OP_RD;
            rd_byte               <= 8'b0;
            wr_byte               <= 8'b0;
            status_byte           <= 8'b0;
            word                  <= 32'b0;
            to_cnt                <= 32'b0;
            tmr                   <= 32'b0;
            rx_hold               <= 8'b0;
            rx_pending            <= 1'b0;
            rx_overrun            <= 1'b0;
        end else begin
            receiver_valid_prev   <= receiver_valid;
            transmitter_busy_prev <= transmitter_busy;

            // one-cycle strobe defaults; states below override as needed
            sram_start        <= 1'b0;
            transmitter_start <= 1'b0;
            imem_we           <= 1'b0;
            run_start         <= 1'b0;

            case (state)
                IDLE: begin
                    idx  <= '0;
                    cnt  <= 8'b0;
                    addr <= '0;
                    len  <= '0;
                    if (rx_consume) begin
                        case (rx_hold)
                            CMD_READ:  begin op <= OP_RD;   state <= RX_ADDR; end
                            CMD_WRITE: begin op <= OP_WR;   state <= RX_ADDR; end
                            CMD_IMEM:  begin op <= OP_IMEM; state <= RX_ADDR; end
                            CMD_GO:    begin op <= OP_GO;   state <= RX_ADDR; end
                            // 'T' is a complete frame on its own: sample the
                            // counter now (see the port comment on why it is
                            // sampled once) and go straight to the reply. No
                            // header, no VALIDATE, no core_busy check.
                            CMD_TIMER: begin tmr <= cycle_count; state <= TMR_TX; end
                            default:   begin status_byte <= STAT_NAK; state <= SEND_STATUS; end
                        endcase
                    end
                end

                RX_ADDR: begin
                    if (rx_consume) begin
                        addr <= {addr[ADDR_BITS-9:0], rx_hold};   // shift in, MSB first
                        cnt  <= cnt + 8'd1;
                        if (cnt == ADDR_BYTES - 1) begin
                            cnt   <= 8'b0;
                            // 'G' carries no length field: validate straight away
                            state <= (op == OP_GO) ? VALIDATE : RX_LEN;
                        end
                    end
                end

                RX_LEN: begin
                    if (rx_consume) begin
                        len <= {len[LEN_BITS-9:0], rx_hold};
                        cnt <= cnt + 8'd1;
                        if (cnt == LENGTH_BYTES - 1) begin
                            cnt   <= 8'b0;
                            state <= VALIDATE;
                        end
                    end
                end

                VALIDATE: begin
                    idx         <= '0;
                    status_byte <= STAT_NAK;   // default reply is NAK; accepts override
                    if (core_busy) begin
                        // core has priority: reject anything while a program runs
                        state <= SEND_STATUS;
                    end else begin
                        case (op)
                            OP_RD:   state <= sram_frame_ok ? RD_ISSUE    : SEND_STATUS;
                            OP_WR:   state <= sram_frame_ok ? WR_RX       : SEND_STATUS;
                            OP_IMEM: state <= imem_frame_ok ? IMEM_RX     : SEND_STATUS;
                            OP_GO: begin
                                state <= SEND_STATUS;
                                if (go_frame_ok) begin
                                    run_start   <= 1'b1;              // pulse host_run
                                    run_pc      <= addr[IMEM_AW-1:0];
                                    status_byte <= STAT_ACK;
                                end
                            end
                            default: state <= SEND_STATUS;
                        endcase
                    end
                end

                // ---------- read SRAM ----------
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

                // ---------- write SRAM ----------
                WR_RX: begin
                    if (rx_consume) begin
                        wr_byte <= rx_hold;
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

                // ---------- write instruction memory ----------
                IMEM_RX: begin
                    if (rx_consume) begin
                        word <= {word[23:0], rx_hold};   // assemble word, MSB first
                        cnt  <= cnt + 8'd1;
                        if (cnt == 8'd3) begin
                            cnt   <= 8'b0;
                            state <= IMEM_WR;
                        end
                    end
                end
                IMEM_WR: begin
                    imem_we    <= 1'b1;
                    imem_waddr <= addr[IMEM_AW-1:0] + idx[IMEM_AW-1:0];
                    imem_wdata <= word;
                    idx        <= idx + 1'b1;
                    if (idx + 1'b1 == nwords) begin
                        status_byte <= STAT_ACK;
                        state       <= SEND_STATUS;
                    end else state <= IMEM_RX;
                end

                // ---------- read the run-length counter ----------
                // Four bytes MSB first out of the snapshot taken in IDLE, then
                // straight back to IDLE — like a read reply, no trailing status.
                // `idx` counts them; IDLE cleared it on the way in.
                TMR_TX: begin
                    if (!transmitter_busy) begin
                        transmitter_start <= 1'b1;
                        data_out          <= tmr[TIMER_BITS-1 -: 8];
                        tmr               <= {tmr[TIMER_BITS-9:0], 8'b0};
                        state             <= TMR_TX_WAIT;
                    end
                end
                TMR_TX_WAIT: begin
                    if (tx_done) begin
                        idx <= idx + 1'b1;
                        if (idx + 1'b1 == TIMER_BYTES) state <= IDLE;
                        else                           state <= TMR_TX;
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

            // ---- receive capture --------------------------------------------
            // Runs in every state, and after the case so that a byte arriving on
            // the same cycle the FSM takes the previous one still lands: the
            // `rx_strobe` branch overrides the `rx_consume` clear. That pair of
            // events is the normal steady state of a streaming write, not a
            // corner case.
            if (rx_consume) rx_pending <= 1'b0;
            if (rx_strobe) begin
                if (rx_pending & ~rx_consume) begin
                    // Genuinely out of room. Keep the older byte so the stream
                    // stays in order and lose the new one, but say so: on this
                    // link the host waits for every reply, so an overrun means
                    // either the host got ahead of the protocol or the FSM is
                    // already desynced. Sticky — a one-clock event forty minutes
                    // into a soak is not something anyone is watching for.
                    rx_overrun <= 1'b1;
                end else begin
                    rx_hold    <= data_in;
                    rx_pending <= 1'b1;
                end
            end

            // Inter-byte timeout: if we sit mid-frame waiting on the host for
            // longer than RX_TIMEOUT clocks, abort silently back to IDLE so a
            // stalled/aborted frame can't wedge the link (docs/uart_host.md §7).
            // This runs after the case, so an abort overrides the state above.
            if (RX_TIMEOUT != 0) begin
                if (rx_waiting) begin
                    if (rx_strobe)                   to_cnt <= 32'b0;
                    else if (to_cnt + 1 >= RX_TIMEOUT) begin
                        to_cnt <= 32'b0;
                        state  <= IDLE;
                    end else                         to_cnt <= to_cnt + 32'd1;
                end else to_cnt <= 32'b0;
            end
        end
    end
endmodule
