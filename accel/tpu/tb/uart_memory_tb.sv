`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_memory_tb.sv — testbench for rtl/uart_memory.sv.
//
//   make mem
//
// Drives the DUT the way host/test_uart_link.py drives the board: 'W' a block of
// external SRAM, 'R' it back, compare. Nothing is poked by backdoor — the only
// path in or out is the serial pins and the behavioral async-SRAM chip model on
// the external memory bus, so a pass here means the whole
// receiver -> uart_interface -> sram_controller -> chip -> back chain is intact.
//
// CLK_PER_BIT is 104, matching the hardware image (12 MHz / 115200), not the 868
// the older UART testbenches use for a 100 MHz core. Turnaround timing is what is
// under investigation and it does not scale with the divisor.
//
// The tests, in the order they run:
//
//   roundtrip       two 64-byte patterns, written and read back whole. The
//                   baseline: if this fails nothing else is worth reading.
//   short frames    16 separate len=1 writes to consecutive addresses, then one
//                   len=16 read. Same byte count as one len=16 write but 16x the
//                   command turnarounds — the localiser from CLAUDE.md's
//                   experiment list, in simulation.
//   rejects         bad command byte, len=0, and a frame running off the end of
//                   the address space. Each must produce exactly one NAK and
//                   leave the FSM in IDLE.
//   quiet           after every exchange: nothing more may come out. A phantom
//                   frame here is the hardware symptom that started all this.
//
//   turnaround      a command byte sent while the device is still transmitting
//                   its previous reply must survive. Runs last because it
//                   deliberately leaves the FSM mid-frame; see the note above
//                   `diagnostic_overlap`.
// -----------------------------------------------------------------------------

module uart_memory_tb;

    // ---- Geometry / protocol -------------------------------------------------
    localparam int CPB        = 104;        // 12 MHz / 115200, as on the board
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;
    localparam int MAXN       = 256;        // longest single transfer here

    localparam [7:0] WCMD = 8'h57, RCMD = 8'h52;
    localparam [7:0] ACK  = 8'h06, NAK  = 8'h15;

    // ---- Clock / reset -------------------------------------------------------
    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;                   // 100 MHz simulation clock

    // ---- Pins ----------------------------------------------------------------
    logic uart_rx = 1'b1;                   // host -> device (idles high)
    wire  uart_tx;                          // device -> host

    wire [MEM_ADDR_W-1:0] sram_addr;
    wire [MEM_DATA_W-1:0] sram_data;        // inout: shared with the chip model
    wire                  sram_we, sram_ce, sram_oen;

    wire blink_slow, blink_fast, activity, collision;

    uart_memory #(
        .UART_CPB        (CPB),
        .UART_RX_TIMEOUT (0),               // as built: cmod_a7 board.tcl sets 0
        .MEM_ADDR_W      (MEM_ADDR_W),
        .MEM_DATA_W      (MEM_DATA_W),
        .SRAM_CPA        (0),
        .IMEM_AW         (10),
        .ACT_W           (6),               // tiny, so the LED logic still toggles
        .HB_W            (8)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),

        .uart_rx    (uart_rx),
        .uart_tx    (uart_tx),

        .sram_addr  (sram_addr),
        .sram_data  (sram_data),
        .sram_we    (sram_we),
        .sram_ce    (sram_ce),
        .sram_oen   (sram_oen),

        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .collision  (collision)
    );

    // =========================================================================
    // Behavioral async SRAM chip model (identical to tpu_top_uart_tb.sv).
    // =========================================================================
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    // =========================================================================
    // Frame counters, taken from inside the DUT rather than inferred from the
    // pins. `rx_byte` is the rising edge of the receiver's valid — exactly what
    // uart_interface counts — and `uart_tx_start` is one strobe per transmitted
    // frame. Counting the wire instead would mean telling a start bit from a
    // data bit that happens to be low.
    // =========================================================================
    int rx_frames = 0, tx_frames = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.rx_byte)       rx_frames++;
            if (dut.uart_tx_start) tx_frames++;
        end
    end

    int errors = 0, checks = 0;

    logic [7:0] sent [0:MAXN-1];
    logic [7:0] got  [0:MAXN-1];

    // =========================================================================
    // Wire tasks.
    // =========================================================================

    // 8N1 out of the host: start low, 8 data bits LSB first, stop high, then
    // `gap_bits` extra idle bit periods (0 = fully back to back).
    task automatic tx_byte(input [7:0] b, input int gap_bits);
        uart_rx = 1'b0;
        repeat (CPB) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            uart_rx = b[i];
            repeat (CPB) @(posedge clk);
        end
        uart_rx = 1'b1;
        repeat (CPB) @(posedge clk);
        if (gap_bits > 0) repeat (gap_bits * CPB) @(posedge clk);
    endtask

    // Decode one frame off uart_tx the way the host's FTDI does: find the
    // falling edge, wait a bit and a half to land in the middle of data bit 0,
    // then step a bit period at a time. Returns at the centre of the stop bit,
    // half a bit before the next start edge can occur, so back-to-back frames
    // are decoded without a race.
    task automatic rx_frame(output logic [7:0] b, output logic framing_ok);
        @(negedge uart_tx);
        repeat (CPB + CPB/2) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            b[i] = uart_tx;
            repeat (CPB) @(posedge clk);
        end
        framing_ok = (uart_tx === 1'b1);
    endtask

    // CMD + 24-bit address (big endian) + 16-bit length (big endian).
    task automatic send_header(input [7:0] cmd, input [23:0] a, input [15:0] n,
                               input int gap_bits);
        tx_byte(cmd,      gap_bits);
        tx_byte(a[23:16], gap_bits);
        tx_byte(a[15:8],  gap_bits);
        tx_byte(a[7:0],   gap_bits);
        tx_byte(n[15:8],  gap_bits);
        tx_byte(n[7:0],   gap_bits);
    endtask

    // Nothing more may come out of the device. Called after every exchange: the
    // hardware symptom is the device sending *more* bytes than it was asked for.
    task automatic expect_quiet(input string what);
        int quiet_tx;
        quiet_tx = tx_frames;
        repeat (20 * CPB) @(posedge clk);
        checks++;
        if (tx_frames != quiet_tx) begin
            $display("  FAIL %s: %0d extra frame(s) after the exchange ended",
                     what, tx_frames - quiet_tx);
            errors++;
        end
    endtask

    // =========================================================================
    // Command tasks. Payload comes from / lands in the module-level arrays.
    // =========================================================================

    // 'W' addr, len n, payload sent[0 .. n-1]; expect one ACK.
    task automatic do_write(input int addr, input int n, input int gap_bits,
                            input string what);
        logic [7:0] st;
        logic       ok;
        fork
            begin : host_tx
                send_header(WCMD, 24'(addr), 16'(n), gap_bits);
                for (int i = 0; i < n; i++) tx_byte(sent[i], gap_bits);
            end
            rx_frame(st, ok);
        join
        checks++;
        if (st !== ACK) begin
            $display("  FAIL %s: write @%05h len %0d replied %02h, expected ACK",
                     what, addr, n, st);
            errors++;
        end else if (!ok) begin
            $display("  FAIL %s: ACK came back with a low stop bit", what);
            errors++;
        end
    endtask

    // 'R' addr, len n -> got[0 .. n-1]. Reads carry no status byte.
    task automatic do_read(input int addr, input int n, input int gap_bits,
                           input string what);
        logic [7:0] b;
        logic       ok;
        fork
            send_header(RCMD, 24'(addr), 16'(n), gap_bits);
            begin : host_rx
                for (int i = 0; i < n; i++) begin
                    // Decoded into a scalar first: passing an array element as a
                    // task output argument is legal SystemVerilog but not
                    // uniformly supported.
                    rx_frame(b, ok);
                    got[i] = b;
                    if (!ok) begin
                        $display("  FAIL %s: read byte %0d had a low stop bit",
                                 what, i);
                        errors++;
                    end
                end
            end
        join
    endtask

    // Compare got[0 .. n-1] against sent[0 .. n-1].
    task automatic check_payload(input int addr, input int n, input string what);
        int bad;
        bad = 0;
        for (int i = 0; i < n; i++) begin
            checks++;
            if (got[i] !== sent[i]) begin
                if (bad == 0)
                    $display("  FAIL %s: @%05h got %02h, wrote %02h (xor %08b)",
                             what, addr + i, got[i], sent[i], got[i] ^ sent[i]);
                bad++;
                errors++;
            end
        end
        if (bad == 0) $display("  ok   %s: %0d bytes round-tripped intact", what, n);
        else          $display("  FAIL %s: %0d of %0d bytes wrong", what, bad, n);
    endtask

    // A frame the device must reject. Sends `hdr_bytes` header bytes (no
    // payload) and expects exactly one NAK.
    task automatic expect_nak(input [7:0] cmd, input [23:0] a, input [15:0] n,
                              input int hdr_bytes, input string what);
        logic [7:0] st;
        logic       ok;
        logic [7:0] hdr [0:5];
        hdr[0] = cmd;
        hdr[1] = a[23:16];
        hdr[2] = a[15:8];
        hdr[3] = a[7:0];
        hdr[4] = n[15:8];
        hdr[5] = n[7:0];
        fork
            begin : host_tx
                for (int i = 0; i < hdr_bytes; i++) tx_byte(hdr[i], 0);
            end
            rx_frame(st, ok);
        join
        checks++;
        if (st !== NAK) begin
            $display("  FAIL %s: replied %02h, expected NAK", what, st);
            errors++;
        end else begin
            $display("  ok   %s: NAK", what);
        end
    endtask

    // =========================================================================
    // Tests.
    // =========================================================================

    logic [15:0] lfsr;

    // Deterministic pseudo-random payload. An LFSR rather than $random so the
    // vector is identical under any simulator.
    task automatic fill_random(input int n);
        for (int i = 0; i < n; i++) begin
            for (int k = 0; k < 8; k++)
                lfsr = {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            sent[i] = lfsr[7:0];
        end
    endtask

    // Address signature: each byte encodes its own address, so a stuck or
    // swapped address line shows up as data that belongs somewhere else.
    task automatic fill_signature(input int addr, input int n);
        for (int i = 0; i < n; i++)
            sent[i] = 8'((addr + i) ^ ((addr + i) >> 8) ^ 8'h5A);
    endtask

    // One block written whole and read back whole.
    task automatic test_roundtrip(input int addr, input int n, input string what);
        do_write(addr, n, 0, what);
        expect_quiet(what);
        do_read(addr, n, 0, what);
        check_payload(addr, n, what);
        expect_quiet(what);
    endtask

    // Same bytes, one command each. Identical traffic volume to a single len=n
    // write, n times the command turnarounds — so a failure rate that tracks
    // this and not the block writes is a turnaround fault, not a byte fault.
    task automatic test_short_frames(input int addr, input int n, input string what);
        logic [7:0] keep [0:MAXN-1];
        for (int i = 0; i < n; i++) keep[i] = sent[i];
        for (int i = 0; i < n; i++) begin
            sent[0] = keep[i];
            do_write(addr + i, 1, 0, what);
        end
        for (int i = 0; i < n; i++) sent[i] = keep[i];
        expect_quiet(what);
        do_read(addr, n, 0, what);
        check_payload(addr, n, what);
        expect_quiet(what);
    endtask

    // =========================================================================
    // TURNAROUND — the byte that arrives while the device is still replying.
    //
    // The host sends a command byte while the device is still transmitting the
    // previous reply. This used to be a diagnostic rather than a check: the FSM
    // consumed `rx_byte` only in IDLE, RX_ADDR, RX_LEN, WR_RX and IMEM_RX, so a
    // byte landing in SEND_STATUS + SEND_STATUS_WAIT — a full byte time, sitting
    // exactly at the turnaround — was dropped, and every frame after it decoded
    // one byte out of phase, permanently.
    //
    // uart_interface now captures into a one-byte holding register in every
    // state (docs/uart_host.md §5), so the byte survives and this is a real
    // assertion. It runs last because it deliberately leaves the FSM mid-frame.
    // =========================================================================
    task automatic diagnostic_overlap();
        logic [7:0] st;
        logic       ok;
        int         rx0;

        $display("");
        $display("---- turnaround: command byte arriving during the reply ----");

        sent[0] = 8'hA5;
        rx0 = rx_frames;

        fork
            // A len=1 write, then — without waiting for the ACK — the next
            // command's first byte. It lands squarely inside SEND_STATUS_WAIT.
            begin : host_tx
                send_header(WCMD, 24'h000100, 16'd1, 0);
                tx_byte(sent[0], 0);
                tx_byte(RCMD, 0);        // pipelined: overlaps the ACK
            end
            rx_frame(st, ok);
        join

        // Let the link settle. With RX_TIMEOUT = 0 there is no mid-frame abort,
        // so whatever state the overlap left the FSM in, it stays there.
        repeat (40 * CPB) @(posedge clk);

        // Written out rather than selected with a ternary: iverilog pads string
        // literals of unequal width in a conditional and prints the padding.
        if (st === ACK) $display("  reply to the write : %02h (ACK)", st);
        else            $display("  reply to the write : %02h (not ACK)", st);

        $display("  receiver delivered : %0d of the 8 bytes the host sent",
                 rx_frames - rx0);

        if (dut.uart_rx_overrun)
            $display("  rx_overrun         : SET — the FSM ran out of room");
        else
            $display("  rx_overrun         : clear — one byte of buffering was enough");

        // The pipelined byte was 'R'. If uart_interface consumed it, the FSM is
        // now in RX_ADDR waiting for three address bytes and host_busy is high.
        // If it dropped it, the FSM fell back to IDLE and host_busy is low —
        // which is the failure: the receiver delivered every byte (the count
        // above says 8) and the command FSM threw the last one away.
        checks++;
        if (dut.uart_host_busy) begin
            $display("  verdict            : CONSUMED — 'R' was accepted; the FSM is mid-frame");
            $display("                       and will sit in RX_ADDR until three more bytes");
            $display("                       arrive (RX_TIMEOUT = 0, so forever).");
        end else begin
            errors++;
            $display("  FAIL verdict       : DROPPED — the receiver delivered every byte and");
            $display("                       uart_interface discarded the last one. The RX");
            $display("                       holding register is not doing its job.");
        end
        $display("------------------------------------------------------------");
        $display("");
    endtask

    // =========================================================================
    // Stimulus.
    // =========================================================================
    initial begin
        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;
        lfsr = 16'hACE1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("==== uart_memory testbench (CPB=%0d, %0d-bit SRAM address) ====",
                 CPB, MEM_ADDR_W);

        // 1) Baseline: a block out and back, twice, with different data.
        fill_random(64);
        test_roundtrip('h00000, 64, "roundtrip-random");

        fill_signature('h3F000, 64);
        test_roundtrip('h3F000, 64, "roundtrip-signature");

        // 2) The turnaround storm.
        fill_random(16);
        test_short_frames('h00200, 16, "short-frames");

        // 3) Reject paths. Each must be exactly one NAK and nothing else.
        expect_nak(8'h5A, 24'h000000, 16'd4, 1, "bad-command");
        expect_quiet("bad-command");

        expect_nak(WCMD, 24'h000000, 16'd0, 6, "zero-length");
        expect_quiet("zero-length");

        // addr + len runs off the end of the 2**19-byte space
        expect_nak(WCMD, 24'h07FFF0, 16'd32, 6, "out-of-range");
        expect_quiet("out-of-range");

        // 4) The link must still work after all that.
        fill_random(32);
        test_roundtrip('h00080, 32, "recovery");

        // Nothing above overlaps a reply, so no byte should ever have arrived
        // while the device was transmitting.
        checks++;
        if (collision) begin
            $display("  FAIL: collision flag set during the protocol tests");
            errors++;
        end

        // Last, because it deliberately leaves the FSM mid-frame.
        diagnostic_overlap();

        if (errors == 0)
            $display("ALL TESTS PASSED (%0d checks, %0d bytes in, %0d frames out)",
                     checks, rx_frames, tx_frames);
        else
            $display("FAIL %0d ERRORS (%0d checks)", errors, checks);

        $finish;
    end

    // Watchdog. The full run is about 6 ms of simulated time; anything past
    // 40 ms means a task is blocked waiting for a frame that never came.
    initial begin
        #40_000_000;
        $display("FAIL timeout — %0d bytes in, %0d frames out", rx_frames, tx_frames);
        $finish;
    end

    initial begin
        $dumpfile("uart_memory_tb.vcd");
        $dumpvars(0, uart_memory_tb);
    end

endmodule
