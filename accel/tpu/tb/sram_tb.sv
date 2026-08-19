// -----------------------------------------------------------------------------
// sram_tb.sv — self-checking testbench for accel/tpu/rtl/sram.sv
//
// Drives the sram_controller through its range request interface and connects
// its chip-side bus to a behavioral model of the CMOD A7 Cellular RAM
// (ISSI IS61/64WV5128, 512K x 8 asynchronous SRAM):
//
//   * output driven only when CE#=0, OE#=0, WE#=1 (a read); Hi-Z otherwise
//   * a word is committed on the WE# rising edge while CE# is low
//   * a small access delay models the asynchronous read path (tAA)
//   * the model *checks* that the address and data it commits have not moved
//     since WE# went low — the hazard the controller's falling-edge WE# exists
//     to avoid, and the one that would silently write a byte to its neighbour
//
// Coverage:
//   * len = 1, which is the old one-byte handshake: patterns, address corners,
//     a block of distinct values, overwrite, `done` a single pulse, `busy`
//     framing the op, the bus released when idle
//   * multi-byte ranges in both directions, including unaligned and long
//   * `stride`, both directions, and stride 0 meaning 1
//   * len = 0: `done` with no bus activity at all
//   * a stalled write producer (din_valid low mid-range)
//   * dout_valid arrives exactly len times, the last one with `done`
//   * throughput: clocks per byte against the beat lengths the DUT promises
//
// Run:  make TEST=sram sim                  (CLOCKS_PER_ACCESS = 0)
//       make TEST=sram IVFLAGS=-DCPA=2 sim  (the old 2-extra-clock beat)
// -----------------------------------------------------------------------------

`timescale 1ns/1ps

`ifndef CPA
`define CPA 0
`endif

module sram_tb;

    // ---- Parameters (must match the DUT instance) ---------------------------
    localparam int CLOCKS_PER_ACCESS = `CPA;
    localparam int ADDR_W            = 19;
    localparam int DATA_W            = 8;
    localparam int LEN_W             = 16;
    localparam int STRIDE_W          = 16;

    // What the DUT promises per byte, restated here so the throughput check is
    // written against the contract rather than against the implementation.
    localparam int RD_BEAT = CLOCKS_PER_ACCESS + 1;
    localparam int WR_BEAT = (CLOCKS_PER_ACCESS + 1 > 2) ? CLOCKS_PER_ACCESS + 1 : 2;

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    int unsigned cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // -------------------------------------------------------------------------
    // DUT interface.
    // -------------------------------------------------------------------------
    logic                  start;
    logic                  we;
    logic [ADDR_W-1:0]     addr;
    logic [LEN_W-1:0]      len;
    logic [STRIDE_W-1:0]   stride;
    logic [DATA_W-1:0]     din;
    logic                  din_valid;
    logic                  din_ready;
    logic [DATA_W-1:0]     dout;
    logic                  dout_valid;
    logic                  busy;
    logic                  done;

    wire  [ADDR_W-1:0]   sram_addr;
    wire  [DATA_W-1:0]   sram_data;
    wire                 sram_we;
    wire                 sram_ce;
    wire                 sram_oen;

    sram_controller #(
        .CLOCKS_PER_ACCESS(CLOCKS_PER_ACCESS),
        .ADDR_W(ADDR_W), .DATA_W(DATA_W),
        .LEN_W(LEN_W), .STRIDE_W(STRIDE_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .we(we), .addr(addr), .len(len), .stride(stride),
        .din(din), .din_valid(din_valid), .din_ready(din_ready),
        .dout(dout), .dout_valid(dout_valid),
        .busy(busy), .done(done),
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen)
    );

    // -------------------------------------------------------------------------
    // Bookkeeping.
    // -------------------------------------------------------------------------
    int errors = 0;
    int checks = 0;

    task automatic check(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL %s (t=%0t)", tag, $time);
        end
    endtask

    // -------------------------------------------------------------------------
    // Behavioral async SRAM model.
    // -------------------------------------------------------------------------
    localparam int MEM_SZ = 1 << ADDR_W;
    logic [DATA_W-1:0] mem [0:MEM_SZ-1];

    // Read: drive the bus (after a modest access delay) only when selected,
    // output-enabled, and not writing. Released to Hi-Z otherwise. The #3 on the
    // continuous assign models the asynchronous access time (tAA).
    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? mem[sram_addr] : {DATA_W{1'bz}};

    // Write: commit the word on the WE# rising edge while the chip is selected.
    // The pins are also latched on the falling edge so the commit can be checked
    // against them: an async SRAM specifies tAS/tAW/tHA around this pulse, and a
    // controller that moves the address inside it writes to the wrong cell on
    // hardware while looking perfect in a functional model.
    logic [ADDR_W-1:0] we_lo_addr;
    logic [DATA_W-1:0] we_lo_data;
    int                write_hazards = 0;

    always @(negedge sram_we) begin
        we_lo_addr <= sram_addr;
        we_lo_data <= sram_data;
    end

    always @(posedge sram_we) begin
        if (sram_ce == 1'b0) begin
            if (sram_addr !== we_lo_addr) begin
                write_hazards++;
                $display("  FAIL write hazard: address moved inside WE# low (%05h -> %05h) t=%0t",
                         we_lo_addr, sram_addr, $time);
            end
            if (sram_data !== we_lo_data) begin
                write_hazards++;
                $display("  FAIL write hazard: data moved inside WE# low (%02h -> %02h) t=%0t",
                         we_lo_data, sram_data, $time);
            end
            mem[sram_addr] <= sram_data;
        end
    end

    // The chip must never be asked to drive while the FPGA is driving.
    always @(posedge clk)
        if (rst_n && sram_ce == 1'b0 && sram_oen == 1'b0 && sram_we == 1'b0) begin
            errors++;
            $display("  FAIL bus contention: OE# and WE# both low (t=%0t)", $time);
        end

    // -------------------------------------------------------------------------
    // Stream agents.
    //
    // Write: `wbuf` is the byte stream, `wptr` the index of the byte currently
    // offered. `din` is combinational off `wptr` so it is stable at every edge
    // the DUT might take it on; `wptr` advances on the accepted handshake.
    // Read: every `dout_valid` appends to `rbuf`.
    // -------------------------------------------------------------------------
    logic [DATA_W-1:0] wbuf [0:1023];
    int                wptr;
    logic [DATA_W-1:0] rbuf [0:1023];
    int                rptr;
    int                rd_valid_after_done;

    assign din = wbuf[wptr];

    always @(posedge clk) begin
        if (din_valid && din_ready) wptr <= wptr + 1;
        if (dout_valid) begin
            rbuf[rptr] <= dout;
            rptr       <= rptr + 1;
        end
    end

    // -------------------------------------------------------------------------
    // Request drivers. Inputs change on negedge; the DUT samples on posedge.
    // Each request asserts start for exactly one cycle, then waits for `done`,
    // asserting along the way that busy frames the transaction and done is a
    // single-cycle pulse.
    // -------------------------------------------------------------------------
    int unsigned t_start, t_done;

    task automatic do_range(input logic wr,
                            input [ADDR_W-1:0] a,
                            input [LEN_W-1:0] n,
                            input [STRIDE_W-1:0] st,
                            input string tag);
        int guard;
        @(negedge clk);
        addr = a; len = n; stride = st; we = wr; start = 1'b1;
        t_start = cyc;
        @(negedge clk);
        start = 1'b0;
        if (n != 0) check(busy, {tag, ": busy asserted after start"});
        guard = 0;
        while (!done) begin
            if (n != 0) check(busy, {tag, ": busy stays high until done"});
            @(negedge clk);
            guard++;
            if (guard > 40000) begin
                $display("  FAIL %s: never completed", tag);
                errors++;
                disable do_range;
            end
        end
        t_done = cyc;
        @(negedge clk);
        check(!done, {tag, ": done is a single-cycle pulse"});
    endtask

    // Write `n` bytes of `wbuf` starting at `a`, stepping by `st`.
    task automatic write_range(input [ADDR_W-1:0] a, input [LEN_W-1:0] n,
                               input [STRIDE_W-1:0] st, input string tag);
        wptr      = 0;
        din_valid = 1'b1;
        do_range(1'b1, a, n, st, tag);
        din_valid = 1'b0;
        check(wptr == int'(n), {tag, ": consumed exactly len bytes"});
    endtask

    // Read `n` bytes starting at `a`, stepping by `st`, into `rbuf`.
    task automatic read_range(input [ADDR_W-1:0] a, input [LEN_W-1:0] n,
                              input [STRIDE_W-1:0] st, input string tag);
        rptr = 0;
        do_range(1'b0, a, n, st, tag);
        check(rptr == int'(n), {tag, ": produced exactly len bytes"});
    endtask

    // Legacy single-byte helpers: the len = 1 case of the range interface.
    task automatic do_write(input [ADDR_W-1:0] a, input [DATA_W-1:0] d);
        wbuf[0] = d;
        write_range(a, 16'd1, 16'd0, "single write");
    endtask

    task automatic do_read(input [ADDR_W-1:0] a, output [DATA_W-1:0] d);
        read_range(a, 16'd1, 16'd0, "single read");
        // A len=1 caller samples `dout` on `done` (uart_interface does exactly
        // this), so that path is checked here and not only via rbuf.
        check(dout === rbuf[0], "single read: dout still valid at done");
        d = dout;
    endtask

    task automatic write_read_check(input [ADDR_W-1:0] a, input [DATA_W-1:0] d,
                                     input string tag);
        logic [DATA_W-1:0] rd;
        do_write(a, d);
        // Bus must be released once the controller is idle again.
        check(sram_data === {DATA_W{1'bz}}, {tag, ": bus Hi-Z when idle"});
        do_read(a, rd);
        check(rd === d, tag);
        if (rd !== d)
            $display("    @0x%05h: wrote 0x%02h, read 0x%02h", a, d, rd);
    endtask

    // -------------------------------------------------------------------------
    // A range round trip: fill wbuf with a pattern, write it, read it back, and
    // compare both the returned stream and the memory model's contents.
    // -------------------------------------------------------------------------
    task automatic range_round_trip(input [ADDR_W-1:0] a, input int n,
                                    input [STRIDE_W-1:0] st, input string tag);
        int unsigned wr_clocks, rd_clocks;
        logic [STRIDE_W-1:0] step;
        step = (st == 0) ? 16'd1 : st;

        for (int i = 0; i < n; i++) wbuf[i] = 8'((i * 37 + a + 11) ^ 8'h5A);
        for (int i = 0; i < n; i++) mem[a + i*step] = 8'hEE;      // poison

        write_range(a, LEN_W'(n), st, {tag, " wr"});
        wr_clocks = t_done - t_start;

        for (int i = 0; i < n; i++) begin
            check(mem[a + i*step] === wbuf[i], $sformatf("%s: mem[%0d]", tag, i));
            if (mem[a + i*step] !== wbuf[i])
                $display("    %s @%05h: exp %02h got %02h", tag, a + i*step,
                         wbuf[i], mem[a + i*step]);
        end

        read_range(a, LEN_W'(n), st, {tag, " rd"});
        rd_clocks = t_done - t_start;

        for (int i = 0; i < n; i++) begin
            check(rbuf[i] === wbuf[i], $sformatf("%s: rbuf[%0d]", tag, i));
            if (rbuf[i] !== wbuf[i])
                $display("    %s stream[%0d]: exp %02h got %02h", tag, i,
                         wbuf[i], rbuf[i]);
        end

        // Sequencing overhead is a small constant per request, not per byte.
        check(wr_clocks <= n*WR_BEAT + 6,
              $sformatf("%s: write %0d bytes in %0d clocks", tag, n, wr_clocks));
        check(rd_clocks <= n*RD_BEAT + 4,
              $sformatf("%s: read %0d bytes in %0d clocks", tag, n, rd_clocks));
        $display("[RANGE %-9s] n=%0d stride=%0d  write %0d clk (%.2f/byte)  read %0d clk (%.2f/byte)",
                 tag, n, step, wr_clocks, real'(wr_clocks)/real'(n),
                 rd_clocks, real'(rd_clocks)/real'(n));
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    logic [DATA_W-1:0] rd;
    int unsigned idle_ce_low;
    initial begin
`ifndef NO_VCD
        $dumpfile("sram_tb.vcd");
        $dumpvars(0, sram_tb);
`endif

        start = 1'b0; we = 1'b0; addr = '0; len = '0; stride = '0;
        din_valid = 1'b0; wptr = 0; rptr = 0;

        // Reset.
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Bus should be idle/Hi-Z coming out of reset.
        check(sram_data === {DATA_W{1'bz}}, "post-reset: bus Hi-Z");
        check(!busy && !done, "post-reset: idle");
        check(sram_we === 1'b1, "post-reset: WE# high");

        $display("==== SRAM controller testbench (CLOCKS_PER_ACCESS=%0d) ====",
                 CLOCKS_PER_ACCESS);

        // ---- len = 1: the old one-byte handshake ----------------------------
        write_read_check(19'h00010, 8'h00, "pattern 0x00");
        write_read_check(19'h00010, 8'hFF, "pattern 0xFF");
        write_read_check(19'h00010, 8'hA5, "pattern 0xA5");
        write_read_check(19'h00010, 8'h5A, "pattern 0x5A");

        write_read_check(19'h00000,       8'h11, "addr min");
        write_read_check({ADDR_W{1'b1}},  8'h22, "addr max");
        write_read_check(19'h40000,       8'h33, "addr bit18");
        write_read_check(19'h2AAAA,       8'h44, "addr walk-1");
        write_read_check(19'h15555,       8'h55, "addr walk-0");

        // Distinct values across a block, then read all back — checks that each
        // write commits to its own address (no address/data aliasing).
        for (int i = 0; i < 32; i++)
            do_write(19'h00100 + i[ADDR_W-1:0], 8'((i * 7 + 3)));
        for (int i = 0; i < 32; i++) begin
            do_read(19'h00100 + i[ADDR_W-1:0], rd);
            check(rd === 8'((i * 7 + 3)), $sformatf("block[%0d]", i));
            if (rd !== 8'((i * 7 + 3)))
                $display("    block[%0d]: exp 0x%02h got 0x%02h", i, 8'((i*7+3)), rd);
        end

        // Overwrite an existing address to confirm old data is replaced.
        do_write(19'h00100, 8'hDE);
        do_read (19'h00100, rd);
        check(rd === 8'hDE, "overwrite");

        // ---- ranges ---------------------------------------------------------
        range_round_trip(19'h01000,   2, 16'd0, "n2");
        range_round_trip(19'h01100,  16, 16'd0, "n16");
        range_round_trip(19'h01203,  64, 16'd0, "unaligned");
        range_round_trip(19'h02000, 256, 16'd0, "n256");
        range_round_trip(19'h7F000, 256, 16'd0, "high");

        // Strided ranges: the wrmem.t case, where consecutive bytes of the
        // transfer are tdrow apart in DRAM.
        range_round_trip(19'h03000,  32, 16'd4,   "stride4");
        range_round_trip(19'h04000,  64, 16'd32,  "stride32");
        range_round_trip(19'h10000, 128, 16'd128, "stride128");

        // Stride 0 must mean 1 (the "zero is not set" convention).
        for (int i = 0; i < 8; i++) wbuf[i] = 8'(i + 8'hA0);
        for (int i = 0; i < 8; i++) mem[19'h05000 + i] = 8'hEE;
        write_range(19'h05000, 16'd8, 16'd0, "stride0");
        for (int i = 0; i < 8; i++)
            check(mem[19'h05000 + i] === 8'(i + 8'hA0), $sformatf("stride0[%0d]", i));

        // ---- len = 0: a handshake and nothing else --------------------------
        idle_ce_low = 0;
        fork
            begin
                @(negedge clk);
                addr = 19'h06000; len = 16'd0; stride = 16'd0; we = 1'b0; start = 1'b1;
                @(negedge clk);
                start = 1'b0;
                check(!busy, "len0: busy never asserts");
                while (!done) @(negedge clk);
                check(1'b1, "len0 completes");
                @(negedge clk);
                check(!done, "len0: done is a single-cycle pulse");
            end
            begin
                repeat (8) begin
                    @(posedge clk);
                    if (sram_ce == 1'b0) idle_ce_low++;
                end
            end
        join
        check(idle_ce_low == 0, "len0: chip never selected");

        // ---- a stalled write producer ---------------------------------------
        // din_valid drops mid-range; the controller must park, then resume with
        // the same byte stream and no gaps or duplicates in memory.
        for (int i = 0; i < 24; i++) wbuf[i] = 8'(8'hC0 + i);
        for (int i = 0; i < 24; i++) mem[19'h07000 + i] = 8'hEE;
        wptr = 0;
        fork
            begin
                din_valid = 1'b1;
                do_range(1'b1, 19'h07000, 16'd24, 16'd0, "stalled");
                din_valid = 1'b0;
            end
            begin
                // Three stalls of growing length at unaligned points.
                for (int s = 1; s <= 3; s++) begin
                    wait (wptr == s * 5);
                    @(negedge clk);
                    din_valid = 1'b0;
                    repeat (s * 3) @(negedge clk);
                    din_valid = 1'b1;
                end
            end
        join
        for (int i = 0; i < 24; i++) begin
            check(mem[19'h07000 + i] === 8'(8'hC0 + i), $sformatf("stalled[%0d]", i));
            if (mem[19'h07000 + i] !== 8'(8'hC0 + i))
                $display("    stalled[%0d]: exp %02h got %02h", i, 8'(8'hC0+i),
                         mem[19'h07000 + i]);
        end
        check(wptr == 24, "stalled: consumed exactly 24 bytes");

        // ---- back-to-back requests, no idle gap -----------------------------
        for (int i = 0; i < 12; i++) wbuf[i] = 8'(8'h30 + i);
        write_range(19'h08000, 16'd6, 16'd0, "b2b a");
        wptr = 6;                                   // continue the same stream
        din_valid = 1'b1;
        do_range(1'b1, 19'h08006, 16'd6, 16'd0, "b2b b");
        din_valid = 1'b0;
        for (int i = 0; i < 12; i++)
            check(mem[19'h08000 + i] === 8'(8'h30 + i), $sformatf("b2b[%0d]", i));
        read_range(19'h08000, 16'd12, 16'd0, "b2b rd");
        for (int i = 0; i < 12; i++)
            check(rbuf[i] === 8'(8'h30 + i), $sformatf("b2b rd[%0d]", i));

        // ---- hazard tally ---------------------------------------------------
        checks++;
        if (write_hazards != 0) begin
            errors++;
            $display("  FAIL %0d write hazards seen", write_hazards);
        end

        // Summary.
        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("SRAM: ALL TESTS PASSED");
        else             $display("SRAM: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #400000;
        $display("SRAM: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
