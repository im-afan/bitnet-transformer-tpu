// -----------------------------------------------------------------------------
// sram_tb.sv — self-checking testbench for accel/tpu/rtl/sram.sv
//
// Drives the sram_controller through its start/busy/done handshake and connects
// its chip-side bus to a behavioral model of the CMOD A7 Cellular RAM
// (ISSI IS61/64WV5128, 512K x 8 asynchronous SRAM):
//
//   * output driven only when CE#=0, OE#=0, WE#=1 (a read); Hi-Z otherwise
//   * a word is committed on the WE# rising edge while CE# is low
//   * a small access delay models the asynchronous read path (tAA)
//
// Checks: write/read round-trips over data patterns and address corners, that
// `done` is a single-cycle pulse, `busy` frames the whole op, and that the bus
// is released (Hi-Z) whenever the controller is idle so there's no contention.
//
// Run:  make TEST=sram sim         (iverilog -g2012 + vvp)
// -----------------------------------------------------------------------------

`timescale 1ns/1ps

module sram_tb;

    // ---- Parameters (must match the DUT instance) ---------------------------
    localparam int CLOCKS_PER_ACCESS = 2;
    localparam int ADDR_W            = 19;
    localparam int DATA_W            = 8;

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT interface.
    // -------------------------------------------------------------------------
    logic                start;
    logic                we;
    logic [ADDR_W-1:0]   addr;
    logic [DATA_W-1:0]   din;
    logic [DATA_W-1:0]   dout;
    logic                busy;
    logic                done;

    wire  [ADDR_W-1:0]   sram_addr;
    wire  [DATA_W-1:0]   sram_data;
    wire                 sram_we;
    wire                 sram_ce;
    wire                 sram_oen;

    sram_controller #(
        .CLOCKS_PER_ACCESS(CLOCKS_PER_ACCESS),
        .ADDR_W(ADDR_W), .DATA_W(DATA_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .we(we), .addr(addr), .din(din),
        .dout(dout), .busy(busy), .done(done),
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen)
    );

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
    always @(posedge sram_we) begin
        if (sram_ce == 1'b0) mem[sram_addr] <= sram_data;
    end

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
    // Handshake drivers. Inputs change on negedge; the DUT samples on posedge.
    // Each op asserts start for exactly one cycle, then waits for `done`,
    // asserting along the way that busy frames the transaction and done is a
    // single-cycle pulse.
    // -------------------------------------------------------------------------
    task automatic do_write(input [ADDR_W-1:0] a, input [DATA_W-1:0] d);
        int done_cycles;
        @(negedge clk);
        addr = a; din = d; we = 1'b1; start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        check(busy, "write: busy asserted after start");
        done_cycles = 0;
        while (!done) begin
            check(busy, "write: busy stays high until done");
            @(negedge clk);
        end
        // done is high now; confirm it drops next cycle (single-cycle pulse).
        @(negedge clk);
        check(!done, "write: done is a single-cycle pulse");
    endtask

    task automatic do_read(input [ADDR_W-1:0] a, output [DATA_W-1:0] d);
        @(negedge clk);
        addr = a; we = 1'b0; start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        check(busy, "read: busy asserted after start");
        while (!done) begin
            check(busy, "read: busy stays high until done");
            @(negedge clk);
        end
        d = dout;
        @(negedge clk);
        check(!done, "read: done is a single-cycle pulse");
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
    // Stimulus.
    // -------------------------------------------------------------------------
    logic [DATA_W-1:0] rd;
    initial begin
        $dumpfile("sram_tb.vcd");
        $dumpvars(0, sram_tb);

        start = 1'b0; we = 1'b0; addr = '0; din = '0;

        // Reset.
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Bus should be idle/Hi-Z coming out of reset.
        check(sram_data === {DATA_W{1'bz}}, "post-reset: bus Hi-Z");
        check(!busy && !done, "post-reset: idle");

        $display("==== SRAM controller testbench ====");

        // Data patterns at a fixed address.
        write_read_check(19'h00010, 8'h00, "pattern 0x00");
        write_read_check(19'h00010, 8'hFF, "pattern 0xFF");
        write_read_check(19'h00010, 8'hA5, "pattern 0xA5");
        write_read_check(19'h00010, 8'h5A, "pattern 0x5A");

        // Address corners.
        write_read_check(19'h00000,       8'h11, "addr min");
        write_read_check({ADDR_W{1'b1}},  8'h22, "addr max");
        write_read_check(19'h40000,       8'h33, "addr bit18");
        write_read_check(19'h2AAAA,       8'h44, "addr walk-1");
        write_read_check(19'h15555,       8'h55, "addr walk-0");

        // Distinct values across a block, then read all back — checks that each
        // write commits to its own address (no address/data aliasing).
        for (int i = 0; i < 32; i++)
            do_write(19'h00100 + i[ADDR_W-1:0], (i * 7 + 3));
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

        // Summary.
        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("SRAM: ALL TESTS PASSED");
        else             $display("SRAM: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #100000;
        $display("SRAM: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
