`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// scratchpad_tb.sv — self-checking testbench for accel/tpu/rtl/scratchpad.sv
//
// Instantiates the scratchpad TWICE from the same driver — once MEM_STYLE="BRAM",
// once MEM_STYLE="REG" — with a shared input bus and separate read-data outputs.
// Both variants must be bit-identical to each other and to a byte-array reference
// model, which simultaneously checks that the register/BRAM knob is purely a
// synthesis choice (same functional behaviour, scratchpad.sv header).
//
// Coverage:
//   - S port  : int32 word write/read roundtrip.
//   - V port  : full-width write, then a byte-strobed partial write (masked bytes
//               must survive) — exercises *_wstrb.
//   - C port  : int32 result-row write + accumulate-style readback.
//   - Cross-port coherence: write via one port, read the same bytes via another
//               (shared storage, differing widths / byte views).
//   - Unaligned windows: reads and writes at every byte offset within a bank
//               row, including windows that straddle the bank wrap — the case
//               the byte-lane banking + barrel rotate exists to handle.
//   - Address wrap: a window running off the top of the memory wraps mod DEPTH.
//   - Port sequencing: A_rd, W_rd and C_rw driven on consecutive cycles (how the
//               MXU actually walks its S_LOAD / S_RUN / S_WB_RD states).
//   - Read-first: a read and a write to the same address in one cycle returns
//               the OLD value.
//
// NOT covered, deliberately: simultaneous requests on two read ports (or two
// write ports). scratchpad.sv arbitrates them onto one banked BRAM port by fixed
// priority and flags the overlap with $error — the TPU never issues them (see
// the exclusivity invariant in the scratchpad.sv header), and supporting them
// would need one storage replica per read port, ~3 Mbit against the A7-35T's
// 1800 Kbit. Driving two enables here would trip the module's own assertion.
//
// Run:  make TEST=scratchpad sim      (iverilog -g2012 + vvp)
// -----------------------------------------------------------------------------

module scratchpad_tb;

    // ---- Small parameters (must match both DUT instances) -------------------
    localparam int ADDR_W    = 12;
    localparam int A_BYTES   = 8;
    localparam int W_BYTES   = 4;
    localparam int C_BYTES   = 16;
    localparam int V_BYTES   = 8;
    localparam int S_BYTES   = 4;
    localparam int DMA_BYTES = 8;
    localparam int MEM_SZ    = 1 << ADDR_W;

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // Shared input bus (driven to both DUTs); separate read-data per DUT.
    // -------------------------------------------------------------------------
    logic                 A_re;   logic [ADDR_W-1:0] A_raddr;
    logic                 W_re;   logic [ADDR_W-1:0] W_raddr;
    logic                 C_re;   logic [ADDR_W-1:0] C_raddr;
    logic                 C_we;   logic [ADDR_W-1:0] C_waddr;
    logic [C_BYTES*8-1:0] C_wdata; logic [C_BYTES-1:0] C_wstrb;
    logic                 V_re;   logic [ADDR_W-1:0] V_raddr;
    logic                 V_we;   logic [ADDR_W-1:0] V_waddr;
    logic [V_BYTES*8-1:0] V_wdata; logic [V_BYTES-1:0] V_wstrb;
    logic                 s_re, s_we; logic [ADDR_W-1:0] s_addr;
    logic [S_BYTES*8-1:0] s_wdata;
    logic                 dma_re; logic [ADDR_W-1:0] dma_raddr;
    logic                 dma_we; logic [ADDR_W-1:0] dma_waddr;
    logic [DMA_BYTES*8-1:0] dma_wdata; logic [DMA_BYTES-1:0] dma_wstrb;

    logic [A_BYTES*8-1:0] A_rdata_b,  A_rdata_r;
    logic [W_BYTES*8-1:0] W_rdata_b,  W_rdata_r;
    logic [C_BYTES*8-1:0] C_rdata_b,  C_rdata_r;
    logic [V_BYTES*8-1:0] V_rdata_b,  V_rdata_r;
    logic [S_BYTES*8-1:0] s_rdata_b,  s_rdata_r;
    logic [DMA_BYTES*8-1:0] dma_rdata_b, dma_rdata_r;

    scratchpad #(
        .MEM_STYLE("BRAM"), .ADDR_W(ADDR_W), .A_BYTES(A_BYTES), .W_BYTES(W_BYTES),
        .C_BYTES(C_BYTES), .V_BYTES(V_BYTES), .S_BYTES(S_BYTES), .DMA_BYTES(DMA_BYTES)
    ) dut_b (
        .clk(clk), .rst_n(rst_n),
        .A_re(A_re), .A_raddr(A_raddr), .A_rdata(A_rdata_b),
        .W_re(W_re), .W_raddr(W_raddr), .W_rdata(W_rdata_b),
        .C_re(C_re), .C_raddr(C_raddr), .C_rdata(C_rdata_b),
        .C_we(C_we), .C_waddr(C_waddr), .C_wdata(C_wdata), .C_wstrb(C_wstrb),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata_b),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        .s_re(s_re), .s_we(s_we), .s_addr(s_addr), .s_wdata(s_wdata), .s_rdata(s_rdata_b),
        .dma_re(dma_re), .dma_raddr(dma_raddr), .dma_rdata(dma_rdata_b),
        .dma_we(dma_we), .dma_waddr(dma_waddr), .dma_wdata(dma_wdata), .dma_wstrb(dma_wstrb)
    );

    scratchpad #(
        .MEM_STYLE("REG"), .ADDR_W(ADDR_W), .A_BYTES(A_BYTES), .W_BYTES(W_BYTES),
        .C_BYTES(C_BYTES), .V_BYTES(V_BYTES), .S_BYTES(S_BYTES), .DMA_BYTES(DMA_BYTES)
    ) dut_r (
        .clk(clk), .rst_n(rst_n),
        .A_re(A_re), .A_raddr(A_raddr), .A_rdata(A_rdata_r),
        .W_re(W_re), .W_raddr(W_raddr), .W_rdata(W_rdata_r),
        .C_re(C_re), .C_raddr(C_raddr), .C_rdata(C_rdata_r),
        .C_we(C_we), .C_waddr(C_waddr), .C_wdata(C_wdata), .C_wstrb(C_wstrb),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata_r),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        .s_re(s_re), .s_we(s_we), .s_addr(s_addr), .s_wdata(s_wdata), .s_rdata(s_rdata_r),
        .dma_re(dma_re), .dma_raddr(dma_raddr), .dma_rdata(dma_rdata_r),
        .dma_we(dma_we), .dma_waddr(dma_waddr), .dma_wdata(dma_wdata), .dma_wstrb(dma_wstrb)
    );

    // -------------------------------------------------------------------------
    // Reference byte model + bookkeeping.
    // -------------------------------------------------------------------------
    logic [7:0] ref_mem [0:MEM_SZ-1];
    int errors = 0, checks = 0;

    // Scratch temporaries for the unaligned sweep (declared here so the stimulus
    // block stays free of mid-block declarations).
    localparam int NBANK_TB = 16;          // = widest port (C_BYTES) rounded to 2^n
    logic [ADDR_W-1:0]    ua;
    logic [V_BYTES*8-1:0] pat;

    task automatic ref_wr(input [ADDR_W-1:0] a, input int nbytes,
                          input [1023:0] data, input [127:0] strb, input logic use_strb);
        for (int k = 0; k < nbytes; k++)
            if (!use_strb || strb[k]) ref_mem[a + k[ADDR_W-1:0]] = data[k*8 +: 8];
    endtask

    function automatic [1023:0] ref_rd(input [ADDR_W-1:0] a, input int nbytes);
        ref_rd = '0;
        for (int k = 0; k < nbytes; k++)
            ref_rd[k*8 +: 8] = ref_mem[a + k[ADDR_W-1:0]];
    endfunction

    // Compare both DUT outputs against the reference gather, byte by byte (a
    // variable-width part-select is illegal, so fold the mismatch over nbytes).
    task automatic check(input [1023:0] got_b, input [1023:0] got_r,
                         input [ADDR_W-1:0] a, input int nbytes, input string tag);
        logic [1023:0] exp;
        logic ok_b, ok_r;
        exp  = ref_rd(a, nbytes);
        ok_b = 1'b1;
        ok_r = 1'b1;
        for (int k = 0; k < nbytes; k++) begin
            if (got_b[k*8 +: 8] !== exp[k*8 +: 8]) ok_b = 1'b0;
            if (got_r[k*8 +: 8] !== exp[k*8 +: 8]) ok_r = 1'b0;
        end
        checks++;
        if (!ok_b) begin
            errors++;
            $display("  FAIL %-16s BRAM @0x%03h: got %h  exp %h", tag, a, got_b, exp);
        end
        if (!ok_r) begin
            errors++;
            $display("  FAIL %-16s REG  @0x%03h: got %h  exp %h", tag, a, got_r, exp);
        end
    endtask

    // -------------------------------------------------------------------------
    // Per-port write helpers (drive shared bus on negedge; DUTs sample posedge).
    // Each updates the reference model to match.
    // -------------------------------------------------------------------------
    task automatic wr_s(input [ADDR_W-1:0] a, input [S_BYTES*8-1:0] d);
        @(negedge clk); s_we = 1'b1; s_addr = a; s_wdata = d;
        @(negedge clk); s_we = 1'b0;
        ref_wr(a, S_BYTES, {992'b0, d}, '0, 1'b0);
    endtask

    task automatic wr_v(input [ADDR_W-1:0] a, input [V_BYTES*8-1:0] d,
                        input [V_BYTES-1:0] strb);
        @(negedge clk); V_we = 1'b1; V_waddr = a; V_wdata = d; V_wstrb = strb;
        @(negedge clk); V_we = 1'b0;
        ref_wr(a, V_BYTES, {{(1024-V_BYTES*8){1'b0}}, d}, {{(128-V_BYTES){1'b0}}, strb}, 1'b1);
    endtask

    task automatic wr_c(input [ADDR_W-1:0] a, input [C_BYTES*8-1:0] d,
                        input [C_BYTES-1:0] strb);
        @(negedge clk); C_we = 1'b1; C_waddr = a; C_wdata = d; C_wstrb = strb;
        @(negedge clk); C_we = 1'b0;
        ref_wr(a, C_BYTES, {{(1024-C_BYTES*8){1'b0}}, d}, {{(128-C_BYTES){1'b0}}, strb}, 1'b1);
    endtask

    task automatic wr_dma(input [ADDR_W-1:0] a, input [DMA_BYTES*8-1:0] d,
                          input [DMA_BYTES-1:0] strb);
        @(negedge clk); dma_we = 1'b1; dma_waddr = a; dma_wdata = d; dma_wstrb = strb;
        @(negedge clk); dma_we = 1'b0;
        ref_wr(a, DMA_BYTES, {{(1024-DMA_BYTES*8){1'b0}}, d}, {{(128-DMA_BYTES){1'b0}}, strb}, 1'b1);
    endtask

    // -------------------------------------------------------------------------
    // Per-port read helpers: assert enable one cycle, sample the cycle after,
    // check both variants against the reference.
    // -------------------------------------------------------------------------
    task automatic rd_s(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); s_re = 1'b1; s_addr = a;
        @(negedge clk); s_re = 1'b0;                 // one posedge elapsed: rdata valid
        check({960'b0, s_rdata_b}, {960'b0, s_rdata_r}, a, S_BYTES, tag);
    endtask

    task automatic rd_v(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); V_re = 1'b1; V_raddr = a;
        @(negedge clk); V_re = 1'b0;
        check({{(1024-V_BYTES*8){1'b0}}, V_rdata_b}, {{(1024-V_BYTES*8){1'b0}}, V_rdata_r},
              a, V_BYTES, tag);
    endtask

    task automatic rd_c(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); C_re = 1'b1; C_raddr = a;
        @(negedge clk); C_re = 1'b0;
        check({{(1024-C_BYTES*8){1'b0}}, C_rdata_b}, {{(1024-C_BYTES*8){1'b0}}, C_rdata_r},
              a, C_BYTES, tag);
    endtask

    task automatic rd_a(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); A_re = 1'b1; A_raddr = a;
        @(negedge clk); A_re = 1'b0;
        check({{(1024-A_BYTES*8){1'b0}}, A_rdata_b}, {{(1024-A_BYTES*8){1'b0}}, A_rdata_r},
              a, A_BYTES, tag);
    endtask

    task automatic rd_w(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); W_re = 1'b1; W_raddr = a;
        @(negedge clk); W_re = 1'b0;
        check({{(1024-W_BYTES*8){1'b0}}, W_rdata_b}, {{(1024-W_BYTES*8){1'b0}}, W_rdata_r},
              a, W_BYTES, tag);
    endtask

    task automatic rd_dma(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); dma_re = 1'b1; dma_raddr = a;
        @(negedge clk); dma_re = 1'b0;
        check({{(1024-DMA_BYTES*8){1'b0}}, dma_rdata_b},
              {{(1024-DMA_BYTES*8){1'b0}}, dma_rdata_r}, a, DMA_BYTES, tag);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        // Idle the whole bus.
        A_re=0; A_raddr=0; W_re=0; W_raddr=0;
        C_re=0; C_raddr=0; C_we=0; C_waddr=0; C_wdata=0; C_wstrb=0;
        V_re=0; V_raddr=0; V_we=0; V_waddr=0; V_wdata=0; V_wstrb=0;
        s_re=0; s_we=0; s_addr=0; s_wdata=0;
        dma_re=0; dma_raddr=0; dma_we=0; dma_waddr=0; dma_wdata=0; dma_wstrb=0;
        for (int i = 0; i < MEM_SZ; i++) ref_mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== Scratchpad testbench (BRAM vs REG) ====");

        // --- S port int32 roundtrip -----------------------------------------
        wr_s(12'h010, 32'hDEAD_BEEF);
        wr_s(12'h020, 32'h0123_4567);
        rd_s(12'h010, "S-roundtrip");
        rd_s(12'h020, "S-roundtrip");

        // --- V full write then strobed partial write ------------------------
        wr_v(12'h040, 64'hFEDC_BA98_7654_3210, 8'hFF);       // all bytes
        rd_v(12'h040, "V-full");
        wr_v(12'h040, 64'h1111_2222_3333_4444, 8'b0101_0101); // mask: even bytes only
        rd_v(12'h040, "V-strobe");                             // odd bytes must survive

        // --- C int32 result row + readback ----------------------------------
        wr_c(12'h100, {4{32'h89AB_CDEF}}, {C_BYTES{1'b1}});
        rd_c(12'h100, "C-row");

        // --- DMA strobed write, read back through V (cross width) ------------
        wr_dma(12'h080, 64'hCAFE_F00D_0BAD_BEEF, 8'hFF);
        rd_v(12'h080, "DMA->V");

        // --- Cross-port coherence: S write, A read (byte views overlap) ------
        wr_s(12'h044, 32'h7788_99AA);   // lands inside the V-region above
        rd_a(12'h040, "S->A-coherent");

        // --- Unaligned windows at every byte offset within a bank row --------
        // The flat model made this trivially correct; the banked model has to
        // skew the row select per bank and barrel-rotate both directions, so
        // sweep every offset rather than spot-checking one.
        for (int off = 0; off < NBANK_TB; off++) begin
            ua = 12'h400 + off;
            for (int k = 0; k < V_BYTES; k++) pat[k*8 +: 8] = off*16 + k;
            wr_v(ua, pat, {V_BYTES{1'b1}});
            rd_v(ua, "UA-V-full");
        end

        // Same sweep with a byte strobe, so the strobe rotate is exercised too
        // (masked bytes must retain what the sweep above left there).
        for (int off = 0; off < NBANK_TB; off++) begin
            ua = 12'h400 + off;
            for (int k = 0; k < V_BYTES; k++) pat[k*8 +: 8] = 8'hF0 + k;
            wr_v(ua, pat, 8'b1010_1010);
            rd_v(ua, "UA-V-strb");
        end

        // A full-NBANK-width unaligned access: spans every bank and forces the
        // +1 row select on the wrapped ones. Read back through a narrower port
        // to confirm the byte view agrees.
        wr_c(12'h505, {4{32'hDEAD_C0DE}}, {C_BYTES{1'b1}});
        rd_c(12'h505, "UA-C-full");
        rd_a(12'h505, "UA-C->A");
        rd_w(12'h50B, "UA-C->W");

        // --- Address wrap: a window running off the top wraps mod DEPTH ------
        wr_v(12'hFFC, 64'h0102_0304_0506_0708, {V_BYTES{1'b1}});
        rd_v(12'hFFC, "wrap-V");
        rd_dma(12'hFFC, "wrap-DMA");
        rd_s(12'h000, "wrap-tail");     // the bytes that wrapped to the bottom

        // --- Port sequencing: A, W, C on consecutive cycles -------------------
        // How the MXU actually drives them (S_LOAD -> S_RUN -> S_WB_RD are
        // distinct states). Each port's data must be valid the cycle after its
        // own enable, with the next port's request already in flight.
        wr_s(12'h200, 32'hA1A2_A3A4);
        wr_s(12'h300, 32'hB1B2_B3B4);

        @(negedge clk); A_re = 1'b1; A_raddr = 12'h040;
        @(negedge clk); A_re = 1'b0; W_re = 1'b1; W_raddr = 12'h200;
        check({{(1024-A_BYTES*8){1'b0}}, A_rdata_b}, {{(1024-A_BYTES*8){1'b0}}, A_rdata_r},
              12'h040, A_BYTES, "SEQ-A");
        @(negedge clk); W_re = 1'b0; C_re = 1'b1; C_raddr = 12'h100;
        check({{(1024-W_BYTES*8){1'b0}}, W_rdata_b}, {{(1024-W_BYTES*8){1'b0}}, W_rdata_r},
              12'h200, W_BYTES, "SEQ-W");
        @(negedge clk); C_re = 1'b0;
        check({{(1024-C_BYTES*8){1'b0}}, C_rdata_b}, {{(1024-C_BYTES*8){1'b0}}, C_rdata_r},
              12'h100, C_BYTES, "SEQ-C");

        // --- Read-first: read and write the same S address in one cycle ------
        wr_s(12'h050, 32'h1234_5678);
        @(negedge clk);
        s_re = 1'b1; s_addr = 12'h050;                 // read old
        s_we = 1'b1; s_wdata = 32'h9999_9999;          // write new, same cycle
        @(negedge clk);
        s_re = 1'b0; s_we = 1'b0;
        checks++;
        if (s_rdata_b !== 32'h1234_5678 || s_rdata_r !== 32'h1234_5678) begin
            errors++;
            $display("  FAIL read-first @0x050: BRAM %h REG %h  exp 12345678",
                     s_rdata_b, s_rdata_r);
        end
        ref_wr(12'h050, S_BYTES, {992'b0, 32'h9999_9999}, '0, 1'b0);
        rd_s(12'h050, "read-first-post");   // new value now visible

        // Summary.
        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("SCRATCHPAD: ALL TESTS PASSED");
        else             $display("SCRATCHPAD: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #200000;
        $display("SCRATCHPAD: TIMEOUT — testbench did not complete");
        $fatal(1);
    end

endmodule
