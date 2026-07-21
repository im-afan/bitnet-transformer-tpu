// -----------------------------------------------------------------------------
// dma_tb.sv — self-checking testbench for accel/tpu/rtl/dma.sv
//
// Wires the real DMA engine to the real sram_controller and the real scratchpad,
// plus the behavioral async-SRAM chip model (same contract as sram_tb.sv), and
// exercises the scalar-unit dispatch:
//
//   FILL  (ReadMemory)  : preload DRAM, DMA -> scratchpad, verify scratchpad
//   SPILL (WriteMemory) : preload scratchpad, DMA -> DRAM, verify DRAM
//   ROUND-TRIP          : spill a region, wipe scratch, fill it back, compare
//
// Reference model is a plain byte memcpy. Also checks corners (len 0/1, unaligned
// bases, len not a multiple of the scratchpad width, a high DRAM address) and the
// handshake (busy frames the transfer, done is a single-cycle pulse).
//
// Multi-file build:
//   make TEST=dma RTL="../rtl/dma.sv ../rtl/sram.sv ../rtl/scratchpad.sv" sim
// -----------------------------------------------------------------------------

`timescale 1ns/1ps

module dma_tb;

    // ---- Parameters ---------------------------------------------------------
    localparam int MEM_ADDR_W        = 19;
    localparam int SCRATCHPAD_ADDR_W = 16;
    localparam int MEM_DATA_W        = 8;
    localparam int SPAD_BYTES        = 64;   // scratchpad DMA port width (bytes)

    // ---- Clock / reset ------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // ---- Scalar-unit dispatch -----------------------------------------------
    logic                          dma_start, dma_write;
    logic [SCRATCHPAD_ADDR_W-1:0]  dma_scratch_addr;
    logic [MEM_ADDR_W-1:0]         dma_dram_addr;
    logic [15:0]                   dma_len;
    logic                          dma_busy, dma_done;

    // ---- DMA <-> sram_controller --------------------------------------------
    logic                    sram_start, sram_we;
    logic [MEM_ADDR_W-1:0]   sram_addr;
    logic [MEM_DATA_W-1:0]   sram_din, sram_dout;
    logic                    sram_busy, sram_done;

    // ---- DMA <-> scratchpad DMA port ----------------------------------------
    logic                        spad_re;
    logic [SCRATCHPAD_ADDR_W-1:0] spad_raddr;
    logic [SPAD_BYTES*8-1:0]     spad_rdata;
    logic                        spad_we;
    logic [SCRATCHPAD_ADDR_W-1:0] spad_waddr;
    logic [SPAD_BYTES*8-1:0]     spad_wdata;
    logic [SPAD_BYTES-1:0]       spad_wstrb;

    // ---- sram_controller <-> SRAM chip pins ---------------------------------
    wire  [MEM_ADDR_W-1:0]   chip_addr;
    wire  [MEM_DATA_W-1:0]   chip_data;
    wire                     chip_we, chip_ce, chip_oen;

    // =========================================================================
    // DUT: DMA engine.
    // =========================================================================
    dma #(
        .MEM_ADDR_W(MEM_ADDR_W), .SCRATCHPAD_ADDR_W(SCRATCHPAD_ADDR_W),
        .MEM_DATA_W(MEM_DATA_W), .SCRATCHPAD_BYTES(SPAD_BYTES)
    ) u_dma (
        .clk(clk), .rst_n(rst_n),
        .sram_start(sram_start), .sram_we(sram_we), .sram_addr(sram_addr),
        .sram_din(sram_din), .sram_dout(sram_dout),
        .sram_busy(sram_busy), .sram_done(sram_done),
        .scratchpad_re(spad_re), .scratchpad_raddr(spad_raddr), .scratchpad_rdata(spad_rdata),
        .scratchpad_we(spad_we), .scratchpad_waddr(spad_waddr),
        .scratchpad_wdata(spad_wdata), .scratchpad_wstrb(spad_wstrb),
        .dma_start(dma_start), .dma_write(dma_write),
        .dma_scratch_addr(dma_scratch_addr), .dma_dram_addr(dma_dram_addr),
        .dma_len(dma_len), .dma_busy(dma_busy), .dma_done(dma_done)
    );

    // =========================================================================
    // Real SRAM controller.
    // =========================================================================
    sram_controller #(
        .CLOCKS_PER_ACCESS(2), .ADDR_W(MEM_ADDR_W), .DATA_W(MEM_DATA_W)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),
        .start(sram_start), .we(sram_we), .addr(sram_addr),
        .din(sram_din), .dout(sram_dout), .busy(sram_busy), .done(sram_done),
        .sram_addr(chip_addr), .sram_data(chip_data),
        .sram_we(chip_we), .sram_ce(chip_ce), .sram_oen(chip_oen)
    );

    // =========================================================================
    // Real scratchpad. Only the DMA port is driven here; the compute ports are
    // tied off. The scratchpad byte-address width matches SCRATCHPAD_ADDR_W.
    // =========================================================================
    scratchpad #(
        .MEM_STYLE("REG"), .ADDR_W(SCRATCHPAD_ADDR_W), .DMA_BYTES(SPAD_BYTES)
    ) u_spad (
        .clk(clk), .rst_n(rst_n),
        .A_re(1'b0), .A_raddr('0), .A_rdata(),
        .W_re(1'b0), .W_raddr('0), .W_rdata(),
        .C_re(1'b0), .C_raddr('0), .C_rdata(),
        .C_we(1'b0), .C_waddr('0), .C_wdata('0), .C_wstrb('0),
        .V_re(1'b0), .V_raddr('0), .V_rdata(),
        .V_we(1'b0), .V_waddr('0), .V_wdata('0), .V_wstrb('0),
        .s_re(1'b0), .s_we(1'b0), .s_addr('0), .s_wdata('0), .s_rdata(),
        .dma_re(spad_re), .dma_raddr(spad_raddr), .dma_rdata(spad_rdata),
        .dma_we(spad_we), .dma_waddr(spad_waddr),
        .dma_wdata(spad_wdata), .dma_wstrb(spad_wstrb)
    );

    // =========================================================================
    // Behavioral async SRAM chip model (matches sram_tb.sv).
    // =========================================================================
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (chip_ce == 1'b0) && (chip_oen == 1'b0) && (chip_we == 1'b1);
    assign #3 chip_data = mem_drives ? sram_mem[chip_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge chip_we) if (chip_ce == 1'b0) sram_mem[chip_addr] <= chip_data;

    // -------------------------------------------------------------------------
    // Reference memory (byte model of what each store should contain).
    // -------------------------------------------------------------------------
    int errors = 0;
    int checks = 0;

    task automatic chk(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL %s (t=%0t)", tag, $time);
        end
    endtask

    // Read one scratchpad byte at absolute address `a` through the DMA port.
    task automatic spad_read_byte(input [SCRATCHPAD_ADDR_W-1:0] a,
                                  output [7:0] b);
        @(negedge clk);
        force u_spad.dma_re    = 1'b1;
        force u_spad.dma_raddr = a;
        @(negedge clk);
        force u_spad.dma_re    = 1'b0;
        @(negedge clk);                 // rdata valid now
        b = u_spad.dma_rdata[7:0];
        release u_spad.dma_re;
        release u_spad.dma_raddr;
    endtask

    // Write one scratchpad byte at absolute address `a` through the DMA port.
    task automatic spad_write_byte(input [SCRATCHPAD_ADDR_W-1:0] a, input [7:0] b);
        @(negedge clk);
        force u_spad.dma_we    = 1'b1;
        force u_spad.dma_waddr = a;
        force u_spad.dma_wdata = { {(SPAD_BYTES-1){8'b0}}, b };
        force u_spad.dma_wstrb = SPAD_BYTES'(1);
        @(negedge clk);
        release u_spad.dma_we;
        release u_spad.dma_waddr;
        release u_spad.dma_wdata;
        release u_spad.dma_wstrb;
    endtask

    // -------------------------------------------------------------------------
    // Dispatch one transfer and wait for done, asserting the handshake.
    // -------------------------------------------------------------------------
    task automatic run_dma(input logic wr,
                           input [SCRATCHPAD_ADDR_W-1:0] saddr,
                           input [MEM_ADDR_W-1:0] daddr,
                           input [15:0] len);
        int guard;
        @(negedge clk);
        dma_write = wr; dma_scratch_addr = saddr; dma_dram_addr = daddr;
        dma_len = len; dma_start = 1'b1;
        @(negedge clk);
        dma_start = 1'b0;                 // one-cycle pulse
        guard = 0;
        while (!dma_done) begin
            @(negedge clk);
            guard++;
            if (guard > 20000) begin
                $display("  FAIL run_dma: no done (wr=%0d len=%0d)", wr, len);
                errors++; disable run_dma;
            end
        end
        @(negedge clk);
        chk(!dma_done, "done is single-cycle pulse");
        chk(!dma_busy, "busy low after done");
    endtask

    // -------------------------------------------------------------------------
    // High-level checks.
    // -------------------------------------------------------------------------
    task automatic test_fill(input [SCRATCHPAD_ADDR_W-1:0] saddr,
                             input [MEM_ADDR_W-1:0] daddr,
                             input [15:0] len, input string tag);
        logic [7:0] b;
        for (int i = 0; i < len; i++)
            sram_mem[daddr + i] = 8'((daddr + i) ^ 8'h3C);   // known pattern
        run_dma(1'b0, saddr, daddr, len);
        for (int i = 0; i < len; i++) begin
            spad_read_byte(saddr + i[SCRATCHPAD_ADDR_W-1:0], b);
            chk(b === 8'((daddr + i) ^ 8'h3C), $sformatf("%s[%0d]", tag, i));
            if (b !== 8'((daddr + i) ^ 8'h3C))
                $display("    fill %s[%0d]: exp %02h got %02h", tag, i,
                         8'((daddr+i)^8'h3C), b);
        end
        $display("[FILL  %-10s] saddr=%05h daddr=%05h len=%0d  (errors: %0d)",
                 tag, saddr, daddr, len, errors);
    endtask

    task automatic test_spill(input [SCRATCHPAD_ADDR_W-1:0] saddr,
                              input [MEM_ADDR_W-1:0] daddr,
                              input [15:0] len, input string tag);
        for (int i = 0; i < len; i++)
            spad_write_byte(saddr + i[SCRATCHPAD_ADDR_W-1:0], 8'((i * 5 + 1)));
        // Poison the DRAM target so a missed write is caught.
        for (int i = 0; i < len; i++) sram_mem[daddr + i] = 8'hEE;
        run_dma(1'b1, saddr, daddr, len);
        for (int i = 0; i < len; i++) begin
            chk(sram_mem[daddr + i] === 8'((i * 5 + 1)), $sformatf("%s[%0d]", tag, i));
            if (sram_mem[daddr + i] !== 8'((i * 5 + 1)))
                $display("    spill %s[%0d]: exp %02h got %02h", tag, i,
                         8'((i*5+1)), sram_mem[daddr + i]);
        end
        $display("[SPILL %-10s] saddr=%05h daddr=%05h len=%0d  (errors: %0d)",
                 tag, saddr, daddr, len, errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    logic [7:0] b;
    initial begin
        $dumpfile("dma_tb.vcd");
        $dumpvars(0, dma_tb);

        dma_start = 0; dma_write = 0; dma_scratch_addr = 0; dma_dram_addr = 0;
        dma_len = 0;
        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== DMA testbench ====");

        // Basic fill / spill.
        test_fill (16'h0100, 19'h00200, 16'd8,  "fill8");
        test_spill(16'h0300, 19'h00400, 16'd8,  "spill8");

        // Spanning more than one scratchpad width (64B) + a non-multiple length.
        test_fill (16'h1000, 19'h05000, 16'd70, "fill70");
        test_spill(16'h2000, 19'h06000, 16'd70, "spill70");

        // Corners: len 0 and len 1.
        run_dma(1'b0, 16'h0000, 19'h00000, 16'd0);   // empty fill: must just done
        chk(1'b1, "len0 completes");
        test_fill (16'h0500, 19'h00700, 16'd1,  "fill1");
        test_spill(16'h0520, 19'h00720, 16'd1,  "spill1");

        // Unaligned bases.
        test_fill (16'h0803, 19'h00901, 16'd37, "fillUA");
        test_spill(16'h0B07, 19'h00C05, 16'd37, "spillUA");

        // High DRAM address (exercises bits above the low 64 KB window).
        test_fill (16'h3000, 19'h7FF00, 16'd16, "fillHi");
        test_spill(16'h3100, 19'h7FE00, 16'd16, "spillHi");

        // Round-trip: spill a region to DRAM, wipe scratch, fill it back.
        for (int i = 0; i < 40; i++)
            spad_write_byte(16'h4000 + i[SCRATCHPAD_ADDR_W-1:0], 8'((i*13+9)));
        run_dma(1'b1, 16'h4000, 19'h10000, 16'd40);        // spill to DRAM
        for (int i = 0; i < 40; i++)
            spad_write_byte(16'h4000 + i[SCRATCHPAD_ADDR_W-1:0], 8'h00);  // wipe
        run_dma(1'b0, 16'h4000, 19'h10000, 16'd40);        // fill back
        for (int i = 0; i < 40; i++) begin
            spad_read_byte(16'h4000 + i[SCRATCHPAD_ADDR_W-1:0], b);
            chk(b === 8'((i*13+9)), $sformatf("roundtrip[%0d]", i));
        end
        $display("[ROUNDTRIP] len=40  (errors: %0d)", errors);

        // Summary.
        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("DMA: ALL TESTS PASSED");
        else             $display("DMA: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #5000000;
        $display("DMA: TIMEOUT");
        $fatal(1);
    end

endmodule
