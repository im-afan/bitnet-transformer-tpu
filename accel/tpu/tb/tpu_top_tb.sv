`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tpu_top_tb.sv — end-to-end testbench for accel/tpu/rtl/tpu_top.sv
//
// Exercises the fully integrated core (scalar_unit + mxu + vpu + scratchpad,
// BRAM working memory) the way a host actually drives it: load an *assembled*
// program and sample input tensors from $readmemh memory files, stream them into
// the DUT over the top-level host ports, pulse host_run, then read the results
// back out of BRAM and compare — no direct poking of any unit's internal ports.
//
// The golden reference is produced entirely outside the RTL: accel/tpulang
// assembles a .tpu program (assembler.py) and simulates the instruction set in
// Python (iss.py) against the same input tensors, and gen_vectors.py exports
//
//     tpu_prog.hex      the assembled program        (imem words, dense)
//     tpu_spad_in.hex   the input tensors            (scratchpad bytes, sparse)
//     tpu_spad_exp.hex  the ISS-computed outputs     (scratchpad bytes, sparse)
//
// This TB is program-agnostic: it streams whatever the program file holds, loads
// whatever bytes the input file holds, and checks exactly the bytes the ISS says
// the program wrote (tpu_spad_exp.hex carries only those). Regenerate the vectors
// after changing the program or tensors:
//
//     cd ../../tpulang && python gen_vectors.py
//
// The default program is examples/relu_layer.tpu — one ternary NN layer plus a
// ReLU (Y = requant(A@W); Z = relu(Y)), the shape the adder model runs.
//
// Small 8x8 array so the systolic drain simulates quickly (RTL default 128x128).
// The geometry below MUST match gen_vectors.py (ROWS/COLS/T).
//
// Run:  make TEST=tpu_top RTL="../rtl/tpu_top.sv ../rtl/scalar_unit.sv \
//            ../rtl/mxu.sv ../rtl/vpu.sv ../rtl/scratchpad.sv"
// Override a vector file:  iverilog ... -DPROG_FILE='"vectors/other_prog.hex"'
// -----------------------------------------------------------------------------

// ---- Vector files (paths relative to the tb/ dir where vvp runs) -------------
`ifndef PROG_FILE
  `define PROG_FILE "vectors/tpu_prog.hex"
`endif
`ifndef SPAD_IN_FILE
  `define SPAD_IN_FILE "vectors/tpu_spad_in.hex"
`endif
`ifndef SPAD_EXP_FILE
  `define SPAD_EXP_FILE "vectors/tpu_spad_exp.hex"
`endif

module tpu_top_tb;

    // ---- Geometry (must match the DUT instance and gen_vectors.py) ----------
    localparam int ROWS      = 8;      // MXU contraction dim d
    localparam int COLS      = 8;      // MXU output features
    localparam int VPU_BYTES = 64;     // VPU port width (512-bit)
    localparam int ADDR_W    = 16;
    localparam int XLEN      = 32;
    localparam int IMEM_AW   = 10;
    localparam int CFG_AW    = 4;
    localparam int REG_AW    = 5;
    localparam int M0_W      = 12;
    localparam int N_W       = 4;
    localparam int DMA_BYTES = 64;     // host memory port width

    localparam int DEPTH     = (1 << ADDR_W);      // scratchpad size in bytes
    localparam int IMEM_DEP  = (1 << IMEM_AW);     // instruction memory depth

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT host interface.
    // -------------------------------------------------------------------------
    logic                  host_run;
    logic [IMEM_AW-1:0]    boot_pc;
    logic                  busy, done;
    logic [IMEM_AW-1:0]    pc_dbg;

    logic                  imem_we;
    logic [IMEM_AW-1:0]    imem_waddr;
    logic [31:0]           imem_wdata;

    logic                  cfg_we;
    logic [CFG_AW-1:0]     cfg_waddr;
    logic [XLEN-1:0]       cfg_wdata;

    logic                  host_mem_re;
    logic [ADDR_W-1:0]     host_mem_raddr;
    logic [DMA_BYTES*8-1:0] host_mem_rdata;
    logic                  host_mem_we;
    logic [ADDR_W-1:0]     host_mem_waddr;
    logic [DMA_BYTES*8-1:0] host_mem_wdata;
    logic [DMA_BYTES-1:0]   host_mem_wstrb;

    tpu_top #(
        .ROWS(ROWS), .COLS(COLS), .VPU_BYTES(VPU_BYTES), .ADDR_W(ADDR_W),
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W),
        .REG_AW(REG_AW), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .MEM_STYLE("BRAM")
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .host_run(host_run), .boot_pc(boot_pc),
        .busy(busy), .done(done), .pc_dbg(pc_dbg),
        .imem_we(imem_we), .imem_waddr(imem_waddr), .imem_wdata(imem_wdata),
        .cfg_we(cfg_we), .cfg_waddr(cfg_waddr), .cfg_wdata(cfg_wdata),
        .host_mem_re(host_mem_re), .host_mem_raddr(host_mem_raddr),
        .host_mem_rdata(host_mem_rdata),
        .host_mem_we(host_mem_we), .host_mem_waddr(host_mem_waddr),
        .host_mem_wdata(host_mem_wdata), .host_mem_wstrb(host_mem_wstrb)
    );

    // -------------------------------------------------------------------------
    // Host memory helpers (drive the top-level host_mem_* port).
    // -------------------------------------------------------------------------
    task automatic spad_wr_byte(input logic [ADDR_W-1:0] addr, input logic [7:0] d);
        @(negedge clk);
        host_mem_we    = 1'b1;
        host_mem_waddr = addr;
        host_mem_wdata = '0;  host_mem_wdata[7:0] = d;
        host_mem_wstrb = '0;  host_mem_wstrb[0]   = 1'b1;   // one byte at addr
        @(negedge clk);
        host_mem_we    = 1'b0;
        host_mem_wstrb = '0;
    endtask

    // Read a 32-bit window starting at addr (byte k in host_mem_rdata[k*8 +: 8]).
    task automatic spad_rd(input logic [ADDR_W-1:0] addr, output logic [31:0] d);
        @(negedge clk);
        host_mem_re    = 1'b1;
        host_mem_raddr = addr;
        @(posedge clk);         // scratchpad latches mem[addr..] into rdata here
        @(negedge clk);         // rdata now stable
        host_mem_re    = 1'b0;
        d = host_mem_rdata[31:0];
    endtask

    task automatic load_instr(input logic [IMEM_AW-1:0] a, input logic [31:0] w);
        @(negedge clk);
        imem_we    = 1'b1;
        imem_waddr = a;
        imem_wdata = w;
        @(negedge clk);
        imem_we    = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Vector-file backing stores (loaded with $readmemh).
    //   prog_words : dense from addr 0; trailing entries stay x (=> program end).
    //   in_bytes   : sparse (@addr runs); a defined byte is a tensor byte to load.
    //   exp_bytes  : sparse (@addr runs); a defined byte is an output to check.
    // -------------------------------------------------------------------------
    logic [31:0] prog_words [0:IMEM_DEP-1];
    logic [7:0]  in_bytes   [0:DEPTH-1];
    logic [7:0]  exp_bytes  [0:DEPTH-1];

    int errors = 0, checks = 0;

    // Fully-defined (no x bits) test for a loaded byte / word.
    function automatic logic defined8(input logic [7:0] b);
        return (^b !== 1'bx);
    endfunction
    function automatic logic defined32(input logic [31:0] w);
        return (^w !== 1'bx);
    endfunction

    // Stream the assembled program into instruction BRAM (stop at first hole).
    // ($readmemh warns that the file has fewer words than prog_words is deep —
    //  that is expected: the unfilled trailing entries stay x and mark the end.)
    task automatic load_program();
        int i;
        $readmemh(`PROG_FILE, prog_words);
        i = 0;
        while (i < IMEM_DEP && defined32(prog_words[i])) begin
            load_instr(i[IMEM_AW-1:0], prog_words[i]);
            i++;
        end
        $display("loaded %0d instruction words from %s", i, `PROG_FILE);
    endtask

    // Stream the sample tensors into scratchpad over the host memory port.
    task automatic load_tensors();
        int a, cnt;
        $readmemh(`SPAD_IN_FILE, in_bytes);
        cnt = 0;
        for (a = 0; a < DEPTH; a++)
            if (defined8(in_bytes[a])) begin
                spad_wr_byte(a[ADDR_W-1:0], in_bytes[a]);
                cnt++;
            end
        $display("loaded %0d scratchpad input bytes from %s", cnt, `SPAD_IN_FILE);
    endtask

    // Read every expected byte back out of BRAM and compare (word at a time).
    task automatic check_outputs();
        int wa, lane;
        logic [31:0] rd;
        logic [7:0]  eb, gb;
        $readmemh(`SPAD_EXP_FILE, exp_bytes);
        for (wa = 0; wa < DEPTH; wa += 4) begin
            // Only read BRAM when this word carries an expected output byte.
            if (defined8(exp_bytes[wa])  || defined8(exp_bytes[wa+1]) ||
                defined8(exp_bytes[wa+2])|| defined8(exp_bytes[wa+3])) begin
                spad_rd(wa[ADDR_W-1:0], rd);
                for (lane = 0; lane < 4; lane++) begin
                    eb = exp_bytes[wa+lane];
                    if (defined8(eb)) begin
                        gb = rd[lane*8 +: 8];
                        checks++;
                        if (gb !== eb) begin
                            errors++;
                            $display("  FAIL @%04h: got %02h  exp %02h",
                                     wa+lane, gb, eb);
                        end
                    end
                end
            end
        end
        $display("[compare] checked %0d output bytes (errors so far: %0d)",
                 checks, errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        host_run = 1'b0; boot_pc = '0;
        imem_we  = 1'b0; imem_waddr = '0; imem_wdata = '0;
        cfg_we   = 1'b0; cfg_waddr  = '0; cfg_wdata  = '0;
        host_mem_re = 1'b0; host_mem_raddr = '0;
        host_mem_we = 1'b0; host_mem_waddr = '0;
        host_mem_wdata = '0; host_mem_wstrb = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== TPU top-level testbench (%0dx%0d array) ====", ROWS, COLS);

        // 1) Load the assembled program + input tensors, 2) run, 3) check.
        load_program();
        load_tensors();

        @(negedge clk); host_run = 1'b1; boot_pc = '0;
        @(negedge clk); host_run = 1'b0;

        // Wait for HALT (watchdog below guards against a hang).
        do @(negedge clk); while (!done);
        $display("program halted at pc=%0d", pc_dbg);

        check_outputs();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("TPU_TOP: ALL TESTS PASSED");
        else             $display("TPU_TOP: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #2000000;
        $display("TPU_TOP: TIMEOUT — program did not halt");
        $fatal(1);
    end

    // Waveform dump.
    initial begin
        $dumpfile("tpu_top_tb.vcd");
        $dumpvars(0, tpu_top_tb);
    end

endmodule
