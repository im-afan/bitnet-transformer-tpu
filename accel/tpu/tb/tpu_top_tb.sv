`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tpu_top_tb.sv — end-to-end testbench for accel/tpu/rtl/tpu_top.sv
//
// Exercises the fully integrated core (scalar_unit + mxu + vpu + scratchpad,
// BRAM working memory) the way a host actually drives it: preload the scratchpad
// through the top-level host memory port, load a program into instruction BRAM,
// pulse host_run, then read the results back out of BRAM — no direct poking of
// any unit's internal ports.
//
// The program simulates one ternary NN layer followed by an activation, the
// shape the adder model runs (TernaryLinear projection, then a pointwise
// nonlinearity):
//
//     Y  = A @ W            MXU matmul: int8 activations × ternary weights,
//                           int32 accumulate, requantized int32→int8 on store
//     Z  = relu(Y)          VPU activation over the int8 layer output (int32 out)
//
// with A : T×d int8, W : d×COLS ternary {−1,0,+1}, Y : T×COLS int8. The requant
// scale is {m0=1, n=0} (identity + int8 clip) so the golden model stays a plain
// integer matmul; inputs are bounded so the accumulator lands inside int8.
//
// Both stages are checked against an in-TB integer reference: the requantized
// layer output Y (validates MXU + requant + scratchpad) and the activation Z
// (validates the VPU op reading the MXU's result back out of scratchpad, i.e.
// the scalar unit's issue-and-wait dependency chain across two units).
//
// Small 8×8 array so the systolic drain simulates quickly (RTL default 128×128).
//
// Run:  make TEST=tpu_top RTL="../rtl/tpu_top.sv ../rtl/scalar_unit.sv \
//            ../rtl/mxu.sv ../rtl/vpu.sv ../rtl/scratchpad.sv"
// -----------------------------------------------------------------------------

module tpu_top_tb;

    // ---- Geometry (must match the DUT instance) -----------------------------
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

    localparam int T          = 4;                 // token rows
    localparam int WCOL_BYTES = (ROWS * 2) / 8;    // packed weight column bytes
    localparam int NELEM      = T * COLS;          // flat layer-output length

    // ---- Requant scale: {m0=1, n=0} => Y = clip(acc) to int8 ----------------
    localparam int RQ_M0 = 1;
    localparam int RQ_N  = 0;

    // ---- Scratchpad address map ---------------------------------------------
    localparam logic [ADDR_W-1:0] ACT = 16'h0000;  // A[t][i] int8, row-major
    localparam logic [ADDR_W-1:0] WGT = 16'h1000;  // W[i][j] ternary, col-major 2-bit
    localparam logic [ADDR_W-1:0] OUT = 16'h3000;  // Y[t][j] int8 (requant store)
    localparam logic [ADDR_W-1:0] RLU = 16'h5000;  // Z[t][j] int32 (relu output)
    localparam logic [ADDR_W-1:0] RQW = 16'h7000;  // requant {m0,n} word (int32)

    // ---- Scalar ISA opcodes (subset used here; must match scalar_unit.sv) ----
    localparam logic [5:0]
        OP_MATMUL = 6'h00, OP_RELU = 6'h04, OP_SETCFG = 6'h15,
        OP_LIS    = 6'h14, OP_HALT = 6'h1F;
    localparam logic [CFG_AW-1:0]
        CFG_TLEN = 4'd0, CFG_VLEN = 4'd1, CFG_SCALAR = 4'd3;

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

    task automatic spad_wr_word(input logic [ADDR_W-1:0] addr, input logic [31:0] d);
        @(negedge clk);
        host_mem_we    = 1'b1;
        host_mem_waddr = addr;
        host_mem_wdata = '0;  host_mem_wdata[31:0] = d;
        host_mem_wstrb = '0;  host_mem_wstrb[3:0]  = 4'hF;  // int32 word at addr
        @(negedge clk);
        host_mem_we    = 1'b0;
        host_mem_wstrb = '0;
    endtask

    // Read a 32-bit window starting at addr (byte k in dma_rdata[k*8 +: 8]).
    task automatic spad_rd(input logic [ADDR_W-1:0] addr, output logic [31:0] d);
        @(negedge clk);
        host_mem_re    = 1'b1;
        host_mem_raddr = addr;
        @(posedge clk);         // scratchpad latches mem[addr..] into rdata here
        @(negedge clk);         // rdata now stable
        host_mem_re    = 1'b0;
        d = host_mem_rdata[31:0];
    endtask

    // -------------------------------------------------------------------------
    // Instruction / config load helpers (scalar unit accepts these while idle).
    // -------------------------------------------------------------------------
    task automatic load_instr(input logic [IMEM_AW-1:0] a, input logic [31:0] w);
        @(negedge clk);
        imem_we    = 1'b1;
        imem_waddr = a;
        imem_wdata = w;
        @(negedge clk);
        imem_we    = 1'b0;
    endtask

    function automatic logic [31:0] enc_rrr(input logic [5:0] op,
                                            input logic [7:0] dst,
                                            input logic [7:0] src0,
                                            input logic [7:0] src1,
                                            input logic [1:0] fl);
        return {op, dst, src0, src1, fl};
    endfunction

    function automatic logic [31:0] enc_imm(input logic [5:0]  op,
                                            input logic [7:0]  dst,
                                            input logic [15:0] imm);
        return {op, dst, imm, 2'b00};   // imm16 occupies ir[17:2]
    endfunction

    // -------------------------------------------------------------------------
    // Reference model.
    // -------------------------------------------------------------------------
    int actv [0:T-1][0:ROWS-1];      // int8 activations
    int wgtv [0:ROWS-1][0:COLS-1];   // ternary weights {-1,0,1}
    int errors = 0, checks = 0;

    function automatic int clip8(input int v);
        if (v >  127) return  127;
        if (v < -128) return -128;
        return v;
    endfunction

    // Ternary -> 2-bit code: 00=0, 01=+1, 11=-1 (scratchpad.md §2).
    function automatic logic [1:0] trit(input int w);
        return (w == 0) ? 2'b00 : (w == 1) ? 2'b01 : 2'b11;
    endfunction

    // Golden layer output Y[t][j] = clip(Σ_i A[t][i]·W[i][j]) with {m0=1,n=0}.
    function automatic int ref_y(input int t, input int j);
        int s; s = 0;
        for (int i = 0; i < ROWS; i++) s += actv[t][i] * wgtv[i][j];
        return clip8(s);
    endfunction

    // Generate bounded, deterministic operands and stream them into scratchpad.
    task automatic gen_and_load_operands();
        logic [8*WCOL_BYTES-1:0] colword;
        // Activations: A[t][i] in [-4,4], row-major int8.
        for (int t = 0; t < T; t++)
            for (int i = 0; i < ROWS; i++) begin
                actv[t][i] = ((t*3 + i) % 9) - 4;
                spad_wr_byte(ACT + ADDR_W'(t*ROWS + i), actv[t][i][7:0]);
            end
        // Weights: W[i][j] in {-1,0,1}, column-major 2-bit packed.
        for (int j = 0; j < COLS; j++) begin
            colword = '0;
            for (int i = 0; i < ROWS; i++) begin
                wgtv[i][j] = ((i + 2*j) % 3) - 1;
                colword[i*2 +: 2] = trit(wgtv[i][j]);
            end
            for (int b = 0; b < WCOL_BYTES; b++)
                spad_wr_byte(WGT + ADDR_W'(j*WCOL_BYTES + b), colword[b*8 +: 8]);
        end
        // Requant {m0,n} word: m0 in low M0_W bits, n above (matches requant8()).
        spad_wr_word(RQW, (RQ_N << M0_W) | RQ_M0);
    endtask

    // -------------------------------------------------------------------------
    // Program: config, address setup, matmul+requant, relu, halt.
    // -------------------------------------------------------------------------
    // Registers: r1=ACT, r2=WGT, r3=OUT(Y), r4=RLU(Z).
    task automatic load_program();
        load_instr(0, enc_imm(OP_SETCFG, CFG_TLEN,   16'(T)));       // MXU token count
        load_instr(1, enc_imm(OP_SETCFG, CFG_VLEN,   16'(NELEM)));   // VPU vector length
        load_instr(2, enc_imm(OP_SETCFG, CFG_SCALAR, RQW));          // requant word addr
        load_instr(3, enc_imm(OP_LIS, 8'd1, ACT));
        load_instr(4, enc_imm(OP_LIS, 8'd2, WGT));
        load_instr(5, enc_imm(OP_LIS, 8'd3, OUT));
        load_instr(6, enc_imm(OP_LIS, 8'd4, RLU));
        // MATMUL out=r3, act=r1, weight=r2; flags[1]=requant, flags[0]=accumulate.
        load_instr(7, enc_rrr(OP_MATMUL, 8'd3, 8'd1, 8'd2, 2'b10));
        // RELU dst=r4, src0=r3 (reads the int8 layer output back from scratchpad).
        load_instr(8, enc_rrr(OP_RELU, 8'd4, 8'd3, 8'd0, 2'b00));
        load_instr(9, enc_rrr(OP_HALT, 8'd0, 8'd0, 8'd0, 2'b00));
    endtask

    // -------------------------------------------------------------------------
    // Checks.
    // -------------------------------------------------------------------------
    task automatic check_layer_output();   // Y = requant(A@W), int8 at OUT
        logic [31:0] rd; int got, exp;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) begin
                spad_rd(OUT + ADDR_W'(t*COLS + j), rd);
                got = $signed(rd[7:0]);
                exp = ref_y(t, j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL Y[%0d][%0d]: got %0d  exp %0d", t, j, got, exp);
                end
            end
        $display("[matmul+requant] checked %0d elems (errors so far: %0d)", NELEM, errors);
    endtask

    task automatic check_activation();     // Z = relu(Y), int32 at RLU
        logic [31:0] rd; int got, exp, idx;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) begin
                idx = t*COLS + j;
                spad_rd(RLU + ADDR_W'(idx*4), rd);
                got = $signed(rd);
                exp = (ref_y(t, j) < 0) ? 0 : ref_y(t, j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL Z[%0d][%0d]: got %0d  exp %0d", t, j, got, exp);
                end
            end
        $display("[relu activation] checked %0d elems (errors so far: %0d)", NELEM, errors);
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

        $display("==== TPU top-level testbench (%0dx%0d array, T=%0d) ====",
                 ROWS, COLS, T);

        // 1) Preload operands into BRAM, 2) load the program, 3) run.
        gen_and_load_operands();
        load_program();

        @(negedge clk); host_run = 1'b1; boot_pc = '0;
        @(negedge clk); host_run = 1'b0;

        // Wait for HALT (watchdog below guards against a hang).
        do @(negedge clk); while (!done);
        $display("program halted at pc=%0d", pc_dbg);

        // 4) Read results back out of BRAM and check both layer stages.
        check_layer_output();
        check_activation();

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
