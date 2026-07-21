// -----------------------------------------------------------------------------
// vpu_tb.sv — self-checking testbench for accel/tpu/rtl/vpu.sv
//
// Models the scratchpad V_rw port with the DUT's memory contract (synchronous
// read: data valid the cycle after V_re; byte-strobed single-cycle write) and
// checks every VPU op against an in-testbench reference:
//
//   elementwise (int8 -> int32): ADD, ELEMENT_MUL, SCALAR_MUL, SCALAR_ADD, RELU,
//                                SQUARE
//   scalar div  (int8 -> int32): SCALAR_DIV (reciprocate-then-multiply, Q15)
//   LUT         (int8 -> int32): GELU, EXP
//   reductions  (int8 -> int32 scalar): DOT, REDUCESUM, REDUCEMAX
//   requant     (int32 -> int8): REQUANT
//
// Coverage includes multi-chunk streaming (vlen > LANES), an exact-boundary
// vlen, and partial-tail vlen so the lane-active predicate / V_wstrb masking is
// exercised. The activation LUTs are filled directly through the DUT's ROM
// arrays (hierarchical reference) so no external $readmemh files are needed;
// the reference reads the same arrays.
//
// Run:  make sim         (iverilog -g2012 + vvp)
// -----------------------------------------------------------------------------

`timescale 1ns/1ps

module vpu_tb;

    // ---- Parameters (must match the DUT instance) ---------------------------
    localparam int SCRATCHPAD_W = 64;
    localparam int ADDR_W       = 16;
    localparam int M0_W         = 12;
    localparam int N_W          = 4;
    localparam int RECIP_Q      = 31;                 // must match the DUT instance
    localparam int DIV_Q        = 15;
    localparam int LANES        = SCRATCHPAD_W / 4;   // 16 int32 lanes

    // ---- Op encoding (matches vpu.sv) ---------------------------------------
    localparam logic [3:0]
        VOP_DOT         = 4'd0,  VOP_ADD        = 4'd1,  VOP_SCALAR_MUL = 4'd2,
        VOP_RELU        = 4'd3,  VOP_GELU       = 4'd4,  VOP_SQUARE     = 4'd5,
        VOP_EXP         = 4'd6,  VOP_REDUCEMAX  = 4'd7,  VOP_REDUCESUM  = 4'd8,
        VOP_ELEMENT_MUL = 4'd9,  VOP_REQUANT    = 4'd10, VOP_SCALAR_ADD = 4'd11,
        VOP_SCALAR_DIV  = 4'd12;

    // ---- Scratchpad address map (generous spacing; int8 chunks over-read) ---
    localparam logic [ADDR_W-1:0] A_ADDR  = 16'h1000;  // src0
    localparam logic [ADDR_W-1:0] B_ADDR  = 16'h2000;  // src1
    localparam logic [ADDR_W-1:0] SC_ADDR = 16'h3000;  // scalar / requant param
    localparam logic [ADDR_W-1:0] D_ADDR  = 16'h4000;  // dst

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT interface.
    // -------------------------------------------------------------------------
    logic                      vpu_start;
    logic [3:0]                vpu_op;
    logic [ADDR_W-1:0]         vpu_src0, vpu_src1, vpu_scalar, vpu_dst;
    logic [9:0]                vpu_vlen;
    logic                      vpu_busy, vpu_done;

    logic                      V_re;
    logic [ADDR_W-1:0]         V_raddr;
    logic [SCRATCHPAD_W*8-1:0] V_rdata;
    logic                      V_we;
    logic [ADDR_W-1:0]         V_waddr;
    logic [SCRATCHPAD_W*8-1:0] V_wdata;
    logic [SCRATCHPAD_W-1:0]   V_wstrb;

    vpu #(
        .SCRATCHPAD_W(SCRATCHPAD_W), .ADDR_W(ADDR_W),
        .M0_W(M0_W), .N_W(N_W)
        // GELU_INIT / EXP_INIT left empty: LUTs are filled from the TB below.
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .vpu_start(vpu_start), .vpu_op(vpu_op),
        .vpu_src0(vpu_src0), .vpu_src1(vpu_src1),
        .vpu_scalar(vpu_scalar), .vpu_dst(vpu_dst),
        .vpu_vlen(vpu_vlen), .vpu_busy(vpu_busy), .vpu_done(vpu_done),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb)
    );

    // -------------------------------------------------------------------------
    // Scratchpad model. Byte-addressed; a read returns SCRATCHPAD_W bytes
    // starting at V_raddr, valid one cycle after V_re. Writes are byte-strobed.
    // -------------------------------------------------------------------------
    localparam int MEM_SZ = 1 << ADDR_W;
    logic [7:0] mem [0:MEM_SZ-1];

    logic [SCRATCHPAD_W*8-1:0] rdata_r;
    always_ff @(posedge clk) begin
        if (V_re)
            for (int k = 0; k < SCRATCHPAD_W; k++)
                rdata_r[k*8 +: 8] <= mem[V_raddr + k];
    end
    assign V_rdata = rdata_r;

    always_ff @(posedge clk) begin
        if (V_we)
            for (int k = 0; k < SCRATCHPAD_W; k++)
                if (V_wstrb[k]) mem[V_waddr + k] <= V_wdata[k*8 +: 8];
    end

    // -------------------------------------------------------------------------
    // Memory access helpers (little-endian).
    // -------------------------------------------------------------------------
    task automatic put8(input [ADDR_W-1:0] a, input int v);
        mem[a] = v[7:0];
    endtask
    task automatic put32(input [ADDR_W-1:0] a, input int v);
        mem[a]   = v[7:0];   mem[a+1] = v[15:8];
        mem[a+2] = v[23:16]; mem[a+3] = v[31:24];
    endtask
    function automatic int get8s(input [ADDR_W-1:0] a);
        return $signed(mem[a]);
    endfunction
    function automatic int get32(input [ADDR_W-1:0] a);
        return {mem[a+3], mem[a+2], mem[a+1], mem[a]};
    endfunction

    // -------------------------------------------------------------------------
    // Test vectors + bookkeeping.
    // -------------------------------------------------------------------------
    int tv_a  [0:255];
    int tv_b  [0:255];
    int tv32  [0:63];
    int errors = 0;
    int checks = 0;

    task automatic gen_i8(input int n);
        logic [7:0] r;
        for (int i = 0; i < n; i++) begin
            r = $random; tv_a[i] = $signed(r);
            r = $random; tv_b[i] = $signed(r);
        end
        // edge values in the first lanes
        if (n > 4) begin
            tv_a[0] = 0;    tv_a[1] = 127;  tv_a[2] = -128; tv_a[3] = -1;
            tv_b[0] = 127;  tv_b[1] = -128; tv_b[2] = 1;    tv_b[3] = 0;
        end
    endtask

    task automatic gen_i32(input int n);
        for (int i = 0; i < n; i++)
            tv32[i] = ($random % 40001) - 20000;   // spans the int8 clip range
        if (n > 4) begin
            tv32[0] = 0; tv32[1] = 1; tv32[2] = -1;
            tv32[3] = 1000000; tv32[4] = -1000000;   // force clip
        end
    endtask

    // -------------------------------------------------------------------------
    // Dispatch one op and wait for completion (issue-and-wait handshake).
    // Inputs are driven on the negedge; the DUT samples on the posedge.
    // -------------------------------------------------------------------------
    task automatic run_op(input logic [3:0] op,
                          input logic [ADDR_W-1:0] s0, s1, sc, d,
                          input int vlen);
        @(negedge clk);
        vpu_op = op; vpu_src0 = s0; vpu_src1 = s1; vpu_scalar = sc;
        vpu_dst = d; vpu_vlen = vlen[9:0]; vpu_start = 1'b1;
        @(negedge clk);
        vpu_start = 1'b0;
        do @(negedge clk); while (!vpu_done);
        @(negedge clk);   // let the DUT settle back to idle
    endtask

    // -------------------------------------------------------------------------
    // Checkers.
    // -------------------------------------------------------------------------
    task automatic exp32(input [ADDR_W-1:0] a, input int exp, input string tag);
        int got;
        got = get32(a);
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %-14s @0x%04h: got %0d  exp %0d", tag, a, got, exp);
        end
    endtask

    task automatic exp8(input [ADDR_W-1:0] a, input int exp, input string tag);
        int got;
        got = get8s(a);
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %-14s @0x%04h: got %0d  exp %0d", tag, a, got, exp);
        end
    endtask

    // Reference for the int8 -> int32 elementwise ops.
    function automatic int ref_elem(input logic [3:0] op,
                                    input int a, input int b, input int scal);
        case (op)
            VOP_ADD:         return a + b;
            VOP_ELEMENT_MUL: return a * b;
            VOP_SCALAR_MUL:  return a * scal;
            VOP_SCALAR_ADD:  return a + scal;
            VOP_RELU:        return (a > 0) ? a : 0;
            VOP_SQUARE:      return a * a;
            default:         return 0;
        endcase
    endfunction

    // Reference for SCALAR_DIV: round(a * 2**DIV_Q / d), mirroring vpu.sv's
    // reciprocate-then-multiply — R = floor(2**RECIP_Q / |d|), a*R rounded down by
    // (RECIP_Q-DIV_Q) with an arithmetic shift, then sign-corrected for d.
    function automatic int ref_div(input int a, input int d);
        longint absd, R, prod, rnd, shifted;
        absd    = (d < 0) ? -longint'(d) : longint'(d);
        R       = (longint'(1) << RECIP_Q) / absd;           // floor; d != 0 in tests
        prod    = longint'(a) * R;
        rnd     = (RECIP_Q == DIV_Q) ? 0 : (longint'(1) << (RECIP_Q - DIV_Q - 1));
        shifted = (prod + rnd) >>> (RECIP_Q - DIV_Q);
        return int'((d < 0) ? -shifted : shifted);
    endfunction

    // Reference for REQUANT: clip((x*m0 + round) >> n), arithmetic shift.
    function automatic int ref_requant(input int x, input int m0, input int n);
        longint prod, round, shifted;
        prod    = longint'(x) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted >  127) return  127;
        else if (shifted < -128) return -128;
        else                     return int'(shifted);
    endfunction

    // -------------------------------------------------------------------------
    // Op-level tests.
    // -------------------------------------------------------------------------
    task automatic test_elem(input logic [3:0] op, input int vlen,
                             input int scal, input string tag);
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) begin
            put8(A_ADDR + i, tv_a[i]);
            put8(B_ADDR + i, tv_b[i]);
        end
        put32(SC_ADDR, scal);
        run_op(op, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int i = 0; i < vlen; i++)
            exp32(D_ADDR + i*4, ref_elem(op, tv_a[i], tv_b[i], scal), tag);
        $display("[%s] vlen=%0d  (errors so far: %0d)", tag, vlen, errors);
    endtask

    task automatic test_lut(input logic [3:0] op, input int vlen, input string tag);
        int idx, exp;
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) put8(A_ADDR + i, tv_a[i]);
        run_op(op, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int i = 0; i < vlen; i++) begin
            idx = tv_a[i] & 32'hFF;
            exp = (op == VOP_GELU) ? $signed(dut.gelu_rom[idx])
                                   : $signed(dut.exp_rom[idx]);
            exp32(D_ADDR + i*4, exp, tag);
        end
        $display("[%s] vlen=%0d  (errors so far: %0d)", tag, vlen, errors);
    endtask

    task automatic test_reduce(input logic [3:0] op, input int vlen, input string tag);
        int exp;
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) begin
            put8(A_ADDR + i, tv_a[i]);
            put8(B_ADDR + i, tv_b[i]);
        end
        run_op(op, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        case (op)
            VOP_REDUCESUM: begin exp = 0; for (int i=0;i<vlen;i++) exp += tv_a[i]; end
            VOP_DOT:       begin exp = 0; for (int i=0;i<vlen;i++) exp += tv_a[i]*tv_b[i]; end
            VOP_REDUCEMAX: begin exp = -2147483648;
                                 for (int i=0;i<vlen;i++) if (tv_a[i] > exp) exp = tv_a[i]; end
            default:       exp = 0;
        endcase
        exp32(D_ADDR, exp, tag);   // single int32 scalar result
        $display("[%s] vlen=%0d  result=%0d  (errors so far: %0d)", tag, vlen, exp, errors);
    endtask

    task automatic test_div(input int vlen, input int d, input string tag);
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) put8(A_ADDR + i, tv_a[i]);
        put32(SC_ADDR, d);
        run_op(VOP_SCALAR_DIV, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int i = 0; i < vlen; i++)
            exp32(D_ADDR + i*4, ref_div(tv_a[i], d), tag);
        $display("[%s] vlen=%0d d=%0d  (errors so far: %0d)", tag, vlen, d, errors);
    endtask

    task automatic test_requant(input int vlen, input int m0, input int n, input string tag);
        gen_i32(vlen);
        for (int i = 0; i < vlen; i++) put32(A_ADDR + i*4, tv32[i]);
        put32(SC_ADDR, (n << M0_W) | m0);
        run_op(VOP_REQUANT, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int i = 0; i < vlen; i++)
            exp8(D_ADDR + i, ref_requant(tv32[i], m0, n), tag);
        $display("[%s] vlen=%0d m0=%0d n=%0d  (errors so far: %0d)", tag, vlen, m0, n, errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        vpu_start = 1'b0; vpu_op = '0;
        vpu_src0 = '0; vpu_src1 = '0; vpu_scalar = '0; vpu_dst = '0; vpu_vlen = '0;
        for (int i = 0; i < MEM_SZ; i++) mem[i] = '0;

        // Reset.
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Fill the activation LUTs (deterministic; reference reads them back).
        for (int i = 0; i < 256; i++) begin
            dut.gelu_rom[i] = $signed(8'((i * 3 + 7)));      // arbitrary but fixed
            dut.exp_rom[i]  = $signed(8'((255 - i) ^ 8'h2A));
        end

        $display("==== VPU testbench ====");

        // Elementwise: multi-chunk (40 = 2 full + tail 8), exact boundary (16),
        // and a couple of small lengths.
        test_elem(VOP_ADD,         40, 0,   "ADD");
        test_elem(VOP_ADD,         16, 0,   "ADD-exact");
        test_elem(VOP_ADD,          1, 0,   "ADD-len1");
        test_elem(VOP_ELEMENT_MUL, 40, 0,   "ELEMENT_MUL");
        test_elem(VOP_SCALAR_MUL,  40, -7,  "SCALAR_MUL-neg");
        test_elem(VOP_SCALAR_MUL,  33, 100, "SCALAR_MUL-pos");
        test_elem(VOP_SCALAR_ADD,  40, -50, "SCALAR_ADD-neg");   // softmax x-max
        test_elem(VOP_SCALAR_ADD,  33, 64,  "SCALAR_ADD-pos");
        test_elem(VOP_RELU,        40, 0,   "RELU");
        test_elem(VOP_SQUARE,      40, 0,   "SQUARE");

        // Scalar divide (reciprocate-then-multiply, Q15 result). Positive and
        // negative divisors, a softmax-sized denominator, and a partial tail.
        test_div(40, 100,  "SCALAR_DIV-pos");
        test_div(33, -37,  "SCALAR_DIV-neg");
        test_div(48, 2039, "SCALAR_DIV-softmax");
        test_div( 7, 3,    "SCALAR_DIV-tail");

        // LUT activations.
        test_lut(VOP_GELU, 40, "GELU");
        test_lut(VOP_EXP,  40, "EXP");

        // Reductions (whole vector -> one int32 scalar).
        test_reduce(VOP_REDUCESUM, 40, "REDUCESUM");
        test_reduce(VOP_REDUCESUM, 48, "REDUCESUM-3chunk");
        test_reduce(VOP_REDUCEMAX, 40, "REDUCEMAX");
        test_reduce(VOP_DOT,       40, "DOT");
        test_reduce(VOP_DOT,        7, "DOT-tail");

        // Requant (int32 -> int8), including clip + rounding.
        test_requant(40, 1500, 10, "REQUANT");
        test_requant(33,  512,  8, "REQUANT-half");
        test_requant(20, 4095,  0, "REQUANT-noshift");

        // Summary.
        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("VPU: ALL TESTS PASSED");
        else             $display("VPU: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #200000;
        $display("VPU: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
