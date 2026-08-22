// -----------------------------------------------------------------------------
// vpu_tb.sv — self-checking testbench for accel/tpu/rtl/vpu.sv
//
// Models the scratchpad V_rw port with the DUT's memory contract (synchronous
// read: data valid the cycle after V_re; byte-strobed single-cycle write) and
// checks every VPU op against an in-testbench reference:
//
//   elementwise (int8 -> int32): ADD, RELU
//   reduction   (int8 -> int32 scalar): DOT
//   narrowing   (int32 -> int8): REQUANT, DYT
//   int4 pack   (int8  -> 4 bits): QUANT4
//
// That is the whole unit now. The ops this TB used to cover as well — GELU,
// EXP, SQUARE, ELEMENT_MUL, SCALAR_MUL/ADD/DIV, REDUCESUM, REDUCEMAX and the
// softmax macro op — were removed from vpu.sv because the current model does
// not issue them (see that file's header), and their tests went with them.
//
// DOT is not reachable from the ISA; it is exercised here because it is the
// inner primitive of VOP_VECMATMUL, whose row/column sequencing is covered at
// the tpu_top level instead.
//
// Everything narrows to **int4** now, in an int8 container: REQUANT clips to
// [-8, 7] and DYT to [-7, 7]. The two still share a fixed point and differ only
// in the lower clip (DyT's hardtanh being odd), so `test_dyt_floor` asserts that
// one input directly: a random spread hits the single value that separates them
// almost never, and without it a DUT that routed `dyt` to requant8 would pass
// everything else here.
//
// QUANT4 is the only op whose destination is narrower than a byte: two nibbles
// share one byte and the write strobe is per byte, so its tests check the packed
// *bytes*, not elements, and `test_quant4_pack` pins the encoding and the
// nibble order directly against hand-written bytes.
//
// Coverage includes multi-chunk streaming (vlen > LANES), an exact-boundary
// vlen, and partial-tail vlen so the lane-active predicate / V_wstrb masking is
// exercised.
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
    localparam int LANES        = SCRATCHPAD_W / 4;   // 16 int32 lanes

    // ---- Op encoding (matches vpu.sv) ---------------------------------------
    localparam logic [4:0]
        VOP_DOT       = 5'd0,  VOP_ADD       = 5'd1,  VOP_RELU = 5'd3,
        VOP_REQUANT   = 5'd10, VOP_VECMATMUL = 5'd13, VOP_DYT  = 5'd16,
        VOP_QUANT4    = 5'd17;

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
    logic [4:0]                vpu_op;
    // Macro-op geometry (docs/macro_ops.md §5). Driven per test; the
    // primitive-op tests leave them at the defaults set in `initial`.
    logic [15:0]               vpu_rows, vpu_cols;
    logic [ADDR_W-1:0]         vpu_row0, vpu_row1, vpu_crow;
    logic [ADDR_W-1:0]         vpu_src0, vpu_src1, vpu_dst;
    // The {m0,n} operand is a literal in the dispatch now rather than a
    // scratchpad address the VPU fetched over the V port before its first chunk
    // (docs/picorv32_migration.md §3). `run_op` still takes an address, and
    // reads the word out of the scratchpad model itself, so every caller and
    // every expected value below is unchanged.
    logic [M0_W+N_W-1:0]       vpu_rq_word;
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
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .vpu_start(vpu_start), .vpu_op(vpu_op),
        .vpu_src0(vpu_src0), .vpu_src1(vpu_src1),
        .vpu_rq_word(vpu_rq_word), .vpu_dst(vpu_dst),
        .vpu_vlen(vpu_vlen),
        // Macro-op geometry: unused by the primitive ops this TB exercises.
        // Rows/cols of 0 read as 1 inside the DUT, so even if a macro op were
        // dispatched here it would degrade to a single pair rather than hang.
        .vpu_rows(vpu_rows), .vpu_cols(vpu_cols),
        .vpu_row0(vpu_row0), .vpu_row1(vpu_row1), .vpu_crow(vpu_crow),
        .vpu_busy(vpu_busy), .vpu_done(vpu_done),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        // No other requester in this TB, so the arbiter always grants.
        .V_rgnt(1'b1), .V_wgnt(1'b1)
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
    task automatic run_op(input logic [4:0] op,
                          input logic [ADDR_W-1:0] s0, s1, sc, d,
                          input int vlen);
        @(negedge clk);
        vpu_op = op; vpu_src0 = s0; vpu_src1 = s1;
        vpu_rq_word = get32(sc);
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
            VOP_ADD:  return a + b;
            VOP_RELU: return (a > 0) ? a : 0;
            default:  return 0;
        endcase
    endfunction

    // Reference for REQUANT: clip((x*m0 + round) >> n), arithmetic shift.
    function automatic int ref_requant(input int x, input int m0, input int n);
        longint prod, round, shifted;
        prod    = longint'(x) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted >  7) return  7;
        else if (shifted < -8) return -8;
        else                   return int'(shifted);
    endfunction

    // Reference for DYT: the same rescale clipped symmetrically. Written out
    // rather than delegating to ref_requant with a bound argument, so that the
    // one constant separating the two ops appears twice and independently —
    // a shared helper would agree with a DUT that had them backwards.
    function automatic int ref_dyt(input int x, input int m0, input int n);
        longint prod, round, shifted;
        prod    = longint'(x) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted >  7) return  7;
        else if (shifted < -7) return -7;
        else                   return int'(shifted);
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

    task automatic test_reduce(input logic [3:0] op, input int vlen, input string tag);
        int exp;
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) begin
            put8(A_ADDR + i, tv_a[i]);
            put8(B_ADDR + i, tv_b[i]);
        end
        run_op(op, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        case (op)
            VOP_DOT: begin exp = 0; for (int i=0;i<vlen;i++) exp += tv_a[i]*tv_b[i]; end
            default: exp = 0;
        endcase
        exp32(D_ADDR, exp, tag);   // single int32 scalar result
        $display("[%s] vlen=%0d  result=%0d  (errors so far: %0d)", tag, vlen, exp, errors);
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

    // DyT / hardtanh. Same operands and same fixed point as REQUANT; the only
    // observable difference is the floor, so this task also asserts that
    // difference directly on a value engineered to land below -127 (see the
    // call sites), which is the one input that separates the two ops.
    task automatic test_dyt(input int vlen, input int m0, input int n, input string tag);
        gen_i32(vlen);
        for (int i = 0; i < vlen; i++) put32(A_ADDR + i*4, tv32[i]);
        put32(SC_ADDR, (n << M0_W) | m0);
        run_op(VOP_DYT, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int i = 0; i < vlen; i++)
            exp8(D_ADDR + i, ref_dyt(tv32[i], m0, n), tag);
        $display("[%s] vlen=%0d m0=%0d n=%0d  (errors so far: %0d)", tag, vlen, m0, n, errors);
    endtask

    // The one input that tells DYT and REQUANT apart: an accumulator that
    // rescales below the int4 floor. REQUANT must return -8, DYT must return -7.
    // Without this, a DUT that routed `dyt` to requant8 would pass every test
    // above, because gen_i32's random spread hits that single value ~never.
    task automatic test_dyt_floor();
        int got_dyt, got_rq;
        put32(A_ADDR, -8);
        put32(SC_ADDR, (0 << M0_W) | 1);        // {m0=1, n=0}: pass through
        run_op(VOP_DYT, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, 1);
        got_dyt = $signed(mem[D_ADDR]);
        exp8(D_ADDR, -7, "DYT-floor");
        run_op(VOP_REQUANT, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, 1);
        got_rq = $signed(mem[D_ADDR]);
        exp8(D_ADDR, -8, "REQUANT-floor");
        $display("[DYT-floor] -8 -> dyt %0d / requant %0d  (errors so far: %0d)",
                 got_dyt, got_rq, errors);
    endtask

    // Reference for QUANT4: the same shifter clipped to [-8, 7], returned as a
    // bare 4-bit two's-complement nibble — the MXU's packed weight element.
    function automatic logic [3:0] ref_quant4(input int x, input int m0, input int n);
        longint prod, round, shifted;
        prod    = longint'(x) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted >  7) return 4'sd7;
        else if (shifted < -8) return -4'sd8;
        else                   return shifted[3:0];
    endfunction

    task automatic expbyte(input [ADDR_W-1:0] a, input logic [7:0] exp,
                           input string tag);
        logic [7:0] got;
        got = mem[a];
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %-14s @0x%04h: got %02h  exp %02h", tag, a, got, exp);
        end
    endtask

    // QUANT4: int8 in at stride 1, 4 bits out, two elements to a byte. Checked
    // a byte at a time, because that is the granularity the DUT can write.
    task automatic test_quant4(input int vlen, input int m0, input int n,
                               input string tag);
        logic [7:0] exp;
        gen_i8(vlen);
        for (int i = 0; i < vlen; i++) put8(A_ADDR + i, tv_a[i]);
        put32(SC_ADDR, (n << M0_W) | m0);
        // Poison the destination so a byte the DUT fails to strobe is visible
        // rather than reading back as a legitimate all-zero nibble pair.
        for (int b = 0; b < (vlen + 1) / 2; b++) mem[D_ADDR + b] = 8'hA5;
        run_op(VOP_QUANT4, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, vlen);
        for (int b = 0; b < (vlen + 1) / 2; b++) begin
            exp = '0;
            for (int j = 0; j < 2; j++)
                if (b*2 + j < vlen)
                    exp[j*4 +: 4] = ref_quant4(tv_a[b*2 + j], m0, n);
            expbyte(D_ADDR + b, exp, tag);
        end
        $display("[%s] vlen=%0d m0=%0d n=%0d  (errors so far: %0d)", tag, vlen, m0, n, errors);
    endtask

    // The encoding itself, against bytes written out by hand. Everything above
    // compares the DUT to a reference that could share a misreading with it;
    // this one cannot. {+1, -1, -8, +7} with {m0=1,n=0} packs low nibble first:
    //   byte0 = 0x1 | (0xF << 4) = 0xF1     (+1, -1)
    //   byte1 = 0x8 | (0x7 << 4) = 0x78     (-8, +7)
    // Both grid ends appear, which is where a sign-extension bug shows: -8 must
    // encode as 0x8 and -1 as 0xF, not as saturated positives.
    task automatic test_quant4_pack();
        put8(A_ADDR + 0,   1);
        put8(A_ADDR + 1,  -1);
        put8(A_ADDR + 2, -99);      // clips to -8
        put8(A_ADDR + 3,  99);      // clips to +7
        put32(SC_ADDR, (0 << M0_W) | 1);        // {m0=1, n=0}: pass through
        mem[D_ADDR] = 8'hA5; mem[D_ADDR+1] = 8'hA5;
        run_op(VOP_QUANT4, A_ADDR, B_ADDR, SC_ADDR, D_ADDR, 4);
        expbyte(D_ADDR + 0, 8'hF1, "QUANT4-pack");
        expbyte(D_ADDR + 1, 8'h78, "QUANT4-pack");
        $display("[QUANT4-pack] {+1,-1,-8,+7} -> %02h %02h  (errors so far: %0d)",
                 mem[D_ADDR], mem[D_ADDR+1], errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        vpu_start = 1'b0; vpu_op = '0;
        vpu_src0 = '0; vpu_src1 = '0; vpu_rq_word = '0; vpu_dst = '0; vpu_vlen = '0;
        vpu_rows = 16'd0; vpu_cols = 16'd0;
        vpu_row0 = '0; vpu_row1 = '0; vpu_crow = '0;
        for (int i = 0; i < MEM_SZ; i++) mem[i] = '0;

        // Reset.
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== VPU testbench ====");

        // Elementwise: multi-chunk (40 = 2 full + tail 8), exact boundary (16),
        // and a couple of small lengths.
        test_elem(VOP_ADD,         40, 0,   "ADD");
        test_elem(VOP_ADD,         16, 0,   "ADD-exact");
        test_elem(VOP_ADD,          1, 0,   "ADD-len1");
        test_elem(VOP_RELU,        40, 0,   "RELU");

        // The one reduction left: DOT, vecmatmul's inner primitive. 48 is three
        // whole chunks, 7 a partial tail.
        test_reduce(VOP_DOT, 40, "DOT");
        test_reduce(VOP_DOT, 48, "DOT-3chunk");
        test_reduce(VOP_DOT,  7, "DOT-tail");

        // Requant (int32 -> int8), including clip + rounding.
        test_requant(40, 1500, 10, "REQUANT");
        test_requant(33,  512,  8, "REQUANT-half");
        test_requant(20, 4095,  0, "REQUANT-noshift");

        // ---- DyT / hardtanh: requant's fixed point, symmetric clip ---------
        test_dyt(40, 1500, 10, "DYT");
        test_dyt(33,  512,  8, "DYT-half");
        test_dyt(20, 4095,  0, "DYT-noshift");
        test_dyt_floor();

        // ---- QUANT4: requant's fixed point, clipped to int4 and packed -----
        // 64 is four whole chunks (LANES = 16), 32 two, 20 one chunk plus a
        // 4-element tail — the case where the strobe must cover exactly one
        // extra byte. m0/n are chosen so the threshold lands inside int8's
        // range and all three codes appear.
        test_quant4(64, 1, 5, "QUANT4");         // /32: lands inside the grid
        test_quant4(32, 1, 6, "QUANT4-2chunk");
        test_quant4(20, 3, 0, "QUANT4-tail");    // no shift: exercises the clip
        test_quant4(18, 1, 4, "QUANT4-odd");     // partial final lane chunk
        test_quant4_pack();

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
