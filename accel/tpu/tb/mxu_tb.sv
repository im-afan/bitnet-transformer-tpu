`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// mxu_tb.sv — self-checking testbench for accel/tpu/rtl/mxu.sv
//
// Models the three scratchpad ports the MXU uses (A_rd activations, W_rd int4
// weights, C_rw int32 results) with the DUT's memory contract (synchronous read:
// data valid the cycle after *_re; single-cycle write) and checks the systolic
// matmul against an in-testbench integer reference C = A_int @ W_int4.
//
// Weights are **int4, row-major**, two nibbles per byte: W[i][j] at
// WGT + i*WROW_BYTES, column j in bits [j*4 +: 4]. `put_wgt` below is the whole
// layout contract.
//
// Coverage: several token counts T (incl. the max-skew case T=1), random int4
// weights + int8 activations with forced edge values, and the `accumulate` tiling
// path (result = existing int32 partial + fresh matmul). Activations are driven
// over the *full* int8 range even though the model only produces int4 values —
// the array is specified for int8 containers and ROWS=4 keeps the PSUM_W=16
// bound (4*127*8 = 4064) satisfied, so this stresses the multiplier harder than
// the real operating point does. Runs a small ROWS x COLS array so the systolic
// drain is quick to simulate; the RTL default is 128x128.
//
// Run:  make -f Makefile.mxu sim      (iverilog -g2012 + vvp)
// -----------------------------------------------------------------------------

module mxu_tb;

    // ---- Parameters (must match the DUT instance) ---------------------------
    localparam int ROWS   = 4;
    localparam int COLS   = 4;
    localparam int ADDR_W = 16;
    localparam int WROW_BYTES = (COLS * 4) / 8;   // packed weight row bytes (int4)
    localparam int RES_STRIDE = COLS * 4;

    // ---- Scratchpad address map ---------------------------------------------
    localparam logic [ADDR_W-1:0] ACT = 16'h1000;  // activations  A[t][i] int8
    localparam logic [ADDR_W-1:0] WGT = 16'h2000;  // weights (row-major, 4-bit)
    localparam logic [ADDR_W-1:0] RES = 16'h3000;  // results C[t][j] int32
    localparam logic [ADDR_W-1:0] SCL = 16'h7000;  // requant {N, M0} word (int32)
    localparam logic [ADDR_W-1:0] RQO = 16'h4000;  // requantized results C[t][j] int8

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // -------------------------------------------------------------------------
    // DUT interface.
    // -------------------------------------------------------------------------
    localparam int M0_W = 12;
    localparam int N_W  = 4;

    logic              start, accumulate, requant, busy, done;
    logic [ADDR_W-1:0] act_addr, weight_addr, out_addr;
    // Tiling geometry, driven per test rather than tied off: the row-major
    // weight layout swaps which tile axis needs the stride, so the tiled path is
    // the one that has to be exercised, not the single-tile default.
    logic              tiled;
    logic [ADDR_W-1:0] a_row, c_row, w_row;
    logic [7:0]        k_tiles, n_tiles;
    // The requant {m0,n} is a literal in the dispatch now, not a scratchpad
    // address the MXU fetches over the C port (docs/picorv32_migration.md §3).
    logic [M0_W+N_W-1:0] rq_word;
    logic [5:0]        t_len;

    logic              A_re;  logic [ADDR_W-1:0] A_raddr;  logic [ROWS*8-1:0] A_rdata;
    logic              W_re;  logic [ADDR_W-1:0] W_raddr;  logic [COLS*4-1:0] W_rdata;
    logic              C_re;  logic [ADDR_W-1:0] C_raddr;  logic [COLS*32-1:0] C_rdata;
    logic              C_we;  logic [ADDR_W-1:0] C_waddr;  logic [COLS*32-1:0] C_wdata;
    logic [COLS*4-1:0] C_wstrb;

    mxu #(.ROWS(ROWS), .COLS(COLS), .ADDR_W(ADDR_W), .M0_W(M0_W), .N_W(N_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .act_addr(act_addr), .weight_addr(weight_addr),
        .out_addr(out_addr), .rq_word(rq_word), .t_len(t_len),
        // Strides zero => the single-tile defaults (a_row=ROWS, c_row=COLS*4,
        // w_row=COLS*4/8), i.e. exactly the constants these replaced. Driving
        // them explicitly rather than leaving them unconnected: an unconnected
        // input floats to z, and `a_row == '0` on z is x, which poisons every
        // address the DUT generates.
        .tiled(tiled), .a_row(a_row), .c_row(c_row), .w_row(w_row),
        .k_tiles(k_tiles), .n_tiles(n_tiles),
        .accumulate(accumulate), .requant(requant),
        .busy(busy), .done(done), .load_active(),
        .A_re(A_re), .A_raddr(A_raddr), .A_rdata(A_rdata),
        .W_re(W_re), .W_raddr(W_raddr), .W_rdata(W_rdata),
        .C_re(C_re), .C_raddr(C_raddr), .C_rdata(C_rdata),
        .C_we(C_we), .C_waddr(C_waddr), .C_wdata(C_wdata), .C_wstrb(C_wstrb)
    );

    // -------------------------------------------------------------------------
    // Scratchpad model. Byte-addressed; each read returns the port's byte width
    // starting at the address, valid one cycle after *_re.
    // -------------------------------------------------------------------------
    localparam int MEM_SZ = 1 << ADDR_W;
    logic [7:0] mem [0:MEM_SZ-1];

    logic [ROWS*8-1:0]  a_rd_r;
    logic [COLS*4-1:0]  w_rd_r;
    logic [COLS*32-1:0] c_rd_r;

    always_ff @(posedge clk) begin
        if (A_re)
            for (int b = 0; b < ROWS; b++)          // ROWS int8 bytes
                a_rd_r[b*8 +: 8] <= mem[A_raddr + b];
        if (W_re)
            for (int b = 0; b < WROW_BYTES; b++)     // packed weight row
                w_rd_r[b*8 +: 8] <= mem[W_raddr + b];
        if (C_re)
            for (int b = 0; b < COLS*4; b++)         // COLS int32 bytes
                c_rd_r[b*8 +: 8] <= mem[C_raddr + b];
    end
    assign A_rdata = a_rd_r;
    assign W_rdata = w_rd_r;
    assign C_rdata = c_rd_r;

    always_ff @(posedge clk) begin
        if (C_we)
            for (int b = 0; b < COLS*4; b++)
                if (C_wstrb[b]) mem[C_waddr + b] <= C_wdata[b*8 +: 8];
    end

    // -------------------------------------------------------------------------
    // Memory helpers.
    // -------------------------------------------------------------------------
    task automatic put_act(input int t, input int i, input int v);
        mem[ACT + t*ROWS + i] = v[7:0];
    endtask

    // int4 weight -> 4-bit two's-complement nibble, ROW-major packed:
    // row i is contiguous, column j occupies nibble j of that row.
    task automatic put_wgt(input int i, input int j, input int wv);
        logic [3:0] nib;
        int         a;
        nib = wv[3:0];
        a   = WGT + i*WROW_BYTES + (j/2);
        mem[a][(j%2)*4 +: 4] = nib;
    endtask

    task automatic put_res(input int t, input int j, input int v);   // int32 LE
        int a; a = RES + t*RES_STRIDE + j*4;
        mem[a]   = v[7:0];   mem[a+1] = v[15:8];
        mem[a+2] = v[23:16]; mem[a+3] = v[31:24];
    endtask

    function automatic int get_res(input int t, input int j);
        int a; a = RES + t*RES_STRIDE + j*4;
        return {mem[a+3], mem[a+2], mem[a+1], mem[a]};
    endfunction

    // Requant {M0, N} word: m0 in low M0_W bits, n above it (matches VPU/DUT).
    // Still written to the scratchpad model so the address map in this TB stays
    // as documented, but the DUT reads the literal below.
    task automatic put_scalar(input int m0, input int n);
        logic [31:0] w;
        w = (n[N_W-1:0] << M0_W) | m0[M0_W-1:0];
        mem[SCL]   = w[7:0];   mem[SCL+1] = w[15:8];
        mem[SCL+2] = w[23:16]; mem[SCL+3] = w[31:24];
    endtask

    function automatic int get_res8(input int t, input int j);   // signed int8 row
        return $signed(mem[RQO + t*COLS + j]);
    endfunction

    // Reference requant: clip((acc*m0 + round) >> n) to **int4**, in an int8
    // container. Mirrors requant8() / mxu.sv ACT_QMIN..ACT_QMAX.
    function automatic int ref_requant(input int acc, input int m0, input int n);
        longint prod, round, shifted;
        prod    = longint'(acc) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if (shifted >  7) return  7;
        if (shifted < -8) return -8;
        return int'(shifted);
    endfunction

    // -------------------------------------------------------------------------
    // Reference model + bookkeeping.
    // -------------------------------------------------------------------------
    int actv [0:63][0:ROWS-1];   // int8 activations
    int wgtv [0:ROWS-1][0:COLS-1];  // int4 weights, [-8, 7]
    int errors = 0, checks = 0;

    task automatic gen_weights();
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) begin
                wgtv[i][j] = int'($unsigned($random) % 16) - 8;  // [-8, 7]
                put_wgt(i, j, wgtv[i][j]);
            end
        // Force both grid ends, which is where a sign-extension bug shows up:
        // -8 has no positive counterpart and +7 is the largest magnitude that
        // does. A nibble read as unsigned would turn -8 into +8 and -1 into +15.
        if (ROWS >= 2 && COLS >= 2) begin
            wgtv[0][0] = -8; put_wgt(0, 0, -8);
            wgtv[0][1] =  7; put_wgt(0, 1,  7);
            wgtv[1][0] = -1; put_wgt(1, 0, -1);
            wgtv[1][1] =  0; put_wgt(1, 1,  0);
        end
    endtask

    task automatic gen_acts(input int T);
        logic [7:0] r;
        for (int t = 0; t < T; t++)
            for (int i = 0; i < ROWS; i++) begin
                r = $random; actv[t][i] = $signed(r);
                put_act(t, i, actv[t][i]);
            end
        // Edge values on token 0.
        if (ROWS >= 4) begin
            actv[0][0]=127; actv[0][1]=-128; actv[0][2]=0; actv[0][3]=-1;
            for (int i = 0; i < 4; i++) put_act(0, i, actv[0][i]);
        end
    endtask

    function automatic int ref_c(input int t, input int j);
        int s; s = 0;
        for (int i = 0; i < ROWS; i++) s += actv[t][i] * wgtv[i][j];
        return s;
    endfunction

    // -------------------------------------------------------------------------
    // Dispatch one matmul and wait for done (issue-and-wait handshake).
    // -------------------------------------------------------------------------
    task automatic run_matmul(input int T, input logic acc);
        @(negedge clk);
        act_addr = ACT; weight_addr = WGT; out_addr = RES;
        t_len = T[5:0]; accumulate = acc; requant = 1'b0; start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        do @(negedge clk); while (!done);
        @(negedge clk);
    endtask

    // Requant path: fresh matmul, narrow int32 -> int8 on store into RQO.
    task automatic run_requant(input int T);
        @(negedge clk);
        act_addr = ACT; weight_addr = WGT; out_addr = RQO;
        t_len = T[5:0]; accumulate = 1'b0; requant = 1'b1; start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        do @(negedge clk); while (!done);
        @(negedge clk);
    endtask

    task automatic check_all(input int T, input string tag);
        int got, exp;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) begin
                got = get_res(t, j);
                exp = ref_c(t, j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL %-12s C[%0d][%0d]: got %0d  exp %0d",
                             tag, t, j, got, exp);
                end
            end
        $display("[%s] T=%0d  (errors so far: %0d)", tag, T, errors);
    endtask

    // Plain matmul: fresh weights + activations, compare against integer reference.
    task automatic test_matmul(input int T, input string tag);
        gen_weights();
        gen_acts(T);
        run_matmul(T, 1'b0);
        check_all(T, tag);
    endtask

    // Requant path: fresh matmul, expect int8 = requant(A@W, {m0, n}).
    task automatic test_requant(input int T, input int m0, input int n, input string tag);
        int got, exp;
        gen_weights();
        gen_acts(T);
        put_scalar(m0, n);
        rq_word = (n[N_W-1:0] << M0_W) | m0[M0_W-1:0];
        run_requant(T);
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) begin
                got = get_res8(t, j);
                exp = ref_requant(ref_c(t, j), m0, n);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL %-12s C8[%0d][%0d]: got %0d  exp %0d  (acc %0d)",
                             tag, t, j, got, exp, ref_c(t, j));
                end
            end
        $display("[%s] T=%0d  m0=%0d n=%0d  (errors so far: %0d)", tag, T, m0, n, errors);
    endtask

    // Accumulate path: preload result region, expect existing + matmul.
    task automatic test_accumulate(input int T, input string tag);
        int got, exp, pre;
        gen_weights();
        gen_acts(T);
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) put_res(t, j, (t*100 + j*7) - 250);
        run_matmul(T, 1'b1);
        for (int t = 0; t < T; t++)
            for (int j = 0; j < COLS; j++) begin
                pre = (t*100 + j*7) - 250;
                got = get_res(t, j);
                exp = pre + ref_c(t, j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL %-12s C[%0d][%0d]: got %0d  exp %0d",
                             tag, t, j, got, exp);
                end
            end
        $display("[%s] T=%0d  (errors so far: %0d)", tag, T, errors);
    endtask


    // -------------------------------------------------------------------------
    // Tiled matmul: a K x N weight matrix larger than the array, walked by the
    // hardware tile loop. This is what the row-major layout actually changed —
    // an n-tile is now a step *along* a weight row (a constant WROW_BYTES) and a
    // k-tile is a step *down* ROWS whole rows (ROWS*w_row), the exact opposite of
    // the column-major arithmetic. tiled=0 above cannot catch a swap of the two.
    // -------------------------------------------------------------------------
    localparam int KT = 2;                    // contraction tiles
    localparam int NT = 2;                    // output tiles
    localparam int KK = ROWS * KT;            // full contraction dim
    localparam int NN = COLS * NT;            // full output dim
    localparam int WROW_FULL = (NN * 4) / 8;  // bytes per full weight row
    localparam int CROW_FULL = NN * 4;        // bytes per full int32 result row

    int actf [0:63][0:KK-1];
    int wgtf [0:KK-1][0:NN-1];

    // Same nibble packing as put_wgt, at the *full* matrix's row stride.
    task automatic put_wgt_full(input int i, input int j, input int wv);
        int a; a = WGT + i*WROW_FULL + (j/2);
        mem[a][(j%2)*4 +: 4] = wv[3:0];
    endtask

    function automatic int get_res_full(input int t, input int j);
        int a; a = RES + t*CROW_FULL + j*4;
        return {mem[a+3], mem[a+2], mem[a+1], mem[a]};
    endfunction

    task automatic test_tiled(input int T, input string tag);
        int got, exp;
        logic [7:0] r;
        for (int i = 0; i < KK; i++)
            for (int j = 0; j < NN; j++) begin
                wgtf[i][j] = int'($unsigned($random) % 16) - 8;
                put_wgt_full(i, j, wgtf[i][j]);
            end
        for (int t = 0; t < T; t++)
            for (int i = 0; i < KK; i++) begin
                r = $random; actf[t][i] = $signed(r);
                mem[ACT + t*KK + i] = actf[t][i][7:0];
            end

        @(negedge clk);
        act_addr = ACT; weight_addr = WGT; out_addr = RES;
        t_len = T[5:0]; accumulate = 1'b0; requant = 1'b0;
        tiled = 1'b1;
        a_row = ADDR_W'(KK); c_row = ADDR_W'(CROW_FULL); w_row = ADDR_W'(WROW_FULL);
        k_tiles = KT[7:0]; n_tiles = NT[7:0];
        start = 1'b1;
        @(negedge clk); start = 1'b0;
        do @(negedge clk); while (!done);
        @(negedge clk);
        tiled = 1'b0; a_row = '0; c_row = '0; w_row = '0;
        k_tiles = 8'd1; n_tiles = 8'd1;

        for (int t = 0; t < T; t++)
            for (int j = 0; j < NN; j++) begin
                exp = 0;
                for (int i = 0; i < KK; i++) exp += actf[t][i] * wgtf[i][j];
                got = get_res_full(t, j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    $display("  FAIL %-12s C[%0d][%0d]: got %0d  exp %0d",
                             tag, t, j, got, exp);
                end
            end
        $display("[%s] T=%0d  %0dx%0d over %0dx%0d array  (errors so far: %0d)",
                 tag, T, KK, NN, ROWS, COLS, errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        start = 1'b0; accumulate = 1'b0; requant = 1'b0; t_len = '0;
        act_addr = '0; weight_addr = '0; out_addr = '0; rq_word = '0;
        tiled = 1'b0; a_row = '0; c_row = '0; w_row = '0;
        k_tiles = 8'd1; n_tiles = 8'd1;
        for (int i = 0; i < MEM_SZ; i++) mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== MXU testbench (%0dx%0d array) ====", ROWS, COLS);

        test_matmul(1, "MM-T1");     // single token: max input-skew corner
        test_matmul(2, "MM-T2");
        test_matmul(4, "MM-T4");
        test_matmul(8, "MM-T8");
        test_accumulate(4, "ACC-T4");
        test_accumulate(6, "ACC-T6");
        // The int4 grid is coarse, so shifts are chosen to land results *inside*
        // [-8, 7] rather than pinned at the clip. |acc| <= ROWS*127*8 = 4064 here,
        // so ~1/512 is the scale that spreads across the grid.
        test_requant(1,   8, 12, "RQ-T1");   // scale = 1/512: mostly in range
        test_requant(4,   8, 12, "RQ-T4");
        test_requant(4,  16, 12, "RQ-T4b");  // 1/256: straddles the clip
        test_requant(8, 205,  8, "RQ-T8");   // 0.80: saturates, tests the clip
        test_tiled(1, "TILE-T1");
        test_tiled(4, "TILE-T4");

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("MXU: ALL TESTS PASSED");
        else             $display("MXU: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog.
    initial begin
        #500000;
        $display("MXU: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
