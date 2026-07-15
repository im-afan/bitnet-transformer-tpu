`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// mxu.sv — TPU weight-stationary ternary systolic matrix unit
//
// Implements the block described in accel/tpu/docs/mxu.md: a ROWS x COLS
// systolic array that computes the transformer's ternary-weight x int8-activation
// projection GEMMs (Wq/Wk/Wv/Wo, FF fc1/fc2). Weights are stationary in the PEs;
// int8 activations stream left->right; int32 partial sums flow top->bottom. With
// ternary weights w in {-1,0,+1} each PE is a select+conditional-negate add, not a
// multiplier (mxu.md §2) — see `macw` below.
//
// Dispatched by the scalar unit (scalar_unit.sv OP_MATMUL) with the issue-and-wait
// handshake: pulse `start` with {act_addr, weight_addr, out_addr, t_len,
// accumulate}, then block on `done` (mxu.md §7).
//
//   Matmul:  C[T x COLS] = A[T x d] @ W[d x COLS],   d = ROWS
//
// Phases (mxu.md §4), sequenced by the FSM:
//   LOAD  stream the d x COLS ternary tile into the PE weight registers, one
//         array-column (ROWS trits, 2-bit packed = W_rd width) per clock.
//   RUN   feed one activation token vector (d int8 = A_rd width) per clock into
//         the array, injected staggered across rows by the input-skew buffer
//         (row i delayed i cycles) so every contribution to an output element
//         lines up on the descending partial sum. Column sums fall out the bottom
//         after ROWS(+skew) cycles; each is scattered into `resbuf[token][col]`
//         (the result buffer absorbs the output skew, so no de-skew shift regs).
//   WB    write the T result rows to out_addr as int32. With `accumulate` the
//         existing int32 partials are read back and summed first (contraction
//         tiling, mxu.md §6).
//
// Numerics (mxu.md §5). Accumulate in int32 — worst case d*127 ~= 16k is well
// inside int32, no mid-accumulate saturation. The store path optionally
// **requantizes on store** (BitNet int32 -> int8): when `requant` is asserted the
// final int32 element is rescaled by the fixed-point pair {M0, N} as
// `clip((acc*M0 + round) >> N)` (the same math as requant.sv / vpu.sv
// VOP_REQUANT) and written as int8; the {M0, N} word is read once from
// `scalar_addr` over the C port. When `requant` is clear the MXU writes int32 as
// before — this is the mode intermediate contraction tiles use so `accumulate`
// can keep running int32 partials in the result bank (mxu.md §6); the final tile
// asserts `requant` to narrow. The int8 output row is COLS bytes (C_wstrb masks
// the upper lanes), consistent with "the scratchpad stores whatever width the
// writer produces" (scratchpad.md §2).
//
// Memory / scratchpad ports (scratchpad.md §4). Synchronous reads: address/enable
// presented one cycle, data valid the next (same contract as vpu.sv / scalar_unit).
//   A_rd : one activation token vector per clock  (ROWS x int8  = ROWS*8 bits)
//   W_rd : one weight array-column per clock       (ROWS trits  = ROWS*2 bits)
//   C_rw : one int32 result row per clock          (COLS x int32 = COLS*32 bits)
//          The doc lists C as a write port (C_wr); the accumulate/tiling path
//          reads back the running partial, so it is exposed here as a read/write
//          C_rw pair. Non-accumulate matmuls never assert C_re.
//
// Scratchpad tile layouts assumed by the address arithmetic below:
//   Activations  row-major  A[t][i] at act_addr    + t*ROWS + i           (int8)
//   Weights      col-major  W[i][j] at weight_addr + j*(ROWS*2/8), 2-bit
//                packed within the column: row i in bits [i*2 +: 2] (col-major so a
//                whole array-column loads in one W_rd). Encoding (scratchpad.md §2):
//                00 = 0, 01 = +1, 11 = -1.
//   Results      row-major  C[t][j] at out_addr     + t*(COLS*4) + j*4     (int32)
//
// Validate bit-exactly against TernaryLinear in model/transformer.py: the array
// computes the integer product A_int @ W_ternary (mxu.md §5); the model's absmean
// scale is folded in by the separate requant step.
//
// Skew note (mxu.md §8 open question): the input stagger uses dedicated triangular
// shift registers (sk_*). Output skew is absorbed by scattering each drained
// column result into resbuf by its propagated token id, so no output de-skew
// registers are needed.
// -----------------------------------------------------------------------------

module mxu #(
    parameter int ROWS   = 128,  // contraction dim d (rows = one activation elem each)
    parameter int COLS   = 128,  // output features (cols = one stationary weight col)
    parameter int ADDR_W = 16,   // scratchpad byte-address width
    parameter int M0_W   = 12,   // requant fixed-point multiplier width (requant.sv)
    parameter int N_W    = 4     // requant shift width
) (
    input  logic clk,
    input  logic rst_n,

    // ---- Dispatch from the scalar unit (mxu.md §7) --------------------------
    input  logic              start,
    input  logic [ADDR_W-1:0] act_addr,     // activation tile base
    input  logic [ADDR_W-1:0] weight_addr,  // ternary weight tile base
    input  logic [ADDR_W-1:0] out_addr,     // result base (int32, or int8 if requant)
    input  logic [ADDR_W-1:0] scalar_addr,  // requant {M0, N} word (used when requant)
    input  logic [5:0]        t_len,        // number of token columns T
    input  logic              accumulate,   // add into existing int32 result (tiling)
    input  logic              requant,      // narrow store int32 -> int8 via {M0, N}
    output logic              busy,
    output logic              done,

    // ---- Scratchpad A_rd port (activation feed, ROWS x int8) ----------------
    output logic                A_re,
    output logic [ADDR_W-1:0]   A_raddr,
    input  logic [ROWS*8-1:0]   A_rdata,     // valid the cycle after A_re

    // ---- Scratchpad W_rd port (weight load, ROWS x 2-bit) -------------------
    output logic                W_re,
    output logic [ADDR_W-1:0]   W_raddr,
    input  logic [ROWS*2-1:0]   W_rdata,     // valid the cycle after W_re

    // ---- Scratchpad C_rw port (int32 result row, COLS x int32) --------------
    // Also carries the accumulate readback and the one-shot requant {M0, N}
    // fetch. Writes are per-byte masked by C_wstrb so an int8 requant row
    // (COLS bytes) does not clobber the neighbouring int32-width lanes.
    output logic                C_re,
    output logic [ADDR_W-1:0]   C_raddr,
    input  logic [COLS*32-1:0]  C_rdata,     // valid the cycle after C_re
    output logic                C_we,
    output logic [ADDR_W-1:0]   C_waddr,
    output logic [COLS*32-1:0]  C_wdata,
    output logic [COLS*4-1:0]   C_wstrb      // per-byte write enable
);

    // -------------------------------------------------------------------------
    // Local widths / derived constants.
    // -------------------------------------------------------------------------
    localparam int ACT_W      = 8;
    localparam int WT_W       = 2;
    localparam int PSUM_W     = 32;
    localparam int TOK_W      = 6;                 // matches t_len width (<= 63 tokens)
    localparam int MAXT       = (1 << TOK_W);      // result-buffer token depth
    localparam int WCOL_BYTES = (ROWS * WT_W) / 8; // bytes per packed weight column
    localparam int RES_STRIDE = COLS * 4;          // bytes per int32 result row
    localparam int RQ_STRIDE  = COLS;              // bytes per int8 (requant) result row
    localparam int RQ_W       = 48;                // requant intermediate width (headroom)

    // int32 -> int8 requantize: clip((acc*m0 + round) >> n). Mirrors requant.sv /
    // vpu.sv requant8 (signed accumulator, int8 clip). m0 is a positive scale.
    function automatic logic [7:0] requant8(input logic signed [PSUM_W-1:0] acc32,
                                            input logic [M0_W-1:0]          m0,
                                            input logic [N_W-1:0]           n);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = acc32 * $signed({1'b0, m0});
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;            // arithmetic (signed) shift
        if (shifted >  127)      requant8 =  8'sd127;
        else if (shifted < -128) requant8 = -8'sd128;
        else                     requant8 = shifted[7:0];
    endfunction

    // Ternary MAC: select + conditional negate (mxu.md §2). w[0]=nonzero, w[1]=sign.
    function automatic logic signed [PSUM_W-1:0] macw(input logic signed [ACT_W-1:0] a,
                                                      input logic [1:0]              w);
        logic signed [PSUM_W-1:0] ae;
        ae = PSUM_W'(a);
        if (!w[0]) macw = '0;
        else       macw = w[1] ? -ae : ae;
    endfunction

    // -------------------------------------------------------------------------
    // FSM.
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_RUN, S_SCL_RD, S_SCL_RDD, S_WB_RD, S_WB_WR, S_DONE
    } state_t;
    state_t state, state_n;

    // Latched dispatch parameters.
    logic [ADDR_W-1:0] act_a, wgt_a, out_a, scl_a;
    logic [5:0]        T_r;
    logic              acc_r, req_r;
    logic [M0_W-1:0]   rq_m0;       // requant multiplier (loaded from scalar_addr)
    logic [N_W-1:0]    rq_n;        // requant shift
    logic [15:0]       expected;    // total bottom completions = T * COLS

    // -------------------------------------------------------------------------
    // PE array. Weights stationary; a-stream (value+token+valid) flows right;
    // int32 partial sums flow down.
    // -------------------------------------------------------------------------
    logic [WT_W-1:0]           w_reg    [0:ROWS-1][0:COLS-1];
    logic signed [ACT_W-1:0]   a_pipe   [0:ROWS-1][0:COLS-1];
    logic [TOK_W-1:0]          tok_pipe [0:ROWS-1][0:COLS-1];
    logic                      av_pipe  [0:ROWS-1][0:COLS-1];
    logic signed [PSUM_W-1:0]  psum     [0:ROWS-1][0:COLS-1];

    // Input skew: sk_*[i][k] = the row-i feed delayed by (k+1) cycles. Row i's
    // left-edge injection reads delay i (sk_*[i][i-1]); row 0 injects with no delay.
    logic signed [ACT_W-1:0]   sk_a [0:ROWS-1][0:ROWS-1];
    logic [TOK_W-1:0]          sk_t [0:ROWS-1][0:ROWS-1];
    logic                      sk_v [0:ROWS-1][0:ROWS-1];

    // Left-edge values presented to the array this cycle (combinational).
    logic signed [ACT_W-1:0]   aL   [0:ROWS-1];
    logic [TOK_W-1:0]          tokL [0:ROWS-1];
    logic                      avL  [0:ROWS-1];

    // Result buffer: drained column sums scattered here by token id, then drained
    // to scratchpad in the writeback phase.
    logic signed [PSUM_W-1:0]  resbuf [0:MAXT-1][0:COLS-1];

    // Weight-load pipeline.
    logic [15:0] wreq, wrcv;   // columns requested / received
    logic        w_last_re;    // a W_rd was issued last cycle (data valid now)

    // Activation-feed pipeline.
    logic [15:0] arq, inj, n_done;
    logic        a_last_re;    // an A_rd was issued last cycle (data valid now)

    // Writeback token pointer.
    logic [15:0] wb_t;

    // -------------------------------------------------------------------------
    // Left-edge feed (combinational). Row 0 injects the just-arrived vector with
    // no delay; row i reads its skew chain at delay i.
    // -------------------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < ROWS; i++) begin
            if (i == 0) begin
                aL[0]   = A_rdata[0 +: ACT_W];
                tokL[0] = inj[TOK_W-1:0];
                avL[0]  = a_last_re;
            end else begin
                aL[i]   = sk_a[i][i-1];
                tokL[i] = sk_t[i][i-1];
                avL[i]  = sk_v[i][i-1];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Scratchpad port drive (combinational per state).
    // -------------------------------------------------------------------------
    logic signed [PSUM_W-1:0] wb_elem;   // final int32 element for the current lane
    always_comb begin
        A_re    = 1'b0; A_raddr = '0;
        W_re    = 1'b0; W_raddr = '0;
        C_re    = 1'b0; C_raddr = '0;
        C_we    = 1'b0; C_waddr = '0; C_wdata = '0; C_wstrb = '0;
        wb_elem = '0;

        unique case (state)
            S_LOAD: begin
                W_re    = (wreq < COLS);
                W_raddr = wgt_a + ADDR_W'(wreq * WCOL_BYTES);
            end
            S_RUN: begin
                A_re    = (arq < {10'b0, T_r});
                A_raddr = act_a + ADDR_W'(arq * ROWS);
            end
            S_SCL_RD: begin           // one-shot fetch of the {M0, N} word
                C_re    = 1'b1;
                C_raddr = scl_a;
            end
            S_WB_RD: begin            // accumulate readback (int32 running partial)
                C_re    = 1'b1;
                C_raddr = out_a + ADDR_W'(wb_t * RES_STRIDE);
            end
            S_WB_WR: begin
                C_we    = 1'b1;
                // int8 requant rows are COLS bytes; int32 rows are COLS*4 bytes.
                C_waddr = out_a + ADDR_W'(wb_t * (req_r ? RQ_STRIDE : RES_STRIDE));
                for (int j = 0; j < COLS; j++) begin
                    wb_elem = resbuf[wb_t[TOK_W-1:0]][j]
                            + (acc_r ? $signed(C_rdata[j*32 +: 32]) : '0);
                    if (req_r) begin
                        C_wdata[j*8 +: 8] = requant8(wb_elem, rq_m0, rq_n);
                        C_wstrb[j]        = 1'b1;              // one int8 byte per col
                    end else begin
                        C_wdata[j*32 +: 32] = wb_elem;
                        C_wstrb[j*4 +: 4]   = 4'b1111;        // full int32 lane
                    end
                end
            end
            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // Next-state logic.
    // -------------------------------------------------------------------------
    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:  if (start) begin
                         if (t_len == 0) state_n = S_DONE;
                         else            state_n = S_LOAD;
                     end
            S_LOAD:  if (wrcv == COLS)       state_n = S_RUN;
            // Requant fetches {M0, N} once before draining the result buffer.
            S_RUN:   if (n_done == expected) begin
                         if      (req_r) state_n = S_SCL_RD;
                         else if (acc_r) state_n = S_WB_RD;
                         else            state_n = S_WB_WR;
                     end
            S_SCL_RD:                        state_n = S_SCL_RDD;
            S_SCL_RDD: if (acc_r) state_n = S_WB_RD;
                       else       state_n = S_WB_WR;
            S_WB_RD:                         state_n = S_WB_WR;
            S_WB_WR: if (wb_t == (16'(T_r) - 16'd1)) state_n = S_DONE;
                     else if (acc_r)         state_n = S_WB_RD;
                     else                    state_n = S_WB_WR;
            S_DONE:                          state_n = S_IDLE;
            default:                         state_n = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential datapath.
    // -------------------------------------------------------------------------
    integer i, j, k, nb;
    logic signed [ACT_W-1:0]  a_l;
    logic [TOK_W-1:0]         t_l;
    logic                     av_l;
    logic signed [PSUM_W-1:0] ptop, ns;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            wreq <= '0; wrcv <= '0; w_last_re <= 1'b0;
            arq  <= '0; inj  <= '0; n_done <= '0; a_last_re <= 1'b0;
            wb_t <= '0;
            // Clear valids so no phantom completions before the first feed.
            for (i = 0; i < ROWS; i++) begin
                for (j = 0; j < COLS; j++) av_pipe[i][j] <= 1'b0;
                for (k = 0; k < ROWS; k++) sk_v[i][k]    <= 1'b0;
            end
        end else begin
            state <= state_n;

            unique case (state)
                // -------------------------------------------------------------
                S_IDLE: if (start && t_len != 0) begin
                    act_a    <= act_addr;
                    wgt_a    <= weight_addr;
                    out_a    <= out_addr;
                    scl_a    <= scalar_addr;
                    T_r      <= t_len;
                    acc_r    <= accumulate;
                    req_r    <= requant;
                    expected <= 16'(t_len) * 16'(COLS);
                    wreq <= '0; wrcv <= '0; w_last_re <= 1'b0;
                    arq  <= '0; inj  <= '0; n_done <= '0; a_last_re <= 1'b0;
                    wb_t <= '0;
                    // Fresh matmul: clear a-stream / skew valids.
                    for (i = 0; i < ROWS; i++) begin
                        for (j = 0; j < COLS; j++) av_pipe[i][j] <= 1'b0;
                        for (k = 0; k < ROWS; k++) sk_v[i][k]    <= 1'b0;
                    end
                end

                // -------------------------------------------------------------
                // Weight load: one array-column (ROWS trits) per clock.
                S_LOAD: begin
                    if (wreq < COLS) wreq <= wreq + 16'd1;
                    w_last_re <= (wreq < COLS);
                    if (w_last_re) begin
                        for (i = 0; i < ROWS; i++)
                            w_reg[i][wrcv[$clog2(COLS)-1:0]] <= W_rdata[i*WT_W +: WT_W];
                        wrcv <= wrcv + 16'd1;
                    end
                end

                // -------------------------------------------------------------
                // Feed + drain.
                S_RUN: begin
                    // Activation read pipeline: issue one token vector per clock;
                    // its data (and injection) lands the following cycle.
                    if (arq < {10'b0, T_r}) arq <= arq + 16'd1;
                    a_last_re <= (arq < {10'b0, T_r});
                    if (a_last_re) inj <= inj + 16'd1;

                    // Input-skew shift registers (triangular; row i => delay i).
                    for (i = 0; i < ROWS; i++) begin
                        sk_a[i][0] <= A_rdata[i*ACT_W +: ACT_W];
                        sk_t[i][0] <= inj[TOK_W-1:0];
                        sk_v[i][0] <= a_last_re;
                        for (k = 1; k < ROWS; k++) begin
                            sk_a[i][k] <= sk_a[i][k-1];
                            sk_t[i][k] <= sk_t[i][k-1];
                            sk_v[i][k] <= sk_v[i][k-1];
                        end
                    end

                    // Systolic PE update. Each PE uses its incoming (left/top)
                    // operands so a-value and psum advance one hop per cycle.
                    nb = 0;
                    for (i = 0; i < ROWS; i++) begin
                        for (j = 0; j < COLS; j++) begin
                            a_l  = (j == 0) ? aL[i]   : a_pipe[i][j-1];
                            t_l  = (j == 0) ? tokL[i] : tok_pipe[i][j-1];
                            av_l = (j == 0) ? avL[i]  : av_pipe[i][j-1];
                            ptop = (i == 0) ? '0      : psum[i-1][j];
                            ns   = ptop + macw(a_l, w_reg[i][j]);

                            a_pipe[i][j]   <= a_l;
                            tok_pipe[i][j] <= t_l;
                            av_pipe[i][j]  <= av_l;
                            psum[i][j]     <= ns;

                            // Bottom row: a completed column dot product. Scatter
                            // into the result buffer by its propagated token id.
                            if (i == ROWS-1 && av_l) begin
                                resbuf[t_l][j] <= ns;
                                nb = nb + 1;
                            end
                        end
                    end
                    n_done <= n_done + nb[15:0];
                end

                // -------------------------------------------------------------
                // Requant {M0, N} lands the cycle after S_SCL_RD (same C-port
                // read contract as the accumulate readback).
                S_SCL_RDD: begin
                    rq_m0 <= C_rdata[M0_W-1:0];
                    rq_n  <= C_rdata[M0_W +: N_W];
                end

                // -------------------------------------------------------------
                // Writeback: one result row per token (int32, or int8 if requant).
                S_WB_WR: wb_t <= wb_t + 16'd1;

                default: ; // S_WB_RD (C_re asserted comb), S_SCL_RD, S_DONE
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Status.
    // -------------------------------------------------------------------------
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

endmodule
