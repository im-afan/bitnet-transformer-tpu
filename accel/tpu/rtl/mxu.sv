`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// mxu.sv — TPU weight-stationary ternary systolic matrix unit
//
// Implements the block described in accel/tpu/docs/mxu.md: a ROWS x COLS
// systolic array that computes the transformer's ternary-weight x int8-activation
// projection GEMMs (Wq/Wk/Wv/Wo, FF fc1/fc2). Weights are stationary in the PEs;
// int8 activations stream left->right; int32 partial sums flow top->bottom. With
// ternary weights w in {-1,0,+1} each PE is a select+conditional-negate add, not a
// multiplier (mxu.md §2) — see `ternary_product` below.
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
//         after ROWS(+skew) cycles; each is scattered into
//         `result_buf[token][col]` (the result buffer absorbs the output skew, so
//         no de-skew shift regs).
//   WB    write the T result rows to out_addr as int32. With `accumulate` the
//         existing int32 partials are read back and summed first (contraction
//         tiling, mxu.md §6).
//
// Numerics (mxu.md §5). Accumulate in int32 — worst case d*127 ~= 16k is well
// inside int32, no mid-accumulate saturation. The store path optionally
// **requantizes on store** (BitNet int32 -> int8): when `requant` is asserted the
// final int32 element is rescaled by the fixed-point pair {M0, N} as
// `clip((acc*M0 + round) >> N)` (the same math as requant.sv / vpu.sv
// VOP_REQUANT) and written as int8; the {M0, N} word arrives in
// the dispatch's own `rq_word` field. When `requant` is clear the MXU writes int32 as
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
// shift registers (skew_*). Output skew is absorbed by scattering each drained
// column result into result_buf by its propagated token id, so no output de-skew
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
    // Requant {M0, N} as a LITERAL, not a scratchpad address. It is M0_W + N_W
    // = 16 bits, exactly the width of the address that used to point at it, so
    // the 128-bit command it now rides in is no tighter for it
    // (docs/picorv32_migration.md §3). What goes away is the one-shot C-port
    // fetch this unit used to run before every requantized store: two states,
    // two scratchpad reads, and one more contender on the arbitrated read port
    // per matmul.
    input  logic [M0_W+N_W-1:0] rq_word,    // {n, m0} (used when requant)
    input  logic [5:0]        t_len,        // number of token columns T
    input  logic              accumulate,   // add into existing int32 result (tiling)
    input  logic              requant,      // narrow store int32 -> int8 via {M0, N}

    // ---- Operand strides (config registers; docs/macro_ops.md §4.1) ---------
    // The three strides that used to be compile-time constants, because each one
    // assumed the operand *was* one tile. A tile of a larger matrix has the
    // larger matrix's stride, so hardware tiling cannot work until these are
    // parameters.
    //
    // `tiled` SELECTS THEM. With `tiled` low the strides are ignored outright
    // and the single-tile constants apply (a_row=ROWS, c_row=COLS*4,
    // w_col=ROWS*2/8), which is exactly the behaviour they replaced. That is not
    // just for backward compatibility: config registers survive across runs (no
    // reset between programs), so a plain matmul keyed off "the strides happen
    // to be zero" would silently inherit whatever the *previous* program left in
    // them. Making the instruction say which mode it is in removes the hazard
    // rather than documenting it. Within tiled mode a zero stride still falls
    // back to the single-tile value, purely so an unset register degrades to
    // something sane instead of addressing everything at offset zero.
    //
    // `c_row` is the **int32** row stride. A requantized store writes int8 and
    // therefore steps `c_row/4`, preserving the exact RQ_ROW_BYTES =
    // RES_ROW_BYTES/4 relationship these were hardcoded to. One register, both
    // strides — which matters because `.acc.rq` uses *both* in the same dispatch:
    // it reads int32 partials back at the wide stride and stores int8 at the
    // narrow one.
    input  logic              tiled,        // use the config strides below
    input  logic [ADDR_W-1:0] a_row,        // activation row stride, bytes (= K)
    input  logic [ADDR_W-1:0] c_row,        // int32 result row stride, bytes (= N*4)
    input  logic [ADDR_W-1:0] w_col,        // weight column stride, bytes (= K*2/8)

    // ---- Tile counts (config; only with `tiled`) ----------------------------
    // How many array-sized tiles the operands span. With both 1 (or 0, which is
    // read as 1) this is a single-tile matmul and the loop below runs once.
    //
    // The loop order is **n outer, k inner**, which is what makes the
    // contraction free: partial sums stay in `result_buf` across the whole k
    // loop and only reach the scratchpad once, when the n-tile is finished.
    // Nothing reads or writes the C port during the contraction, so `.acc`'s
    // readback path is not on the critical path of a tiled matmul at all — it
    // survives only for accumulating into a *pre-existing* C.
    input  logic [7:0]        k_tiles,      // contraction tiles = K / ROWS
    input  logic [7:0]        n_tiles,      // output tiles      = N / COLS
    output logic              busy,
    output logic              done,
    // High during the weight-load phase (S_LOAD). Broken out so perf_counters.sv
    // can separate weight-load time from streaming time over a run — the
    // measurement that says whether double-buffering the PE weight registers
    // (docs/macro_ops.md §4.5) would pay, and the one that shows why decode with
    // t_len=1 is weight-load bound. Debug/telemetry only.
    output logic              load_active,

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
    localparam int ACT_W         = 8;                 // activation element, int8
    localparam int WT_W          = 2;                 // weight element, packed trit
    localparam int PSUM_W        = 16;                // partial-sum accumulator, int32
    localparam int TOK_W         = 5;                 // EXPERIMENT: 32 token slots
    localparam int MAX_TOKENS    = (1 << TOK_W);      // result-buffer token depth
    localparam int WGT_COL_BYTES = (ROWS * WT_W) / 8; // bytes per packed weight column
    localparam int RES_ROW_BYTES = COLS * 4;          // bytes per int32 result row
    localparam int RQ_ROW_BYTES  = COLS;              // bytes per int8 (requant) result row
    localparam int REQUANT_W     = 48;                // requant intermediate width (headroom)

    // int32 -> int8 requantize: clip((acc*m0 + round) >> n). Mirrors requant.sv /
    // vpu.sv requant8 (signed accumulator, int8 clip). m0 is a positive scale.
    function automatic logic [7:0] requant8(input logic signed [PSUM_W-1:0] acc_i32,
                                            input logic [M0_W-1:0]          mult_m0,
                                            input logic [N_W-1:0]           shift_n);
        logic signed [REQUANT_W-1:0] product, round_bias, shifted;
        product    = acc_i32 * $signed({1'b0, mult_m0});
        round_bias = (shift_n == 0) ? '0 : (REQUANT_W'(1) <<< (shift_n - 1));
        shifted    = (product + round_bias) >>> shift_n;  // arithmetic (signed) shift
        if (shifted >  127)      requant8 =  8'sd127;
        else if (shifted < -128) requant8 = -8'sd128;
        else                     requant8 = shifted[7:0];
    endfunction

    // Ternary product: select + conditional negate, no multiplier (mxu.md §2).
    // wt[0] = nonzero flag, wt[1] = sign. Encoding 00 = 0, 01 = +1, 11 = -1.
    function automatic logic signed [PSUM_W-1:0] ternary_product(
            input logic signed [ACT_W-1:0] act,
            input logic [1:0]              wt);
        logic signed [PSUM_W-1:0] act_ext;
        act_ext = PSUM_W'(act);
        if (!wt[0]) ternary_product = '0;
        else        ternary_product = wt[1] ? -act_ext : act_ext;
    endfunction

    // -------------------------------------------------------------------------
    // FSM.
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,           // waiting for a dispatch
        S_LOAD,           // streaming a weight tile into the PE registers
        S_RUN,            // feeding activations / draining column sums
        S_WB_ACC_RD,      // accumulate readback of one existing int32 result row
        S_WB_WRITE,       // write one result row out
        S_DONE            // one-cycle completion pulse
    } state_t;
    state_t state, state_n;

    // Latched dispatch parameters.
    logic [ADDR_W-1:0] act_base, wgt_base, out_base;
    // Effective strides, resolved once at dispatch (zero -> single-tile default).
    // Latched rather than read live so a dispatch cannot straddle a config write;
    // under issue-and-wait it cannot anyway, but the address generators below
    // then match the rest of the latched dispatch state.
    logic [ADDR_W-1:0] act_row_stride;   // bytes between activation rows
    logic [ADDR_W-1:0] res_row_stride;   // bytes between int32 result rows
    logic [ADDR_W-1:0] wgt_col_stride;   // bytes between packed weight columns
    // int8 store stride for a requantized writeback: res_row_stride/4 (see the
    // port comment). Only meaningful when requant_mode.
    wire  [ADDR_W-1:0] res_row_stride_i8 = res_row_stride >> 2;

    // ---- Tile loop state ----------------------------------------------------
    logic [7:0]        k_tile_count, n_tile_count;  // latched tile counts (>= 1)
    logic [7:0]        k_tile_idx,   n_tile_idx;    // current contraction / output tile
    // Running operand bases, stepped at tile boundaries rather than recomputed
    // with a multiplier per access.
    logic [ADDR_W-1:0] act_tile_base;    // act_base + k_tile_idx*ROWS
    logic [ADDR_W-1:0] wgt_ntile_base;   // wgt_base + n_tile_idx*COLS*w_col (n-tile origin)
    logic [ADDR_W-1:0] wgt_tile_base;    // wgt_ntile_base + k_tile_idx*WGT_COL_BYTES
    logic [ADDR_W-1:0] out_tile_base;    // out_base + n_tile_idx*COLS*(4 or 1)
    logic [ADDR_W-1:0] wgt_ntile_step;   // COLS*wgt_col_stride, computed once at dispatch
    logic [ADDR_W-1:0] out_ntile_step;   // COLS*4 (int32) or COLS (requant int8)

    // Stride selection, as combinational wires so the dispatch latch and the
    // derived per-tile steps below cannot disagree about how a stride resolved.
    wire [ADDR_W-1:0] act_row_stride_sel =
        (!tiled || a_row == '0) ? ADDR_W'(ROWS)           : a_row;
    wire [ADDR_W-1:0] res_row_stride_sel =
        (!tiled || c_row == '0) ? ADDR_W'(RES_ROW_BYTES)  : c_row;
    wire [ADDR_W-1:0] wgt_col_stride_sel =
        (!tiled || w_col == '0) ? ADDR_W'(WGT_COL_BYTES)  : w_col;

    wire k_tile_last = (k_tile_idx == k_tile_count - 8'd1);
    wire n_tile_last = (n_tile_idx == n_tile_count - 8'd1);
    // First contraction tile of an n-tile initialises result_buf; later ones add
    // to it. That is the entire cost of accumulating across the contraction — no
    // clear pass, no extra state.
    wire k_tile_first = (k_tile_idx == 8'd0);

    logic [5:0]        tok_count;      // T, tokens in this dispatch
    logic              acc_mode;       // latched `accumulate`
    logic              requant_mode;   // latched `requant`
    logic [M0_W-1:0]   rq_mult_m0;     // requant multiplier (latched from rq_word)
    logic [N_W-1:0]    rq_shift_n;     // requant shift
    logic [15:0]       expected_completions;  // total bottom completions = T * COLS

    // -------------------------------------------------------------------------
    // PE array. Weights stationary; a-stream (value+token+valid) flows right;
    // int32 partial sums flow down.
    // -------------------------------------------------------------------------
    logic [WT_W-1:0]           pe_weight    [0:ROWS-1][0:COLS-1];
    logic signed [ACT_W-1:0]   pe_act       [0:ROWS-1][0:COLS-1];
    logic [TOK_W-1:0]          pe_tok       [0:ROWS-1][0:COLS-1];
    logic                      pe_act_valid [0:ROWS-1][0:COLS-1];
    logic signed [PSUM_W-1:0]  pe_psum      [0:ROWS-1][0:COLS-1];

    // Input skew: skew_*[i][k] = the row-i feed delayed by (k+1) cycles. Row i's
    // left-edge injection reads delay i (skew_*[i][i-1]); row 0 injects with no
    // delay. Declared square but only the k <= i-1 triangle is ever read.
    logic signed [ACT_W-1:0]   skew_act   [0:ROWS-1][0:ROWS-1];
    logic [TOK_W-1:0]          skew_tok   [0:ROWS-1][0:ROWS-1];
    logic                      skew_valid [0:ROWS-1][0:ROWS-1];

    // Left-edge values presented to the array this cycle (combinational).
    logic signed [ACT_W-1:0]   edge_act   [0:ROWS-1];
    logic [TOK_W-1:0]          edge_tok   [0:ROWS-1];
    logic                      edge_valid [0:ROWS-1];

    // Result buffer: drained column sums scattered here by token id, then drained
    // to scratchpad in the writeback phase.
    logic signed [PSUM_W-1:0]  result_buf [0:MAX_TOKENS-1][0:COLS-1];

    // Weight-load pipeline. Two pointers, one cycle apart, because the scratchpad
    // read has one cycle of latency: req counts what has been asked for, rcv what
    // has landed, and the inflight flag bridges the two.
    logic [15:0] wgt_req_cnt, wgt_rcv_cnt;
    logic        wgt_rd_inflight;   // a W_rd was issued last cycle (data valid now)

    // Activation-feed pipeline (same two-pointer structure).
    logic [15:0] act_req_cnt;       // token vectors requested
    logic [15:0] inj_tok_idx;       // token id of the vector being injected now
    logic [15:0] completions;       // bottom-row results scattered so far this tile
    logic        act_rd_inflight;   // an A_rd was issued last cycle (data valid now)

    // Writeback token pointer.
    logic [15:0] wb_tok_idx;

    // -------------------------------------------------------------------------
    // Left-edge feed (combinational). Row 0 injects the just-arrived vector with
    // no delay; row i reads its skew chain at delay i.
    // -------------------------------------------------------------------------
    always_comb begin
        for (int row_i = 0; row_i < ROWS; row_i++) begin
            if (row_i == 0) begin
                edge_act[0]   = A_rdata[0 +: ACT_W];
                edge_tok[0]   = inj_tok_idx[TOK_W-1:0];
                edge_valid[0] = act_rd_inflight;
            end else begin
                edge_act[row_i]   = skew_act[row_i][row_i-1];
                edge_tok[row_i]   = skew_tok[row_i][row_i-1];
                edge_valid[row_i] = skew_valid[row_i][row_i-1];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Scratchpad port drive (combinational per state).
    // -------------------------------------------------------------------------
    logic signed [PSUM_W-1:0] wb_elem_i32;   // final int32 element for the current lane
    always_comb begin
        A_re    = 1'b0; A_raddr = '0;
        W_re    = 1'b0; W_raddr = '0;
        C_re    = 1'b0; C_raddr = '0;
        C_we    = 1'b0; C_waddr = '0; C_wdata = '0; C_wstrb = '0;
        wb_elem_i32 = '0;

        unique case (state)
            S_LOAD: begin
                W_re    = (wgt_req_cnt < COLS);
                W_raddr = wgt_tile_base + ADDR_W'(wgt_req_cnt * wgt_col_stride);
            end
            S_RUN: begin
                A_re    = (act_req_cnt < {10'b0, tok_count});
                A_raddr = act_tile_base + ADDR_W'(act_req_cnt * act_row_stride);
            end
            S_WB_ACC_RD: begin        // accumulate readback (int32 running partial)
                C_re    = 1'b1;
                // always the int32 stride: the partials are int32 whatever the
                // store width ends up being.
                C_raddr = out_tile_base + ADDR_W'(wb_tok_idx * res_row_stride);
            end
            S_WB_WRITE: begin
                C_we    = 1'b1;
                // int8 requant rows step res_row_stride/4; int32 rows step the full one.
                C_waddr = out_tile_base
                        + ADDR_W'(wb_tok_idx * (requant_mode ? res_row_stride_i8
                                                             : res_row_stride));
                for (int col_j = 0; col_j < COLS; col_j++) begin
                    wb_elem_i32 = result_buf[wb_tok_idx[TOK_W-1:0]][col_j]
                                + (acc_mode ? $signed(C_rdata[col_j*32 +: 32]) : '0);
                    if (requant_mode) begin
                        C_wdata[col_j*8 +: 8] = requant8(wb_elem_i32, rq_mult_m0, rq_shift_n);
                        C_wstrb[col_j]        = 1'b1;          // one int8 byte per col
                    end else begin
                        C_wdata[col_j*32 +: 32] = wb_elem_i32;
                        C_wstrb[col_j*4 +: 4]   = 4'b1111;     // full int32 lane
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
            S_LOAD:  if (wgt_rcv_cnt == COLS) state_n = S_RUN;
            // End of a contraction tile. If more remain, go straight back to
            // LOAD for the next weight tile with the partials still sitting in
            // result_buf — the scratchpad is not touched between tiles. Only
            // after the last one does the result drain out. The requant {M0, N}
            // arrives in the dispatch itself, so there is nothing to fetch
            // before the drain any more.
            S_RUN:   if (completions == expected_completions) begin
                         if      (!k_tile_last)  state_n = S_LOAD;
                         else if (acc_mode)      state_n = S_WB_ACC_RD;
                         else                    state_n = S_WB_WRITE;
                     end
            S_WB_ACC_RD:                         state_n = S_WB_WRITE;
            // Output tile finished. If more remain, restart the contraction for
            // the next one; otherwise the whole matmul is done.
            S_WB_WRITE: if (wb_tok_idx == (16'(tok_count) - 16'd1)) begin
                            // if/else rather than a ternary: mixing two enum
                            // values in a conditional expression loses the enum
                            // type and needs an explicit cast.
                            if (n_tile_last) state_n = S_DONE;
                            else             state_n = S_LOAD;
                        end
                        else if (acc_mode)       state_n = S_WB_ACC_RD;
                        else                     state_n = S_WB_WRITE;
            S_DONE:                              state_n = S_IDLE;
            default:                             state_n = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential datapath.
    // -------------------------------------------------------------------------
    integer row_i, col_j, dly_k;      // loop indices: array row, array column, skew stage
    integer bottom_hits;              // bottom-row results produced this cycle

    // Operands entering the PE currently being evaluated, and the sum leaving it.
    logic signed [ACT_W-1:0]  in_act;    // from the left neighbour (or the array edge)
    logic [TOK_W-1:0]         in_tok;    // token id riding along with in_act
    logic                     in_valid;  // in_act carries a real activation
    logic signed [PSUM_W-1:0] in_psum;   // from the row above (0 at the top row)
    logic signed [PSUM_W-1:0] out_psum;  // in_psum + this PE's ternary product

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            wgt_req_cnt <= '0; wgt_rcv_cnt <= '0; wgt_rd_inflight <= 1'b0;
            act_req_cnt <= '0; inj_tok_idx <= '0; completions <= '0;
            act_rd_inflight <= 1'b0;
            wb_tok_idx <= '0;
            // Clear valids so no phantom completions before the first feed.
            for (row_i = 0; row_i < ROWS; row_i++) begin
                for (col_j = 0; col_j < COLS; col_j++) pe_act_valid[row_i][col_j] <= 1'b0;
                for (dly_k = 0; dly_k < ROWS; dly_k++) skew_valid[row_i][dly_k]   <= 1'b0;
            end
        end else begin
            state <= state_n;

            unique case (state)
                // -------------------------------------------------------------
                S_IDLE: if (start && t_len != 0) begin
                    act_base       <= act_addr;
                    wgt_base       <= weight_addr;
                    out_base       <= out_addr;
                    rq_mult_m0     <= rq_word[M0_W-1:0];
                    rq_shift_n     <= rq_word[M0_W +: N_W];
                    tok_count      <= t_len;
                    acc_mode       <= accumulate;
                    requant_mode   <= requant;
                    act_row_stride <= act_row_stride_sel;
                    res_row_stride <= res_row_stride_sel;
                    wgt_col_stride <= wgt_col_stride_sel;

                    // Tile loop. A count of 0 reads as 1 so an unset config
                    // register is a plain single-tile matmul rather than a
                    // dispatch that computes nothing.
                    k_tile_count <= (tiled && k_tiles != 8'd0) ? k_tiles : 8'd1;
                    n_tile_count <= (tiled && n_tiles != 8'd0) ? n_tiles : 8'd1;
                    k_tile_idx   <= '0;
                    n_tile_idx   <= '0;
                    act_tile_base  <= act_addr;
                    wgt_ntile_base <= weight_addr;
                    wgt_tile_base  <= weight_addr;
                    out_tile_base  <= out_addr;
                    // Per-output-tile steps: one array width along each axis.
                    wgt_ntile_step <= ADDR_W'(COLS) * wgt_col_stride_sel;
                    out_ntile_step <= requant ? ADDR_W'(RQ_ROW_BYTES)
                                              : ADDR_W'(RES_ROW_BYTES);
                    expected_completions <= 16'(t_len) * 16'(COLS);
                    wgt_req_cnt <= '0; wgt_rcv_cnt <= '0; wgt_rd_inflight <= 1'b0;
                    act_req_cnt <= '0; inj_tok_idx <= '0; completions <= '0;
                    act_rd_inflight <= 1'b0;
                    wb_tok_idx <= '0;
                    // Fresh matmul: clear a-stream / skew valids.
                    for (row_i = 0; row_i < ROWS; row_i++) begin
                        for (col_j = 0; col_j < COLS; col_j++) pe_act_valid[row_i][col_j] <= 1'b0;
                        for (dly_k = 0; dly_k < ROWS; dly_k++) skew_valid[row_i][dly_k]   <= 1'b0;
                    end
                end

                // -------------------------------------------------------------
                // Weight load: one array-column (ROWS trits) per clock.
                S_LOAD: begin
                    if (wgt_req_cnt < COLS) wgt_req_cnt <= wgt_req_cnt + 16'd1;
                    wgt_rd_inflight <= (wgt_req_cnt < COLS);
                    if (wgt_rd_inflight) begin
                        for (row_i = 0; row_i < ROWS; row_i++)
                            pe_weight[row_i][wgt_rcv_cnt[$clog2(COLS)-1:0]]
                                <= W_rdata[row_i*WT_W +: WT_W];
                        wgt_rcv_cnt <= wgt_rcv_cnt + 16'd1;
                    end
                end

                // -------------------------------------------------------------
                // Feed + drain.
                S_RUN: begin
                    // Activation read pipeline: issue one token vector per clock;
                    // its data (and injection) lands the following cycle.
                    if (act_req_cnt < {10'b0, tok_count}) act_req_cnt <= act_req_cnt + 16'd1;
                    act_rd_inflight <= (act_req_cnt < {10'b0, tok_count});
                    if (act_rd_inflight) inj_tok_idx <= inj_tok_idx + 16'd1;

                    // Input-skew shift registers (triangular; row i => delay i).
                    for (row_i = 0; row_i < ROWS; row_i++) begin
                        skew_act[row_i][0]   <= A_rdata[row_i*ACT_W +: ACT_W];
                        skew_tok[row_i][0]   <= inj_tok_idx[TOK_W-1:0];
                        skew_valid[row_i][0] <= act_rd_inflight;
                        for (dly_k = 1; dly_k < ROWS; dly_k++) begin
                            skew_act[row_i][dly_k]   <= skew_act[row_i][dly_k-1];
                            skew_tok[row_i][dly_k]   <= skew_tok[row_i][dly_k-1];
                            skew_valid[row_i][dly_k] <= skew_valid[row_i][dly_k-1];
                        end
                    end

                    // Systolic PE update. Each PE uses its incoming (left/top)
                    // operands so pe_act and pe_psum advance one hop per cycle.
                    // Note pe_psum is not an accumulator: it is recomputed every
                    // cycle from the row above plus this PE's product, so the
                    // sum over the contraction dim is spatial (down the array),
                    // not temporal.
                    bottom_hits = 0;
                    for (row_i = 0; row_i < ROWS; row_i++) begin
                        for (col_j = 0; col_j < COLS; col_j++) begin
                            in_act   = (col_j == 0) ? edge_act[row_i]
                                                    : pe_act[row_i][col_j-1];
                            in_tok   = (col_j == 0) ? edge_tok[row_i]
                                                    : pe_tok[row_i][col_j-1];
                            in_valid = (col_j == 0) ? edge_valid[row_i]
                                                    : pe_act_valid[row_i][col_j-1];
                            in_psum  = (row_i == 0) ? '0
                                                    : pe_psum[row_i-1][col_j];
                            out_psum = in_psum + ternary_product(in_act,
                                                                 pe_weight[row_i][col_j]);

                            pe_act[row_i][col_j]       <= in_act;
                            pe_tok[row_i][col_j]       <= in_tok;
                            pe_act_valid[row_i][col_j] <= in_valid;
                            pe_psum[row_i][col_j]      <= out_psum;

                            // Bottom row: a completed column dot product. Scatter
                            // into the result buffer by its propagated token id.
                            // Contraction tile 0 initialises the running sum;
                            // every later tile adds to it, which is how the
                            // partials stay resident across the k loop instead
                            // of round-tripping through the scratchpad.
                            if (row_i == ROWS-1 && in_valid) begin
                                result_buf[in_tok][col_j] <=
                                    k_tile_first ? out_psum
                                                 : (result_buf[in_tok][col_j] + out_psum);
                                bottom_hits = bottom_hits + 1;
                            end
                        end
                    end
                    completions <= completions + bottom_hits[15:0];

                    // Contraction tile finished and more remain: step the two
                    // operand bases one array width along the contraction axis.
                    // The result stays in result_buf — nothing is written out here.
                    if (completions == expected_completions && !k_tile_last) begin
                        k_tile_idx    <= k_tile_idx    + 8'd1;
                        act_tile_base <= act_tile_base + ADDR_W'(ROWS);
                        wgt_tile_base <= wgt_tile_base + ADDR_W'(WGT_COL_BYTES);
                    end
                end

                // -------------------------------------------------------------
                // Writeback: one result row per token (int32, or int8 if requant).
                // On the last row, if output tiles remain, step to the next one:
                // reset the contraction, rewind the activation base to column 0,
                // and advance the weight/result bases by one array width.
                S_WB_WRITE: begin
                    wb_tok_idx <= wb_tok_idx + 16'd1;
                    if (wb_tok_idx == (16'(tok_count) - 16'd1) && !n_tile_last) begin
                        n_tile_idx     <= n_tile_idx + 8'd1;
                        k_tile_idx     <= '0;
                        wb_tok_idx     <= '0;
                        act_tile_base  <= act_base;
                        wgt_ntile_base <= wgt_ntile_base + wgt_ntile_step;
                        wgt_tile_base  <= wgt_ntile_base + wgt_ntile_step;
                        out_tile_base  <= out_tile_base  + out_ntile_step;
                    end
                end

                default: ; // S_WB_ACC_RD (C_re asserted comb), S_DONE
            endcase

            // ---------------------------------------------------------------
            // Per-tile streaming reset. Every *entry* into S_LOAD begins a fresh
            // weight load and activation stream, whether that is the first tile
            // of a dispatch, the next contraction tile, or the first tile of the
            // next output tile — so the counters are cleared in one place rather
            // than at each of the three transitions.
            //
            // This sits AFTER the case on purpose. On the cycle S_RUN hands over
            // to S_LOAD the systolic block above still runs and assigns
            // `completions <= completions + bottom_hits`; the later assignment in
            // the same always_ff is the one that takes effect, so the clear has
            // to come last or the next tile would start with the previous tile's
            // completion count and drain immediately.
            //
            // result_buf is deliberately NOT cleared: carrying it across the
            // contraction is the whole point, and `k_tile_first` selects load-vs-
            // accumulate at the scatter.
            // ---------------------------------------------------------------
            if (state_n == S_LOAD && state != S_LOAD) begin
                wgt_req_cnt <= '0; wgt_rcv_cnt <= '0; wgt_rd_inflight <= 1'b0;
                act_req_cnt <= '0; inj_tok_idx <= '0; completions <= '0;
                act_rd_inflight <= 1'b0;
                for (row_i = 0; row_i < ROWS; row_i++) begin
                    for (col_j = 0; col_j < COLS; col_j++) pe_act_valid[row_i][col_j] <= 1'b0;
                    for (dly_k = 0; dly_k < ROWS; dly_k++) skew_valid[row_i][dly_k]   <= 1'b0;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Status.
    // -------------------------------------------------------------------------
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign load_active = (state == S_LOAD);

endmodule
