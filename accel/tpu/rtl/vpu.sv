`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// vpu.sv — TPU SIMD vector unit
//
// Implements the block described in accel/tpu/docs/vpu.md: LANES int32 lanes fed
// from a single wide scratchpad read/modify/write port (V_rw), computing every
// pointwise / reduction op that is not an int4-weight matmul.
//
// SCOPE. This unit implements exactly the ops
// accel/tpulang/examples/adder_model.tpu issues, and nothing else:
//
//     VOP_DOT  VOP_ADD  VOP_RELU  VOP_REQUANT  VOP_DYT  VOP_QUANT4
//     VOP_VECMATMUL
//
// VOP_DOT is the odd one out: no shipped kernel issues `vecdot`, but it is the
// inner primitive VOP_VECMATMUL runs once per (row, col) pair, so its datapath
// is not optional and keeping the opcode costs nothing.
//
// **VOP_VECMATMUL now has no caller either.** Packing K and V to int4
// (VOP_QUANT4) moved both attention matmuls onto the MXU, and making Model.fc an
// Int4Linear moved the output head — its last caller — with them. The pair
// is kept because a model shape that needs an int8 x int8 matmul is one
// checkpoint away, and because removing them is a real area decision rather
// than a deletion: VOP_DOT is the reduction path (the accumulator, the fold and
// the S_WB scalar store), and VOP_VECMATMUL is the counter block around it.
// Nothing else in this unit uses either.
//
// Everything else was removed deliberately (GELU, EXP, SQUARE, ELEMENT_MUL,
// SCALAR_MUL/ADD/DIV, REDUCEMAX, REDUCESUM, SOFTMAX and its fused SM_EXP), along
// with the two 256-entry activation ROMs and the restoring divider that only
// SCALAR_DIV needed. They existed for a softmax-attention, LayerNorm, GELU-FFN
// model; model/transformer.py is now ReLU attention, DyT normalization and a
// ReLU feed-forward, so none of them had a caller. See docs/vpu.md §Removed ops
// for the list and for what coming back would cost.
//
// **The reduction path is now sum-only.** VOP_DOT is the sole reduction, so
// there is no max fold and `acc` always starts at zero. Restoring REDUCEMAX
// means restoring both.
//
// Dispatched by the scalar unit (scalar_unit.sv) with the issue-and-wait
// handshake: pulse `vpu_start`, then block on `vpu_done`. The op selector
// `vpu_op` matches VOP_* in scalar_unit.sv.
//
// Numerics / requant. Compute ops read int8 operands, accumulate in int32, and
// write their **int32** result straight to scratchpad — the VPU no longer
// narrows on the writeback path. Going int32 -> int4 is an *explicit*
// instruction, VOP_REQUANT, which applies the fixed-point rescale
// `clip((acc*m0 + round) >> n)` with a per-tensor {m0, n} read from the scalar
// operand. This lets the scalar unit keep the residual stream wide and
// requantize only where a narrow activation is actually needed (e.g. before
// feeding the MXU).
//
// **The narrow lands int4, in an int8 container.** REQUANT and DYT still write
// one byte per element and every activation buffer is still byte-per-element;
// only the value range narrows, to [-8, 7] and [-7, 7] respectively. Clipping
// there rather than at int8 is what keeps mxu.sv's PSUM_W bound true — see the
// numerics note in that file.
//
// DyT (VOP_DYT). Dynamic tanh — `hardtanh(alpha*x, -1, 1)`, arxiv 2503.10622 —
// is the normalization model/transformer.py uses in place of LayerNorm, and on
// a narrow integer datapath it is *the same instruction as a requant with a
// different clip*. If the output scale is pinned to 1/7 then a saturating narrow
// from int32 already computes the hardtanh: the multiplier `m0/2**n` carries
// `alpha * s_in * 7`, and every value the clip catches is exactly one that
// hardtanh would have flattened to +-1. The only difference from VOP_REQUANT is
// the *lower* bound, -7 instead of int4's -8: hardtanh is an odd function,
// so a floor of -8/7 = -1.143 would be off-spec at one end only. One
// comparison constant is the whole cost of the op, and having it be an op at
// all is what lets a kernel say "this is a normalization" instead of "this is a
// rescale that happens to clip". See accel/tpulang/adder_kernel.md §4.
//
// int4 narrow + pack (VOP_QUANT4, opcode 0x22, formerly `tquant`). The third op
// sharing that fixed point, and the only one whose *destination* is not int8: it
// clips to [-8, 7] and writes a bare 4-bit two's-complement nibble, two elements
// per byte, which is the MXU's packed weight layout. That is what makes the
// result directly addressable as a `matmul_t` weight row, with no repacking pass
// and no host round trip — and therefore what lets an *activation* be a weight
// operand. The shipped kernel packs K and V with it, which is how attention's
// Q@K^T and P@V left VOP_VECMATMUL (one serial dot product per output element)
// for the array.
//
// Now that weights and activations are the same width this is usually a pure
// repack — the source was already clipped to [-8, 7] by whatever requant
// produced it, so `{m0,n} = {1,0}` is lossless. Under the ternary encoding it
// was a genuine second narrow onto {-1, 0, 1}.
//
// Its write strobe is per byte, so `vpu_vlen` must be a multiple of 2 (it was 4
// when four trits shared a byte); a partial final byte writes nibble 0 into the
// slot the tail did not fill rather than preserving what was there.
//
// The scalar operand. VOP_REQUANT, VOP_DYT and VOP_QUANT4 are the only ops that
// use `vpu_rq_word`, and all three use the same thing: one {m0, n} pair, latched
// at dispatch and held for the whole vector. It used to be a scratchpad address
// that cost a two-state fetch over the V port before the first chunk; it is now
// a 16-bit literal in the command (docs/picorv32_migration.md §3).
//
// Streaming: a vector of `vpu_vlen` elements is processed LANES elements at a
// time. Each chunk reads its operand(s), computes all lanes in one cycle, then
// either writes the chunk back (elementwise / requant) or folds it into a
// running int32 accumulator (reduction). A partial tail chunk is masked by a
// lane-active predicate and by V_wstrb.
//
// Element widths on the 512-bit port: int8 operands occupy the low LANES bytes
// of an access; int32 operands/results occupy the full width (LANES*32 bits).
// Source/destination byte strides therefore differ per op (see *_stride below).
//
// Memory contract (matches scalar_unit.sv): synchronous read — address/enable
// presented one cycle, V_rdata valid the next; writes complete in one cycle
// under V_wstrb.
// -----------------------------------------------------------------------------

module vpu #(
    parameter int SCRATCHPAD_W = 64,  // scratchpad port width in bytes (512-bit port)
    parameter int ADDR_W       = 16,  // scratchpad byte-address width
    parameter int M0_W         = 12,  // requant fixed-point multiplier width (requant.sv)
    parameter int N_W          = 4    // requant shift width
) (
    input  logic clk,
    input  logic rst_n,

    // ---- Dispatch from the scalar unit (scalar_unit.sv / vpu.md §Interface) --
    input  logic                 vpu_start,
    input  logic [4:0]           vpu_op,
    input  logic [ADDR_W-1:0]    vpu_src0,
    input  logic [ADDR_W-1:0]    vpu_src1,
    // REQUANT / DYT / QUANT4 {n,m0} as a LITERAL, carried in the dispatch
    // rather than fetched from the scratchpad (docs/picorv32_migration.md §3).
    // M0_W + N_W = 16 bits, the same width as the address it replaces. The two
    // states that used to fetch it (S_RDS/S_RDSD) are gone with it, so every
    // requant/dyt/quant4 dispatch is two clocks and one scratchpad read shorter.
    input  logic [M0_W+N_W-1:0]  vpu_rq_word,
    input  logic [ADDR_W-1:0]    vpu_dst,
    input  logic [9:0]           vpu_vlen,     // vector length in elements
    // ---- Macro-op geometry (config registers; docs/macro_ops.md §5) --------
    // Only read by the macro ops. `vpu_rows`/`vpu_cols` are independent because
    // vecmatmul is not square in general: prefill attends T queries against T
    // keys, decode attends *one* query against t+1 keys.
    input  logic [15:0]          vpu_rows,     // query rows      (outer loop)
    input  logic [15:0]          vpu_cols,     // key rows        (inner loop)
    input  logic [ADDR_W-1:0]    vpu_row0,     // src0 row stride, bytes
    input  logic [ADDR_W-1:0]    vpu_row1,     // src1 row stride, bytes
    input  logic [ADDR_W-1:0]    vpu_crow,     // dst row stride, bytes (int32)
    output logic                 vpu_busy,
    output logic                 vpu_done,
    // High on exactly the clocks this unit is executing a VOP_VECMATMUL
    // dispatch — a strict subset of `vpu_busy`. Purely observational: nothing
    // in the datapath reads it. tpu_top.sv feeds it to perf_counters.sv so the
    // macro op's share of VPU time is separable from the primitive ops', which
    // is the one split `vpu_busy` alone cannot show (docs/macro_ops.md §7).
    output logic                 vpu_mm_busy,

    // ---- Scratchpad V_rw port (single logical read/modify/write, 512-bit) ----
    output logic                      V_re,
    output logic [ADDR_W-1:0]         V_raddr,
    input  logic [SCRATCHPAD_W*8-1:0] V_rdata,     // valid the cycle after V_re
    output logic                      V_we,
    output logic [ADDR_W-1:0]         V_waddr,
    output logic [SCRATCHPAD_W*8-1:0] V_wdata,
    output logic [SCRATCHPAD_W-1:0]   V_wstrb,      // per-byte write enable
    // Grants. The scratchpad arbitrates now that per-unit command queues let the
    // MXU, the VPU and the DMA be mid-dispatch at once (scratchpad.sv, "grants").
    // The MXU wins both chains because its writeback cannot stall; this unit can,
    // so when a grant comes back low the whole FSM freezes for that clock and
    // re-presents the same access. `stalled` below is the only thing that reads
    // these -- no state, no skid, no change to the datapath.
    input  logic                      V_rgnt,
    input  logic                      V_wgnt
);

    localparam int LANES = SCRATCHPAD_W / 4;   // int32 lanes per access (512b -> 16)
    localparam int ACC_W = 32;
    localparam int RQ_W  = 48;                 // requant intermediate width (headroom)

    // Op encoding. The surviving codes keep the values they had before the
    // unused ops were removed, so vpu.sv, scalar_unit.sv, iss.py and the
    // assembler need no re-synchronization; the gaps are retired opcodes. The
    // field stays 5 bits because VOP_DYT is 16 — compacting the encoding would
    // save two flops on the dispatch bus and cost a four-way rename.
    localparam logic [4:0]
        VOP_DOT         = 5'd0,   // also VOP_VECMATMUL's inner primitive
        VOP_ADD         = 5'd1,
        VOP_RELU        = 5'd3,
        VOP_REQUANT     = 5'd10,
        // ---- Macro op (docs/macro_ops.md §5) --------------------------------
        // Sequenced internally out of VOP_DOT. No new datapath: the wrapper
        // re-runs the existing inner FSM with different operand addresses,
        // which is why it costs counters rather than lanes.
        VOP_VECMATMUL   = 5'd13,   // S[t][s] = Σ_d src0[t][d]·src1[s][d]
        // dst[i] = clip_pm127((src0[i]*m0 + round) >> n) — DyT / hardtanh.
        // int32 in, int8 out, {m0,n} from the scalar operand, exactly like
        // VOP_REQUANT; see the header note for why the clip is symmetric.
        VOP_DYT         = 5'd16,
        // dst = 2-bit trit codes, 4 per byte, from int8 inputs. The same
        // fixed point again, clipped to +-1; see the header note.
        VOP_QUANT4      = 5'd17;

    // -------------------------------------------------------------------------
    // Op-class helpers.
    // -------------------------------------------------------------------------
    function automatic logic needs_src1(input logic [4:0] o);
        return (o == VOP_DOT) || (o == VOP_ADD);
    endfunction
    // Sum-only since REDUCEMAX/REDUCESUM were removed; VOP_DOT is the only
    // reduction left, so `acc` never needs a most-negative initial value.
    function automatic logic is_reduction(input logic [4:0] o);
        return (o == VOP_DOT);
    endfunction

    // The int4 grid. Activations are int4 *values* in int8 containers, so these
    // are the clips, not the container's. Matches mxu.sv ACT_QMIN/ACT_QMAX and
    // model/transformer.py INT4_QMIN/INT4_QMAX; `dyt` clips symmetrically at
    // +-QMAX because hardtanh is odd, exactly as it used to at +-127.
    localparam int Q4_MIN = -8;
    localparam int Q4_MAX =  7;

    // int32 -> int4 requantize: clip((acc*m0 + round) >> n), sign-extended into
    // an int8 container.
    function automatic logic [7:0] requant8(input logic signed [ACC_W-1:0] acc32,
                                            input logic [M0_W-1:0]         m0,
                                            input logic [N_W-1:0]          n);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = acc32 * $signed({1'b0, m0});          // m0 is a positive scale
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;                 // arithmetic (signed) shift
        if (shifted > Q4_MAX)      requant8 = 8'(signed'(Q4_MAX));
        else if (shifted < Q4_MIN) requant8 = 8'(signed'(Q4_MIN));
        else                       requant8 = shifted[7:0];
    endfunction

    // DyT / hardtanh: the same rescale, clipped **symmetrically**. Sharing the
    // shifter with requant8 is deliberate — the two ops differ in exactly one
    // constant, and writing that difference out here rather than hiding it in a
    // flag is what keeps the numerics reviewable against iss.py / model/quant.py.
    function automatic logic [7:0] dyt8(input logic signed [ACC_W-1:0] acc32,
                                        input logic [M0_W-1:0]         m0,
                                        input logic [N_W-1:0]          n);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = acc32 * $signed({1'b0, m0});
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;
        if (shifted >  Q4_MAX)      dyt8 = 8'(signed'(Q4_MAX));
        else if (shifted < -Q4_MAX) dyt8 = 8'(signed'(-Q4_MAX));  // odd: no -8
        else                        dyt8 = shifted[7:0];
    endfunction

    // int4 narrow + pack (VOP_QUANT4, formerly `tquant`). The same rescale as
    // requant8, returned as a bare 4-bit two's-complement nibble for the MXU's
    // packed weight layout — two per byte, where the trit code was four.
    //
    // With the model's weights and activations both int4 this is *usually* an
    // identity on the value: whatever produced the source already clipped to
    // [-8, 7], so `{m0,n} = {1,0}` makes this a pure repack. The rescale is kept
    // because it costs one shifter that already exists and because a repack that
    // cannot also requantize would force a second pass wherever a scale changes.
    function automatic logic [3:0] quant4(input logic signed [ACC_W-1:0] acc32,
                                          input logic [M0_W-1:0]         m0,
                                          input logic [N_W-1:0]          n);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = acc32 * $signed({1'b0, m0});
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;
        if (shifted > Q4_MAX)      quant4 = 4'(signed'(Q4_MAX));
        else if (shifted < Q4_MIN) quant4 = 4'(signed'(Q4_MIN));
        else                       quant4 = shifted[3:0];
    endfunction

    // -------------------------------------------------------------------------
    // FSM.
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE, S_RD0, S_RD0D, S_RD1, S_RD1D, S_EXEC, S_WB, S_DONE
    } state_t;
    state_t state, state_n;

    // Scratchpad denial. V_re/V_we are Moore outputs of `state`, and each state
    // makes at most one access, so freezing the state register for a clock and
    // re-driving the same request is exactly a retry. Nothing downstream sees a
    // difference: `V_rdata` is only ever sampled the cycle after a *granted*
    // read, because the state that samples it is one the FSM could not have
    // reached without the grant.
    wire stalled = (V_re && !V_rgnt) || (V_we && !V_wgnt);

    // Latched operands (captured on start; scalar unit may move on immediately).
    logic [4:0]        op_r;
    logic [ADDR_W-1:0] dst_r;
    logic [ADDR_W-1:0] p_src0, p_src1, p_dst;   // running chunk pointers
    logic [10:0]       remaining;               // elements left to process (0..1024)

    logic [M0_W+N_W-1:0]     rq_word_r;         // latched {n,m0} literal
    logic signed [ACC_W-1:0] acc;               // reduction accumulator (sum only)

    // Requant parameters. Latched at dispatch, so they are valid from S_RD0 on.
    wire [M0_W-1:0] rq_m0 = rq_word_r[M0_W-1:0];
    wire [N_W-1:0]  rq_n  = rq_word_r[M0_W +: N_W];

    // Element widths are per-op, and the two are independent: ADD/RELU/DOT read
    // int8 and write int32; REQUANT and DYT read int32 and write int8. src1 is
    // always an int8 vector (binary ops).
    // VOP_QUANT4 is the one op that reads int8 *and* writes narrower than int8,
    // so the two predicates are genuinely independent rather than two names for
    // one thing.
    wire src0_is32   = (op_r == VOP_REQUANT) || (op_r == VOP_DYT);
    wire dst_is4     = (op_r == VOP_QUANT4);
    wire dst_is8     = (op_r == VOP_REQUANT) || (op_r == VOP_DYT);
    wire [ADDR_W-1:0] src0_stride = src0_is32 ? ADDR_W'(LANES*4) : ADDR_W'(LANES);
    wire [ADDR_W-1:0] src1_stride = ADDR_W'(LANES);
    wire [ADDR_W-1:0] dst_stride  = dst_is4 ? ADDR_W'(LANES/2)
                                  : dst_is8 ? ADDR_W'(LANES)
                                            : ADDR_W'(LANES*4);

    // Chunk geometry.
    wire [10:0] chunk_active = (remaining >= LANES) ? 11'(LANES) : remaining;
    wire        last_chunk   = (remaining <= LANES);

    // -------------------------------------------------------------------------
    // vecmatmul macro-op state.
    //
    // The whole op is "run VOP_DOT once per (row, col) pair". `op_r` is set to
    // VOP_DOT at dispatch so every existing path — needs_src1, is_reduction, the
    // accumulator, the S_WB scalar store — works unchanged; `mm_active` is the
    // only thing that remembers we are inside a macro op, and its sole effect is
    // to send S_WB back to S_RD0 with the next pair's addresses instead of to
    // S_DONE.
    //
    // Bases are stepped rather than multiplied out: one add per pair, no
    // multiplier in the address path.
    // -------------------------------------------------------------------------
    logic              mm_active;
    logic [15:0]       mm_rows, mm_cols;      // latched counts (>= 1)
    logic [15:0]       mm_t, mm_s;            // current query row / key row
    logic [9:0]        mm_vlen;               // contraction length, per pair
    logic [ADDR_W-1:0] mm_row0, mm_row1, mm_crow;
    logic [ADDR_W-1:0] mm_src1_base;          // src1 origin, to rewind each row
    logic [ADDR_W-1:0] mm_a_base;             // src0 + t*row0
    logic [ADDR_W-1:0] mm_b_base;             // src1 + s*row1
    logic [ADDR_W-1:0] mm_d_row;              // dst  + t*crow   (row origin)
    logic [ADDR_W-1:0] mm_d_ptr;              // dst  + t*crow + s*4

    // Next-pair addresses, combinational so the sequential block can commit the
    // counter and the pointers derived from it in the same cycle.
    wire mm_s_last = ((mm_s + 16'd1) == mm_cols);
    wire mm_last   = mm_s_last && ((mm_t + 16'd1) == mm_rows);
    wire [ADDR_W-1:0] mm_a_next = mm_s_last ? (mm_a_base + mm_row0) : mm_a_base;
    wire [ADDR_W-1:0] mm_b_next = mm_s_last ? mm_src1_base : (mm_b_base + mm_row1);
    wire [ADDR_W-1:0] mm_drow_next = mm_s_last ? (mm_d_row + mm_crow) : mm_d_row;
    wire [ADDR_W-1:0] mm_d_next = mm_s_last ? (mm_d_row + mm_crow)
                                            : (mm_d_ptr + ADDR_W'(4));

    // Operand chunk registers (full port width; interpreted per element size).
    logic [SCRATCHPAD_W*8-1:0] V_data0;
    logic [SCRATCHPAD_W*8-1:0] V_data1;

    // -------------------------------------------------------------------------
    // Per-lane datapath (combinational, over the latched operand chunks).
    // -------------------------------------------------------------------------
    logic                    lane_active [0:LANES-1];
    logic signed [ACC_W-1:0] res32 [0:LANES-1];   // int32 elementwise result
    logic        [7:0]       res8  [0:LANES-1];    // int8 requant result
    logic        [3:0]       res4  [0:LANES-1];    // 4-bit int4 nibble (QUANT4)
    logic signed [ACC_W-1:0] red_val [0:LANES-1];  // per-lane value into reduction

    logic signed [7:0]       a8;
    logic signed [7:0]       b8;
    logic signed [ACC_W-1:0] a32;
    always_comb begin
        a8 = '0; b8 = '0; a32 = '0;
        for (int l = 0; l < LANES; l++) begin
            a8   = V_data0[l*8  +: 8];
            b8   = V_data1[l*8  +: 8];
            a32  = V_data0[l*32 +: 32];
            lane_active[l] = (l < chunk_active);
            res32[l] = '0;
            res8[l]  = '0;
            res4[l]  = '0;
            red_val[l] = '0;
            unique case (op_r)
                VOP_ADD:         res32[l] = ACC_W'(a8) + ACC_W'(b8);
                VOP_RELU:        res32[l] = (a8 > 0) ? ACC_W'(a8) : '0;
                VOP_REQUANT:     res8[l]  = requant8(a32, rq_m0, rq_n);
                VOP_DYT:         res8[l]  = dyt8(a32, rq_m0, rq_n);
                // int8 source, unlike the two above: this narrows an activation
                // that a requant already produced.
                VOP_QUANT4:      res4[l]  = quant4(ACC_W'(a8), rq_m0, rq_n);
                VOP_DOT:         red_val[l] = a8 * b8;
                default: ;
            endcase
        end
    end

    // Reduction fold across active lanes, then combine with the accumulator.
    // Sum only: VOP_DOT is the last reduction, so the max fold REDUCEMAX needed
    // is gone and with it `acc`'s most-negative initial value.
    logic signed [ACC_W-1:0] sum_chunk, acc_next;
    always_comb begin
        sum_chunk = '0;
        for (int l = 0; l < LANES; l++) begin
            if (lane_active[l]) sum_chunk += red_val[l];
        end
        acc_next = acc + sum_chunk;
    end

    // -------------------------------------------------------------------------
    // Writeback payload + byte strobe (elementwise / requant chunk).
    // int32 dst: 4 bytes per lane; int8 dst (REQUANT/DYT): 1 byte per lane;
    // 4-bit dst (QUANT4): 1 byte per *two* lanes.
    // -------------------------------------------------------------------------
    logic [SCRATCHPAD_W*8-1:0] wb_data;
    logic [SCRATCHPAD_W-1:0]   wb_strb;
    always_comb begin
        wb_data = '0;
        wb_strb = '0;
        for (int l = 0; l < LANES; l++) begin
            if (dst_is4) begin
                // Two nibbles share a byte, so the strobe cannot be per lane;
                // it is asserted below, once per pair. An inactive lane
                // contributes nibble 0, which is why the tail of a vlen that is
                // not a multiple of 2 zero-fills.
                if (lane_active[l]) wb_data[l*4 +: 4] = res4[l];
            end else if (dst_is8) begin
                wb_data[l*8 +: 8] = res8[l];
                if (lane_active[l]) wb_strb[l] = 1'b1;
            end else begin
                wb_data[l*32 +: 32] = res32[l];
                if (lane_active[l]) wb_strb[l*4 +: 4] = 4'b1111;
            end
        end
        if (dst_is4) begin
            for (int b = 0; b < LANES/2; b++) begin
                if (lane_active[b*2]) wb_strb[b] = 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Scratchpad port drive (combinational per state).
    // -------------------------------------------------------------------------
    always_comb begin
        V_re    = 1'b0;
        V_raddr = '0;
        V_we    = 1'b0;
        V_waddr = '0;
        V_wdata = '0;
        V_wstrb = '0;

        unique case (state)
            S_RD0: begin V_re = 1'b1; V_raddr = p_src0; end
            S_RD1: begin V_re = 1'b1; V_raddr = p_src1; end
            S_EXEC: if (!is_reduction(op_r)) begin
                V_we    = 1'b1;
                V_waddr = p_dst;
                V_wdata = wb_data;
                V_wstrb = wb_strb;
            end
            S_WB: begin   // reduction result: one int32 scalar into low 4 bytes
                V_we               = 1'b1;
                V_waddr            = dst_r;
                V_wdata[ACC_W-1:0] = acc;
                V_wstrb[3:0]       = 4'b1111;
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
            S_IDLE: if (vpu_start) begin
                        if (vpu_vlen == 0) state_n = S_DONE;
                        else               state_n = S_RD0;
                    end
            S_RD0:  state_n = S_RD0D;
            S_RD0D: begin
                        if (needs_src1(op_r)) state_n = S_RD1;
                        else                  state_n = S_EXEC;
                    end
            S_RD1:  state_n = S_RD1D;
            S_RD1D: state_n = S_EXEC;
            S_EXEC: begin
                        if (last_chunk) begin
                            if (is_reduction(op_r)) state_n = S_WB;
                            else                    state_n = S_DONE;
                        end else state_n = S_RD0;
                    end
            // A reduction's scalar store. For a macro op this is one (row,col)
            // pair finished: go straight back for the next one.
            S_WB:   if (mm_active && !mm_last) state_n = S_RD0;
                    else                       state_n = S_DONE;
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential datapath.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            op_r          <= '0;
            mm_active     <= 1'b0;
            dst_r         <= '0;
            p_src0        <= '0;
            p_src1        <= '0;
            p_dst         <= '0;
            rq_word_r     <= '0;
            remaining     <= '0;
            acc           <= '0;
            V_data0       <= '0;
            V_data1       <= '0;
        end else if (!stalled) begin
            state <= state_n;

            unique case (state)
                S_IDLE: if (vpu_start) begin
                    // A macro op runs as its inner primitive; `mm_active` is the
                    // only trace of the wrapper. A count of 0 reads as 1 so an
                    // unset config register degrades to a single pair rather
                    // than a dispatch that computes nothing.
                    // A macro op enters as its inner primitive; `mm_active` is
                    // the only trace of the wrapper.
                    op_r          <= (vpu_op == VOP_VECMATMUL) ? VOP_DOT : vpu_op;
                    mm_active     <= (vpu_op == VOP_VECMATMUL);

                    mm_rows       <= (vpu_rows == '0) ? 16'd1 : vpu_rows;
                    mm_cols       <= (vpu_cols == '0) ? 16'd1 : vpu_cols;
                    mm_t          <= '0;
                    mm_s          <= '0;
                    mm_vlen       <= vpu_vlen;
                    mm_row0       <= vpu_row0;
                    mm_row1       <= vpu_row1;
                    mm_crow       <= vpu_crow;
                    mm_src1_base  <= vpu_src1;
                    mm_a_base     <= vpu_src0;
                    mm_b_base     <= vpu_src1;
                    mm_d_row      <= vpu_dst;
                    mm_d_ptr      <= vpu_dst;

                    dst_r         <= vpu_dst;
                    p_src0        <= vpu_src0;
                    p_src1        <= vpu_src1;
                    p_dst         <= vpu_dst;
                    rq_word_r     <= vpu_rq_word;
                    remaining     <= {1'b0, vpu_vlen};
                    // VOP_DOT is the only reduction left, so the accumulator
                    // always opens at zero.
                    acc           <= '0;
                end

                // One (row,col) pair has just been stored. Advance to the next
                // and restart the inner dot product over the fresh operands.
                S_WB: if (mm_active && !mm_last) begin
                    mm_s      <= mm_s_last ? 16'd0 : (mm_s + 16'd1);
                    mm_t      <= mm_s_last ? (mm_t + 16'd1) : mm_t;
                    mm_a_base <= mm_a_next;
                    mm_b_base <= mm_b_next;
                    mm_d_row  <= mm_drow_next;
                    mm_d_ptr  <= mm_d_next;

                    p_src0    <= mm_a_next;
                    p_src1    <= mm_b_next;
                    dst_r     <= mm_d_next;
                    remaining <= {1'b0, mm_vlen};
                    acc       <= '0;
                end

                S_RD0D: V_data0 <= V_rdata;
                S_RD1D: V_data1 <= V_rdata;
                S_EXEC: begin
                    if (is_reduction(op_r)) acc <= acc_next;
                    // Advance to the next chunk (harmless on the last chunk).
                    p_src0    <= p_src0 + src0_stride;
                    p_src1    <= p_src1 + src1_stride;
                    p_dst     <= p_dst  + dst_stride;
                    remaining <= remaining - chunk_active;
                end

                default: ;
            endcase

        end
    end

    // -------------------------------------------------------------------------
    // Status.
    // -------------------------------------------------------------------------
    assign vpu_busy = (state != S_IDLE);
    assign vpu_done = (state == S_DONE);

    // `mm_active` is written at dispatch and then *held* until the next
    // dispatch overwrites it, so on its own it stays high through the following
    // idle. Qualifying it with `vpu_busy` is what makes this a per-op event;
    // it also matches `vpu_busy`'s edges exactly, so the vecmatmul counter is a
    // subset of the VPU counter clock for clock.
    assign vpu_mm_busy = vpu_busy && mm_active;

endmodule
