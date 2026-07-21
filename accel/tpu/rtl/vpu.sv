`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// vpu.sv — TPU SIMD vector unit
//
// Implements the block described in accel/tpu/docs/vpu.md: LANES int32 lanes fed
// from a single wide scratchpad read/modify/write port (V_rw), computing every
// pointwise / reduction op that is not a ternary-weight matmul (activations,
// residual adds, vector*scalar, dot products, softmax/LayerNorm reductions).
//
// Dispatched by the scalar unit (scalar_unit.sv) with the issue-and-wait
// handshake: pulse `vpu_start`, then block on `vpu_done`. The op selector
// `vpu_op` matches VOP_* in scalar_unit.sv.
//
// Numerics / requant. Compute ops read int8 operands, accumulate in int32, and
// write their **int32** result straight to scratchpad — the VPU no longer
// narrows on the writeback path. Going int32 -> int8 is an *explicit*
// instruction, VOP_REQUANT, which applies the BitNet fixed-point rescale
// `clip((acc*m0 + round) >> n)` (the same math as requant.sv) with a per-tensor
// {m0, n} read from the scalar operand. This lets the scalar unit keep the
// residual stream / softmax temporaries wide and requantize only where an int8
// activation is actually needed (e.g. before feeding the MXU).
//
// Broadcast scalar ops. VOP_SCALAR_ADD / VOP_SCALAR_MUL / VOP_SCALAR_DIV read one
// int32 word from `vpu_scalar` (once, on the first chunk) and apply it to every
// lane — the broadcast forms softmax's `x - max` and LayerNorm's `x - mean` need.
// Dividing by a *runtime* scalar (softmax's Σexp, LayerNorm's variance) cannot be
// a compile-time {m0, n} like REQUANT, so VOP_SCALAR_DIV reciprocates the divisor
// once into R = 2**RECIP_Q / |d| using a single shared restoring divider
// (RECIP_Q+1 cycles, amortized over the whole vector) and then reuses each lane's
// existing 8x32 multiplier. The result is emitted in a *fixed* Q(DIV_Q) format,
// `dst[i] = round(src0[i] * 2**DIV_Q / d)`: pinning the format in hardware is what
// keeps the result's scale known at compile time even though the divisor is not,
// so the compiler can requantize a division like any other int32 buffer.
//
// Activation LUTs. GELU/EXP are 256-entry int8 tables indexed by the signed input
// byte, generated on the host at a *canonical* input scale (accel/compiler/luts.py)
// rather than per-program — the table is a fixed hardware artifact, and getting an
// operand into the table's input scale is the compiler's job (an ordinary REQUANT
// ahead of the op). See vpu.md §Activation LUTs.
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
    parameter int N_W          = 4,   // requant shift width
    // VOP_SCALAR_DIV fixed-point. The divisor is reciprocated to R = 2**RECIP_Q /
    // |d| (RECIP_Q-bit unsigned) and each quotient is emitted in Q(DIV_Q): dst[i]
    // = round(src0[i] * 2**DIV_Q / d). DIV_Q pins the result scale at compile time
    // (see vpu.md §Divide); RECIP_Q carries enough reciprocal precision for it.
    parameter int RECIP_Q      = 31,  // reciprocal numerator exponent (2**RECIP_Q / |d|)
    parameter int DIV_Q        = 15,  // quotient fractional bits (Q15 result)
    // Host-generated activation LUTs (256 x int8), indexed by the signed input
    // byte. Left empty by default; a top-level with real activations points
    // these at $readmemh files generated from the reference model.
    parameter     GELU_INIT    = "",
    parameter     EXP_INIT     = ""
) (
    input  logic clk,
    input  logic rst_n,

    // ---- Dispatch from the scalar unit (scalar_unit.sv / vpu.md §Interface) --
    input  logic                 vpu_start,
    input  logic [3:0]           vpu_op,
    input  logic [ADDR_W-1:0]    vpu_src0,
    input  logic [ADDR_W-1:0]    vpu_src1,
    input  logic [ADDR_W-1:0]    vpu_scalar,   // SCALAR_MUL multiplier / REQUANT {n,m0}
    input  logic [ADDR_W-1:0]    vpu_dst,
    input  logic [9:0]           vpu_vlen,     // vector length in elements
    output logic                 vpu_busy,
    output logic                 vpu_done,

    // ---- Scratchpad V_rw port (single logical read/modify/write, 512-bit) ----
    output logic                      V_re,
    output logic [ADDR_W-1:0]         V_raddr,
    input  logic [SCRATCHPAD_W*8-1:0] V_rdata,     // valid the cycle after V_re
    output logic                      V_we,
    output logic [ADDR_W-1:0]         V_waddr,
    output logic [SCRATCHPAD_W*8-1:0] V_wdata,
    output logic [SCRATCHPAD_W-1:0]   V_wstrb       // per-byte write enable
);

    localparam int LANES = SCRATCHPAD_W / 4;   // int32 lanes per access (512b -> 16)
    localparam int ACC_W = 32;
    localparam int RQ_W  = 48;                 // requant intermediate width (headroom)
    // Reciprocal / divide intermediate width: a8 (int8) * R (RECIP_Q+1 bits),
    // plus a sign bit and rounding headroom.
    localparam int DIV_W = RECIP_Q + 12;

    // Op encoding (codes 0..4 match VOP_* in scalar_unit.sv).
    localparam logic [3:0]
        VOP_DOT         = 4'd0,
        VOP_ADD         = 4'd1,
        VOP_SCALAR_MUL  = 4'd2,
        VOP_RELU        = 4'd3,
        VOP_GELU        = 4'd4,
        VOP_SQUARE      = 4'd5,
        VOP_EXP         = 4'd6,
        VOP_REDUCEMAX   = 4'd7,
        VOP_REDUCESUM   = 4'd8,
        VOP_ELEMENT_MUL = 4'd9,
        VOP_REQUANT     = 4'd10,
        VOP_SCALAR_ADD  = 4'd11,   // dst[i] = src0[i] + scalar   (broadcast add)
        VOP_SCALAR_DIV  = 4'd12;   // dst[i] = round(src0[i] * 2**DIV_Q / scalar)

    // -------------------------------------------------------------------------
    // Activation LUTs (256 x int8). Preloaded on the host from the reference.
    // -------------------------------------------------------------------------
    logic signed [7:0] gelu_rom [0:255];
    logic signed [7:0] exp_rom  [0:255];
    initial begin
        if (GELU_INIT != "") $readmemh(GELU_INIT, gelu_rom);
        if (EXP_INIT  != "") $readmemh(EXP_INIT,  exp_rom);
    end

    // -------------------------------------------------------------------------
    // Op-class helpers.
    // -------------------------------------------------------------------------
    function automatic logic needs_src1(input logic [3:0] o);
        return (o == VOP_DOT) || (o == VOP_ADD) || (o == VOP_ELEMENT_MUL);
    endfunction
    function automatic logic needs_scalar(input logic [3:0] o);
        return (o == VOP_SCALAR_MUL) || (o == VOP_REQUANT) ||
               (o == VOP_SCALAR_ADD) || (o == VOP_SCALAR_DIV);
    endfunction
    function automatic logic is_reduction(input logic [3:0] o);
        return (o == VOP_DOT) || (o == VOP_REDUCESUM) || (o == VOP_REDUCEMAX);
    endfunction

    // int32 -> int8 requantize: clip((acc*m0 + round) >> n). Mirrors requant.sv,
    // extended to signed accumulators and int8 clip.
    function automatic logic [7:0] requant8(input logic signed [ACC_W-1:0] acc32,
                                            input logic [M0_W-1:0]         m0,
                                            input logic [N_W-1:0]          n);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = acc32 * $signed({1'b0, m0});          // m0 is a positive scale
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;                 // arithmetic (signed) shift
        if (shifted >  127)      requant8 =  8'sd127;
        else if (shifted < -128) requant8 = -8'sd128;
        else                     requant8 = shifted[7:0];
    endfunction

    // -------------------------------------------------------------------------
    // FSM.
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE, S_RD0, S_RD0D, S_RD1, S_RD1D, S_RDS, S_RDSD, S_RECIP, S_EXEC, S_WB, S_DONE
    } state_t;
    state_t state, state_n;

    // Latched operands (captured on start; scalar unit may move on immediately).
    logic [3:0]        op_r;
    logic [ADDR_W-1:0] dst_r;
    logic [ADDR_W-1:0] p_src0, p_src1, p_dst;   // running chunk pointers
    logic [ADDR_W-1:0] scalar_addr_r;
    logic [10:0]       remaining;               // elements left to process (0..1024)
    logic              scalar_loaded;

    logic [ACC_W-1:0]  scalar_reg;              // SCALAR_MUL multiplier / REQUANT {n,m0}
    logic signed [ACC_W-1:0] acc;               // reduction accumulator

    // Reciprocal unit (VOP_SCALAR_DIV): one shared restoring divider computing
    // R = floor(2**RECIP_Q / |scalar|) in RECIP_Q+1 steps after the scalar loads.
    // A div-by-zero saturates R to all-ones (largest representable reciprocal).
    logic [RECIP_Q:0]   recip_R;   // reciprocal magnitude, held across all chunks
    logic               div_neg;   // sign of the divisor (applied per lane in EXEC)
    logic [RECIP_Q+1:0] div_rem;   // running remainder (headroom for the shift)
    logic [RECIP_Q:0]   div_quot;  // quotient built MSB-first
    logic [RECIP_Q:0]   div_n;     // numerator shift register (2**RECIP_Q, then <<1)
    logic [ACC_W-1:0]   div_d;     // |divisor|
    logic [5:0]         div_cnt;   // iterations remaining

    // One restoring-division step, shared by every S_RECIP cycle.
    wire [RECIP_Q+1:0] div_rem_sh = {div_rem[RECIP_Q:0], div_n[RECIP_Q]};
    wire               div_ge     = (div_rem_sh >= {1'b0, div_d});

    // Requant parameters (valid once scalar_reg is loaded).
    wire [M0_W-1:0] rq_m0 = scalar_reg[M0_W-1:0];
    wire [N_W-1:0]  rq_n  = scalar_reg[M0_W +: N_W];

    // int8 source, int32 destination for compute ops; int32 source, int8
    // destination for REQUANT. src1 is always an int8 vector (binary ops).
    wire is_requant = (op_r == VOP_REQUANT);
    wire [ADDR_W-1:0] src0_stride = is_requant ? ADDR_W'(LANES*4) : ADDR_W'(LANES);
    wire [ADDR_W-1:0] src1_stride = ADDR_W'(LANES);
    wire [ADDR_W-1:0] dst_stride  = is_requant ? ADDR_W'(LANES)   : ADDR_W'(LANES*4);

    // Chunk geometry.
    wire [10:0] chunk_active = (remaining >= LANES) ? 11'(LANES) : remaining;
    wire        last_chunk   = (remaining <= LANES);

    // Operand chunk registers (full port width; interpreted per element size).
    logic [SCRATCHPAD_W*8-1:0] V_data0;
    logic [SCRATCHPAD_W*8-1:0] V_data1;

    // -------------------------------------------------------------------------
    // Per-lane datapath (combinational, over the latched operand chunks).
    // -------------------------------------------------------------------------
    logic                    lane_active [0:LANES-1];
    logic signed [ACC_W-1:0] res32 [0:LANES-1];   // int32 elementwise result
    logic        [7:0]       res8  [0:LANES-1];    // int8 requant result
    logic signed [ACC_W-1:0] red_val [0:LANES-1];  // per-lane value into reduction

    localparam int DIV_SHIFT = RECIP_Q - DIV_Q;   // post-multiply right shift

    logic signed [7:0]       a8;
    logic signed [7:0]       b8;
    logic        [7:0]       idx8;    // unsigned byte for LUT indexing
    logic signed [ACC_W-1:0] a32;
    logic signed [DIV_W-1:0] div_prod;   // a8 * reciprocal, pre-shift
    logic signed [DIV_W-1:0] div_q;      // rounded quotient magnitude
    always_comb begin
        a8 = '0; b8 = '0; idx8 = '0; a32 = '0; div_prod = '0; div_q = '0;
        for (int l = 0; l < LANES; l++) begin
            a8   = V_data0[l*8  +: 8];
            b8   = V_data1[l*8  +: 8];
            idx8 = V_data0[l*8  +: 8];
            a32  = V_data0[l*32 +: 32];
            lane_active[l] = (l < chunk_active);
            res32[l] = '0;
            res8[l]  = '0;
            red_val[l] = '0;
            // round(a8 * 2**DIV_Q / |d|) = round((a8 * R) >> (RECIP_Q - DIV_Q)),
            // then sign-corrected below. R is unsigned, so widen it through 0.
            div_prod = $signed(a8) * $signed({1'b0, recip_R});
            // The round term must be *signed*, else the sum goes unsigned and `>>>`
            // degrades to a logical shift (mirrors requant8's signed `round`).
            div_q    = (DIV_SHIFT == 0) ? div_prod
                     : ((div_prod + $signed(DIV_W'(1) <<< (DIV_SHIFT - 1))) >>> DIV_SHIFT);
            unique case (op_r)
                VOP_ADD:         res32[l] = ACC_W'(a8) + ACC_W'(b8);
                VOP_ELEMENT_MUL: res32[l] = a8 * b8;
                VOP_SCALAR_MUL:  res32[l] = a8 * $signed(scalar_reg);
                VOP_SCALAR_ADD:  res32[l] = ACC_W'(a8) + $signed(scalar_reg);
                VOP_SCALAR_DIV:  res32[l] = div_neg ? ACC_W'(-div_q) : ACC_W'(div_q);
                VOP_RELU:        res32[l] = (a8 > 0) ? ACC_W'(a8) : '0;
                VOP_SQUARE:      res32[l] = a8 * a8;
                VOP_GELU:        res32[l] = ACC_W'(gelu_rom[idx8]);
                VOP_EXP:         res32[l] = ACC_W'(exp_rom [idx8]);
                VOP_REQUANT:     res8[l]  = requant8(a32, rq_m0, rq_n);
                VOP_DOT:         red_val[l] = a8 * b8;
                VOP_REDUCESUM:   red_val[l] = ACC_W'(a8);
                VOP_REDUCEMAX:   red_val[l] = ACC_W'(a8);
                default: ;
            endcase
        end
    end

    // Reduction fold across active lanes, then combine with the accumulator.
    logic signed [ACC_W-1:0] sum_chunk, max_chunk, acc_next;
    always_comb begin
        sum_chunk = '0;
        max_chunk = {1'b1, {(ACC_W-1){1'b0}}};   // most-negative int32
        for (int l = 0; l < LANES; l++) begin
            if (lane_active[l]) begin
                sum_chunk += red_val[l];
                if (red_val[l] > max_chunk) max_chunk = red_val[l];
            end
        end
        acc_next = (op_r == VOP_REDUCEMAX) ? ((max_chunk > acc) ? max_chunk : acc)
                                           : (acc + sum_chunk);
    end

    // -------------------------------------------------------------------------
    // Writeback payload + byte strobe (elementwise / requant chunk).
    // int32 dst: 4 bytes per lane; int8 dst (REQUANT): 1 byte per lane.
    // -------------------------------------------------------------------------
    logic [SCRATCHPAD_W*8-1:0] wb_data;
    logic [SCRATCHPAD_W-1:0]   wb_strb;
    always_comb begin
        wb_data = '0;
        wb_strb = '0;
        for (int l = 0; l < LANES; l++) begin
            if (is_requant) begin
                wb_data[l*8 +: 8] = res8[l];
                if (lane_active[l]) wb_strb[l] = 1'b1;
            end else begin
                wb_data[l*32 +: 32] = res32[l];
                if (lane_active[l]) wb_strb[l*4 +: 4] = 4'b1111;
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
            S_RDS: begin V_re = 1'b1; V_raddr = scalar_addr_r; end
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
                        if (needs_src1(op_r))                          state_n = S_RD1;
                        else if (needs_scalar(op_r) && !scalar_loaded) state_n = S_RDS;
                        else                                           state_n = S_EXEC;
                    end
            S_RD1:  state_n = S_RD1D;
            S_RD1D: state_n = S_EXEC;
            S_RDS:  state_n = S_RDSD;
            S_RDSD: if (op_r == VOP_SCALAR_DIV) state_n = S_RECIP;
                    else                        state_n = S_EXEC;
            S_RECIP: if (div_cnt == 6'd1) state_n = S_EXEC;
                     else                 state_n = S_RECIP;
            S_EXEC: begin
                        if (last_chunk) begin
                            if (is_reduction(op_r)) state_n = S_WB;
                            else                    state_n = S_DONE;
                        end else state_n = S_RD0;
                    end
            S_WB:   state_n = S_DONE;
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
            dst_r         <= '0;
            p_src0        <= '0;
            p_src1        <= '0;
            p_dst         <= '0;
            scalar_addr_r <= '0;
            remaining     <= '0;
            scalar_loaded <= 1'b0;
            scalar_reg    <= '0;
            acc           <= '0;
            V_data0       <= '0;
            V_data1       <= '0;
            recip_R       <= '0;
            div_neg       <= 1'b0;
            div_rem       <= '0;
            div_quot      <= '0;
            div_n         <= '0;
            div_d         <= '0;
            div_cnt       <= '0;
        end else begin
            state <= state_n;

            unique case (state)
                S_IDLE: if (vpu_start) begin
                    op_r          <= vpu_op;
                    dst_r         <= vpu_dst;
                    p_src0        <= vpu_src0;
                    p_src1        <= vpu_src1;
                    p_dst         <= vpu_dst;
                    scalar_addr_r <= vpu_scalar;
                    remaining     <= {1'b0, vpu_vlen};
                    scalar_loaded <= 1'b0;
                    // Reduction accumulator init: 0 for sum/dot, most-negative for max.
                    acc <= (vpu_op == VOP_REDUCEMAX) ? {1'b1, {(ACC_W-1){1'b0}}} : '0;
                end

                S_RD0D: V_data0 <= V_rdata;
                S_RD1D: V_data1 <= V_rdata;
                S_RDSD: begin
                    scalar_reg    <= V_rdata[ACC_W-1:0];   // multiplier / {n,m0} / divisor
                    scalar_loaded <= 1'b1;
                    // Kick off the reciprocal for SCALAR_DIV: |d|, numerator 2**RECIP_Q.
                    div_neg  <= V_rdata[ACC_W-1];
                    div_d    <= V_rdata[ACC_W-1] ? (~V_rdata[ACC_W-1:0] + 1'b1)
                                                 :   V_rdata[ACC_W-1:0];
                    div_rem  <= '0;
                    div_quot <= '0;
                    div_n    <= {1'b1, {RECIP_Q{1'b0}}};   // 2**RECIP_Q
                    div_cnt  <= 6'(RECIP_Q + 1);
                end

                // Restoring division: one quotient bit (MSB-first) per cycle.
                S_RECIP: begin
                    div_rem  <= div_ge ? (div_rem_sh - {1'b0, div_d}) : div_rem_sh;
                    div_quot <= {div_quot[RECIP_Q-1:0], div_ge};
                    div_n    <= div_n << 1;
                    div_cnt  <= div_cnt - 1'b1;
                    if (div_cnt == 6'd1) recip_R <= {div_quot[RECIP_Q-1:0], div_ge};
                end

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

endmodule
