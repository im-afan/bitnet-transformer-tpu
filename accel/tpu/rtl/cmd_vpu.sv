`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmd_vpu.sv — VPU macro-op front end: command queue + decode + sequencing
//
// Same shape as cmd_mxu.sv (read that one first — the reasoning about why a
// GEOM command is not the config register file coming back applies verbatim).
//
//   VPU_OP    one vector pass: the VOP_* selector, dst/src0/src1, `vlen`, and
//             the requant {m0,n} as a literal.
//   VPU_GEOM  vecmatmul's row/column counts and the four row strides. Sticky,
//             retires in one clock.
//
// The split is cheaper here than on the MXU: a plain pointwise op fits in 128
// bits with room to spare, and only `vecmatmul` reads the geometry at all.
// Nothing in the shipped adder kernel issues `vecmatmul` — int4 K/V moved
// both attention matmuls onto the array — so in practice the VPU sees a single
// command type and GEOM never appears.
//
// `src1` and the requant word occupy different fields even though no op uses
// both: requant/dyt/quant4 used to pass the {m0,n} *address* in the src1 slot
// (scalar_unit.sv's `vpu_scalar = r_src1`). Now that the word is a literal there
// is no reason to overlap them, and keeping them apart means the decode does not
// depend on the opcode.
// -----------------------------------------------------------------------------

module cmd_vpu #(
    parameter int ADDR_W = 16,
    parameter int M0_W   = 12,
    parameter int N_W    = 4,
    parameter int DEPTH  = 8
) (
    input  logic clk,
    input  logic rst_n,

    // ---- producer side ------------------------------------------------------
    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    // ---- VPU dispatch (vpu.sv) ----------------------------------------------
    output logic                 vpu_start,
    output logic [4:0]           vpu_op,
    output logic [ADDR_W-1:0]    vpu_src0,
    output logic [ADDR_W-1:0]    vpu_src1,
    output logic [M0_W+N_W-1:0]  vpu_rq_word,
    output logic [ADDR_W-1:0]    vpu_dst,
    output logic [9:0]           vpu_vlen,
    output logic [15:0]          vpu_rows,
    output logic [15:0]          vpu_cols,
    output logic [ADDR_W-1:0]    vpu_row0,
    output logic [ADDR_W-1:0]    vpu_row1,
    output logic [ADDR_W-1:0]    vpu_crow,
    input  logic                 vpu_done,

    // ---- status -------------------------------------------------------------
    output logic [31:0] issued,
    output logic [31:0] retired,
    output logic [15:0] level,
    output logic        idle
);

    localparam logic [7:0] VPU_CMD_OP   = 8'h01,
                           VPU_CMD_GEOM = 8'h02;

    logic [127:0] head;
    logic         empty, pop;

    cmd_queue #(.WIDTH(128), .DEPTH(DEPTH)) u_q (
        .clk (clk), .rst_n (rst_n),
        .wr_en (cmd_we), .wr_data (cmd_wdata), .full (cmd_full),
        .empty (empty), .head (head), .pop (pop),
        .count (level)
    );

    wire [7:0]  c_op = head[7:0];
    wire [31:0] w0   = head[31:0];
    wire [31:0] w1   = head[63:32];
    wire [31:0] w2   = head[95:64];

    wire is_op   = (c_op == VPU_CMD_OP);
    wire is_geom = (c_op == VPU_CMD_GEOM);

    // ---- sticky macro-op geometry ------------------------------------------
    logic [15:0]       g_rows, g_cols;
    logic [ADDR_W-1:0] g_row0, g_row1, g_crow;

    assign vpu_rows = g_rows;
    assign vpu_cols = g_cols;
    assign vpu_row0 = g_row0;
    assign vpu_row1 = g_row1;
    assign vpu_crow = g_crow;

    // ---- per-op operands ----------------------------------------------------
    assign vpu_op      = w0[12:8];
    assign vpu_dst     = w0[31:16];
    assign vpu_src0    = w1[15:0];
    assign vpu_src1    = w1[31:16];
    assign vpu_vlen    = w2[9:0];
    assign vpu_rq_word = w2[16 +: (M0_W+N_W)];

    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    assign vpu_start = (state == S_HEAD) && !empty && is_op;
    assign pop       = ((state == S_HEAD) && !empty && !is_op) ||
                       ((state == S_RUN)  && vpu_done);

    assign idle    = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_HEAD;
            g_rows <= '0;
            g_cols <= '0;
            g_row0 <= '0;
            g_row1 <= '0;
            g_crow <= '0;
        end else begin
            case (state)
                S_HEAD: begin
                    if (!empty) begin
                        if (is_geom) begin
                            g_rows <= w0[31:16];
                            g_cols <= w1[15:0];
                            g_row0 <= w1[31:16];
                            g_row1 <= w2[15:0];
                            g_crow <= w2[31:16];
                        end else if (is_op) begin
                            state <= S_RUN;
                        end
                    end
                end
                S_RUN: if (vpu_done) state <= S_HEAD;
                default: state <= S_HEAD;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issued  <= '0;
            retired <= '0;
        end else begin
            if (cmd_we && !cmd_full) issued  <= issued  + 32'd1;
            if (pop)                 retired <= retired + 32'd1;
        end
    end

// synthesis translate_off
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && state == S_HEAD && !empty && !is_op && !is_geom)
            $display("[%0t] cmd_vpu: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
