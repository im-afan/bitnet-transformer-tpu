`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmd_mxu.sv — MXU macro-op front end: command queue + decode + sequencing
//
// Pops 128-bit commands and turns them into the MXU's existing
// start/busy/done dispatch (mxu.md §7). The array itself is unchanged; only
// where its operands come from moved.
//
// Two commands, both 128 bits (docs/picorv32_migration.md §3):
//
//   MXU_GEOM  the operand strides and tile counts. Latched into sticky
//             registers and retired in one clock — it starts nothing.
//   MXU_MM    one matmul: three addresses, the requant word, and the acc/rq/
//             tiled flags. Runs on the array.
//
// WHY THE SPLIT. A self-contained matmul command needs three 16-bit addresses,
// a 16-bit requant word, three 16-bit strides, two tile counts and `tlen` —
// 146 bits. Squeezing that into 128 needs 12-bit strides and 6-bit tile counts
// and lands at exactly 128 with no margin, which is not worth doing. So the
// geometry rides its own command.
//
// That is NOT the config-register model coming back. `cfg` was a global
// register file: writable by any instruction, shared by every unit, and
// surviving across programs — which is exactly how one program's leftover
// `arow` corrupted the next program's plain `matmul` (docs/macro_ops.md §4.0).
// MXU_GEOM travels in *this unit's* in-order queue, so what a matmul sees is
// whatever GEOM precedes it in the same command stream. The producer is free to
// skip a GEOM whose value has not changed, and that is a pure optimization
// rather than a hazard: there is no path by which anything else can write it.
//
// `tiled` stays a per-matmul flag rather than "the strides are nonzero", for
// the reason macro_ops.md §4.0 gives — the mode is a property of the
// instruction, not of leftover machine state.
//
// The requant {m0,n} is a LITERAL in the command, not a scratchpad address.
// It is 12 + 4 = 16 bits, exactly the width of the address that used to point
// at it, so the command budget is unchanged and the MXU loses the two-state
// one-shot fetch it used to run over the C port before every requantized
// store.
// -----------------------------------------------------------------------------

module cmd_mxu #(
    parameter int ADDR_W = 16,
    parameter int M0_W   = 12,
    parameter int N_W    = 4,
    parameter int DEPTH  = 8      // queue entries
) (
    input  logic clk,
    input  logic rst_n,

    // ---- producer side (scalar unit or CPU) ---------------------------------
    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    // ---- MXU dispatch (mxu.sv) ----------------------------------------------
    output logic                     mxu_start,
    output logic [ADDR_W-1:0]        mxu_act_addr,
    output logic [ADDR_W-1:0]        mxu_weight_addr,
    output logic [ADDR_W-1:0]        mxu_out_addr,
    output logic [M0_W+N_W-1:0]      mxu_rq_word,
    output logic [5:0]               mxu_t_len,
    output logic                     mxu_tiled,
    output logic [ADDR_W-1:0]        mxu_a_row,
    output logic [ADDR_W-1:0]        mxu_c_row,
    output logic [ADDR_W-1:0]        mxu_w_col,
    output logic [7:0]               mxu_k_tiles,
    output logic [7:0]               mxu_n_tiles,
    output logic                     mxu_accumulate,
    output logic                     mxu_requant,
    input  logic                     mxu_done,

    // ---- status (MMIO / sync; docs/picorv32_migration.md §4) ----------------
    output logic [31:0] issued,     // commands accepted into the queue
    output logic [31:0] retired,    // commands completed
    output logic [15:0] level,      // queue occupancy
    output logic        idle        // queue empty AND nothing in flight
);

    // Command opcodes. Mirrored by the producers (scalar_unit.sv's packer and
    // the firmware's tpu.h) and by iss.py — house convention is to duplicate
    // the constants with this note rather than to add a package to the file
    // list, the same way VOP_* is duplicated between scalar_unit.sv and vpu.sv.
    localparam logic [7:0] MXU_GEOM = 8'h01,
                           MXU_MM   = 8'h02;

    logic [127:0] head;
    logic         empty, pop;

    cmd_queue #(.WIDTH(128), .DEPTH(DEPTH)) u_q (
        .clk (clk), .rst_n (rst_n),
        .wr_en (cmd_we), .wr_data (cmd_wdata), .full (cmd_full),
        .empty (empty), .head (head), .pop (pop),
        .count (level)
    );

    // ---- field extraction (see the layout table in the doc) -----------------
    wire [7:0]  c_op   = head[7:0];
    wire [31:0] w0     = head[31:0];
    wire [31:0] w1     = head[63:32];
    wire [31:0] w2     = head[95:64];

    wire is_geom = (c_op == MXU_GEOM);
    wire is_mm   = (c_op == MXU_MM);

    // ---- sticky geometry ----------------------------------------------------
    logic [ADDR_W-1:0] g_arow, g_crow, g_wcol;
    logic [7:0]        g_ktiles, g_ntiles;
    logic [5:0]        g_tlen;

    assign mxu_a_row   = g_arow;
    assign mxu_c_row   = g_crow;
    assign mxu_w_col   = g_wcol;
    assign mxu_k_tiles = g_ktiles;
    assign mxu_n_tiles = g_ntiles;
    assign mxu_t_len   = g_tlen;

    // ---- per-matmul operands (combinational off the head) -------------------
    assign mxu_out_addr    = w0[31:16];
    assign mxu_act_addr    = w1[15:0];
    assign mxu_weight_addr = w1[31:16];
    assign mxu_rq_word     = w2[M0_W+N_W-1:0];
    assign mxu_accumulate  = w0[8];
    assign mxu_requant     = w0[9];
    assign mxu_tiled       = w0[10];

    // ---- sequencer ----------------------------------------------------------
    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    // A GEOM retires in the same clock it is seen. An MM asserts `start` for one
    // clock out of S_HEAD and then waits for `done`, which is the handshake the
    // array already had — the scalar unit's S_EXEC/S_WAIT pair did exactly this.
    assign mxu_start = (state == S_HEAD) && !empty && is_mm;
    assign pop       = ((state == S_HEAD) && !empty && !is_mm) ||   // GEOM or unknown
                       ((state == S_RUN)  && mxu_done);

    assign idle    = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_HEAD;
            g_arow   <= '0;
            g_crow   <= '0;
            g_wcol   <= '0;
            g_ktiles <= '0;
            g_ntiles <= '0;
            g_tlen   <= '0;
        end else begin
            case (state)
                S_HEAD: begin
                    if (!empty) begin
                        if (is_geom) begin
                            g_arow   <= w0[31:16];
                            g_crow   <= w1[15:0];
                            g_wcol   <= w1[31:16];
                            g_ktiles <= w2[7:0];
                            g_ntiles <= w2[15:8];
                            g_tlen   <= w2[21:16];
                        end else if (is_mm) begin
                            state <= S_RUN;
                        end
                        // anything else falls through and is discarded by `pop`
                    end
                end
                S_RUN: if (mxu_done) state <= S_HEAD;
                default: state <= S_HEAD;
            endcase
        end
    end

    // ---- issued / retired counters -----------------------------------------
    // Both count *commands accepted*, GEOM included, so the producer can compute
    // a sequence number by counting its own pushes and wait for exactly that one
    // (docs/picorv32_migration.md §4). Free-running; the host reads them over the
    // 'T' block and takes differences.
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
        if (rst_n && state == S_HEAD && !empty && !is_geom && !is_mm)
            $display("[%0t] cmd_mxu: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
