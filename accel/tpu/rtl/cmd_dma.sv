`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmd_dma.sv — DMA macro-op front end: command queue + decode + sequencing
//
// Same shape as cmd_mxu.sv / cmd_vpu.sv, but with **no sticky state at all**: a
// DMA transfer is a scratchpad address, a 19-bit DRAM address, a byte count, a
// direction, and the three transpose-geometry registers — 99 bits, which fits a
// single 128-bit command with room over. So `cfg len` / `cfg tcols` / `cfg
// tsrow` / `cfg tdrow` simply cease to exist rather than becoming a GEOM
// command.
//
// That removes the ISA's single most common silent bug by construction. The
// pitfall list in docs/isa.md §10 opens with "Set `len` before every DMA whose
// tensor size differs from the last one ... stale config is the most common
// silent bug", and `adder_model.tpu` re-issues `setcfg len` eleven times per
// layer for exactly that reason. A transfer now carries its own length.
//
// The transpose geometry keeps its zero-means-unset defaults (dma.md §5,
// `tcols -> len`, `tsrow -> tcols`, `tdrow -> 1`), which live in dma.sv and are
// unchanged — a `.t` command that leaves them zero still degrades to a plain
// copy rather than to garbage.
// -----------------------------------------------------------------------------

module cmd_dma #(
    parameter int ADDR_W     = 16,   // scratchpad byte address
    parameter int MEM_ADDR_W = 19,   // external DRAM byte address
    parameter int DEPTH      = 8
) (
    input  logic clk,
    input  logic rst_n,

    // ---- producer side ------------------------------------------------------
    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    // ---- DMA dispatch (dma.sv) ----------------------------------------------
    output logic                    dma_start,
    output logic                    dma_write,
    output logic [ADDR_W-1:0]       dma_scratch_addr,
    output logic [MEM_ADDR_W-1:0]   dma_dram_addr,
    output logic [15:0]             dma_len,
    output logic                    dma_transpose,
    output logic [15:0]             dma_tcols,
    output logic [15:0]             dma_tsrow,
    output logic [15:0]             dma_tdrow,
    input  logic                    dma_done,

    // ---- status -------------------------------------------------------------
    output logic [31:0] issued,
    output logic [31:0] retired,
    output logic [15:0] level,
    output logic        idle
);

    localparam logic [7:0] DMA_MOVE = 8'h01;

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
    wire [31:0] w3   = head[127:96];

    wire is_move = (c_op == DMA_MOVE);

    assign dma_scratch_addr = w0[31:16];
    assign dma_write        = w0[8];
    assign dma_transpose    = w0[9];
    assign dma_dram_addr    = w1[MEM_ADDR_W-1:0];
    assign dma_len          = w2[15:0];
    assign dma_tcols        = w2[31:16];
    assign dma_tsrow        = w3[15:0];
    assign dma_tdrow        = w3[31:16];

    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    assign dma_start = (state == S_HEAD) && !empty && is_move;
    assign pop       = ((state == S_HEAD) && !empty && !is_move) ||
                       ((state == S_RUN)  && dma_done);

    assign idle    = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                             state <= S_HEAD;
        else case (state)
            S_HEAD:  if (!empty && is_move)     state <= S_RUN;
            S_RUN:   if (dma_done)              state <= S_HEAD;
            default:                            state <= S_HEAD;
        endcase
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
        if (rst_n && state == S_HEAD && !empty && !is_move)
            $display("[%0t] cmd_dma: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
