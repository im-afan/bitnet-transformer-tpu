`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmd_queue.sv — generic command FIFO for the macro-op dispatch path
//
// One of these sits in front of each accelerator (cmd_mxu.sv, cmd_vpu.sv,
// cmd_dma.sv). A producer — the scalar unit today, PicoRV32 once the firmware
// path lands — pushes fixed-width command words in; the unit's front end pops
// them one at a time and runs them in order.
//
// This is what replaces the config-register model. A command carries its own
// geometry, so nothing a *previous* dispatch (or a previous program) left in a
// register can reach this one. Where per-unit state genuinely amortizes — the
// MXU's operand strides — it is carried by a `GEOM` command that flows through
// this same queue, so its scope is a position in the unit's command stream
// rather than a globally visible register. See docs/picorv32_migration.md §3.
//
// Depth is the run-ahead budget: it is how far the producer may get in front of
// the unit before it stalls. `full` is what stalls it, and the producer needs no
// software flow control as a result — a store to a full queue simply does not
// complete. Data dependencies are still the program's problem (§4 of the doc).
//
// Storage is a plain 2-D array indexed by two wrapping pointers, which infers
// LUTRAM (distributed) at these depths rather than a block RAM. `head` is a
// combinational read of the storage, so a pop and the next command's decode are
// back to back with no bubble.
// -----------------------------------------------------------------------------

module cmd_queue #(
    parameter int WIDTH = 128,   // command width in bits
    parameter int DEPTH = 8      // entries; must be a power of two
) (
    input  logic             clk,
    input  logic             rst_n,

    // ---- producer side ------------------------------------------------------
    input  logic             wr_en,     // ignored when `full`
    input  logic [WIDTH-1:0] wr_data,
    output logic             full,

    // ---- consumer side ------------------------------------------------------
    output logic             empty,
    output logic [WIDTH-1:0] head,      // valid whenever !empty
    input  logic             pop,       // retire `head` (ignored when empty)

    // ---- observation --------------------------------------------------------
    output logic [15:0]      count      // entries currently held
);

    localparam int PTR_W = $clog2(DEPTH);

    logic [WIDTH-1:0]  mem [0:DEPTH-1];
    logic [PTR_W-1:0]  wptr, rptr;
    logic [PTR_W:0]    level;           // one bit wider: 0..DEPTH inclusive

    assign empty = (level == '0);
    assign full  = (level == (PTR_W+1)'(DEPTH));
    assign head  = mem[rptr];
    assign count = 16'(level);

    wire do_wr = wr_en && !full;
    wire do_rd = pop   && !empty;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr  <= '0;
            rptr  <= '0;
            level <= '0;
        end else begin
            if (do_wr) begin
                mem[wptr] <= wr_data;
                wptr      <= wptr + PTR_W'(1);
            end
            if (do_rd) rptr <= rptr + PTR_W'(1);

            case ({do_wr, do_rd})
                2'b10:   level <= level + (PTR_W+1)'(1);
                2'b01:   level <= level - (PTR_W+1)'(1);
                default: ;                       // both or neither: unchanged
            endcase
        end
    end

// synthesis translate_off
`ifndef SYNTHESIS
    // A dropped command is a silent wrong answer, so say so. The producer is
    // supposed to be stalled by `full` rather than to push through it.
    // Plain `always`: a reporting process, not synthesis intent (same convention
    // as scratchpad.sv's trace block).
    always @(posedge clk) begin
        if (rst_n && wr_en && full)
            $display("[%0t] cmd_queue: WRITE DROPPED, queue full (depth %0d)", $time, DEPTH);
    end
`endif
// synthesis translate_on

endmodule
