`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// perf_counters.sv — per-run event counters for the TPU core
//
// Replaces cycle_timer.sv, which counted exactly one event (the scalar unit's
// `busy`). Counter 0 here is bit-identical to what that module reported, so the
// first word of the UART 'T' reply is unchanged; the remaining counters are new.
//
// WHY THIS EXISTS. Total run length answers "how long did it take" and nothing
// else. Once the MXU sequences its own tile loop (docs/macro_ops.md §4) the loop
// is no longer visible in the instruction stream, so an instruction trace cannot
// tell you whether a run was weight-load bound, VPU bound, DMA bound, or just
// waiting on issue-and-wait. These counters are the only way to see that split,
// which is what docs/macro_ops.md §7 calls for.
//
// SEMANTICS. `run` (the scalar unit's `busy`) defines the measurement window:
//
//   rising edge of run  -> every counter restarts, loading ev[i] (so an event
//                          already active on the run's first clock is counted)
//   run high            -> counter i increments on each clock where ev[i] is high
//   run low             -> every counter freezes and stays readable until the
//                          next run starts
//
// Counters therefore describe *one run*, and are directly comparable with each
// other and with counter 0 (their common denominator). Events are sampled only
// inside the window: an event asserted while the core is idle is not counted,
// which is what makes "fraction of the run spent in X" a well-formed question.
//
// EDGE SEMANTICS — the subtle part, and the source of a real bug in the module
// this replaces. `run` and the unit `busy` flags are all registered outputs, so
// at the clock edge where this block first samples `run` high, the core has
// already been busy for one whole clock. Loading ev[i] on that edge (rather than
// 0) is the single compensation that makes the count exact. The predecessor
// applied that compensation *and* kept counting for one edge after `run` fell
// (its increment was gated on the delayed `run_q`, not on `run`), so every run
// read one clock long. Do not reintroduce that: the increment below is gated on
// `run`, and `run_q` is edge-detect state only.
//
// The ground truth is tb/tpu_top_uart_tb.sv's `expected_run_clocks()` —
// `(t_busy_fall - t_busy_rise) / CLK_NS`, derived from wall-clock timestamps
// rather than from this counter — checked per program on every run.
//
// OVERFLOW saturates per counter instead of wrapping, for the reason the
// predecessor gave: a wrapped count is indistinguishable from a short run, which
// is the one reading that would be believed without question. A pinned all-ones
// is at least obviously suspicious.
//
// PACKING. `counts` is the natural little-end packing — counter i occupies
// `counts[i*W +: W]`. The UART wants counter 0 transmitted *first* and shifts
// from the high end, so tpu_top.sv reverses the word order explicitly when it
// builds the 'T' bus. That reversal is deliberately not hidden in here.
// -----------------------------------------------------------------------------

module perf_counters #(
    parameter int N = 6,        // number of counters
    parameter int W = 32        // width of each counter
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           run,      // scalar_unit.busy — defines the run window
    input  logic [N-1:0]   ev,       // ev[i] high on a clock counter i should count
    output logic [N*W-1:0] counts    // counter i at counts[i*W +: W]
);

    logic run_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_q  <= 1'b0;
            counts <= '0;
        end else begin
            run_q <= run;
            for (int i = 0; i < N; i++) begin
                if (run && !run_q) begin
                    // Run starts: restart, counting this clock if already active.
                    counts[i*W +: W] <= ev[i] ? W'(1) : W'(0);
                end else if (run && ev[i] && !(&counts[i*W +: W])) begin
                    counts[i*W +: W] <= counts[i*W +: W] + W'(1);
                end
                // else: idle, event inactive, or saturated — hold.
            end
        end
    end

endmodule
