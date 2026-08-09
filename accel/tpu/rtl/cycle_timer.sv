`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cycle_timer.sv — run-length counter for the scalar unit
//
// Times one program run in core clocks. `busy` is the scalar unit's status
// output (scalar_unit.sv: high from the S_FETCH that follows `host_run` until
// the program reaches S_HALT), so this counts exactly the interval between
// "the host said go" and "the program halted", with no software instrumentation
// in the program itself and no free-running timebase to subtract.
//
//   rising edge of busy   -> the count restarts at this clock
//   busy high             -> count up, one per clock
//   falling edge of busy  -> freeze; the value stays readable until the next run
//
// So after a run `cycles` is the number of clocks `busy` was high, and while a
// run is in progress it is the number elapsed so far. At 12 MHz one clock is
// 83.3 ns, so a 32-bit count spans 358 s — longer than any run this machine is
// built for, and longer than the host's own run timeout.
//
// EDGE SEMANTICS. `busy` is a registered output, so at the clock edge where this
// block first samples it high, the core has already been busy for one whole
// clock. That first clock is the reason the rising-edge branch loads 1 rather
// than 0, and the reason the count continues off `busy_q` for one more edge
// after `busy` drops: between them the two make `cycles` the exact number of
// clocks `busy` was high, rather than that number minus one.
//
// OVERFLOW saturates instead of wrapping. A wrapped count is indistinguishable
// from a short run, which is the one reading that would be believed without
// question; a pinned 0xFFFF_FFFF is at least obviously suspicious. It is still
// ambiguous with a genuine full-scale count, which is why the width is chosen so
// that reaching it means something has gone wrong rather than something has run
// for a while.
//
// Reset clears the count. There is no run to report before the first one.
// -----------------------------------------------------------------------------

module cycle_timer #(
    parameter int W = 32       // counter width in bits
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         busy,      // scalar_unit.busy
    output logic [W-1:0] cycles     // clocks the last (or current) run has taken
);

    logic busy_q;
    wire  full = &cycles;           // saturated (see header)

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            cycles <= '0;
        end else begin
            busy_q <= busy;
            if (busy && !busy_q)          cycles <= W'(1);          // run starts
            else if (busy_q && !full)     cycles <= cycles + W'(1); // running
            // else: idle, or saturated — hold.
        end
    end

endmodule
