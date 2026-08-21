`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cmd_queue_tb.sv — the macro-op command FIFO
//
// Everything else in the suite drives the queues through a producer that waits
// after every dispatch, so they never hold more than one entry and `full` never
// asserts. That is exactly the state the design is supposed to leave behind, so
// the depth behaviour needs a test of its own: fill to capacity, confirm the
// back pressure, drain, and wrap the pointers several times over.
//
// The one behaviour worth being emphatic about is that a write to a full queue
// is DROPPED, not queued. That is why `full` has to stall the producer rather
// than merely inform it, and why both producers treat it as a hold: the scalar
// unit stays in S_PUSH, and the CPU's store does not get its write response.
// -----------------------------------------------------------------------------

module cmd_queue_tb;

    localparam int WIDTH  = 128;
    localparam int DEPTH  = 8;
    localparam int CLK_NS = 10;

    int errors = 0, checks = 0;

    logic clk = 0, rst_n = 0;
    always #(CLK_NS/2) clk = ~clk;

    logic             wr_en = 0, pop = 0;
    logic [WIDTH-1:0] wr_data = '0;
    logic             full, empty;
    logic [WIDTH-1:0] head;
    logic [15:0]      count;

    cmd_queue #(.WIDTH(WIDTH), .DEPTH(DEPTH)) dut (
        .clk (clk), .rst_n (rst_n),
        .wr_en (wr_en), .wr_data (wr_data), .full (full),
        .empty (empty), .head (head), .pop (pop), .count (count)
    );

    task automatic chk(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL: %s", tag);
        end
    endtask

    // A value that is distinctive in every 32-bit lane, so a word-order or
    // pointer mistake cannot look like a pass.
    function automatic logic [WIDTH-1:0] pat(input int i);
        pat = {32'hDEAD_0000 + i, 32'hC0DE_0000 + i,
               32'hBEEF_0000 + i, 32'h5A5A_0000 + i};
    endfunction

    task automatic push(input int i);
        @(negedge clk);
        wr_data = pat(i);
        wr_en   = 1'b1;
        @(negedge clk);
        wr_en   = 1'b0;
    endtask

    task automatic take(input int i, input string tag);
        chk(!empty,             {tag, ": not empty"});
        chk(head === pat(i),    {tag, ": head value"});
        @(negedge clk);
        pop = 1'b1;
        @(negedge clk);
        pop = 1'b0;
    endtask

    initial begin
        $display("==== cmd_queue testbench (depth %0d) ====", DEPTH);
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        chk(empty && !full && count == 0, "reset: empty");

        // ---- fill to capacity ------------------------------------------------
        for (int i = 0; i < DEPTH; i++) begin
            chk(!full, $sformatf("fill %0d: not full yet", i));
            push(i);
        end
        chk(full,            "full after DEPTH pushes");
        chk(!empty,          "not empty when full");
        chk(count == DEPTH,  "count == DEPTH");

        // ---- a write to a full queue must not disturb it ---------------------
        wr_data = pat(999);
        @(negedge clk); wr_en = 1'b1;
        @(negedge clk); wr_en = 1'b0;
        chk(count == DEPTH,     "overflow write did not change the level");
        chk(head === pat(0),    "overflow write did not disturb the head");

        // ---- drain in order --------------------------------------------------
        for (int i = 0; i < DEPTH; i++)
            take(i, $sformatf("drain %0d", i));
        chk(empty,       "empty after draining");
        chk(!full,       "not full after draining");
        chk(count == 0,  "count back to 0");

        // ---- wrap the pointers several times ---------------------------------
        // Push/pop one at a time past the end of the storage array, which is
        // where a pointer that does not wrap, or a level that drifts, shows up.
        for (int i = 0; i < DEPTH*3 + 3; i++) begin
            push(100 + i);
            chk(count == 1, $sformatf("wrap %0d: one entry held", i));
            take(100 + i, $sformatf("wrap %0d", i));
        end
        chk(empty, "empty after wrapping");

        // ---- simultaneous push and pop ---------------------------------------
        // The level must be unchanged; this is the steady state a producer that
        // keeps up with its unit actually runs in.
        push(200);
        push(201);
        chk(count == 2, "two queued");
        @(negedge clk);
        wr_data = pat(202);
        wr_en   = 1'b1;
        pop     = 1'b1;
        @(negedge clk);
        wr_en   = 1'b0;
        pop     = 1'b0;
        chk(count == 2,      "push+pop on one clock: level unchanged");
        chk(head === pat(201), "push+pop on one clock: head advanced");
        take(201, "concurrent 1");
        take(202, "concurrent 2");
        chk(empty, "drained after concurrent test");

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("CMD_QUEUE: ALL TESTS PASSED");
        else             $display("CMD_QUEUE: %0d FAILURES", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("CMD_QUEUE: WATCHDOG");
        $fatal(1);
    end

endmodule
