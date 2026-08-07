`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// uart_memory_cosim_tb.sv — rtl/uart_memory.sv driven by the *real* Python host.
//
//   make cosim                       (see host/sim_link.py for the other half)
//
// uart_memory_tb.sv drives the DUT from hand-written SystemVerilog tasks: it is
// a model of what the host does. This one removes the model. The serial pins are
// bridged to two text files, host/sim_link.py presents the far end of that bridge
// as a `serial.Serial` work-alike, and host/test_uart_link.py — unmodified in
// every way that matters — talks to it exactly as it talks to the board. Same
// frame encoder, same reply parser, same reject cases, same resync logic.
//
// That is the point: the RTL half of the link has been simulated for a while and
// passes, so if the failure is a host/device disagreement rather than a bad flop,
// a testbench that encodes the host's behaviour by hand cannot see it — it would
// have to make the same mistake twice, in two languages.
//
// The DUT is instantiated with the cmod_a7_mem bitstream's own parameters
// (boards/cmod_a7_mem/board.tcl): CPB 104, RX_TIMEOUT 0, 19-bit SRAM, CPA 2.
// The clock here is 100 MHz rather than the board's 12 MHz, which changes
// nothing the design can observe — every timing in this link is counted in
// clocks, and CLK_PER_BIT is the same 104 — but it makes a byte cost 10.4 us of
// simulated time instead of 86.7 us, so a run finishes 8x sooner. sim_link.py
// therefore measures its own timeouts in bit times, not seconds.
//
// ---- Wire protocol of the bridge --------------------------------------------
//
// host -> device (`+h2d=`, appended by Python, one record per line):
//
//   0xx   a byte to shift out of uart_rx, 8N1, back to back with its neighbours
//   100   shut the simulation down
//
// device -> host (`+d2h=`, appended here, one record per line):
//
//   P <cpb> <clk_ns> <bit_ns>        preamble: the bridge's timing constants
//   R <time>                         reset released, link live
//   B <hh> <time> <stop_ok>          a byte was decoded off uart_tx
//   T <time> <consumed> <sent>       heartbeat: sim clock + both byte counters
//   E <what> <time>                  rx_overrun / collision went high (sticky)
//   F <time> <state> <pend> <hold>   uart_interface FSM step   (+fsm only)
//   X <time> <consumed> <sent>       simulation finished
//
// `consumed` is the number of host bytes fully clocked in, so the host can tell
// "the device owes me a reply" from "the device has not been given the command
// yet" — without it, every timeout would be ambiguous. Heartbeats are what let
// the host time out at all: with no wall clock in here, silence is only
// measurable against simulated time.
//
// ---- Plusargs ---------------------------------------------------------------
//
//   +h2d=FILE    host -> device stream          (default cosim_h2d.txt)
//   +d2h=FILE    device -> host stream          (default cosim_d2h.txt)
//   +fsm         log every uart_interface state change (large; on demand)
//   +vcd=FILE    dump waves (off by default — it costs ~3x the run time)
//   +idle_ms=N   give up after N ms of simulated time with no host byte
// -----------------------------------------------------------------------------

module uart_memory_cosim_tb #(
    // Overridable from the command line, so the one open question in CLAUDE.md's
    // experiment list can be answered without editing this file:
    //
    //   iverilog -Puart_memory_cosim_tb.RX_TIMEOUT=2080 ...
    //
    // 0 is what both board targets build, and what compiles uart_interface's
    // mid-frame abort out of the design entirely.
    parameter int RX_TIMEOUT = 0
);

    // ---- Geometry / protocol, as boards/cmod_a7_mem/board.tcl builds it ------
    localparam int CPB        = 104;        // UART_CPB: 12 MHz / 115200
    localparam int CLK_NS     = 10;         // 100 MHz simulation clock
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int HB_BITS    = 20;         // heartbeat period, in bit times

    // Idle poll backoff. Checking the h2d stream costs three host system calls
    // and buys 8 clocks of simulated time, so polling at a fixed fast rate makes
    // the *device's* own 128-byte reply — 133 000 clocks in which the host by
    // definition has nothing to say — the dominant cost of a run. Backing off
    // after a stretch of silence cuts that by ~30x; the price is up to 4 bit
    // times of extra latency on the first byte of a command, which is smaller
    // than the USB scheduling jitter the real link has anyway.
    localparam int POLL_FAST  = 8;          // clocks, while the host is talking
    localparam int POLL_MED   = 104;        // one bit time
    localparam int POLL_SLOW  = 416;        // four bit times
    localparam int FAST_POLLS = 64;         // empty polls before stepping down
    localparam int MED_POLLS  = 256;

    // ---- Clock / reset -------------------------------------------------------
    logic clk = 1'b0, rst_n = 1'b0;
    always #(CLK_NS/2) clk = ~clk;

    // ---- Pins ----------------------------------------------------------------
    logic uart_rx = 1'b1;                   // host -> device (idles high)
    wire  uart_tx;                          // device -> host

    wire [MEM_ADDR_W-1:0] sram_addr;
    wire [MEM_DATA_W-1:0] sram_data;        // inout: shared with the chip model
    wire                  sram_we, sram_ce, sram_oen;

    wire blink_slow, blink_fast, activity, collision;

    uart_memory #(
        .UART_CPB        (CPB),
        .UART_RX_TIMEOUT (RX_TIMEOUT),      // as built: board.tcl sets 0
        .MEM_ADDR_W      (MEM_ADDR_W),
        .MEM_DATA_W      (MEM_DATA_W),
        .SRAM_CPA        (2),
        .IMEM_AW         (10),
        .ACT_W           (6),               // tiny, so the LED logic still toggles
        .HB_W            (8)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),

        .uart_rx    (uart_rx),
        .uart_tx    (uart_tx),

        .sram_addr  (sram_addr),
        .sram_data  (sram_data),
        .sram_we    (sram_we),
        .sram_ce    (sram_ce),
        .sram_oen   (sram_oen),

        .blink_slow (blink_slow),
        .blink_fast (blink_fast),
        .activity   (activity),
        .collision  (collision)
    );

    // =========================================================================
    // Behavioral async SRAM chip model (identical to uart_memory_tb.sv).
    // =========================================================================
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    // =========================================================================
    // Bridge state.
    // =========================================================================
    integer fh, fd;                   // h2d (read), d2h (write)
    integer consumed = 0;             // host bytes fully clocked into the DUT
    integer sent     = 0;             // bytes decoded off uart_tx
    bit     bridge_up = 1'b0;
    bit     trace_fsm = 1'b0;
    integer idle_limit_ns;            // no host byte for this long -> give up
    integer last_byte_ns = 0;

    // =========================================================================
    // Host -> device.
    // =========================================================================

    // 8N1 out of the host: start low, 8 data bits LSB first, stop high. No
    // trailing idle — consecutive bytes go out back to back, which is what an
    // FTDI streaming at line rate does and what the RX holding register is
    // there to survive.
    task automatic drive_byte(input logic [7:0] b);
        uart_rx = 1'b0;
        repeat (CPB) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            uart_rx = b[i];
            repeat (CPB) @(posedge clk);
        end
        uart_rx = 1'b1;
        repeat (CPB) @(posedge clk);
    endtask

    // Pull one record off the h2d stream, or report that none is ready.
    //
    // Two things stop this from being a plain $fgets. A read that hits EOF
    // leaves the EOF flag set, and every later read returns nothing even after
    // Python appends — so the position is restored with $fseek, which clears it.
    // And Python's append is not atomic with respect to this read, so a record
    // can be seen half written; a line whose last character is not a newline is
    // put back and retried rather than parsed into a byte that was never sent.
    task automatic next_record(output bit ok, output integer val);
        integer pos, code;
        reg [8*32-1:0] line;
        begin
            ok   = 1'b0;
            val  = 0;
            pos  = $ftell(fh);
            code = $fgets(line, fh);
            if (code == 0)                   code = $fseek(fh, pos, 0);  // EOF
            else if (line[7:0] !== 8'h0A)    code = $fseek(fh, pos, 0);  // partial
            else if ($sscanf(line, "%h", val) == 1) ok = 1'b1;
            else                             code = $fseek(fh, pos, 0);
        end
    endtask

    task automatic shutdown(input reg [8*40-1:0] why);
        $fdisplay(fd, "X %0d %0d %0d", $time, consumed, sent);
        $fflush(fd);
        $display("cosim: %0s at %0t — %0d bytes in, %0d bytes out, rx_overrun=%0b, collision=%0b",
                 why, $time, consumed, sent, dut.uart_rx_overrun, collision);
        $fclose(fd);
        $finish;
    endtask

    initial begin : host_to_device
        bit     ok;
        integer val;
        integer quiet;
        quiet = 0;
        wait (bridge_up);
        forever begin
            next_record(ok, val);
            if (ok) begin
                quiet = 0;
                if (val == 'h100) shutdown("host closed the link");
                else begin
                    drive_byte(val[7:0]);
                    consumed++;
                    last_byte_ns = $time;
                end
            end else begin
                if      (quiet < FAST_POLLS) repeat (POLL_FAST) @(posedge clk);
                else if (quiet < MED_POLLS)  repeat (POLL_MED)  @(posedge clk);
                else                         repeat (POLL_SLOW) @(posedge clk);
                quiet++;
                if ($time - last_byte_ns > idle_limit_ns)
                    shutdown("host went silent (idle watchdog)");
            end
        end
    end

    // =========================================================================
    // Device -> host. Decoded the way the host's FTDI decodes it: find the
    // falling edge, wait a bit and a half to land mid-data-bit-0, then step a
    // bit at a time, finishing at the centre of the stop bit — half a bit before
    // the next start edge can occur, so back-to-back frames do not race.
    // =========================================================================
    initial begin : device_to_host
        logic [7:0] b;
        bit         stop_ok;
        wait (bridge_up);
        forever begin
            @(negedge uart_tx);
            repeat (CPB + CPB/2) @(posedge clk);
            for (int i = 0; i < 8; i++) begin
                b[i] = uart_tx;
                repeat (CPB) @(posedge clk);
            end
            stop_ok = (uart_tx === 1'b1);
            sent++;
            $fdisplay(fd, "B %02h %0d %0d", b, $time, stop_ok);
            $fflush(fd);
        end
    end

    // Heartbeat. The host has no wall clock on this side, so silence is only
    // measurable against simulated time: these lines are what let a missing
    // reply become a timeout instead of a hang.
    initial begin : heartbeat
        wait (bridge_up);
        forever begin
            repeat (HB_BITS * CPB) @(posedge clk);
            $fdisplay(fd, "T %0d %0d %0d", $time, consumed, sent);
            $fflush(fd);
        end
    end

    // =========================================================================
    // Instrumentation. Both flags are sticky in the DUT, so edge-detect them
    // here to report *when* — a one-clock event is the whole diagnosis and the
    // host cannot see it any other way.
    // =========================================================================
    bit ovr_seen = 1'b0, col_seen = 1'b0;
    logic [3:0] fsm_prev = 4'hF;

    always @(posedge clk) if (bridge_up) begin
        if (dut.uart_rx_overrun && !ovr_seen) begin
            ovr_seen <= 1'b1;
            $fdisplay(fd, "E rx_overrun %0d", $time);
            $fflush(fd);
        end
        if (collision && !col_seen) begin
            col_seen <= 1'b1;
            $fdisplay(fd, "E collision %0d", $time);
            $fflush(fd);
        end
        if (trace_fsm && dut.u_uart.state !== fsm_prev) begin
            fsm_prev <= dut.u_uart.state;
            $fdisplay(fd, "F %0d %0d %0d %02h", $time, dut.u_uart.state,
                      dut.u_uart.rx_pending, dut.u_uart.rx_hold);
            $fflush(fd);
        end
    end

    // =========================================================================
    // Bring-up.
    // =========================================================================
    initial begin : bringup
        reg [8*256-1:0] hp, dp, vcd;
        integer idle_ms;
        integer got;

        hp  = "cosim_h2d.txt";
        dp  = "cosim_d2h.txt";
        got = $value$plusargs("h2d=%s", hp);
        got = $value$plusargs("d2h=%s", dp);
        trace_fsm = ($test$plusargs("fsm") != 0);

        idle_ms = 60;
        got = $value$plusargs("idle_ms=%d", idle_ms);
        idle_limit_ns = idle_ms * 1000000;

        if ($value$plusargs("vcd=%s", vcd)) begin
            $dumpfile(vcd);
            $dumpvars(0, uart_memory_cosim_tb);
        end

        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;

        fh = $fopen(hp, "rb");
        if (fh == 0) begin
            $display("cosim: cannot open %0s for reading", hp);
            $finish;
        end
        fd = $fopen(dp, "wb");
        if (fd == 0) begin
            $display("cosim: cannot open %0s for writing", dp);
            $finish;
        end

        $fdisplay(fd, "P %0d %0d %0d", CPB, CLK_NS, CPB * CLK_NS);
        $fflush(fd);

        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (8) @(posedge clk);

        last_byte_ns = $time;
        $fdisplay(fd, "R %0d", $time);
        $fflush(fd);
        $display("cosim: link up (CPB=%0d, %0d ns clock, %0d ns/bit) — %0s <-> %0s",
                 CPB, CLK_NS, CPB * CLK_NS, hp, dp);
        bridge_up = 1'b1;
    end

endmodule
