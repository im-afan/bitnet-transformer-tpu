// -----------------------------------------------------------------------------
// uart_interface_tb.sv — self-checking testbench for accel/tpu/rtl/uart_interface.sv
//
// Wires the real command FSM (uart_interface) to the real uart_receiver and
// uart_transmitter, and to the real sram_controller plus the behavioral async-SRAM
// chip model (same contract as sram_tb.sv / dma_tb.sv). The host side is driven by
// bit-banging uart_rx and decoding uart_tx, exactly like uart_receiver_tb.sv /
// uart_transmitter_tb.sv, so this is a full end-to-end test of the serial link.
//
// The instruction-memory write port and the run trigger are captured by small
// behavioral models (an IMEM array and a run-pulse counter) so the 'I' and 'G'
// commands can be checked without pulling in the whole scalar unit.
//
// A small CLK_PER_BIT is used purely to keep simulation fast — the FSM is
// baud-agnostic, so functionally it is identical to the 868-clock default.
//
// Covers docs/uart_host.md §10 plus the extended command set:
//   1. write N bytes to SRAM, read back, compare (round-trip)
//   2. len==1 and a burst crossing a 256-byte boundary (address carry)
//   3. len==0, top-bits-set address, addr+len overflow -> NAK, memory untouched
//   4. unknown command byte -> NAK, FSM accepts a valid command next
//   5. mid-frame abort + RX_TIMEOUT elapse, then a fresh command decodes
//   6. back-to-back commands with no idle gap
//   7. 'I' write instruction memory + read back the IMEM model
//   8. 'G' start program: run pulse fires with the right boot PC
//   9. arbitration: any command while core_busy -> NAK; works again once idle
//  10. new-command validation: bad IMEM length/range and bad boot PC -> NAK
//  11. a 256-byte payload at an odd base (length high byte, idx carry, page cross)
//  12. turnaround: a byte sent while the device is mid-reply must NOT be lost
//  13. every exchange is followed by "and nothing else came out"
//  14. 'T' read timer: the 4-byte reply is `cycle_count`, MSB first, sampled
//      once; it is answered while core_busy (the one command that is)
//  15. outrunning the one-byte holding register sets `rx_overrun` (last: it
//      deliberately desyncs the FSM)
//
// Build:
//   make TEST=uart_interface sim        (RTL_uart_interface in tb/Makefile)
//
// RX_TIMEOUT is a *module parameter*, not a localparam, so the as-built
// configuration can be simulated without editing this file:
//
//   iverilog -g2012 -Puart_interface_tb.RX_TIMEOUT=0 -o t.vvp <sources> && vvp t.vvp
//
// Both board targets set UART_RX_TIMEOUT = 0 (synth/vivado/boards/*/board.tcl),
// which compiles the mid-frame abort out of the design entirely. A testbench that
// only ever runs with a non-zero timeout is testing a configuration nobody
// builds; test 5 is skipped when it is 0 and everything else must still pass.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module uart_interface_tb #(
    // clocks; > one byte (10*CPB) so normal frames never trip it, but small
    // enough to exercise mid-frame abort quickly. 0 = as built on both boards.
    parameter int RX_TIMEOUT = 1000
);

    // ---- Parameters ---------------------------------------------------------
    localparam int CPB         = 16;    // UART clocks per bit (fast, for sim)
    localparam int MEM_ADDR_W  = 19;
    localparam int MEM_DATA_W  = 8;
    localparam int IMEM_AW     = 10;
    localparam int MAXLEN      = 256;   // largest SRAM payload used by any test
    localparam int MAXW        = 8;     // largest IMEM payload (words) used

    localparam [7:0] WCMD = 8'h57, RCMD = 8'h52, ICMD = 8'h49, GCMD = 8'h47,
                     TCMD = 8'h54;
    localparam [7:0] ACK  = 8'h06, NAK  = 8'h15;

    // ---- Clock / reset ------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // ---- Serial pins --------------------------------------------------------
    logic uart_rx = 1'b1;   // host -> device (idles high)
    logic uart_tx;          // device -> host

    // ---- Arbitration --------------------------------------------------------
    logic core_busy = 1'b0; // driven by the TB to model a running program

    // ---- Run-length counter -------------------------------------------------
    // Stands in for cycle_timer.sv (tested on its own terms in tpu_top_uart_tb,
    // where a real program runs). Here it only has to be something the 'T' reply
    // can be compared against, and something that *moves* — a constant would not
    // catch a reply assembled from a stale or re-sampled snapshot.
    logic [31:0] cycle_count = 32'h0000_0000;

    // ---- receiver <-> FSM ---------------------------------------------------
    logic [7:0] rx_data;
    logic       rx_valid;

    // ---- FSM <-> transmitter ------------------------------------------------
    logic       tx_start;
    logic [7:0] tx_data;
    logic       tx_busy;

    // ---- FSM <-> sram_controller --------------------------------------------
    logic                   sram_start, sram_we;
    logic [MEM_ADDR_W-1:0]  sram_addr;
    logic [MEM_DATA_W-1:0]  sram_din, sram_dout;
    logic                   sram_busy, sram_done;
    logic                   host_busy;
    logic                   rx_overrun;

    // ---- FSM -> IMEM write / run trigger ------------------------------------
    logic                imem_we;
    logic [IMEM_AW-1:0]  imem_waddr;
    logic [31:0]         imem_wdata;
    logic                run_start;
    logic [IMEM_AW-1:0]  run_pc;

    // ---- sram_controller <-> SRAM chip pins ---------------------------------
    wire  [MEM_ADDR_W-1:0]  chip_addr;
    wire  [MEM_DATA_W-1:0]  chip_data;
    wire                    chip_we, chip_ce, chip_oen;

    // =========================================================================
    // DUT + real UART blocks + real SRAM controller
    // =========================================================================
    uart_receiver #(.CLK_PER_BIT(CPB)) u_rx (
        .clk(clk), .rst_n(rst_n),
        .uart_rx(uart_rx), .data(rx_data), .valid(rx_valid)
    );

    uart_transmitter #(.CLK_PER_BIT(CPB)) u_tx (
        .clk(clk), .rst_n(rst_n),
        .start(tx_start), .data(tx_data), .uart_tx(uart_tx), .busy(tx_busy)
    );

    uart_interface #(
        .ADDR_W(MEM_ADDR_W), .LENGTH_W(16), .IMEM_AW(IMEM_AW), .RX_TIMEOUT(RX_TIMEOUT)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .core_busy(core_busy), .cycle_count(cycle_count),
        .data_in(rx_data), .receiver_valid(rx_valid),
        .transmitter_start(tx_start), .data_out(tx_data), .transmitter_busy(tx_busy),
        .sram_start(sram_start), .sram_we(sram_we), .sram_addr(sram_addr),
        .sram_din(sram_din), .sram_dout(sram_dout),
        .sram_busy(sram_busy), .sram_done(sram_done),
        .imem_we(imem_we), .imem_waddr(imem_waddr), .imem_wdata(imem_wdata),
        .run_start(run_start), .run_pc(run_pc),
        .host_busy(host_busy), .rx_overrun(rx_overrun)
    );

    // One strobe per transmitted frame, taken from inside the DUT. Counting the
    // wire instead would mean telling a start bit from a data bit that happens
    // to be low — see `expect_quiet`, which is the whole reason this exists.
    int tx_frames = 0;
    always @(posedge clk) if (rst_n && tx_start) tx_frames++;

    sram_controller #(
        .CLOCKS_PER_ACCESS(2), .ADDR_W(MEM_ADDR_W), .DATA_W(MEM_DATA_W)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),
        .start(sram_start), .we(sram_we), .addr(sram_addr),
        .din(sram_din), .dout(sram_dout), .busy(sram_busy), .done(sram_done),
        .sram_addr(chip_addr), .sram_data(chip_data),
        .sram_we(chip_we), .sram_ce(chip_ce), .sram_oen(chip_oen)
    );

    // =========================================================================
    // Behavioral async SRAM chip model (matches sram_tb.sv / dma_tb.sv)
    // =========================================================================
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (chip_ce == 1'b0) && (chip_oen == 1'b0) && (chip_we == 1'b1);
    assign #3 chip_data = mem_drives ? sram_mem[chip_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge chip_we) if (chip_ce == 1'b0) sram_mem[chip_addr] <= chip_data;

    // =========================================================================
    // Behavioral IMEM + run-trigger models (stand in for the scalar unit)
    // =========================================================================
    logic [31:0] imem_model [0:(1 << IMEM_AW)-1];
    always @(posedge clk) if (imem_we) imem_model[imem_waddr] <= imem_wdata;

    int                run_count = 0;   // number of run pulses seen
    logic [IMEM_AW-1:0] run_pc_last;    // boot PC of the most recent run pulse
    always @(posedge clk) if (run_start) begin
        run_count   <= run_count + 1;
        run_pc_last <= run_pc;
    end

    // =========================================================================
    // Host-side bit-bang / decode helpers
    // =========================================================================
    int errors = 0;
    int checks = 0;

    logic [7:0]  wbuf   [0:MAXLEN-1];   // SRAM write payload
    logic [7:0]  rbuf   [0:MAXLEN-1];   // SRAM read-back
    logic [31:0] iwords [0:MAXW-1];     // IMEM write payload (32-bit words)

    task automatic chk(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL: %s  (t=%0t)", tag, $time);
        end
    endtask

    // Drive one 8N1 byte onto uart_rx (start, 8 data LSB-first, stop).
    task automatic tx_bit_byte(input [7:0] b);
        uart_rx = 1'b0;                         // start bit
        repeat (CPB) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            uart_rx = b[i];
            repeat (CPB) @(posedge clk);
        end
        uart_rx = 1'b1;                         // stop bit
        repeat (CPB) @(posedge clk);
    endtask

    // Decode one 8N1 byte from uart_tx, sampling each bit at its center.
    task automatic rx_bit_byte(output [7:0] b);
        @(negedge uart_tx);                     // start bit edge
        repeat (CPB / 2) @(posedge clk);        // align to bit center
        for (int i = 0; i < 8; i++) begin
            repeat (CPB) @(posedge clk);
            b[i] = uart_tx;
        end
        repeat (CPB) @(posedge clk);            // into stop bit
    endtask

    // Send a 6-byte command header: CMD + 24-bit addr (BE) + 16-bit len (BE).
    task automatic send_header(input [7:0] cmd, input [23:0] a, input [15:0] n);
        tx_bit_byte(cmd);
        tx_bit_byte(a[23:16]); tx_bit_byte(a[15:8]); tx_bit_byte(a[7:0]);
        tx_bit_byte(n[15:8]);  tx_bit_byte(n[7:0]);
    endtask

    // Full SRAM write: header + wbuf[0..n-1], then check the ACK reply.
    task automatic do_write(input [23:0] a, input [15:0] n);
        logic [7:0] st;
        fork
            begin
                send_header(WCMD, a, n);
                for (int i = 0; i < n; i++) tx_bit_byte(wbuf[i]);
            end
            rx_bit_byte(st);            // ACK arrives after the last byte commits
        join
        chk(st === ACK, $sformatf("write ack (got %02h)", st));
    endtask

    // Full SRAM read: header, then collect n data bytes back into rbuf[0..n-1].
    task automatic do_read(input [23:0] a, input [15:0] n);
        fork
            send_header(RCMD, a, n);
            begin
                logic [7:0] t;
                for (int i = 0; i < n; i++) begin
                    rx_bit_byte(t);
                    rbuf[i] = t;
                end
            end
        join
    endtask

    // 'I' write instruction memory: header (len = nwords*4 bytes) + word payload
    // (each word big-endian, MSB byte first), then check ACK.
    task automatic do_imem(input [23:0] a, input [15:0] nwords);
        logic [7:0] st;
        fork
            begin
                send_header(ICMD, a, 16'(nwords << 2));
                for (int w = 0; w < nwords; w++) begin
                    tx_bit_byte(iwords[w][31:24]);
                    tx_bit_byte(iwords[w][23:16]);
                    tx_bit_byte(iwords[w][15:8]);
                    tx_bit_byte(iwords[w][7:0]);
                end
            end
            rx_bit_byte(st);
        join
        chk(st === ACK, $sformatf("imem ack (got %02h)", st));
    endtask

    // 'G' go/run: 4-byte frame (CMD + 24-bit boot PC, no length), check status.
    task automatic do_go(input [23:0] a, input logic exp_ack);
        logic [7:0] st;
        fork
            begin
                tx_bit_byte(GCMD);
                tx_bit_byte(a[23:16]); tx_bit_byte(a[15:8]); tx_bit_byte(a[7:0]);
            end
            rx_bit_byte(st);
        join
        chk(st === (exp_ack ? ACK : NAK), $sformatf("go status (got %02h)", st));
    endtask

    // 'T' read timer: a bare 1-byte frame, answered with 4 bytes MSB first and
    // no status. Returns what the device reported.
    task automatic do_timer(output [31:0] v);
        logic [7:0] b;
        fork
            tx_bit_byte(TCMD);
            begin
                for (int i = 0; i < 4; i++) begin
                    rx_bit_byte(b);
                    v = {v[23:0], b};
                end
            end
        join
    endtask

    // Nothing more may come out of the device. The hardware symptom this whole
    // branch exists for is the device sending *more* bytes than the host asked
    // for, and none of the tasks above would notice: each reads exactly the
    // number of frames it expects and leaves anything extra on the wire.
    task automatic expect_quiet(input string tag);
        int quiet_tx;
        quiet_tx = tx_frames;
        repeat (20 * CPB) @(posedge clk);
        chk(tx_frames == quiet_tx,
            $sformatf("%s: %0d extra frame(s) after the exchange",
                      tag, tx_frames - quiet_tx));
    endtask

    // Send a header only (no payload) and expect a single NAK — used for
    // commands the device rejects at validation, before any data phase.
    task automatic expect_nak(input [7:0] cmd, input [23:0] a, input [15:0] n);
        logic [7:0] st;
        fork
            send_header(cmd, a, n);
            rx_bit_byte(st);
        join
        chk(st === NAK, $sformatf("expected NAK (got %02h)", st));
    endtask

    // =========================================================================
    // Stimulus
    // =========================================================================
    initial begin
        // wipe model memory
        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        chk(host_busy === 1'b0, "idle at start");

        // ---- Test 1: round-trip a multi-byte block --------------------------
        for (int i = 0; i < 8; i++) wbuf[i] = 8'(8'hA0 + i);
        do_write(24'h00_0100, 16'd8);
        expect_quiet("write ack");
        for (int i = 0; i < 8; i++)
            chk(sram_mem[24'h000100 + i] === wbuf[i],
                $sformatf("mem[%0h]", 24'h000100 + i));

        do_read(24'h00_0100, 16'd8);
        expect_quiet("read reply");
        for (int i = 0; i < 8; i++)
            chk(rbuf[i] === wbuf[i],
                $sformatf("readback[%0d] got %02h exp %02h", i, rbuf[i], wbuf[i]));

        // ---- Test 2a: len == 1 ---------------------------------------------
        wbuf[0] = 8'h5A;
        do_write(24'h00_7FFF, 16'd1);
        chk(sram_mem[24'h007FFF] === 8'h5A, "single-byte write");
        do_read(24'h00_7FFF, 16'd1);
        chk(rbuf[0] === 8'h5A, "single-byte read");

        // ---- Test 2b: burst crossing a 256-byte boundary (address carry) ----
        wbuf[0] = 8'h11; wbuf[1] = 8'h22; wbuf[2] = 8'h33; wbuf[3] = 8'h44;
        do_write(24'h00_00FE, 16'd4);         // 0x0FE,0x0FF,0x100,0x101
        chk(sram_mem[24'h0000FE] === 8'h11, "carry mem 0FE");
        chk(sram_mem[24'h0000FF] === 8'h22, "carry mem 0FF");
        chk(sram_mem[24'h000100] === 8'h33, "carry mem 100");
        chk(sram_mem[24'h000101] === 8'h44, "carry mem 101");
        do_read(24'h00_00FE, 16'd4);
        chk(rbuf[0] === 8'h11 && rbuf[1] === 8'h22 &&
            rbuf[2] === 8'h33 && rbuf[3] === 8'h44, "carry readback");

        // ---- Test 3: rejected SRAM commands leave memory untouched ----------
        sram_mem[24'h000200] = 8'hEE;
        expect_nak(WCMD, 24'h00_0200, 16'd0);                 // len == 0
        chk(sram_mem[24'h000200] === 8'hEE, "len0 leaves mem untouched");

        sram_mem[24'h000201] = 8'hEE;
        expect_nak(WCMD, 24'h08_0000, 16'd4);                 // top bit set
        chk(sram_mem[24'h000201] === 8'hEE, "bad-addr leaves mem untouched");

        sram_mem[24'h07FFFF] = 8'hEE;
        expect_nak(WCMD, 24'h07_FFFF, 16'd2);                 // addr+len overflow
        chk(sram_mem[24'h07FFFF] === 8'hEE, "overflow write not committed");

        expect_nak(RCMD, 24'h08_0000, 16'd4);                 // rejected read
        expect_nak(RCMD, 24'h00_0100, 16'd0);

        // ---- Test 4: unknown command byte, then a valid command still works --
        begin
            logic [7:0] st;
            fork
                tx_bit_byte(8'h99);   // not a known command
                rx_bit_byte(st);
            join
            chk(st === NAK, $sformatf("bad cmd NAK (got %02h)", st));
        end
        do_read(24'h00_0100, 16'd8);          // FSM recovered; reads a valid region
        for (int i = 0; i < 8; i++)
            chk(rbuf[i] === sram_mem[24'h000100 + i], "valid cmd after bad cmd");

        // ---- Test 5: mid-frame abort + RX_TIMEOUT, then fresh command --------
        // Only meaningful when the abort exists. Both board targets build with
        // UART_RX_TIMEOUT = 0, so under that configuration a half-sent frame
        // wedges the FSM until reset by design — skip rather than hang.
        if (RX_TIMEOUT != 0) begin
            tx_bit_byte(WCMD);
            tx_bit_byte(8'h00);
            repeat (RX_TIMEOUT + 4 * CPB) @(posedge clk);   // let the timeout elapse
            chk(host_busy === 1'b0, "timeout returned FSM to idle");

            wbuf[0] = 8'hDE; wbuf[1] = 8'hAD;
            do_write(24'h00_0300, 16'd2);
            do_read(24'h00_0300, 16'd2);
            chk(rbuf[0] === 8'hDE && rbuf[1] === 8'hAD, "command after timeout abort");
        end else begin
            $display("  skip: mid-frame abort (RX_TIMEOUT = 0, as built)");
        end

        // ---- Test 6: back-to-back commands, no idle gap ---------------------
        wbuf[0] = 8'h01; wbuf[1] = 8'h02; wbuf[2] = 8'h03;
        do_write(24'h00_0400, 16'd3);
        do_read (24'h00_0400, 16'd3);         // fired right after the ACK
        chk(rbuf[0] === 8'h01 && rbuf[1] === 8'h02 && rbuf[2] === 8'h03,
            "back-to-back write then read");

        // ---- Test 7: 'I' write instruction memory, read the model back ------
        iwords[0] = 32'hDEAD_BEEF;
        iwords[1] = 32'h0011_2233;
        iwords[2] = 32'hCAFE_F00D;
        iwords[3] = 32'h8000_0001;
        do_imem(24'h00_0010, 16'd4);          // 4 words at instr addr 0x010
        for (int w = 0; w < 4; w++)
            chk(imem_model[24'h000010 + w] === iwords[w],
                $sformatf("imem[%0h] got %08h exp %08h",
                          24'h000010 + w, imem_model[24'h000010 + w], iwords[w]));

        // ---- Test 8: 'G' start program — run pulse + correct boot PC --------
        begin
            int c0;
            c0 = run_count;                   // capture at block entry (not time 0)
            do_go(24'h00_0005, 1'b1);         // boot PC = 5, accepted
            chk(run_count === c0 + 1, "exactly one run pulse");
            chk(run_pc_last === 10'd5, $sformatf("boot pc (got %0d)", run_pc_last));
        end

        // ---- Test 9: arbitration — any command while core_busy is NAK'd ------
        core_busy = 1'b1;
        repeat (2) @(posedge clk);
        begin
            int c0;
            c0 = run_count;
            expect_nak(RCMD, 24'h00_0100, 16'd4);   // read rejected while busy
            expect_nak(WCMD, 24'h00_0100, 16'd4);   // write rejected while busy
            expect_nak(ICMD, 24'h00_0010, 16'd4);   // imem rejected while busy
            do_go     (24'h00_0005, 1'b0);          // go rejected while busy
            chk(run_count === c0, "no run pulse while busy");
        end
        // memory and the region under the rejected write must be unchanged
        chk(sram_mem[24'h000100] === 8'h33, "no SRAM write while busy");

        core_busy = 1'b0;
        repeat (2) @(posedge clk);
        do_read(24'h00_0100, 16'd4);               // works again once idle
        for (int i = 0; i < 4; i++)
            chk(rbuf[i] === sram_mem[24'h000100 + i], "command works after busy");

        // ---- Test 10: validation of the new commands ------------------------
        expect_nak(ICMD, 24'h00_0010, 16'd3);   // IMEM len not a multiple of 4
        expect_quiet("imem bad length");
        expect_nak(ICMD, 24'h00_0400, 16'd4);   // IMEM addr out of range (>=1024)
        expect_nak(ICMD, 24'h00_03FF, 16'd8);   // IMEM addr+words overflow
        do_go     (24'h00_0400, 1'b0);          // boot PC out of range -> NAK
        expect_quiet("bad boot pc");

        // ---- Test 11: a payload longer than one address byte ----------------
        // Every payload above is <= 16 bytes, which exercises neither the length
        // field's high byte nor `idx`'s carry out of bit 7. 256 bytes at an odd
        // base does both, and crosses a 256-byte SRAM page while it is at it.
        for (int i = 0; i < 256; i++) wbuf[i] = 8'((i * 8'h1B) ^ 8'h5A);
        do_write(24'h00_10FF, 16'd256);
        expect_quiet("256B write");
        do_read (24'h00_10FF, 16'd256);
        expect_quiet("256B read");
        begin
            int bad;
            bad = 0;
            for (int i = 0; i < 256; i++) if (rbuf[i] !== wbuf[i]) bad++;
            chk(bad == 0, $sformatf("256-byte round trip (%0d bytes wrong)", bad));
        end

        // ---- Test 12: turnaround — a byte arriving mid-reply is not lost -----
        //
        // The regression test for the RX holding register, and the one gap that
        // let the hardware bug through: no test above ever sends a byte while
        // the device is transmitting. SEND_STATUS + SEND_STATUS_WAIT last a full
        // byte time and sit exactly where a pipelined host puts its next command
        // byte, and before the holding register existed the FSM consumed
        // `rx_byte` only in IDLE/RX_ADDR/RX_LEN/WR_RX/IMEM_RX and silently threw
        // that byte away — one lost byte, permanently desynced link.
        wbuf[0] = 8'h7E;
        begin
            logic [7:0] st, b;
            fork
                begin
                    send_header(WCMD, 24'h00_0500, 16'd1);
                    tx_bit_byte(wbuf[0]);
                    tx_bit_byte(RCMD);     // pipelined: lands inside the ACK
                end
                rx_bit_byte(st);
            join
            chk(st === ACK, $sformatf("pipelined write ack (got %02h)", st));
            chk(rx_overrun === 1'b0, "single pipelined byte is not an overrun");

            repeat (4 * CPB) @(posedge clk);
            chk(host_busy === 1'b1,
                "pipelined 'R' survived the reply turnaround (FSM is in RX_ADDR)");

            if (host_busy === 1'b1) begin
                // Finish the frame the pipelined byte started and check the data.
                fork
                    begin
                        tx_bit_byte(8'h00); tx_bit_byte(8'h05); tx_bit_byte(8'h00);
                        tx_bit_byte(8'h00); tx_bit_byte(8'h01);
                    end
                    rx_bit_byte(b);
                join
                chk(b === wbuf[0],
                    $sformatf("read from the pipelined command (got %02h exp %02h)",
                              b, wbuf[0]));
                expect_quiet("pipelined read");
            end
        end

        // ---- Test 14: 'T' read the run-length counter -----------------------
        begin
            logic [31:0] t;

            // 14a: the reply is the counter, MSB first, and nothing else.
            cycle_count = 32'h1234_5678;
            do_timer(t);
            chk(t === 32'h1234_5678,
                $sformatf("timer reply (got %08h exp 12345678)", t));
            expect_quiet("timer reply");

            // 14b: the counter is sampled once, at the command byte. It is a
            // free-running count in the real design, so a reply assembled from
            // four separate samples would mix bytes from different values —
            // change it a byte into the reply and none of that may show up.
            cycle_count = 32'hAABB_CCDD;
            fork
                do_timer(t);
                begin
                    repeat (12 * CPB) @(posedge clk);   // ~1 reply byte in
                    cycle_count = 32'h0000_0001;
                end
            join
            chk(t === 32'hAABB_CCDD,
                $sformatf("timer snapshot is coherent (got %08h exp AABBCCDD)", t));

            // 14c: 'T' is the one command answered while the core is running —
            // it touches no memory, and watching the count rise mid-run is the
            // reason it exists.
            core_busy   = 1'b1;
            cycle_count = 32'h0000_2A00;
            repeat (2) @(posedge clk);
            do_timer(t);
            chk(t === 32'h0000_2A00,
                $sformatf("timer answered while core_busy (got %08h)", t));
            expect_quiet("timer while busy");
            core_busy = 1'b0;
            repeat (2) @(posedge clk);

            // ...and the FSM is still in step afterwards.
            do_read(24'h00_0100, 16'd4);
            for (int i = 0; i < 4; i++)
                chk(rbuf[i] === sram_mem[24'h000100 + i], "read after timer command");
        end

        // Nothing above overlapped by more than one byte, so the device must
        // never have run out of room.
        chk(rx_overrun === 1'b0, "no RX overrun across the whole protocol suite");

        // Last, because it deliberately overruns the holding register and leaves
        // the FSM desynced.
        test_overrun_is_reported();

        repeat (4) @(posedge clk);

        if (errors == 0) $display("ALL TESTS PASSED (%0d checks)", checks);
        else             $display("FAIL %0d ERRORS (%0d checks)", errors, checks);

        $finish;
    end

    // =========================================================================
    // Overrun reporting — runs last, because it leaves the FSM desynced.
    //
    // One byte of buffering is what docs/uart_host.md §5 specifies and what the
    // protocol needs (the host cannot be more than one byte ahead without having
    // seen the reply). Two is beyond it. What matters is that going beyond it is
    // *reported* rather than silent: `rx_overrun` is the difference between "a
    // byte was lost, and here is the clock it happened on" and four kilobytes of
    // NAKs with no idea where they started.
    // =========================================================================
    task automatic test_overrun_is_reported();
        logic [7:0] b;
        $display("");
        $display("---- overrun reporting ----");
        chk(rx_overrun === 1'b0, "overrun clear before the deliberate overrun");

        // Read 4 bytes and shove 4 command bytes in behind it. The first is held,
        // the rest have nowhere to go.
        fork
            begin
                send_header(RCMD, 24'h00_0100, 16'd4);
                repeat (2 * CPB) @(posedge clk);
                tx_bit_byte(8'hAA);
                tx_bit_byte(8'hBB);
                tx_bit_byte(8'hCC);
                tx_bit_byte(8'hDD);
            end
            begin
                for (int i = 0; i < 4; i++) rx_bit_byte(b);
            end
        join
        chk(rx_overrun === 1'b1,
            "rx_overrun set when the host outran the holding register");
        $display("---------------------------");
        $display("");
    endtask

    // Safety net so a hang (e.g. an expected reply that never comes) fails loudly
    // instead of running forever.
    initial begin
        #40_000_000;   // 40 ms
        $display("  FAIL: global timeout — DUT appears hung");
        $finish;
    end

    initial begin
        $dumpfile("uart_interface_tb.vcd");
        $dumpvars(0, uart_interface_tb);
    end

endmodule
