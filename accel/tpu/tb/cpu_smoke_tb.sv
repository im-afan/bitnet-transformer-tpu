`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cpu_smoke_tb.sv — PicoRV32 as the command producer, end to end
//
// The scalar unit's path is covered by tpu_top_tb / tpu_top_uart_tb, which run
// real tpulang programs against golden vectors. This test covers the *other*
// producer: boot the CPU out of firmware RAM and check that a store to a command
// port actually moves bytes.
//
// What it exercises, in one run: CPU reset release via `boot_pc`'s producer bit,
// AXI4-Lite instruction fetch out of firmware RAM, AXI4-Lite stores into the DMA
// command aperture, commit-on-the-fourth-word, cmd_dma's queue and decode, the
// real DMA engine and SRAM controller, the sequence-number handshake the
// firmware polls on, and MMIO_DONE raising the core's `done`.
//
// NO RISC-V TOOLCHAIN IS NEEDED. `fw_smoke.hex` is checked in, hand-encoded, and
// its listing is below. That is deliberate: this testbench is the bring-up path
// for the CPU, and it should not be gated on an installed cross-compiler. Real
// kernels come from gcc; this one does not need to.
//
//   00: 800000b7  lui   x1, 0x80000        ; x1 = MMIO base
//   04: 01000137  lui   x2, 0x01000
//   08: 00110113  addi  x2, x2, 1          ; w0 = spad 0x0100, op=MOVE, fill
//   0c: 0220a023  sw    x2, 0x20(x1)
//   10: 000011b7  lui   x3, 0x1            ; w1 = DRAM 0x1000
//   14: 0230a223  sw    x3, 0x24(x1)
//   18: 04000213  addi  x4, x0, 64         ; w2 = len 64, tcols 0
//   1c: 0240a423  sw    x4, 0x28(x1)
//   20: 0200a623  sw    x0, 0x2c(x1)       ; w3 = 0 -> COMMIT
//   24: 01000137  lui   x2, 0x01000
//   28: 10110113  addi  x2, x2, 0x101      ; w0 = spad 0x0100, op=MOVE, spill
//   2c: 0220a023  sw    x2, 0x20(x1)
//   30: 000021b7  lui   x3, 0x2            ; w1 = DRAM 0x2000
//   34: 0230a223  sw    x3, 0x24(x1)
//   38: 0240a423  sw    x4, 0x28(x1)       ; w2 = len 64 (x4 still holds it)
//   3c: 0200a623  sw    x0, 0x2c(x1)       ; COMMIT
//   40: 00200313  addi  x6, x0, 2
//   44: 0640a283  wait: lw x5, 0x64(x1)    ; dma_retired
//   48: fe629ee3  bne   x5, x6, wait
//   4c: 0600a823  sw    x0, 0x70(x1)       ; MMIO_DONE
//   50: 0000006f  spin: jal x0, spin
//
// The firmware copies DRAM 0x1000..0x103F into the scratchpad and back out to
// DRAM 0x2000, so the check is a 64-byte compare against what was seeded — a
// round trip through the scratchpad that only completes if every layer above
// worked.
// -----------------------------------------------------------------------------

`ifndef WATCHDOG_NS
  `define WATCHDOG_NS 2000000
`endif

module cpu_smoke_tb;

    localparam int ROWS       = 8;
    localparam int COLS       = 8;
    localparam int VPU_BYTES  = 32;
    localparam int ADDR_W     = 16;
    localparam int XLEN       = 32;
    localparam int M0_W       = 12;
    localparam int N_W        = 4;
    localparam int IMEM_AW    = 10;
    localparam int CFG_AW     = 5;
    localparam int FW_AW      = 12;
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int CLK_NS = 10;

    localparam [MEM_ADDR_W-1:0] SRC_DRAM = 19'h1000;
    localparam [MEM_ADDR_W-1:0] DST_DRAM = 19'h2000;
    localparam int              NBYTES   = 64;

    int errors = 0, checks = 0;

    logic clk = 0, rst_n = 0;
    always #(CLK_NS/2) clk = ~clk;

    logic                  host_run;
    logic [FW_AW:0]        boot_pc;
    logic                  busy, done;
    logic [IMEM_AW-1:0]    pc_dbg;

    wire  [MEM_ADDR_W-1:0] sram_addr;
    wire  [MEM_DATA_W-1:0] sram_data;
    wire                   sram_we, sram_ce, sram_oen;
    wire                   uart_tx;

    tpu_top #(
        .ROWS(ROWS), .COLS(COLS), .VPU_BYTES(VPU_BYTES), .ADDR_W(ADDR_W),
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .FW_AW(FW_AW), .FW_INIT("fw_smoke.hex"),
        .MEM_STYLE("BRAM"), .MEM_ADDR_W(MEM_ADDR_W), .MEM_DATA_W(MEM_DATA_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .host_run(host_run), .boot_pc(boot_pc),
        .busy(busy), .done(done), .pc_dbg(pc_dbg),
        .imem_we(1'b0), .imem_waddr('0), .imem_wdata('0),
        .cfg_we(1'b0), .cfg_waddr('0), .cfg_wdata('0),
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen),
        .uart_rx(1'b1), .uart_tx(uart_tx)
    );

    // ---- behavioural async SRAM chip (same model as tpu_top_tb) -------------
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    task automatic expect_byte(input [MEM_ADDR_W-1:0] a, input [7:0] want,
                               input string tag);
        checks++;
        if (sram_mem[a] !== want) begin
            errors++;
            $display("  FAIL [%s] DRAM[0x%05h] = 0x%02h, expected 0x%02h",
                     tag, a, sram_mem[a], want);
        end
    endtask

    // ---- stimulus -----------------------------------------------------------
    initial begin
        $display("==== PicoRV32 command-producer smoke test ====");

        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;
        // A pattern that is not a constant and not the address, so a stuck bus
        // or an off-by-one both show up.
        for (int i = 0; i < NBYTES; i++)
            sram_mem[SRC_DRAM + MEM_ADDR_W'(i)] = 8'((i * 7) + 3);

        host_run = 1'b0;
        boot_pc  = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        // Top bit of boot_pc selects the producer: 1 = the CPU's firmware.
        @(negedge clk);
        boot_pc  = (1 << FW_AW);
        host_run = 1'b1;
        @(negedge clk);
        host_run = 1'b0;

        checks++;
        if (!busy) begin
            errors++;
            $display("  FAIL: core did not go busy after starting the CPU");
        end

        // The firmware polls dma_retired, so `done` only arrives if the queue,
        // the DMA and the status readback all work.
        wait (done);
        $display("firmware signalled done after %0t ns", $time);

        repeat (4) @(negedge clk);
        checks++;
        if (busy) begin
            errors++;
            $display("  FAIL: core still busy after done");
        end

        // `done` must LATCH, not pulse. Raising it takes the CPU back into reset
        // (tpu_top drops cpu_run in response), which clears both of the transient
        // signals that produced it -- so a naive implementation gives a two-clock
        // pulse that a polling host misses. It has to hold until the next run,
        // exactly as the scalar unit parks in S_HALT.
        repeat (50) @(negedge clk);
        checks++;
        if (!done) begin
            errors++;
            $display("  FAIL: done did not latch (pulsed instead of holding)");
        end

        for (int i = 0; i < NBYTES; i++)
            expect_byte(DST_DRAM + MEM_ADDR_W'(i), 8'((i * 7) + 3), "roundtrip");

        // The copy must not have disturbed the byte just past the range.
        expect_byte(DST_DRAM + MEM_ADDR_W'(NBYTES), 8'h00, "no-overrun");

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("CPU_SMOKE: ALL TESTS PASSED");
        else             $display("CPU_SMOKE: %0d FAILURES", errors);
        $finish;
    end

    initial begin
        #(`WATCHDOG_NS);
        $display("CPU_SMOKE: WATCHDOG - firmware never signalled done (busy=%b done=%b)", busy, done);
        $fatal(1);
    end

endmodule
