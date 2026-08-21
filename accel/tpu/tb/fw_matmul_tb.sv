`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// fw_matmul_tb.sv — the C firmware matmul through the whole core.
//
// cpu_smoke_tb proves the CPU can push a DMA command. This proves it can drive
// the array: ../fw/matmul.c stages A and W in, issues one `matmul_t` over a 4x2
// tile grid, and spills the int32 C back — the first exercise of the CPU -> MXU
// path, and the simulation half of host/run_fw_matmul.py.
//
// The firmware is loaded through cpu_subsys.sv's FW_INIT ($readmemh), not over
// the UART, so this is a backdoor test in the same shape as tpu_top_tb: seed
// DRAM directly, pulse host_run with the producer bit set, wait for `done`,
// check DRAM. Nothing drives the serial pins.
//
//   make fw                     ../fw/matmul.hex
//   make fw FWPROG=matmul_loop  ../fw/matmul_loop.hex — same product, tile grid
//                               walked in C, so the same expectations apply
//
// It needs a built image, which needs a RISC-V cross gcc, which is why it is not
// part of `make all` — the same reason tb/fw_smoke.hex is hand-encoded.
//
// Operands and reference are generated here with the same formulas
// host/run_fw_matmul.py uses, so the two agree by construction:
//
//   A[m][k] = (m*3 + k*5) % 9 - 4      int8, row-major at 0x0000
//   W[k][n] = (k + 3*n) % 3 - 1        trits, col-major 2-bit at 0x0400
//   C[m][n] = sum_k A[m][k] * W[k][n]  int32, row-major at 0x1000
// -----------------------------------------------------------------------------

`ifndef FW_HEX
  `define FW_HEX "../fw/matmul.hex"
`endif
// Problem shape. Must match what the firmware was built with -- ../fw/Makefile
// takes the same three numbers, and run_fw_sweep.sh passes both from one place.
`ifndef FW_M
  `define FW_M 8
`endif
`ifndef FW_KT
  `define FW_KT 4
`endif
`ifndef FW_NT
  `define FW_NT 2
`endif
`ifndef WATCHDOG_NS
  `define WATCHDOG_NS 2000000
`endif

module fw_matmul_tb;

    localparam int ROWS       = 8;
    localparam int COLS       = 8;
    localparam int VPU_BYTES  = 32;
    localparam int ADDR_W     = 16;
    localparam int XLEN       = 32;
    localparam int M0_W       = 12;
    localparam int N_W        = 4;
    localparam int REG_AW     = 5;
    localparam int IMEM_AW    = 10;
    localparam int CFG_AW     = 5;
    localparam int FW_AW      = 12;
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int CLK_NS = 10;

    // ---- the problem (mirrors fw/matmul.c) ----------------------------------
    localparam int M      = `FW_M;
    localparam int KTILES = `FW_KT;
    localparam int NTILES = `FW_NT;
    localparam int K      = KTILES * ROWS;
    localparam int N      = NTILES * COLS;

    localparam int AROW = K;                 // A row stride, bytes
    localparam int WCOL = (K * 2) / 8;       // W column stride, bytes
    localparam int CROW = N * 4;             // C row stride, bytes

    // Bases spaced for shapes past the default: A <= 8 KB, W <= 8 KB, C <= 48 KB.
    localparam [MEM_ADDR_W-1:0] A_ADDR = 19'h0_0000;
    localparam [MEM_ADDR_W-1:0] W_ADDR = 19'h0_2000;
    localparam [MEM_ADDR_W-1:0] C_ADDR = 19'h0_4000;
    localparam int              A_BYTES = M * K;
    localparam int              W_BYTES = N * WCOL;
    localparam int              C_BYTES = M * N * 4;

    int errors = 0, checks = 0;
    int unsigned cyc = 0, run_clk = 0;

    int a_full [0:M-1][0:K-1];
    int w_full [0:N-1][0:K-1];               // w_full[n][k] is W[k][n]
    int c_ref  [0:M-1][0:N-1];

    logic clk = 0, rst_n = 0;
    always #(CLK_NS/2) clk = ~clk;
    always @(posedge clk) cyc++;

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
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W),
        .REG_AW(REG_AW), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .FW_AW(FW_AW), .FW_INIT(`FW_HEX),
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

    // ---- operands ------------------------------------------------------------
    logic [7:0] wbyte;
    logic [1:0] code;

    task automatic seed_dram();
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) begin
                a_full[m][k] = ((m * 3 + k * 5) % 9) - 4;
                sram_mem[A_ADDR + MEM_ADDR_W'(m * AROW + k)] = 8'(a_full[m][k]);
            end

        // W: one column of K trits per output feature, 2-bit packed four to a
        // byte (00 = 0, 01 = +1, 11 = -1 — scratchpad.md §2).
        for (int n = 0; n < N; n++) begin
            for (int k = 0; k < K; k++)
                w_full[n][k] = ((k + 3 * n) % 3) - 1;
            for (int b = 0; b < WCOL; b++) begin
                wbyte = 8'h00;
                for (int t = 0; t < 4; t++) begin
                    code = (w_full[n][b * 4 + t] == 0) ? 2'b00 :
                           (w_full[n][b * 4 + t] == 1) ? 2'b01 : 2'b11;
                    wbyte[t * 2 +: 2] = code;
                end
                sram_mem[W_ADDR + MEM_ADDR_W'(n * WCOL + b)] = wbyte;
            end
        end

        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) begin
                c_ref[m][n] = 0;
                for (int k = 0; k < K; k++)
                    c_ref[m][n] += a_full[m][k] * w_full[n][k];
            end
    endtask

    // ---- checks --------------------------------------------------------------
    int got;

    task automatic check_c();
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) begin
                got = {sram_mem[C_ADDR + MEM_ADDR_W'(m * CROW + n * 4 + 3)],
                       sram_mem[C_ADDR + MEM_ADDR_W'(m * CROW + n * 4 + 2)],
                       sram_mem[C_ADDR + MEM_ADDR_W'(m * CROW + n * 4 + 1)],
                       sram_mem[C_ADDR + MEM_ADDR_W'(m * CROW + n * 4 + 0)]};
                checks++;
                if (got !== c_ref[m][n]) begin
                    errors++;
                    if (errors <= 8)
                        $display("  FAIL C[%0d][%0d] = %0d, expected %0d",
                                 m, n, got, c_ref[m][n]);
                end
            end

        // Nothing may land past the spill.
        checks++;
        if (sram_mem[C_ADDR + MEM_ADDR_W'(C_BYTES)] !== 8'h00) begin
            errors++;
            $display("  FAIL: byte past C was written (0x%02h)",
                     sram_mem[C_ADDR + MEM_ADDR_W'(C_BYTES)]);
        end
    endtask

    // ---- stimulus ------------------------------------------------------------
    initial begin
        $display("==== firmware matmul: %s ====", `FW_HEX);
        $display("problem: [%0dx%0d] @ [%0dx%0d], %0dx%0d tiles of %0dx%0d",
                 M, K, K, N, KTILES, NTILES, ROWS, COLS);

        // The scratchpad map is fixed; the shape is not. Fail loudly rather than
        // letting one tensor land on top of another.
        if (M > 32 || A_BYTES > 32'h2000 || W_BYTES > 32'h2000 ||
            C_BYTES > 32'hC000) begin
            $display("FW_MATMUL: M=%0d K=%0d N=%0d does not fit the map (A=%0d W=%0d C=%0d bytes)",
                     M, K, N, A_BYTES, W_BYTES, C_BYTES);
            $fatal(1);
        end

        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;
        seed_dram();

        host_run = 1'b0;
        boot_pc  = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        // FW_INIT is a $readmemh at elaboration: if the image is missing the
        // firmware RAM is all x and the CPU would run garbage until the
        // watchdog. Say so instead.
        if (dut.u_cpu.fw_mem[0] === 32'hxxxx_xxxx) begin
            $display("FW_MATMUL: %s did not load - build it first (make -C ../fw)",
                     `FW_HEX);
            $fatal(1);
        end

        // Top bit of boot_pc selects the producer: 1 = the CPU's firmware.
        @(negedge clk);
        boot_pc  = (1 << FW_AW);
        host_run = 1'b1;
        run_clk  = cyc;
        @(negedge clk);
        host_run = 1'b0;

        wait (done);
        run_clk = cyc - run_clk;
        $display("firmware signalled done after %0d clocks", run_clk);
        // Same counters the 'T' command reports. `idlec` is what CPU issue costs:
        // clocks in which no unit was busy at all.
        $display("  counters: run=%0d mxu=%0d mload=%0d vpu=%0d dma=%0d",
                 dut.u_perf.counts[0*32 +: 32], dut.u_perf.counts[1*32 +: 32],
                 dut.u_perf.counts[2*32 +: 32], dut.u_perf.counts[3*32 +: 32],
                 dut.u_perf.counts[4*32 +: 32]);
        $display("            idlec=%0d qfull=%0d ovlap=%0d",
                 dut.u_perf.counts[7*32 +: 32], dut.u_perf.counts[8*32 +: 32],
                 dut.u_perf.counts[9*32 +: 32]);
        // One machine-readable line for run_fw_sweep.sh. The command counts come
        // from the queues' own `issued`, so "how many dispatches did this shape
        // cost the CPU" is measured rather than assumed.
        $display("SWEEP m=%0d kt=%0d nt=%0d run=%0d mxu=%0d dma=%0d idlec=%0d qfull=%0d mxucmd=%0d dmacmd=%0d",
                 M, KTILES, NTILES,
                 dut.u_perf.counts[0*32 +: 32], dut.u_perf.counts[1*32 +: 32],
                 dut.u_perf.counts[4*32 +: 32], dut.u_perf.counts[7*32 +: 32],
                 dut.u_perf.counts[8*32 +: 32], dut.mxu_issued, dut.dma_issued);

        check_c();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("FW_MATMUL: ALL TESTS PASSED");
        else             $display("FW_MATMUL: %0d FAILURES", errors);
        $finish;
    end

    initial begin
        #(`WATCHDOG_NS);
        $display("FW_MATMUL: WATCHDOG - firmware never signalled done (busy=%b done=%b)",
                 busy, done);
        $fatal(1);
    end

endmodule
