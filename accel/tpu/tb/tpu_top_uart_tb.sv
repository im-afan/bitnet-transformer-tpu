`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tpu_top_uart_tb.sv — UART-driven end-to-end testbench for tpu_top.sv
//
// The pure-serial counterpart to tpu_top_tb.sv. Where that test pokes the program
// into instruction BRAM and the tensors into the scratchpad by backdoor, this one
// drives the whole flow the way a real host PC would — over the UART pins only:
//
//   1. 'I'  load the assembled program into instruction memory
//   2. 'W'  write the input tensors into external DRAM
//   3. 'G'  start the program at PC 0
//   4.      the program DMAs DRAM -> scratchpad, computes, DMAs results -> DRAM
//   5. 'T'  read back how many core clocks the run took, and check that against
//           the `busy` interval this testbench timed itself
//   6. 'R'  read the result bytes back out of DRAM and compare to the golden image
//
// That whole sequence runs *twice*, for two different programs, with no reset in
// between — the same thing two back-to-back run_program.py invocations do. It is
// not redundant coverage: the scalar unit used to accept host IMEM/config writes
// only in S_IDLE, which is reachable only out of reset, so the second program's
// 'I' load was ACK'd and silently dropped and the board re-ran the first program
// until the reset button was pressed. Keep PROG2 different from PROG.
//
// Nothing touches the scratchpad or IMEM by backdoor; the only path in/out is the
// serial link + the real sram_controller driving the behavioral async-SRAM chip
// model on the external-DRAM pins.
//
// The program must therefore be self-contained over DRAM (stage inputs in with
// rdmem, spill outputs out with wrmem) — all the examples/ programs now are. Each
// uses identical DRAM and scratchpad addresses so the ISS golden (which models DMA
// as a no-op with tensors pre-placed in the scratchpad) matches the real DMA
// byte-copy. The three $readmemh vector files are produced by gen_vectors.py
// exactly as for tpu_top_tb; regenerate them (or use the Makefile `uart` target)
// after editing the program:
//
//     cd ../../tpulang && python gen_vectors.py -p examples/relu_layer.tpu -o ../tpu/tb/vectors_uart
//
//     cd ../../tpulang && python gen_vectors.py -p examples/vector_add.tpu -o ../tpu/tb/vectors_uart2
//
// Geometry below MUST match gen_vectors.py (ROWS/COLS/T). A small UART_CPB keeps
// the bit-banged serial traffic fast in simulation; the FSM is baud-agnostic.
//
// Run (or use `make uart`, which regenerates the vectors first):
//   make TEST=tpu_top_uart        (RTL_tpu_top_uart in tb/Makefile = UART_RTL)
// -----------------------------------------------------------------------------

// Own vector directory (vectors_uart/) so `make uart` never clobbers the
// backdoor tpu_top_tb's vectors/, which are generated from a different program.
`ifndef PROG_FILE
  `define PROG_FILE "vectors_uart/tpu_prog.hex"
`endif
`ifndef SPAD_IN_FILE
  `define SPAD_IN_FILE "vectors_uart/tpu_spad_in.hex"
`endif
`ifndef SPAD_EXP_FILE
  `define SPAD_EXP_FILE "vectors_uart/tpu_spad_exp.hex"
`endif

// Second program, loaded over the same link *without* an intervening reset —
// this is the "run one program, then a different one" case that fails on the
// board until the button is pressed. Point them at a different vector dir.
`ifndef PROG2_FILE
  `define PROG2_FILE "vectors_uart2/tpu_prog.hex"
`endif
`ifndef SPAD_IN2_FILE
  `define SPAD_IN2_FILE "vectors_uart2/tpu_spad_in.hex"
`endif
`ifndef SPAD_EXP2_FILE
  `define SPAD_EXP2_FILE "vectors_uart2/tpu_spad_exp.hex"
`endif

// VPU activation LUTs — not vectors but fixed ROM contents, shared by both
// programs and identical to what the bitstream burns in (accel/tpulang/luts.py).
`ifndef GELU_LUT_FILE
  `define GELU_LUT_FILE "../rtl/luts/gelu_lut.hex"
`endif
`ifndef EXP_LUT_FILE
  `define EXP_LUT_FILE "../rtl/luts/exp_lut.hex"
`endif

module tpu_top_uart_tb;

    // ---- Geometry (must match the DUT instance and gen_vectors.py) ----------
    localparam int ROWS      = 8;
    localparam int COLS      = 8;
    localparam int VPU_BYTES = 64;
    localparam int ADDR_W    = 16;
    localparam int XLEN      = 32;
    localparam int IMEM_AW   = 10;
    localparam int CFG_AW    = 5;   // 32 cfg regs: 15..17 are the DMA transpose geometry
    localparam int REG_AW    = 5;
    localparam int M0_W      = 12;
    localparam int N_W       = 4;
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int CPB       = 16;             // UART clocks-per-bit (fast, for sim)
    localparam int DEPTH     = (1 << ADDR_W);  // address span we scan for tensors
    localparam int IMEM_DEP  = (1 << IMEM_AW);

    localparam [7:0] WCMD = 8'h57, RCMD = 8'h52, ICMD = 8'h49, GCMD = 8'h47,
                     TCMD = 8'h54;

    // Performance-counter block returned by 'T'. Must match tpu_top.sv's NPERF
    // and PERF_* indices — those define the wire order, this decodes it.
    localparam int NPERF   = 6;
    localparam int P_RUN   = 0, P_MXU = 1, P_MLOAD = 2,
                   P_VPU   = 3, P_DMA = 4, P_SWAIT = 5;

    // Word w of a 'T' reply. uart_read_timer() shifts bytes in from the low end,
    // so the first word received (counter 0) ends up in the highest slice.
    function automatic logic [31:0] ctr_w(input logic [NPERF*32-1:0] ctr,
                                          input int w);
        return ctr[(NPERF-1-w)*32 +: 32];
    endfunction
    localparam [7:0] ACK  = 8'h06, NAK  = 8'h15;

    localparam int   CLK_NS = 10;   // clock period in ns; matches `always #5` below

    // ---- Clock / reset ------------------------------------------------------
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz

    // ---- DUT host debug ports (observed, not driven — the host uses UART) ----
    logic                  host_run = 1'b0;
    logic [IMEM_AW-1:0]    boot_pc  = '0;
    logic                  busy, done;
    logic [IMEM_AW-1:0]    pc_dbg;

    logic                  imem_we = 1'b0;
    logic [IMEM_AW-1:0]    imem_waddr = '0;
    logic [31:0]           imem_wdata = '0;
    logic                  cfg_we = 1'b0;
    logic [CFG_AW-1:0]     cfg_waddr = '0;
    logic [XLEN-1:0]       cfg_wdata = '0;

    // ---- External DRAM pins + UART serial pins ------------------------------
    wire  [MEM_ADDR_W-1:0] sram_addr;
    wire  [MEM_DATA_W-1:0] sram_data;   // inout: bus shared with the chip model
    wire                   sram_we, sram_ce, sram_oen;

    logic                  uart_rx = 1'b1; // host -> device (idles high)
    wire                   uart_tx;        // device -> host

    tpu_top #(
        .ROWS(ROWS), .COLS(COLS), .VPU_BYTES(VPU_BYTES), .ADDR_W(ADDR_W),
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W),
        .REG_AW(REG_AW), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .MEM_STYLE("BRAM"), .MEM_ADDR_W(MEM_ADDR_W), .MEM_DATA_W(MEM_DATA_W),
        .UART_CPB(CPB), .UART_RX_TIMEOUT(0),
        .GELU_INIT(`GELU_LUT_FILE), .EXP_INIT(`EXP_LUT_FILE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .host_run(host_run), .boot_pc(boot_pc),
        .busy(busy), .done(done), .pc_dbg(pc_dbg),
        .imem_we(imem_we), .imem_waddr(imem_waddr), .imem_wdata(imem_wdata),
        .cfg_we(cfg_we), .cfg_waddr(cfg_waddr), .cfg_wdata(cfg_wdata),
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen),
        .uart_rx(uart_rx), .uart_tx(uart_tx)
    );

    // ---- Behavioral async SRAM chip model (matches tpu_top_tb.sv) -----------
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    // ---- Independent measure of the run length ------------------------------
    //
    // What `cycle_timer` inside the DUT should be reporting, derived a different
    // way: the simulation time between the edges of the `busy` pin, divided by
    // the clock period. `busy` only ever changes at a clock edge, so that
    // division is exact, and it shares no logic with the counter under test —
    // a counter that miscounted, failed to restart, or never froze would have
    // to be wrong in the same way as the simulator's clock to agree with this.
    time t_busy_rise, t_busy_fall;
    always @(posedge busy) t_busy_rise = $time;
    always @(negedge busy) t_busy_fall = $time;

    function automatic int unsigned expected_run_clocks();
        return int'((t_busy_fall - t_busy_rise) / CLK_NS);
    endfunction

    // ---- Vector-file backing stores -----------------------------------------
    logic [31:0] prog_words [0:IMEM_DEP-1];
    logic [7:0]  in_bytes   [0:DEPTH-1];
    logic [7:0]  exp_bytes  [0:DEPTH-1];

    int errors = 0, checks = 0;

    function automatic logic defined8(input logic [7:0] b);
        return (^b !== 1'bx);
    endfunction
    function automatic logic defined32(input logic [31:0] w);
        return (^w !== 1'bx);
    endfunction

    // =========================================================================
    // UART host: bit-bang uart_rx, decode uart_tx (same style as the unit TBs)
    // =========================================================================
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

    task automatic rx_bit_byte(output [7:0] b);
        @(negedge uart_tx);
        repeat (CPB / 2) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            repeat (CPB) @(posedge clk);
            b[i] = uart_tx;
        end
        repeat (CPB) @(posedge clk);
    endtask

    // CMD + 24-bit addr (BE) + 16-bit len (BE)
    task automatic send_header(input [7:0] cmd, input [23:0] a, input [15:0] n);
        tx_bit_byte(cmd);
        tx_bit_byte(a[23:16]); tx_bit_byte(a[15:8]); tx_bit_byte(a[7:0]);
        tx_bit_byte(n[15:8]);  tx_bit_byte(n[7:0]);
    endtask

    // 'I' load nwords instruction words (dense from addr 0), MSB byte first.
    task automatic uart_load_program(input int nwords);
        logic [7:0] st;
        fork
            begin
                send_header(ICMD, 24'd0, 16'(nwords << 2));
                for (int w = 0; w < nwords; w++) begin
                    tx_bit_byte(prog_words[w][31:24]);
                    tx_bit_byte(prog_words[w][23:16]);
                    tx_bit_byte(prog_words[w][15:8]);
                    tx_bit_byte(prog_words[w][7:0]);
                end
            end
            rx_bit_byte(st);
        join
        checks++;
        if (st !== ACK) begin errors++; $display("  FAIL: imem load, got %02h", st); end
    endtask

    // 'W' write a run of `n` input bytes (in_bytes[start .. start+n-1]) to DRAM.
    task automatic uart_write_dram(input int start, input int n);
        logic [7:0] st;
        fork
            begin
                send_header(WCMD, 24'(start), 16'(n));
                for (int i = 0; i < n; i++) tx_bit_byte(in_bytes[start + i]);
            end
            rx_bit_byte(st);
        join
        checks++;
        if (st !== ACK) begin
            errors++; $display("  FAIL: dram write @%05h, got %02h", start, st);
        end
    endtask

    // 'R' read a run of `n` bytes from DRAM and compare to exp_bytes.
    task automatic uart_read_check(input int start, input int n);
        logic [7:0] b;
        fork
            send_header(RCMD, 24'(start), 16'(n));
            begin
                for (int i = 0; i < n; i++) begin
                    rx_bit_byte(b);
                    checks++;
                    if (b !== exp_bytes[start + i]) begin
                        errors++;
                        $display("  FAIL @%05h: got %02h  exp %02h",
                                 start + i, b, exp_bytes[start + i]);
                    end
                end
            end
        join
    endtask

    // 'G' start at PC 0; expect ACK.
    task automatic uart_go(input [23:0] pc);
        logic [7:0] st;
        fork
            begin
                tx_bit_byte(GCMD);
                tx_bit_byte(pc[23:16]); tx_bit_byte(pc[15:8]); tx_bit_byte(pc[7:0]);
            end
            rx_bit_byte(st);
        join
        checks++;
        if (st !== ACK) begin errors++; $display("  FAIL: go, got %02h", st); end
    endtask

    // 'T' read the performance counters: one command byte, NPERF 32-bit words
    // back, MSB first both within and across words, no status. Word 0 is the
    // run length (tpu_top.sv PERF_RUN), which is why a 4-byte reader still
    // decodes it; the rest must be drained regardless or the link desyncs for
    // the next command.
    // Packed rather than an unpacked array of words: Icarus does not support
    // unpacked dimensions on subroutine ports. Word w is ctr[w*32 +: 32], and
    // `ctr_w(ctr, w)` below does that indexing for the checks.
    task automatic uart_read_timer(output [NPERF*32-1:0] ctr);
        logic [7:0] b;
        fork
            tx_bit_byte(TCMD);
            begin
                ctr = '0;
                // Words arrive high-index-first on the wire, so shifting the
                // whole block left by a byte per byte received lands word 0 in
                // the top slice — see ctr_w().
                for (int i = 0; i < NPERF*4; i++) begin
                    rx_bit_byte(b);
                    ctr = {ctr[NPERF*32-9:0], b};
                end
            end
        join
    endtask

    // Walk the sparse tensor/expected images and emit one command per contiguous
    // run of defined bytes (keeps the frame count small).
    task automatic uart_write_all_inputs();
        int a, start, n;
        a = 0;
        while (a < DEPTH) begin
            if (defined8(in_bytes[a])) begin
                start = a; n = 0;
                while (a < DEPTH && defined8(in_bytes[a])) begin a = a + 1; n = n + 1; end
                uart_write_dram(start, n);
            end else a = a + 1;
        end
    endtask

    task automatic uart_check_all_outputs();
        int a, start, n;
        a = 0;
        while (a < DEPTH) begin
            if (defined8(exp_bytes[a])) begin
                start = a; n = 0;
                while (a < DEPTH && defined8(exp_bytes[a])) begin a = a + 1; n = n + 1; end
                uart_read_check(start, n);
            end else a = a + 1;
        end
    endtask

    int nprog;

    // Blank the vector arrays so a second $readmemh cannot leave bytes of the
    // previous program's image behind — the run-scanners key off `defined8`.
    task automatic clear_vectors();
        for (int i = 0; i < IMEM_DEP; i++) prog_words[i] = 32'bx;
        for (int i = 0; i < DEPTH; i++)   begin in_bytes[i] = 8'bx; exp_bytes[i] = 8'bx; end
    endtask

    // Wait for the run to finish. `done` is *level*-held in S_HALT, so after the
    // first program it is already high when the second 'G' goes out; waiting on
    // `done` alone would fall straight through and "observe" a halt that never
    // happened. Wait for the core to actually pick the run up (`busy`) first,
    // and bound that wait so a core that never starts is reported as such.
    task automatic wait_for_halt(input int start_limit);
        int n;
        n = 0;
        while (!busy && n < start_limit) begin @(negedge clk); n++; end
        checks++;
        if (!busy) begin
            errors++;
            $display("  FAIL: core never started after 'G' (still done=%0b)", done);
        end else begin
            do @(negedge clk); while (!done);
        end
    endtask

    // Does the loaded image contain a matmul? Scans the program words actually
    // loaded (opcode is the top 6 bits; OP_MATMUL is 0x00) rather than taking
    // the answer as a parameter, so the MXU counter is checked against whatever
    // PROG/PROG2 the Makefile was given instead of against a hardcoded guess
    // about the default pair. Only [0,nprog) is scanned, which clear_vectors()
    // guarantees is the defined region — an all-zero word would otherwise
    // decode as a matmul.
    // OP_MATMUL (0x00) or OP_MATMUL_T (0x1D) — both drive the MXU.
    function automatic bit image_has_matmul(input int nwords);
        for (int i = 0; i < nwords; i++)
            if (prog_words[i][31:26] == 6'h00 || prog_words[i][31:26] == 6'h1D)
                return 1'b1;
        return 1'b0;
    endfunction

    // Same idea for the VPU: mirrors scalar_unit.sv's is_vpu_op(). Needed
    // because not every program touches every unit — tiled_matmul.tpu is pure
    // MXU with no VPU op at all, so "the VPU counter must be nonzero" is only
    // true of programs that actually dispatch one.
    function automatic bit is_vpu_opcode(input logic [5:0] o);
        return (o == 6'h01) || (o == 6'h02) || (o == 6'h03) || (o == 6'h04) ||
               (o == 6'h05) || (o == 6'h09) || (o == 6'h0A) || (o == 6'h0B) ||
               (o == 6'h0C) || (o == 6'h0D) || (o == 6'h0E) || (o == 6'h0F) ||
               (o == 6'h1B) ||
               (o == 6'h1E) ||  // OP_VECMM  (vecmatmul macro op)
               (o == 6'h20);    // OP_SOFTMAX (softmax macro op)
    endfunction

    function automatic bit image_has_vpu(input int nwords);
        for (int i = 0; i < nwords; i++)
            if (is_vpu_opcode(prog_words[i][31:26])) return 1'b1;
        return 1'b0;
    endfunction

    task automatic test_program(input string label);
        bit expect_mxu, expect_vpu;
        nprog = 0;
        while (nprog < IMEM_DEP && defined32(prog_words[nprog])) nprog++;
        expect_mxu = image_has_matmul(nprog);
        expect_vpu = image_has_vpu(nprog);

        $display("---- %0s ----", label);

        // 1) program -> IMEM, 2) tensors -> DRAM, all over UART
        uart_load_program(nprog);
        $display("loaded %0d instruction words over UART", nprog);
        uart_write_all_inputs();
        $display("wrote input tensors to DRAM over UART");

        // 3) start, 4) wait for the program to halt (observed via `done`)
        uart_go(24'd0);
        $display("issued run; waiting for halt...");
        wait_for_halt(200_000);
        $display("program halted at pc=%0d (expected %0d)", pc_dbg, nprog - 1);

        checks++;
        if (busy) begin errors++; $display("  FAIL: core still busy after halt"); end

        // 5) 'T' — the device's own measure of how long that run took. Checked
        //    per program, not once at the end: the counter must restart on every
        //    run, and program B is the only thing that can prove it does.
        begin
            logic [NPERF*32-1:0] ctr;
            int unsigned want;
            uart_read_timer(ctr);
            want = expected_run_clocks();
            checks++;
            if (ctr_w(ctr, P_RUN) !== want) begin
                errors++;
                $display("  FAIL: 'T' reported %0d clocks, busy was high for %0d",
                         ctr_w(ctr, P_RUN), want);
            end else begin
                $display("run took %0d core clocks (device timer agrees)",
                         ctr_w(ctr, P_RUN));
            end
            $display("  counters: mxu=%0d mload=%0d vpu=%0d dma=%0d swait=%0d",
                     ctr_w(ctr, P_MXU), ctr_w(ctr, P_MLOAD), ctr_w(ctr, P_VPU),
                     ctr_w(ctr, P_DMA), ctr_w(ctr, P_SWAIT));

            // Structural invariants. These hold for *any* program, so they test
            // the counter block rather than this particular pair of programs.
            // Every counter is gated by the same run window, so none can exceed
            // it; and the MXU's load phase is a sub-state of MXU busy.
            for (int i = 0; i < NPERF; i++) begin
                checks++;
                if (ctr_w(ctr, i) > ctr_w(ctr, P_RUN)) begin
                    errors++;
                    $display("  FAIL: counter %0d = %0d exceeds run length %0d",
                             i, ctr_w(ctr, i), ctr_w(ctr, P_RUN));
                end
            end
            checks++;
            if (ctr_w(ctr, P_MLOAD) > ctr_w(ctr, P_MXU)) begin
                errors++;
                $display("  FAIL: mload=%0d exceeds mxu=%0d (must be a subset)",
                         ctr_w(ctr, P_MLOAD), ctr_w(ctr, P_MXU));
            end

            // Every example stages tensors over DMA; VPU use is per-program.
            checks++;
            if (expect_vpu && ctr_w(ctr, P_VPU) == 0) begin
                errors++;
                $display("  FAIL: vpu counter is 0, but the program ran VPU ops");
            end else if (!expect_vpu && ctr_w(ctr, P_VPU) != 0) begin
                errors++;
                $display("  FAIL: vpu counter is %0d, but the program has no VPU op",
                         ctr_w(ctr, P_VPU));
            end
            checks++;
            if (ctr_w(ctr, P_DMA) == 0) begin
                errors++;
                $display("  FAIL: dma counter is 0, but the program staged tensors");
            end

            // The discriminating check (see the task header).
            checks++;
            if (expect_mxu && ctr_w(ctr, P_MXU) == 0) begin
                errors++;
                $display("  FAIL: mxu counter is 0, but the program has a matmul");
            end else if (!expect_mxu && ctr_w(ctr, P_MXU) != 0) begin
                errors++;
                $display("  FAIL: mxu counter is %0d, but the program has no matmul",
                         ctr_w(ctr, P_MXU));
            end
            // A matmul must load weights, so mload is nonzero exactly when the
            // MXU ran at all — this is what proves S_LOAD is really being
            // isolated rather than mload just mirroring mxu.
            checks++;
            if (expect_mxu && ctr_w(ctr, P_MLOAD) == 0) begin
                errors++;
                $display("  FAIL: mload is 0, but the MXU ran a matmul");
            end
            checks++;
            if (expect_mxu && ctr_w(ctr, P_MLOAD) >= ctr_w(ctr, P_MXU)) begin
                errors++;
                $display("  FAIL: mload=%0d is not a strict subset of mxu=%0d",
                         ctr_w(ctr, P_MLOAD), ctr_w(ctr, P_MXU));
            end
        end

        // 6) read results back over UART and compare
        uart_check_all_outputs();
        $display("---- %0s: %0d checks, %0d errors so far ----", label, checks, errors);
    endtask

    // =========================================================================
    // Stimulus
    // =========================================================================
    initial begin
        // int nprog;

        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("==== TPU UART end-to-end testbench (%0dx%0d array) ====", ROWS, COLS);

        // Two *different* programs back to back over the same link, with no
        // reset in between — exactly what run_program.py does on the board when
        // it is invoked twice with different --program arguments.
        clear_vectors();
        $readmemh(`PROG_FILE, prog_words);
        $readmemh(`SPAD_IN_FILE, in_bytes);
        $readmemh(`SPAD_EXP_FILE, exp_bytes);
        test_program("program A");

        repeat (100) @(posedge clk);

        clear_vectors();
        $readmemh(`PROG2_FILE, prog_words);
        $readmemh(`SPAD_IN2_FILE, in_bytes);
        $readmemh(`SPAD_EXP2_FILE, exp_bytes);
        test_program("program B (no reset in between)");

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("TPU_TOP_UART: ALL TESTS PASSED");
        else             $display("TPU_TOP_UART: FAILED (%0d errors)", errors);

        $finish;
    end

    // Watchdog.
    initial begin
        #50_000_000;   // 50 ms
        $display("TPU_TOP_UART: TIMEOUT — no halt / stuck link");
        $fatal(1);
    end

    initial begin
        $dumpfile("tpu_top_uart_tb.vcd");
        $dumpvars(0, tpu_top_uart_tb);
    end

endmodule
