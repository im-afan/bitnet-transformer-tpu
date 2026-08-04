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
//   5. 'R'  read the result bytes back out of DRAM and compare to the golden image
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
// Geometry below MUST match gen_vectors.py (ROWS/COLS/T). A small UART_CPB keeps
// the bit-banged serial traffic fast in simulation; the FSM is baud-agnostic.
//
// Run (or use `make uart`):
//   make TEST=tpu_top_uart RTL="../rtl/tpu_top.sv ../rtl/scalar_unit.sv \
//        ../rtl/mxu.sv ../rtl/vpu.sv ../rtl/scratchpad.sv ../rtl/dma.sv \
//        ../rtl/sram.sv ../rtl/uart_interface.sv ../rtl/uart_receiver.sv \
//        ../rtl/uart_transmitter.sv"
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

module tpu_top_uart_tb;

    // ---- Geometry (must match the DUT instance and gen_vectors.py) ----------
    localparam int ROWS      = 8;
    localparam int COLS      = 8;
    localparam int VPU_BYTES = 64;
    localparam int ADDR_W    = 16;
    localparam int XLEN      = 32;
    localparam int IMEM_AW   = 10;
    localparam int CFG_AW    = 4;
    localparam int REG_AW    = 5;
    localparam int M0_W      = 12;
    localparam int N_W       = 4;
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int CPB       = 16;             // UART clocks-per-bit (fast, for sim)
    localparam int DEPTH     = (1 << ADDR_W);  // address span we scan for tensors
    localparam int IMEM_DEP  = (1 << IMEM_AW);

    localparam [7:0] WCMD = 8'h57, RCMD = 8'h52, ICMD = 8'h49, GCMD = 8'h47;
    localparam [7:0] ACK  = 8'h06, NAK  = 8'h15;

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
        .UART_CPB(CPB), .UART_RX_TIMEOUT(0)
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
    task automatic test_program();
        nprog = 0;
        while (nprog < IMEM_DEP && defined32(prog_words[nprog])) nprog++;

        // 1) program -> IMEM, 2) tensors -> DRAM, all over UART
        uart_load_program(nprog);
        $display("loaded %0d instruction words over UART", nprog);
        uart_write_all_inputs();
        $display("wrote input tensors to DRAM over UART");

        // 3) start, 4) wait for the program to halt (observed via `done`)
        uart_go(24'd0);
        $display("issued run; waiting for halt...");
        do @(negedge clk); while (!done);
        $display("program halted at pc=%0d", pc_dbg);

        checks++;
        if (busy) begin errors++; $display("  FAIL: core still busy after halt"); end

        // 5) read results back over UART and compare
        uart_check_all_outputs();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("TPU_TOP_UART: ALL TESTS PASSED");
        else             $display("TPU_TOP_UART: FAILED (%0d errors)", errors);
    endtask

    // =========================================================================
    // Stimulus
    // =========================================================================
    initial begin
        // int nprog;

        for (int i = 0; i < SRAM_SZ; i++) sram_mem[i] = 8'h00;
        $readmemh(`PROG_FILE, prog_words);
        $readmemh(`SPAD_IN_FILE, in_bytes);
        $readmemh(`SPAD_EXP_FILE, exp_bytes);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);


        $display("==== TPU UART end-to-end testbench (%0dx%0d array) ====", ROWS, COLS);

        // run program twice without reset
        test_program();
        repeat (100) @(posedge clk);
        test_program();
        
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
