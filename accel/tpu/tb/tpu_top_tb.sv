`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tpu_top_tb.sv — end-to-end testbench for accel/tpu/rtl/tpu_top.sv
//
// Exercises the fully integrated core (scalar_unit + mxu + vpu + scratchpad,
// BRAM working memory) the way a host actually drives it: load an *assembled*
// program and sample input tensors from $readmemh memory files, load them into
// scratchpad, pulse host_run, then read the results back out of BRAM and compare.
//
// DRAM preload/readback. The example programs are now self-contained over DRAM:
// each stages its inputs in from external DRAM (rdmem) and spills its outputs back
// (wrmem), so the host's only job is to seed DRAM and read DRAM. This TB does that
// by *backdoor* poke/peek of the behavioral async-SRAM chip model attached to the
// external-DRAM pins (`sram_mem[...]`), a zero-time load/readback. The DMA engine
// then moves the bytes DRAM<->scratchpad exactly as it would for a real host; the
// scratchpad and compute-unit ports are never poked directly. This is the backdoor
// analog of tpu_top_uart_tb.sv, which drives the identical flow over the UART pins.
//
// The golden reference is produced entirely outside the RTL: accel/tpulang
// assembles a .tpu program (assembler.py) and simulates the instruction set in
// Python (iss.py) against the same input tensors, and gen_vectors.py exports
//
//     tpu_prog.hex      the assembled program        (imem words, dense)
//     tpu_spad_in.hex   the input tensors            (scratchpad bytes, sparse)
//     tpu_spad_exp.hex  the ISS-computed outputs     (scratchpad bytes, sparse)
//
// This TB is program-agnostic: it streams whatever the program file holds, loads
// whatever bytes the input file holds, and checks exactly the bytes the ISS says
// the program wrote (tpu_spad_exp.hex carries only those). Regenerate the vectors
// after changing the program or tensors:
//
//     cd ../../tpulang && python gen_vectors.py
//
// The default program is examples/relu_layer.tpu — one ternary NN layer plus a
// ReLU (Y = requant(A@W); Z = relu(Y)), the shape the adder model runs.
//
// Small 8x8 array so the systolic drain simulates quickly (RTL default 128x128).
// The geometry below MUST match gen_vectors.py (ROWS/COLS/T).
//
// Run:  make TEST=tpu_top          (RTL_tpu_top in tb/Makefile supplies the
//                                   sources; tpu_top instantiates the UART
//                                   blocks and cycle_timer even though this TB
//                                   drives neither, so they must be compiled in)
// Override a vector file:  iverilog ... -DPROG_FILE='"vectors/other_prog.hex"'
// -----------------------------------------------------------------------------

// ---- Vector files (paths relative to the tb/ dir where vvp runs) -------------
`ifndef PROG_FILE
  `define PROG_FILE "vectors/tpu_prog.hex"
`endif
`ifndef SPAD_IN_FILE
  `define SPAD_IN_FILE "vectors/tpu_spad_in.hex"
`endif
`ifndef WATCHDOG_NS
  `define WATCHDOG_NS 2000000
`endif
`ifndef SPAD_EXP_FILE
  `define SPAD_EXP_FILE "vectors/tpu_spad_exp.hex"
`endif

module tpu_top_tb;

    // ---- Geometry (must match the DUT instance and gen_vectors.py) ----------
    localparam int ROWS      = 8;      // MXU contraction dim d
    localparam int COLS      = 8;      // MXU output features
    localparam int VPU_BYTES = 64;     // VPU port width (512-bit)
    localparam int ADDR_W    = 16;
    localparam int XLEN      = 32;
    localparam int IMEM_AW   = 10;
    localparam int CFG_AW    = 5;   // 32 cfg regs: 15..17 are the DMA transpose geometry
    localparam int REG_AW    = 5;
    localparam int M0_W      = 12;
    localparam int N_W       = 4;
    localparam int MEM_ADDR_W = 19;    // external SRAM byte address (DUT default)
    localparam int MEM_DATA_W = 8;     // external SRAM data width (bits)

    // in_bytes/exp_bytes are **DRAM** byte maps (the TB seeds and checks DRAM
    // through the backdoor), so this spans MEM_ADDR_W. It was ADDR_W, which
    // silently discarded any expectation above 64 KB — see the same note in
    // tpu_top_uart_tb.sv.
    localparam int DEPTH     = (1 << MEM_ADDR_W);  // DRAM span scanned for tensors
    localparam int IMEM_DEP  = (1 << IMEM_AW);     // instruction memory depth

    // -------------------------------------------------------------------------
    // Clock / reset.
    // -------------------------------------------------------------------------
    int unsigned cyc = 0, run_clk = 0;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;   // 100 MHz
    always @(posedge clk) cyc <= cyc + 1;

    // -------------------------------------------------------------------------
    // DUT host interface.
    // -------------------------------------------------------------------------
    logic                  host_run;
    logic [IMEM_AW-1:0]    boot_pc;
    logic                  busy, done;
    logic [IMEM_AW-1:0]    pc_dbg;

    logic                  imem_we;
    logic [IMEM_AW-1:0]    imem_waddr;
    logic [31:0]           imem_wdata;

    logic                  cfg_we;
    logic [CFG_AW-1:0]     cfg_waddr;
    logic [XLEN-1:0]       cfg_wdata;

    // External DRAM pins (driven by the DUT's internal DMA + sram_controller).
    wire  [MEM_ADDR_W-1:0] sram_addr;
    wire  [MEM_DATA_W-1:0] sram_data;   // inout: bus shared with the chip model
    wire                   sram_we, sram_ce, sram_oen;

    // UART host link is unused in this core-focused test: hold RX idle-high so the
    // receiver never triggers, and leave TX observed but unchecked.
    wire                   uart_tx;

    tpu_top #(
        .ROWS(ROWS), .COLS(COLS), .VPU_BYTES(VPU_BYTES), .ADDR_W(ADDR_W),
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W),
        .REG_AW(REG_AW), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .MEM_STYLE("BRAM"), .MEM_ADDR_W(MEM_ADDR_W), .MEM_DATA_W(MEM_DATA_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .host_run(host_run), .boot_pc(boot_pc),
        .busy(busy), .done(done), .pc_dbg(pc_dbg),
        .imem_we(imem_we), .imem_waddr(imem_waddr), .imem_wdata(imem_wdata),
        .cfg_we(cfg_we), .cfg_waddr(cfg_waddr), .cfg_wdata(cfg_wdata),
        // External DRAM interface
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen),
        // UART host link (idle in this test)
        .uart_rx(1'b1), .uart_tx(uart_tx)
    );

    // -------------------------------------------------------------------------
    // Behavioral async SRAM chip model on the external-DRAM pins (matches
    // sram_tb.sv / dma_tb.sv). The chip drives the bus only when selected, output
    // enabled, and not being written; otherwise it releases to Hi-Z. A write
    // commits on the WE# rising edge. Present so DMA-using programs work; the
    // default program leaves it idle.
    // -------------------------------------------------------------------------
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    // -------------------------------------------------------------------------
    // DRAM preload/readback via backdoor poke/peek of the SRAM chip model.
    // Zero-time: preload runs before host_run (no DMA writer is active) and
    // readback runs after HALT (memory is stable). Byte k of a word lives at
    // addr+k.
    //
    // These are MEM_ADDR_W wide, not ADDR_W. They used to be ADDR_W on the
    // assumption that "tensor addresses (< 2^ADDR_W) index directly into the
    // wider DRAM" — true only while the scalar unit could not name DRAM above
    // 64 KB. Once it could, a 16-bit backdoor silently poked every high input
    // byte to `addr & 0xFFFF` and read every high expectation from the wrong
    // place, which reads back as x rather than as a wrong value.
    // -------------------------------------------------------------------------
    task automatic dram_wr_byte(input logic [MEM_ADDR_W-1:0] addr, input logic [7:0] d);
        sram_mem[addr] = d;
    endtask

    // Read a 32-bit window starting at addr (byte k at addr+k).
    task automatic dram_rd(input logic [MEM_ADDR_W-1:0] addr, output logic [31:0] d);
        for (int k = 0; k < 4; k++)
            d[k*8 +: 8] = sram_mem[MEM_ADDR_W'(addr + k)];
    endtask

    task automatic load_instr(input logic [IMEM_AW-1:0] a, input logic [31:0] w);
        @(negedge clk);
        imem_we    = 1'b1;
        imem_waddr = a;
        imem_wdata = w;
        @(negedge clk);
        imem_we    = 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Vector-file backing stores (loaded with $readmemh).
    //   prog_words : dense from addr 0; trailing entries stay x (=> program end).
    //   in_bytes   : sparse (@addr runs); a defined byte is a tensor byte to load.
    //   exp_bytes  : sparse (@addr runs); a defined byte is an output to check.
    // -------------------------------------------------------------------------
    logic [31:0] prog_words [0:IMEM_DEP-1];
    logic [7:0]  in_bytes   [0:DEPTH-1];
    logic [7:0]  exp_bytes  [0:DEPTH-1];

    int errors = 0, checks = 0;

    // Fully-defined (no x bits) test for a loaded byte / word.
    function automatic logic defined8(input logic [7:0] b);
        return (^b !== 1'bx);
    endfunction
    function automatic logic defined32(input logic [31:0] w);
        return (^w !== 1'bx);
    endfunction

    // Stream the assembled program into instruction BRAM (stop at first hole).
    // ($readmemh warns that the file has fewer words than prog_words is deep —
    //  that is expected: the unfilled trailing entries stay x and mark the end.)
    task automatic load_program();
        int i;
        $readmemh(`PROG_FILE, prog_words);
        i = 0;
        while (i < IMEM_DEP && defined32(prog_words[i])) begin
            load_instr(i[IMEM_AW-1:0], prog_words[i]);
            i++;
        end
        $display("loaded %0d instruction words from %s", i, `PROG_FILE);
    endtask

    // Stream the sample tensors into scratchpad over the host memory port.
    task automatic load_tensors();
        int a, cnt;
        $readmemh(`SPAD_IN_FILE, in_bytes);
        cnt = 0;
        for (a = 0; a < DEPTH; a++)
            if (defined8(in_bytes[a])) begin
                dram_wr_byte(a[MEM_ADDR_W-1:0], in_bytes[a]);
                cnt++;
            end
        $display("loaded %0d DRAM input bytes from %s", cnt, `SPAD_IN_FILE);
    endtask

    // Read every expected byte back out of BRAM and compare (word at a time).
    task automatic check_outputs();
        int wa, lane;
        logic [31:0] rd;
        logic [7:0]  eb, gb;
        $readmemh(`SPAD_EXP_FILE, exp_bytes);
        for (wa = 0; wa < DEPTH; wa += 4) begin
            // Only read BRAM when this word carries an expected output byte.
            if (defined8(exp_bytes[wa])  || defined8(exp_bytes[wa+1]) ||
                defined8(exp_bytes[wa+2])|| defined8(exp_bytes[wa+3])) begin
                dram_rd(wa[MEM_ADDR_W-1:0], rd);
                for (lane = 0; lane < 4; lane++) begin
                    eb = exp_bytes[wa+lane];
                    if (defined8(eb)) begin
                        gb = rd[lane*8 +: 8];
                        checks++;
                        if (gb !== eb) begin
                            errors++;
                            $display("  FAIL @%04h: got %02h  exp %02h",
                                     wa+lane, gb, eb);
                        end
                    end
                end
            end
        end
        $display("[compare] checked %0d output bytes (errors so far: %0d)",
                 checks, errors);
    endtask

    // -------------------------------------------------------------------------
    // Stimulus.
    // -------------------------------------------------------------------------
    initial begin
        host_run = 1'b0; boot_pc = '0;
        imem_we  = 1'b0; imem_waddr = '0; imem_wdata = '0;
        cfg_we   = 1'b0; cfg_waddr  = '0; cfg_wdata  = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== TPU top-level testbench (%0dx%0d array) ====", ROWS, COLS);

        // 1) Load the assembled program + input tensors, 2) run, 3) check.
        load_program();
        load_tensors();

        @(negedge clk); host_run = 1'b1; boot_pc = '0;
        run_clk = cyc;
        @(negedge clk); host_run = 1'b0;

        // Wait for HALT (watchdog below guards against a hang).
        do @(negedge clk); while (!done);
        run_clk = cyc - run_clk;
        $display("program halted at pc=%0d after %0d clocks", pc_dbg, run_clk);
        // The same six counters the 'T' command reports, read out of the block
        // directly since this TB has no serial link. `dma` against `run` is the
        // memory-bound share, which is what the sram.sv range interface moves.
        $display("  counters: run=%0d mxu=%0d mload=%0d vpu=%0d dma=%0d swait=%0d",
                 dut.u_perf.counts[0*32 +: 32], dut.u_perf.counts[1*32 +: 32],
                 dut.u_perf.counts[2*32 +: 32], dut.u_perf.counts[3*32 +: 32],
                 dut.u_perf.counts[4*32 +: 32], dut.u_perf.counts[5*32 +: 32]);

        check_outputs();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("TPU_TOP: ALL TESTS PASSED");
        else             $display("TPU_TOP: FAILED (%0d errors)", errors);
        $finish;
    end

    // Watchdog. Overridable because run length is a property of the *program*,
    // not of this testbench: relu_layer halts in microseconds, but a whole
    // transformer is ~190 KB of DMA at 1-2 clocks/byte plus its MXU/VPU work,
    // i.e. milliseconds. Default unchanged, so every existing invocation is
    // unaffected.
    //   iverilog ... -DWATCHDOG_NS=200000000
    initial begin
        #(`WATCHDOG_NS);
        $display("TPU_TOP: TIMEOUT — program did not halt after %0d ns",
                 `WATCHDOG_NS);
        $fatal(1);
    end

    // Waveform dump. Also guardable: at this timescale a full-model run dumps
    // every net of the whole core for tens of milliseconds, which is hundreds of
    // megabytes of VCD that nobody reads. `-DNO_VCD` skips it.
    //   iverilog ... -DNO_VCD
`ifndef NO_VCD
    initial begin
        $dumpfile("tpu_top_tb.vcd");
        $dumpvars(0, tpu_top_tb);
    end
`endif

endmodule
