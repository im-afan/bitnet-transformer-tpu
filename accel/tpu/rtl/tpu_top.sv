`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// tpu_top.sv — TPU top-level integration
//
// Stitches the four implemented blocks into one core, per the README §1 block
// diagram:
//
//     scalar_unit  (control processor / ISA sequencer, scalar_unit.sv)
//        │  issue-and-wait dispatch (start → busy/done)
//        ├──────────────► mxu   (weight-stationary ternary systolic array)
//        └──────────────► vpu   (SIMD vector unit)
//                                 │           │
//     scratchpad (BRAM working memory, scratchpad.sv) ◄── all three units
//        A_rd / W_rd / C_rw (MXU) · V_rw (VPU) · S_rw (scalar)
//
// The scalar unit fetches a program from its own instruction BRAM, does the
// scalar math (loop counters, address arithmetic, requant scalars), and
// dispatches OP_MATMUL to the MXU and the VPU ops to the VPU, blocking on each
// unit's `done` before the next dependent op (scalar_unit.md §2). Every math
// instruction names scratchpad byte addresses; the scratchpad is the single
// owner of the on-chip storage and every unit drives it on the same-named typed
// ports it already exposes (scratchpad.md §6), so integration here is almost
// entirely name-matched wiring.
//
// Memory. The top-level working memory is FPGA BRAM: the scratchpad is
// instantiated with MEM_STYLE="BRAM" (scratchpad.md §6). DRAM / weight streaming
// is out of scope here — there is no DMA engine block yet — so the scratchpad's
// DMA port is brought out to the top level as a host preload/readback interface
// (`host_mem_*`), letting a testbench or host fill activations/weights and read
// results back out of BRAM directly.
//
// Deliberately left out (see notes at the stub tie-offs below):
//   * comms / inter-TPU LINK — the scalar unit's WriteNeighbor dispatch is
//     stubbed (`nb_done` tied high) so a LINK op is a completing no-op. Wiring a
//     real comms block (comms.md) is a later step.
//   * DMA engine — the scalar unit's Read/WriteMemory dispatch is likewise
//     stubbed (`dma_done` tied high). Memory is moved in/out through the
//     `host_mem_*` port instead until a DMA engine lands.
//
// Everything is parameterized off the systolic array size (ROWS/COLS), the VPU
// port width, and the scalar datapath; the scratchpad's per-port byte widths are
// *derived* from those below so the whole core has a single source of truth for
// the bus widths (A = ROWS int8, W = ROWS trits, C = COLS int32, V = VPU port,
// S = scalar word). With the defaults (128×128 array, 512-bit VPU) these resolve
// to exactly the scratchpad.sv defaults.
// -----------------------------------------------------------------------------

module tpu_top #(
    // ---- Systolic array / datapath sizing -----------------------------------
    parameter int ROWS     = 128,   // MXU contraction dim d
    parameter int COLS     = 128,   // MXU output features
    parameter int VPU_BYTES = 64,   // VPU scratchpad port width in bytes (512-bit)
    parameter int ADDR_W   = 16,    // scratchpad byte-address width
    parameter int XLEN     = 32,    // scalar datapath width
    parameter int M0_W     = 12,    // requant fixed-point multiplier width
    parameter int N_W      = 4,     // requant shift width

    // ---- Scalar unit sizing -------------------------------------------------
    parameter int REG_AW   = 5,     // scalar register file: 2**REG_AW regs
    parameter int IMEM_AW  = 10,    // instruction memory depth = 2**IMEM_AW
    parameter int CFG_AW   = 4,     // config register file: 2**CFG_AW regs

    // ---- Memory / init ------------------------------------------------------
    parameter     MEM_STYLE = "BRAM",  // top-level working memory primitive
    parameter     SPAD_INIT = "",      // optional scratchpad $readmemh preload
    parameter     GELU_INIT = "",      // VPU GELU LUT (256 × int8) $readmemh
    parameter     EXP_INIT  = "",      // VPU EXP  LUT (256 × int8) $readmemh

    // ---- Derived scratchpad byte widths (do not override) -------------------
    parameter int A_BYTES   = ROWS,          // A_rd  : ROWS int8 activation column
    parameter int W_BYTES   = (ROWS * 2) / 8, // W_rd  : ROWS trits, 2-bit packed
    parameter int C_BYTES   = COLS * 4,      // C_rw  : COLS int32 result row
    parameter int V_BYTES   = VPU_BYTES,     // V_rw  : VPU SIMD access
    parameter int S_BYTES   = XLEN / 8,      // S_rw  : scalar word
    parameter int DMA_BYTES  = 64            // host memory port burst width
) (
    input  logic clk,
    input  logic rst_n,

    // ---- Host control: program run + status (scalar_unit host ports) --------
    input  logic                 host_run,     // pulse: start program at boot_pc
    input  logic [IMEM_AW-1:0]   boot_pc,
    output logic                 busy,         // program executing
    output logic                 done,         // HALT reached
    output logic [IMEM_AW-1:0]   pc_dbg,

    // ---- Host: instruction memory load (while idle) -------------------------
    input  logic                 imem_we,
    input  logic [IMEM_AW-1:0]   imem_waddr,
    input  logic [31:0]          imem_wdata,

    // ---- Host: config register preset (while idle) --------------------------
    input  logic                 cfg_we,
    input  logic [CFG_AW-1:0]    cfg_waddr,
    input  logic [XLEN-1:0]      cfg_wdata,

    // ---- Host memory port: scratchpad DMA port brought out for preload /
    //      readback (stands in for the future DMA engine) ---------------------
    input  logic                  host_mem_re,
    input  logic [ADDR_W-1:0]     host_mem_raddr,
    output logic [DMA_BYTES*8-1:0] host_mem_rdata,   // valid the cycle after re
    input  logic                  host_mem_we,
    input  logic [ADDR_W-1:0]     host_mem_waddr,
    input  logic [DMA_BYTES*8-1:0] host_mem_wdata,
    input  logic [DMA_BYTES-1:0]   host_mem_wstrb
);

    // =========================================================================
    // Interconnect nets.
    // =========================================================================

    // ---- scalar_unit → MXU dispatch -----------------------------------------
    logic                mxu_start;
    logic [ADDR_W-1:0]   mxu_act_addr, mxu_weight_addr, mxu_out_addr, mxu_scalar_addr;
    logic [5:0]          mxu_t_len;
    logic                mxu_accumulate, mxu_requant;
    logic                mxu_busy, mxu_done;

    // ---- scalar_unit → VPU dispatch -----------------------------------------
    logic                vpu_start;
    logic [3:0]          vpu_op;
    logic [ADDR_W-1:0]   vpu_src0, vpu_src1, vpu_scalar, vpu_dst;
    logic [9:0]          vpu_vlen;
    logic                vpu_busy, vpu_done;

    // ---- scalar_unit ↔ scratchpad S_rw --------------------------------------
    logic                s_re, s_we;
    logic [ADDR_W-1:0]   s_addr;
    logic [XLEN-1:0]     s_wdata, s_rdata;

    // ---- MXU ↔ scratchpad A_rd / W_rd / C_rw --------------------------------
    logic                A_re;
    logic [ADDR_W-1:0]   A_raddr;
    logic [A_BYTES*8-1:0] A_rdata;

    logic                W_re;
    logic [ADDR_W-1:0]   W_raddr;
    logic [W_BYTES*8-1:0] W_rdata;

    logic                C_re;
    logic [ADDR_W-1:0]   C_raddr;
    logic [C_BYTES*8-1:0] C_rdata;
    logic                C_we;
    logic [ADDR_W-1:0]   C_waddr;
    logic [C_BYTES*8-1:0] C_wdata;
    logic [C_BYTES-1:0]  C_wstrb;

    // ---- VPU ↔ scratchpad V_rw ----------------------------------------------
    logic                V_re;
    logic [ADDR_W-1:0]   V_raddr;
    logic [V_BYTES*8-1:0] V_rdata;
    logic                V_we;
    logic [ADDR_W-1:0]   V_waddr;
    logic [V_BYTES*8-1:0] V_wdata;
    logic [V_BYTES-1:0]  V_wstrb;

    // ---- scalar_unit DMA / LINK dispatch (stubbed — see tie-offs below) -----
    logic                dma_start, dma_write;
    logic [ADDR_W-1:0]   dma_scratch_addr, dma_dram_addr;
    logic [15:0]         dma_len;
    logic                nb_start;
    logic [1:0]          nb_dir;
    logic [ADDR_W-1:0]   nb_src_addr, nb_dst_addr;
    logic [15:0]         nb_len;

    // =========================================================================
    // Scalar unit — control processor / ISA sequencer.
    // =========================================================================
    scalar_unit #(
        .XLEN    (XLEN),
        .REG_AW  (REG_AW),
        .IMEM_AW (IMEM_AW),
        .ADDR_W  (ADDR_W),
        .CFG_AW  (CFG_AW)
    ) u_scalar (
        .clk   (clk),
        .rst_n (rst_n),

        .host_run   (host_run),
        .boot_pc    (boot_pc),
        .busy       (busy),
        .done       (done),
        .pc_dbg     (pc_dbg),
        .imem_we    (imem_we),
        .imem_waddr (imem_waddr),
        .imem_wdata (imem_wdata),
        .cfg_we     (cfg_we),
        .cfg_waddr  (cfg_waddr),
        .cfg_wdata  (cfg_wdata),

        // Scratchpad scalar port
        .s_re    (s_re),
        .s_we    (s_we),
        .s_addr  (s_addr),
        .s_wdata (s_wdata),
        .s_rdata (s_rdata),

        // MXU dispatch
        .mxu_start       (mxu_start),
        .mxu_act_addr    (mxu_act_addr),
        .mxu_weight_addr (mxu_weight_addr),
        .mxu_out_addr    (mxu_out_addr),
        .mxu_scalar_addr (mxu_scalar_addr),
        .mxu_t_len       (mxu_t_len),
        .mxu_accumulate  (mxu_accumulate),
        .mxu_requant     (mxu_requant),
        .mxu_busy        (mxu_busy),
        .mxu_done        (mxu_done),

        // VPU dispatch
        .vpu_start  (vpu_start),
        .vpu_op     (vpu_op),
        .vpu_src0   (vpu_src0),
        .vpu_src1   (vpu_src1),
        .vpu_scalar (vpu_scalar),
        .vpu_dst    (vpu_dst),
        .vpu_vlen   (vpu_vlen),
        .vpu_busy   (vpu_busy),
        .vpu_done   (vpu_done),

        // DMA dispatch (no engine yet — stubbed below)
        .dma_start        (dma_start),
        .dma_write        (dma_write),
        .dma_scratch_addr (dma_scratch_addr),
        .dma_dram_addr    (dma_dram_addr),
        .dma_len          (dma_len),
        .dma_busy         (1'b0),      // stub: no DMA engine integrated
        .dma_done         (1'b1),      // stub: Read/WriteMemory completes as a no-op

        // Inter-TPU LINK dispatch (comms left out — stubbed below)
        .nb_start    (nb_start),
        .nb_dir      (nb_dir),
        .nb_src_addr (nb_src_addr),
        .nb_dst_addr (nb_dst_addr),
        .nb_len      (nb_len),
        .nb_done     (1'b1)            // stub: WriteNeighbor completes as a no-op
    );

    // dma_*/nb_* dispatch outputs are intentionally unconnected: no DMA or comms
    // block is integrated yet. Their `done` inputs are tied high above so a
    // program that issues those ops does not deadlock (the op is a no-op).

    // =========================================================================
    // MXU — weight-stationary ternary systolic matrix unit.
    // =========================================================================
    mxu #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .ADDR_W (ADDR_W),
        .M0_W   (M0_W),
        .N_W    (N_W)
    ) u_mxu (
        .clk   (clk),
        .rst_n (rst_n),

        .start       (mxu_start),
        .act_addr    (mxu_act_addr),
        .weight_addr (mxu_weight_addr),
        .out_addr    (mxu_out_addr),
        .scalar_addr (mxu_scalar_addr),
        .t_len       (mxu_t_len),
        .accumulate  (mxu_accumulate),
        .requant     (mxu_requant),
        .busy        (mxu_busy),
        .done        (mxu_done),

        // A_rd (activation feed)
        .A_re    (A_re),
        .A_raddr (A_raddr),
        .A_rdata (A_rdata),

        // W_rd (weight load)
        .W_re    (W_re),
        .W_raddr (W_raddr),
        .W_rdata (W_rdata),

        // C_rw (int32 result + accumulate readback)
        .C_re    (C_re),
        .C_raddr (C_raddr),
        .C_rdata (C_rdata),
        .C_we    (C_we),
        .C_waddr (C_waddr),
        .C_wdata (C_wdata),
        .C_wstrb (C_wstrb)
    );

    // =========================================================================
    // VPU — SIMD vector unit.
    // =========================================================================
    vpu #(
        .SCRATCHPAD_W (VPU_BYTES),
        .ADDR_W       (ADDR_W),
        .M0_W         (M0_W),
        .N_W          (N_W),
        .GELU_INIT    (GELU_INIT),
        .EXP_INIT     (EXP_INIT)
    ) u_vpu (
        .clk   (clk),
        .rst_n (rst_n),

        .vpu_start  (vpu_start),
        .vpu_op     (vpu_op),
        .vpu_src0   (vpu_src0),
        .vpu_src1   (vpu_src1),
        .vpu_scalar (vpu_scalar),
        .vpu_dst    (vpu_dst),
        .vpu_vlen   (vpu_vlen),
        .vpu_busy   (vpu_busy),
        .vpu_done   (vpu_done),

        // V_rw (SIMD read/modify/write)
        .V_re    (V_re),
        .V_raddr (V_raddr),
        .V_rdata (V_rdata),
        .V_we    (V_we),
        .V_waddr (V_waddr),
        .V_wdata (V_wdata),
        .V_wstrb (V_wstrb)
    );

    // =========================================================================
    // Scratchpad — shared BRAM working memory (single storage owner).
    // The DMA port is brought out to the top level as the host preload/readback
    // interface (host_mem_*), standing in for a future DMA engine.
    // =========================================================================
    scratchpad #(
        .MEM_STYLE (MEM_STYLE),
        .ADDR_W    (ADDR_W),
        .A_BYTES   (A_BYTES),
        .W_BYTES   (W_BYTES),
        .C_BYTES   (C_BYTES),
        .V_BYTES   (V_BYTES),
        .S_BYTES   (S_BYTES),
        .DMA_BYTES (DMA_BYTES),
        .INIT_FILE (SPAD_INIT)
    ) u_scratchpad (
        .clk   (clk),
        .rst_n (rst_n),

        // A_rd (MXU activation feed)
        .A_re    (A_re),
        .A_raddr (A_raddr),
        .A_rdata (A_rdata),

        // W_rd (MXU weight load)
        .W_re    (W_re),
        .W_raddr (W_raddr),
        .W_rdata (W_rdata),

        // C_rw (MXU result + accumulate)
        .C_re    (C_re),
        .C_raddr (C_raddr),
        .C_rdata (C_rdata),
        .C_we    (C_we),
        .C_waddr (C_waddr),
        .C_wdata (C_wdata),
        .C_wstrb (C_wstrb),

        // V_rw (VPU SIMD)
        .V_re    (V_re),
        .V_raddr (V_raddr),
        .V_rdata (V_rdata),
        .V_we    (V_we),
        .V_waddr (V_waddr),
        .V_wdata (V_wdata),
        .V_wstrb (V_wstrb),

        // S_rw (scalar unit word)
        .s_re    (s_re),
        .s_we    (s_we),
        .s_addr  (s_addr),
        .s_wdata (s_wdata),
        .s_rdata (s_rdata),

        // DMA port → host preload/readback
        .dma_re    (host_mem_re),
        .dma_raddr (host_mem_raddr),
        .dma_rdata (host_mem_rdata),
        .dma_we    (host_mem_we),
        .dma_waddr (host_mem_waddr),
        .dma_wdata (host_mem_wdata),
        .dma_wstrb (host_mem_wstrb)
    );

endmodule
