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
// instantiated with MEM_STYLE="BRAM" (scratchpad.md §6). External DRAM is the
// async SRAM chip: the DMA engine (dma.sv) owns the scratchpad's DMA port and,
// behind it, a single sram_controller whose chip pins (`sram_*`) are brought out
// to the top level. The scalar unit's Read/WriteMemory ops dispatch to the DMA
// engine, which byte-moves DRAM <-> scratchpad. The scalar unit emits a full
// MEM_ADDR_W DRAM address, so a program reaches the whole SRAM chip; only the
// scratchpad side is ADDR_W. (It used to emit ADDR_W and be zero-extended here,
// which silently confined every program to the low 64 KB of a 512 KB part.)
//
// Deliberately left out (see notes at the stub tie-off below):
//   * comms / inter-TPU LINK — the scalar unit's WriteNeighbor dispatch is
//     stubbed (`nb_done` tied high) so a LINK op is a completing no-op. Wiring a
//     real comms block (comms.md) is a later step.
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
    parameter int CFG_AW   = 5,     // config register file: 2**CFG_AW regs

    // ---- External DRAM (async SRAM chip) ------------------------------------
    parameter int MEM_ADDR_W = 19,   // external SRAM byte address (CMOD A7: 512K×8)
    parameter int MEM_DATA_W = 8,    // external SRAM data width (bits) = 1 byte
    parameter int SRAM_CPA   = 0,    // sram_controller CLOCKS_PER_ACCESS (extra)

    // ---- UART host link (bring-up / debug over serial; docs/uart_host.md) ----
    parameter int UART_CPB        = 104, // UART clocks-per-bit (115200 @ 12 MHz)
    parameter int UART_RX_TIMEOUT = 0,   // UART inter-byte abort (clocks; 0 = off)

    // ---- Memory / init ------------------------------------------------------
    parameter     MEM_STYLE = "BRAM",  // top-level working memory primitive
    parameter     SPAD_INIT = "",      // optional scratchpad $readmemh preload
    // GELU_INIT / EXP_INIT are gone with the `gelu` / `exp` instructions and
    // their 256-entry ROMs (vpu.sv header). The VPU now has no $readmemh
    // artifact at all, so rtl/luts/ and accel/tpulang/luts.py were deleted too.

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

    // ---- External DRAM pins: async SRAM chip interface ----------------------
    //      Driven by the on-chip DMA engine (while a program runs) or the UART
    //      host (while idle) through the shared sram_controller. The scalar unit's
    //      Read/WriteMemory ops move data over this port when running.
    output logic [MEM_ADDR_W-1:0] sram_addr,
    inout  wire  [MEM_DATA_W-1:0] sram_data,   // inout must be a net, not a var
    output logic                  sram_we,
    output logic                  sram_ce,
    output logic                  sram_oen,

    // ---- UART host serial link ----------------------------------------------
    //      Lets a host PC read/write external SRAM, load the program image into
    //      instruction memory, and start the core — all while the core is idle
    //      (commands arriving mid-run are NAK'd). See docs/uart_host.md.
    input  logic                  uart_rx,     // serial in from host
    output logic                  uart_tx      // serial out to host
);

    // =========================================================================
    // Interconnect nets.
    // =========================================================================

    // ---- scalar_unit → MXU dispatch -----------------------------------------
    logic                mxu_start;
    logic [ADDR_W-1:0]   mxu_act_addr, mxu_weight_addr, mxu_out_addr, mxu_scalar_addr;
    logic [5:0]          mxu_t_len;
    logic                mxu_tiled;
    logic [ADDR_W-1:0]   mxu_a_row, mxu_c_row, mxu_w_col;
    logic [7:0]          mxu_k_tiles, mxu_n_tiles;
    logic                mxu_accumulate, mxu_requant;
    logic                mxu_busy, mxu_done;

    // ---- scalar_unit → VPU dispatch -----------------------------------------
    logic                vpu_start;
    logic [4:0]          vpu_op;
    logic [ADDR_W-1:0]   vpu_src0, vpu_src1, vpu_scalar, vpu_dst;
    logic [9:0]          vpu_vlen;
    logic [15:0]         vpu_rows, vpu_cols;
    logic [ADDR_W-1:0]   vpu_row0, vpu_row1, vpu_crow;
    logic                vpu_busy, vpu_done;
    logic                vpu_mm_busy;   // VPU busy on a vecmatmul (perf only)

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

    // ---- scalar_unit → DMA dispatch -----------------------------------------
    logic                dma_start, dma_write;
    logic [ADDR_W-1:0]     dma_scratch_addr;
    logic [MEM_ADDR_W-1:0] dma_dram_addr;   // the DRAM side is the wider space
    logic [15:0]         dma_len;
    logic                dma_transpose;                       // `.t` instruction flag
    logic [15:0]         dma_tcols, dma_tsrow, dma_tdrow;     // transpose geometry (cfg)
    logic                dma_busy, dma_done;

    // ---- DMA ↔ scratchpad DMA port ------------------------------------------
    logic                 spad_dma_re;
    logic [ADDR_W-1:0]    spad_dma_raddr;
    logic [DMA_BYTES*8-1:0] spad_dma_rdata;
    logic                 spad_dma_we;
    logic [ADDR_W-1:0]    spad_dma_waddr;
    logic [DMA_BYTES*8-1:0] spad_dma_wdata;
    logic [DMA_BYTES-1:0]   spad_dma_wstrb;

    // ---- sram_controller (user side) — shared, arbitrated below -------------
    //   DMA drives the controller while a program runs; the UART host drives it
    //   while idle. `mem_*` are the muxed signals actually wired to the
    //   controller; `dma_mem_*` / `uart_mem_*` are the two candidate drivers.
    logic                  mem_start, mem_we, mem_busy, mem_done;
    logic [MEM_ADDR_W-1:0] mem_addr;
    logic [15:0]           mem_len, mem_stride;
    logic [MEM_DATA_W-1:0] mem_din, mem_dout;
    logic                  mem_din_valid, mem_din_ready, mem_dout_valid;

    logic                  dma_mem_start, dma_mem_we;
    logic [MEM_ADDR_W-1:0] dma_mem_addr;
    logic [15:0]           dma_mem_len, dma_mem_stride;
    logic [MEM_DATA_W-1:0] dma_mem_din;
    logic                  dma_mem_din_valid;

    logic                  uart_mem_start, uart_mem_we;
    logic [MEM_ADDR_W-1:0] uart_mem_addr;
    logic [MEM_DATA_W-1:0] uart_mem_din;

    // ---- UART host interface ------------------------------------------------
    logic [7:0]            uart_rx_data;
    logic                  uart_rx_valid;
    logic                  uart_tx_start, uart_tx_busy;
    logic [7:0]            uart_tx_data;

    logic                  uart_imem_we;
    logic [IMEM_AW-1:0]    uart_imem_waddr;
    logic [31:0]           uart_imem_wdata;
    logic                  uart_run_start;
    logic [IMEM_AW-1:0]    uart_run_pc;

    // ---- Performance counters → UART 'T' command ----------------------------
    //   NPERF event bits, integrated over one run by perf_counters.sv. See the
    //   PERF_* indices and the wire-order remap where the block is instantiated.
    localparam int NPERF = 7;
    logic [NPERF-1:0]      perf_ev;
    logic [NPERF*32-1:0]   perf_counts;   // counter i at [i*32 +: 32]
    logic [NPERF*32-1:0]   perf_wire;     // same, reordered for transmission
    logic                  su_wait_active, mxu_load_active;

    // ---- Host program/run path: external host ports OR'd with the UART host --
    //   Both paths write instruction memory and pulse host_run; they are used
    //   only while the core is idle, so a simple priority-mux is safe.
    logic                  su_host_run;
    logic [IMEM_AW-1:0]    su_boot_pc;
    logic                  su_imem_we;
    logic [IMEM_AW-1:0]    su_imem_waddr;
    logic [31:0]           su_imem_wdata;

    assign su_host_run   = host_run  | uart_run_start;
    assign su_boot_pc    = uart_run_start ? uart_run_pc : boot_pc;
    assign su_imem_we    = imem_we   | uart_imem_we;
    assign su_imem_waddr = uart_imem_we ? uart_imem_waddr : imem_waddr;
    assign su_imem_wdata = uart_imem_we ? uart_imem_wdata : imem_wdata;

    // ---- SRAM arbitration: the core has priority ----------------------------
    //   While a program runs (`busy`) the DMA engine owns the controller; while
    //   idle the UART host does. The UART FSM independently NAKs any command that
    //   arrives while `busy`, so the two drivers never actually contend.
    //   The host side moves one byte per command turnaround, so it takes the
    //   controller's range interface at len = 1 and holds `din` for the whole
    //   transaction; the DMA drives the range and stream signals for real.
    assign mem_start     = busy ? dma_mem_start     : uart_mem_start;
    assign mem_we        = busy ? dma_mem_we        : uart_mem_we;
    assign mem_addr      = busy ? dma_mem_addr      : uart_mem_addr;
    assign mem_len       = busy ? dma_mem_len       : 16'd1;
    assign mem_stride    = busy ? dma_mem_stride    : 16'd0;
    assign mem_din       = busy ? dma_mem_din       : uart_mem_din;
    assign mem_din_valid = busy ? dma_mem_din_valid : 1'b1;

    // ---- scalar_unit → LINK dispatch (comms left out — stubbed below) -------
    logic                nb_start;
    logic [1:0]          nb_dir;
    logic [ADDR_W-1:0]   nb_src_addr, nb_dst_addr;
    logic [15:0]         nb_len;

    // =========================================================================
    // Scalar unit — control processor / ISA sequencer.
    // =========================================================================
    scalar_unit #(
        .XLEN       (XLEN),
        .REG_AW     (REG_AW),
        .IMEM_AW    (IMEM_AW),
        .ADDR_W     (ADDR_W),
        .MEM_ADDR_W (MEM_ADDR_W),
        .CFG_AW     (CFG_AW)
    ) u_scalar (
        .clk   (clk),
        .rst_n (rst_n),

        .host_run   (su_host_run),
        .boot_pc    (su_boot_pc),
        .busy        (busy),
        .done        (done),
        .pc_dbg      (pc_dbg),
        .wait_active (su_wait_active),
        .imem_we    (su_imem_we),
        .imem_waddr (su_imem_waddr),
        .imem_wdata (su_imem_wdata),
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
        .mxu_tiled       (mxu_tiled),
        .mxu_a_row       (mxu_a_row),
        .mxu_c_row       (mxu_c_row),
        .mxu_w_col       (mxu_w_col),
        .mxu_k_tiles     (mxu_k_tiles),
        .mxu_n_tiles     (mxu_n_tiles),
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
        .vpu_rows   (vpu_rows),
        .vpu_cols   (vpu_cols),
        .vpu_row0   (vpu_row0),
        .vpu_row1   (vpu_row1),
        .vpu_crow   (vpu_crow),
        .vpu_busy   (vpu_busy),
        .vpu_done   (vpu_done),

        // DMA dispatch → DMA engine (below)
        .dma_start        (dma_start),
        .dma_write        (dma_write),
        .dma_scratch_addr (dma_scratch_addr),
        .dma_dram_addr    (dma_dram_addr),
        .dma_len          (dma_len),
        .dma_transpose    (dma_transpose),
        .dma_tcols        (dma_tcols),
        .dma_tsrow        (dma_tsrow),
        .dma_tdrow        (dma_tdrow),
        .dma_busy         (dma_busy),
        .dma_done         (dma_done),

        // Inter-TPU LINK dispatch (comms left out — stubbed below)
        .nb_start    (nb_start),
        .nb_dir      (nb_dir),
        .nb_src_addr (nb_src_addr),
        .nb_dst_addr (nb_dst_addr),
        .nb_len      (nb_len),
        .nb_done     (1'b1)            // stub: WriteNeighbor completes as a no-op
    );

    // nb_* dispatch outputs are intentionally unconnected: no comms block is
    // integrated yet. Its `nb_done` input is tied high above so a program that
    // issues a LINK op does not deadlock (the op is a no-op). The DMA dispatch is
    // now wired to the real DMA engine below.

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
        .tiled       (mxu_tiled),
        .a_row       (mxu_a_row),
        .c_row       (mxu_c_row),
        .w_col       (mxu_w_col),
        .k_tiles     (mxu_k_tiles),
        .n_tiles     (mxu_n_tiles),
        .accumulate  (mxu_accumulate),
        .requant     (mxu_requant),
        .busy        (mxu_busy),
        .done        (mxu_done),
        .load_active (mxu_load_active),

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
        .N_W          (N_W)
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
        .vpu_rows   (vpu_rows),
        .vpu_cols   (vpu_cols),
        .vpu_row0   (vpu_row0),
        .vpu_row1   (vpu_row1),
        .vpu_crow   (vpu_crow),
        .vpu_busy    (vpu_busy),
        .vpu_done    (vpu_done),
        .vpu_mm_busy (vpu_mm_busy),

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
    // The DMA port is owned by the on-chip DMA engine (instantiated below).
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

        // DMA port → DMA engine
        .dma_re    (spad_dma_re),
        .dma_raddr (spad_dma_raddr),
        .dma_rdata (spad_dma_rdata),
        .dma_we    (spad_dma_we),
        .dma_waddr (spad_dma_waddr),
        .dma_wdata (spad_dma_wdata),
        .dma_wstrb (spad_dma_wstrb)
    );

    // =========================================================================
    // DMA engine — byte-moves DRAM <-> scratchpad on scalar Read/WriteMemory.
    // Owns the scratchpad DMA port (above) and the sram_controller (below). The
    // scalar unit's DRAM address is ADDR_W wide; zero-extend to the wider
    // external-SRAM byte address.
    // =========================================================================
    dma #(
        .MEM_ADDR_W        (MEM_ADDR_W),
        .SCRATCHPAD_ADDR_W (ADDR_W),
        .MEM_DATA_W        (MEM_DATA_W),
        .SCRATCHPAD_BYTES  (DMA_BYTES)
    ) u_dma (
        .clk   (clk),
        .rst_n (rst_n),

        // sram_controller interface (user side) — arbitrated with the UART host
        .sram_start      (dma_mem_start),
        .sram_we         (dma_mem_we),
        .sram_addr       (dma_mem_addr),
        .sram_len        (dma_mem_len),
        .sram_stride     (dma_mem_stride),
        .sram_din        (dma_mem_din),
        .sram_din_valid  (dma_mem_din_valid),
        .sram_din_ready  (mem_din_ready),
        .sram_dout       (mem_dout),
        .sram_dout_valid (mem_dout_valid),
        .sram_busy       (mem_busy),
        .sram_done       (mem_done),

        // scratchpad DMA port
        .scratchpad_re    (spad_dma_re),
        .scratchpad_raddr (spad_dma_raddr),
        .scratchpad_rdata (spad_dma_rdata),
        .scratchpad_we    (spad_dma_we),
        .scratchpad_waddr (spad_dma_waddr),
        .scratchpad_wdata (spad_dma_wdata),
        .scratchpad_wstrb (spad_dma_wstrb),

        // scalar-unit dispatch (issue-and-wait)
        .dma_start        (dma_start),
        .dma_write        (dma_write),
        .dma_scratch_addr (dma_scratch_addr),
        .dma_dram_addr    (dma_dram_addr),   // MEM_ADDR_W on both sides
        .dma_len          (dma_len),
        .dma_transpose    (dma_transpose),
        .dma_tcols        (dma_tcols),
        .dma_tsrow        (dma_tsrow),
        .dma_tdrow        (dma_tdrow),
        .dma_busy         (dma_busy),
        .dma_done         (dma_done)
    );

    // =========================================================================
    // SRAM controller — single owner is the DMA engine. Its chip-side pins are
    // the top-level external DRAM interface (sram_*).
    // =========================================================================
    sram_controller #(
        .CLOCKS_PER_ACCESS (SRAM_CPA),
        .ADDR_W            (MEM_ADDR_W),
        .DATA_W            (MEM_DATA_W)
    ) u_sram (
        .clk   (clk),
        .rst_n (rst_n),

        // user side ← DMA engine (or the UART host while idle)
        .start  (mem_start),
        .we     (mem_we),
        .addr   (mem_addr),
        .len    (mem_len),
        .stride (mem_stride),

        .din       (mem_din),
        .din_valid (mem_din_valid),
        .din_ready (mem_din_ready),

        .dout       (mem_dout),
        .dout_valid (mem_dout_valid),

        .busy (mem_busy),
        .done (mem_done),

        // chip side → top-level pins
        .sram_addr (sram_addr),
        .sram_data (sram_data),
        .sram_we   (sram_we),
        .sram_ce   (sram_ce),
        .sram_oen  (sram_oen)
    );

    // =========================================================================
    // Performance counters — integrate seven event bits over one program run
    // (from 'G' to HALT), read over UART with the 'T' command. Nothing else in
    // the core observes them.
    //
    // Counter 0 is the run length itself, which makes it the denominator for
    // every other counter: each one is directly readable as a fraction of the
    // run. Together they are the bottleneck breakdown docs/macro_ops.md §7 calls
    // for — without them, a hardware-sequenced tile loop is invisible, because
    // the loop no longer appears in the instruction stream.
    //
    //   0 run    total clocks busy  (denominator; == the old cycle_timer)
    //   1 mxu    MXU busy           (systolic array utilization)
    //   2 mload  MXU in S_LOAD      (weight-load share of MXU time)
    //   3 vpu    VPU busy           (attention share vs. GEMM)
    //   4 dma    DMA busy           (memory-bound vs. compute-bound)
    //   5 swait  scalar in S_WAIT   (what issue-and-wait costs)
    //   6 vmm    VPU on a vecmatmul (macro-op share of VPU time)
    //
    // These overlap by construction: under issue-and-wait `swait` covers nearly
    // all of `mxu`+`vpu`+`dma`, `mload` is a subset of `mxu`, and `vmm` is a
    // subset of `vpu`. They are fractions of a run, not a partition of it.
    //
    // Counter 6 exists because `vpu` on its own conflates two very different
    // costs. The primitive ops (add / relu / requant / dyt) stream LANES
    // elements per chunk and are cheap per element; `vecmatmul` re-runs the
    // inner dot product once per (row, col) pair and pays the S_RD0..S_WB round
    // trip on each one, so a VPU share that looks high may be almost entirely
    // one macro op. Which of the two it is decides whether the thing worth
    // optimizing is the pointwise path or `vecmatmul`'s per-pair overhead.
    // =========================================================================
    localparam int PERF_RUN = 0, PERF_MXU  = 1, PERF_MLOAD = 2,
                   PERF_VPU = 3, PERF_DMA  = 4, PERF_SWAIT = 5,
                   PERF_VMM = 6;

    always_comb begin
        perf_ev              = '0;
        perf_ev[PERF_RUN]    = busy;
        perf_ev[PERF_MXU]    = mxu_busy;
        perf_ev[PERF_MLOAD]  = mxu_load_active;
        perf_ev[PERF_VPU]    = vpu_busy;
        perf_ev[PERF_DMA]    = dma_busy;
        perf_ev[PERF_SWAIT]  = su_wait_active;
        perf_ev[PERF_VMM]    = vpu_mm_busy;
    end

    perf_counters #(
        .N (NPERF),
        .W (32)
    ) u_perf (
        .clk    (clk),
        .rst_n  (rst_n),
        .run    (busy),
        .ev     (perf_ev),
        .counts (perf_counts)
    );

    // Wire order: uart_interface shifts the 'T' reply out of the *high* end, so
    // reverse the word order here to put counter 0 (the run length) on the wire
    // first. That keeps a 'T' reply prefix-compatible with the single-word reply
    // this command used to give, and keeps the natural little-end packing inside
    // perf_counters.sv rather than hiding a transmission detail in it.
    always_comb begin
        for (int i = 0; i < NPERF; i++)
            perf_wire[(NPERF-1-i)*32 +: 32] = perf_counts[i*32 +: 32];
    end

    // =========================================================================
    // UART host link — serial bring-up/debug path (docs/uart_host.md).
    //   uart_receiver/transmitter do the 8N1 wire framing; uart_interface is the
    //   command FSM. It shares the sram_controller with the DMA engine (mux above,
    //   core priority), writes the scalar unit's instruction memory, and can start
    //   the program. Any command arriving while the core is busy is NAK'd.
    // =========================================================================
    uart_receiver #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_rx (
        .clk   (clk),
        .rst_n (rst_n),
        .uart_rx (uart_rx),
        .data    (uart_rx_data),
        .valid   (uart_rx_valid)
    );

    uart_transmitter #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_tx (
        .clk   (clk),
        .rst_n (rst_n),
        .start   (uart_tx_start),
        .data    (uart_tx_data),
        .uart_tx (uart_tx),
        .busy    (uart_tx_busy)
    );

    uart_interface #(
        .ADDR_W     (MEM_ADDR_W),
        .LENGTH_W   (16),
        .IMEM_AW     (IMEM_AW),
        .RX_TIMEOUT  (UART_RX_TIMEOUT),
        .TIMER_WORDS (NPERF)
    ) u_uart (
        .clk   (clk),
        .rst_n (rst_n),

        // arbitration: core has priority ('T' excepted — see uart_interface.sv)
        .core_busy   (busy),
        .cycle_count (perf_wire),

        // receiver / transmitter
        .data_in           (uart_rx_data),
        .receiver_valid    (uart_rx_valid),
        .transmitter_start (uart_tx_start),
        .data_out          (uart_tx_data),
        .transmitter_busy  (uart_tx_busy),

        // sram_controller user side (muxed with DMA above)
        .sram_start (uart_mem_start),
        .sram_we    (uart_mem_we),
        .sram_addr  (uart_mem_addr),
        .sram_din   (uart_mem_din),
        .sram_dout  (mem_dout),
        .sram_busy  (mem_busy),
        .sram_done  (mem_done),

        // instruction-memory write → scalar unit (OR'd into su_imem_* above)
        .imem_we    (uart_imem_we),
        .imem_waddr (uart_imem_waddr),
        .imem_wdata (uart_imem_wdata),

        // run trigger → scalar unit (OR'd into su_host_run above)
        .run_start (uart_run_start),
        .run_pc    (uart_run_pc),

        .host_busy  (),  // informational; unused at top level
        .rx_overrun ()   // ditto — cmod_a7_mem is the image that surfaces it
    );

endmodule
