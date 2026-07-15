`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// scratchpad.sv — TPU on-chip working memory
//
// Implements the block described in accel/tpu/docs/scratchpad.md: the shared,
// byte-addressed working memory that every compute unit reads and writes. DRAM
// is only ever touched through the DMA port; every math instruction names
// scratchpad byte addresses (scratchpad.md §1). This module is the single owner
// of the storage and exposes the typed ports the existing units already drive,
// so a top-level just wires same-named signals:
//
//   A_rd  (read only)   MXU activation feed   — mxu.sv A_re/A_raddr/A_rdata
//   W_rd  (read only)   MXU weight load       — mxu.sv W_re/W_raddr/W_rdata
//   C_rw  (read/write)  MXU int32 result +    — mxu.sv C_re.../C_we...
//                       accumulate readback
//   V_rw  (read/write)  VPU SIMD lanes        — vpu.sv V_re.../V_we...
//   S_rw  (read/write)  scalar unit word      — scalar_unit.sv s_re/s_we/s_addr...
//   DMA   (read/write)  DMA engine            — DRAM <-> scratchpad (future block)
//
// Memory contract (matches every unit and both testbenches): **synchronous
// read** — address/enable presented one cycle, `*_rdata` valid the next; writes
// complete in one cycle under a per-byte write strobe. Read-during-write to the
// same byte returns the OLD value (read-first), which is what the units and the
// inline TB memory models assume.
//
// The doc's typed ports have different widths (§4): each is parameterized by its
// byte width and gathers/scatters that many contiguous bytes starting at the
// byte address. int8 operands occupy one byte/element, int32 four; the widths
// below default to a 128x128 array (A=128 B activation column, C=512 B int32
// result row, V=64 B / 512-bit VPU access, S=4 B scalar word).
//
// Arbitration (scratchpad.md §4). Access is statically arbitrated by the scalar
// unit's issue-and-wait model: the MXU owns A_rd/W_rd/C_rw for a Matmul, the VPU
// owns V_rw, DMA fills idle cycles. Reads are independent per port (the MXU reads
// A_rd, W_rd and C_rw concurrently), so all read ports are live every cycle. The
// write ports (C/V/S/DMA) are not expected to target the same byte in the same
// cycle; if they ever do, the priority below (C < V < S < DMA) is last-writer-wins.
//
// Registers vs BRAM (`MEM_STYLE`). The storage is selected at elaboration:
//   MEM_STYLE = "BRAM" (default) -> ram_style="block":    dense FPGA block RAM.
//   MEM_STYLE = "REG"            -> ram_style="registers": flip-flop / LUTRAM
//                                   register file.
// Both honour the identical synchronous-read timing above, so the rest of the
// TPU is agnostic to the choice — it only trades area/primitive: a small, wide,
// many-read-port scratch region is cheap as a register file; a large one wants
// block RAM. (A true block-RAM mapping of the full multi-read/multi-write port
// set requires the per-bank replication in scratchpad.md §3; until a board is
// fixed in constraints/ this behavioural model plus the ram_style hint captures
// the intent and the two synthesis targets.)
// -----------------------------------------------------------------------------

module scratchpad #(
    parameter     MEM_STYLE = "BRAM",  // "BRAM" -> block RAM; "REG" -> flip-flops
    parameter int ADDR_W    = 16,      // byte-address width; storage = 2**ADDR_W bytes
    parameter int A_BYTES   = 128,     // A_rd  width (bytes): MXU activation column
    parameter int W_BYTES   = 32,      // W_rd  width (bytes): packed ternary column
    parameter int C_BYTES   = 512,     // C_rw  width (bytes): int32 result row
    parameter int V_BYTES   = 64,      // V_rw  width (bytes): VPU SIMD access (512b)
    parameter int S_BYTES   = 4,       // S_rw  width (bytes): scalar word (int32)
    parameter int DMA_BYTES = 64,      // DMA   width (bytes): DMA burst
    parameter     INIT_FILE = ""       // optional $readmemh preload (one byte/line)
) (
    input  logic clk,
    input  logic rst_n,   // clears the read-data output registers; storage is not reset

    // ---- A_rd : MXU activation feed (read only) -----------------------------
    input  logic                 A_re,
    input  logic [ADDR_W-1:0]    A_raddr,
    output logic [A_BYTES*8-1:0] A_rdata,     // valid the cycle after A_re

    // ---- W_rd : MXU weight load (read only) ---------------------------------
    input  logic                 W_re,
    input  logic [ADDR_W-1:0]    W_raddr,
    output logic [W_BYTES*8-1:0] W_rdata,     // valid the cycle after W_re

    // ---- C_rw : MXU int32 result store + accumulate readback ----------------
    input  logic                 C_re,
    input  logic [ADDR_W-1:0]    C_raddr,
    output logic [C_BYTES*8-1:0] C_rdata,     // valid the cycle after C_re
    input  logic                 C_we,
    input  logic [ADDR_W-1:0]    C_waddr,
    input  logic [C_BYTES*8-1:0] C_wdata,
    input  logic [C_BYTES-1:0]   C_wstrb,     // per-byte write enable

    // ---- V_rw : VPU SIMD read/modify/write ----------------------------------
    input  logic                 V_re,
    input  logic [ADDR_W-1:0]    V_raddr,
    output logic [V_BYTES*8-1:0] V_rdata,     // valid the cycle after V_re
    input  logic                 V_we,
    input  logic [ADDR_W-1:0]    V_waddr,
    input  logic [V_BYTES*8-1:0] V_wdata,
    input  logic [V_BYTES-1:0]   V_wstrb,     // per-byte write enable

    // ---- S_rw : scalar unit single word (shared addr for read + write) ------
    input  logic                 s_re,
    input  logic                 s_we,
    input  logic [ADDR_W-1:0]    s_addr,
    input  logic [S_BYTES*8-1:0] s_wdata,
    output logic [S_BYTES*8-1:0] s_rdata,     // valid the cycle after s_re

    // ---- DMA : DRAM <-> scratchpad engine (read/modify/write) ---------------
    input  logic                 dma_re,
    input  logic [ADDR_W-1:0]    dma_raddr,
    output logic [DMA_BYTES*8-1:0] dma_rdata,   // valid the cycle after dma_re
    input  logic                 dma_we,
    input  logic [ADDR_W-1:0]    dma_waddr,
    input  logic [DMA_BYTES*8-1:0] dma_wdata,
    input  logic [DMA_BYTES-1:0]   dma_wstrb    // per-byte write enable
);

    localparam int DEPTH = (1 << ADDR_W);

    // -------------------------------------------------------------------------
    // Shared byte-addressed storage. The two elaboration arms carry the same
    // block name `g_mem`, so `g_mem.mem` resolves regardless of MEM_STYLE and
    // the port logic below stays single-sourced; only the ram_style hint (and
    // hence the inferred FPGA primitive) differs.
    // -------------------------------------------------------------------------
    generate
        if (MEM_STYLE == "REG" || MEM_STYLE == "registers" ||
            MEM_STYLE == "distributed") begin : g_mem
            (* ram_style = "registers" *) logic [7:0] mem [0:DEPTH-1];
        end else begin : g_mem
            (* ram_style = "block" *)     logic [7:0] mem [0:DEPTH-1];
        end
    endgenerate

    initial if (INIT_FILE != "") $readmemh(INIT_FILE, g_mem.mem);

    // -------------------------------------------------------------------------
    // Read ports (synchronous, read-first). Each latches its byte window on its
    // own enable; all read ports are independent and may fire the same cycle.
    // Address arithmetic is ADDR_W-wide, so a window near the top wraps mod
    // DEPTH rather than indexing out of range.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            A_rdata   <= '0;
            W_rdata   <= '0;
            C_rdata   <= '0;
            V_rdata   <= '0;
            s_rdata   <= '0;
            dma_rdata <= '0;
        end else begin
            if (A_re)
                for (int k = 0; k < A_BYTES; k++)
                    A_rdata[k*8 +: 8]   <= g_mem.mem[A_raddr + ADDR_W'(k)];
            if (W_re)
                for (int k = 0; k < W_BYTES; k++)
                    W_rdata[k*8 +: 8]   <= g_mem.mem[W_raddr + ADDR_W'(k)];
            if (C_re)
                for (int k = 0; k < C_BYTES; k++)
                    C_rdata[k*8 +: 8]   <= g_mem.mem[C_raddr + ADDR_W'(k)];
            if (V_re)
                for (int k = 0; k < V_BYTES; k++)
                    V_rdata[k*8 +: 8]   <= g_mem.mem[V_raddr + ADDR_W'(k)];
            if (s_re)
                for (int k = 0; k < S_BYTES; k++)
                    s_rdata[k*8 +: 8]   <= g_mem.mem[s_addr + ADDR_W'(k)];
            if (dma_re)
                for (int k = 0; k < DMA_BYTES; k++)
                    dma_rdata[k*8 +: 8] <= g_mem.mem[dma_raddr + ADDR_W'(k)];
        end
    end

    // -------------------------------------------------------------------------
    // Write ports (byte-strobed, single cycle). One process owns the storage.
    // Priority on a same-byte collision (not expected under static arbitration):
    // C < V < S < DMA, last-listed wins.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (C_we)
            for (int k = 0; k < C_BYTES; k++)
                if (C_wstrb[k]) g_mem.mem[C_waddr + ADDR_W'(k)]   <= C_wdata[k*8 +: 8];
        if (V_we)
            for (int k = 0; k < V_BYTES; k++)
                if (V_wstrb[k]) g_mem.mem[V_waddr + ADDR_W'(k)]   <= V_wdata[k*8 +: 8];
        if (s_we)   // scalar word write has no sub-word strobe; whole S_BYTES word
            for (int k = 0; k < S_BYTES; k++)
                g_mem.mem[s_addr + ADDR_W'(k)] <= s_wdata[k*8 +: 8];
        if (dma_we)
            for (int k = 0; k < DMA_BYTES; k++)
                if (dma_wstrb[k]) g_mem.mem[dma_waddr + ADDR_W'(k)] <= dma_wdata[k*8 +: 8];
    end

endmodule
