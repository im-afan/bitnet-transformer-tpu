`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// cpu_subsys.sv — PicoRV32 + AXI4-Lite fabric + the macro-op MMIO aperture
//
// The second command producer. `scalar_unit.sv` packs the 32-bit tpulang ISA
// into commands; this packs whatever the firmware stores into the command ports.
// Both push into the same cmd_mxu/cmd_vpu/cmd_dma queues, and tpu_top arbitrates
// (docs/picorv32_migration.md §11 phase 4 — both live at once so the existing
// .tpu suite stays a regression while the C path is brought up).
//
//     picorv32_axi ──AXI4-Lite──► [ decode ] ──► firmware RAM   0x0000_0000
//                                            ├─► MMIO / cmd     0x8000_0000
//                                            └─► scratchpad     0x9000_0000
//
// ONE SLAVE, NOT AN INTERCONNECT. PicoRV32 has a single outstanding transaction,
// so the three regions are decoded inside one AXI4-Lite slave FSM rather than
// behind a crossbar. That is the whole "fabric": ~120 lines, no arbitration, no
// outstanding-transaction tracking.
//
// TOPOLOGY A of the migration doc §6: everything, instruction fetch included,
// goes over AXI4-Lite. It is the simple one and it costs about 1.4% of total
// runtime against tightly coupling the RAM to the core's native port, which is
// affordable at the adder model's arithmetic intensity and measurable with the
// `starved` counters if it ever stops being. Topology B is a later swap of this
// file's insides, not of its interface.
//
// COMMAND PORTS ARE 4-WORD APERTURES, COMMIT ON THE LAST WORD. Writes to +0x0,
// +0x4, +0x8 stage into a register; the write to +0xC assembles the 128 bits and
// pushes. Four stores are therefore atomic with respect to the queue with no
// separate trigger store, and a torn command cannot be enqueued. If the target
// queue is full the +0xC write does not complete — `bvalid` is withheld, the CPU
// stalls inside the store, and flow control needs no software at all. Only data
// dependencies are the program's problem (§4).
//
// HALT. The firmware writes MMIO_DONE and then spins; `ebreak` traps and does
// the same thing. Either raises `cpu_done`, which tpu_top reports to the host
// exactly where the scalar unit's HALT used to.
// -----------------------------------------------------------------------------

module cpu_subsys #(
    parameter int ADDR_W     = 16,   // scratchpad byte address
    parameter int MEM_ADDR_W = 19,   // (unused here; kept for symmetry)
    parameter int XLEN       = 32,
    parameter int FW_AW      = 12,   // firmware RAM depth = 2**FW_AW words (16 KB)
    parameter     FW_INIT    = ""    // optional $readmemh preload (one word/line)
) (
    input  logic clk,
    input  logic rst_n,

    // ---- run control --------------------------------------------------------
    input  logic              cpu_run,     // level: high releases the core
    output logic              cpu_busy,
    output logic              cpu_done,    // firmware wrote DONE, or trapped
    output logic              cpu_trap,

    // ---- host firmware load (while the core is held in reset) ---------------
    input  logic              fw_we,
    input  logic [FW_AW-1:0]  fw_waddr,
    input  logic [31:0]       fw_wdata,

    // ---- command push (to the three queues; tpu_top arbitrates) -------------
    output logic              cmd_we,
    output logic [1:0]        cmd_unit,
    output logic [127:0]      cmd_data,
    input  logic              cmd_full,

    // ---- unit status (read back through the MMIO aperture) -----------------
    input  logic [31:0]       mxu_issued, mxu_retired,
    input  logic [31:0]       vpu_issued, vpu_retired,
    input  logic [31:0]       dma_issued, dma_retired,
    // Queue occupancy. Advisory only -- a full queue stalls the store, so
    // firmware never has to check before pushing. It is here for a kernel that
    // wants to know how much run-ahead it is actually getting.
    input  logic [15:0]       mxu_level, vpu_level, dma_level,
    input  logic [3:0]        unit_idle,

    // ---- scratchpad S port (shared with the scalar unit; tpu_top muxes) -----
    output logic              s_re,
    output logic              s_we,
    output logic [ADDR_W-1:0] s_addr,
    output logic [XLEN-1:0]   s_wdata,
    input  logic [XLEN-1:0]   s_rdata,
    input  logic              s_rgnt,
    input  logic              s_wgnt
);

    // MMIO offsets inside the 0x8xxx_xxxx region.
    // Command apertures are at +0x000 (MXU), +0x010 (VPU), +0x020 (DMA); the
    // unit index is `wr_addr[5:4]` and the word index `wr_addr[3:2]`, so they
    // need no constants of their own.
    localparam logic [11:0] MMIO_STATUS = 12'h040,   // read-only block, below
                            MMIO_DONE   = 12'h070;

    // =========================================================================
    // AXI4-Lite master from the core.
    // =========================================================================
    wire        m_awvalid, m_wvalid, m_bready, m_arvalid, m_rready;
    wire [31:0] m_awaddr, m_wdata, m_araddr;
    wire [3:0]  m_wstrb;
    logic       m_awready, m_wready, m_bvalid, m_arready, m_rvalid;
    logic [31:0] m_rdata;

    wire core_resetn = rst_n && cpu_run;

    picorv32_axi #(
        .ENABLE_COUNTERS   (1),
        .ENABLE_COUNTERS64 (0),
        .BARREL_SHIFTER    (1),
        .COMPRESSED_ISA    (1),
        .ENABLE_MUL        (0),
        .ENABLE_FAST_MUL   (1),
        .ENABLE_DIV        (0),
        .ENABLE_IRQ        (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (32'h0000_0000 + (1 << (FW_AW + 2)))
    ) u_cpu (
        .clk   (clk),
        .resetn(core_resetn),
        .trap  (cpu_trap),

        .mem_axi_awvalid (m_awvalid), .mem_axi_awready (m_awready),
        .mem_axi_awaddr  (m_awaddr),  .mem_axi_awprot  (),
        .mem_axi_wvalid  (m_wvalid),  .mem_axi_wready  (m_wready),
        .mem_axi_wdata   (m_wdata),   .mem_axi_wstrb   (m_wstrb),
        .mem_axi_bvalid  (m_bvalid),  .mem_axi_bready  (m_bready),
        .mem_axi_arvalid (m_arvalid), .mem_axi_arready (m_arready),
        .mem_axi_araddr  (m_araddr),  .mem_axi_arprot  (),
        .mem_axi_rvalid  (m_rvalid),  .mem_axi_rready  (m_rready),
        .mem_axi_rdata   (m_rdata),

        .pcpi_valid (), .pcpi_insn (), .pcpi_rs1 (), .pcpi_rs2 (),
        .pcpi_wr (1'b0), .pcpi_rd (32'b0), .pcpi_wait (1'b0), .pcpi_ready (1'b0),
        .irq (32'b0), .eoi (),
        .trace_valid (), .trace_data ()
    );

    // =========================================================================
    // Firmware RAM. Simple dual-port: the AXI side reads/writes, the host side
    // writes while the core is in reset. Byte strobes so the C compiler's `sb`
    // and `sh` work on globals.
    // =========================================================================
    logic [31:0] fw_mem [0:(1<<FW_AW)-1];
    logic [31:0] fw_q;

    initial if (FW_INIT != "") $readmemh(FW_INIT, fw_mem);

    // =========================================================================
    // Transaction bookkeeping. One outstanding transaction, so this is two
    // flags and a small state.
    // =========================================================================
    logic        aw_got, w_got;
    logic [31:0] wr_addr, wr_data;
    logic [3:0]  wr_strb;

    logic        rd_busy;      // an AR has been accepted, R not yet returned
    logic [31:0] rd_addr;
    logic [1:0]  rd_wait;      // read-latency countdown

    // Region decode (of whichever address is in play).
    function automatic logic is_fw   (input logic [31:0] a); return (a[31:28] == 4'h0); endfunction
    function automatic logic is_mmio (input logic [31:0] a); return (a[31:28] == 4'h8); endfunction
    function automatic logic is_spad (input logic [31:0] a); return (a[31:28] == 4'h9); endfunction

    // ---- command staging ----------------------------------------------------
    logic [31:0] stage [0:2];    // words 0..2; word 3 arrives with the commit
    logic [1:0]  stage_unit;

    wire        wr_is_cmd  = is_mmio(wr_addr) && (wr_addr[11:8] == 4'h0) &&
                             (wr_addr[7:4] <= 4'h2);
    wire [1:0]  wr_cmd_unit= wr_addr[5:4];
    wire        wr_cmd_commit = wr_is_cmd && (wr_addr[3:2] == 2'd3);

    // A commit only lands when the target queue has room; otherwise the whole
    // write transaction is held open and the core stalls in its store.
    wire        cmd_stalled = wr_cmd_commit && cmd_full;

    assign cmd_we   = aw_got && w_got && wr_cmd_commit && !cmd_full && !m_bvalid;
    assign cmd_unit = wr_cmd_unit;
    assign cmd_data = {wr_data, stage[2], stage[1], stage[0]};

    // ---- scratchpad window --------------------------------------------------
    wire wr_is_spad = is_spad(wr_addr);
    wire rd_is_spad = is_spad(rd_addr);

    assign s_we    = aw_got && w_got && wr_is_spad && !m_bvalid;
    assign s_re    = rd_busy && rd_is_spad && (rd_wait == 2'd1);
    assign s_addr  = s_we ? wr_addr[ADDR_W-1:0] : rd_addr[ADDR_W-1:0];
    assign s_wdata = wr_data;

    // ---- status readback ----------------------------------------------------
    wire [9:0] mmio_sel = rd_addr[11:2];   // hoisted: see scalar_unit.sv's note
    logic [31:0] mmio_rdata;
    always_comb begin
        unique case (mmio_sel)
            (MMIO_STATUS >> 2) + 10'd0: mmio_rdata = mxu_issued;
            (MMIO_STATUS >> 2) + 10'd1: mmio_rdata = mxu_retired;
            (MMIO_STATUS >> 2) + 10'd2: mmio_rdata = {16'b0, mxu_level};
            (MMIO_STATUS >> 2) + 10'd4: mmio_rdata = vpu_issued;
            (MMIO_STATUS >> 2) + 10'd5: mmio_rdata = vpu_retired;
            (MMIO_STATUS >> 2) + 10'd6: mmio_rdata = {16'b0, vpu_level};
            (MMIO_STATUS >> 2) + 10'd8: mmio_rdata = dma_issued;
            (MMIO_STATUS >> 2) + 10'd9: mmio_rdata = dma_retired;
            (MMIO_STATUS >> 2) + 10'd10:mmio_rdata = {16'b0, dma_level};
            (MMIO_STATUS >> 2) + 10'd12:mmio_rdata = {28'b0, unit_idle};
            default:                    mmio_rdata = 32'h0;
        endcase
    end

    // =========================================================================
    // AXI4-Lite slave FSM.
    // =========================================================================
    assign m_awready = !aw_got;
    assign m_wready  = !w_got;
    assign m_arready = !rd_busy && !(aw_got || w_got);

    // `done` latches and holds until the next run, mirroring the scalar unit
    // parking in S_HALT. Both terms that raise it are transient -- cpu_run drops
    // in response to it, which takes the core back into reset and clears
    // `trap` -- so neither can be reported directly.
    logic done_r, cpu_run_q;
    assign cpu_done = done_r;
    assign cpu_busy = cpu_run && !done_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_got   <= 1'b0;
            w_got    <= 1'b0;
            wr_addr  <= '0;
            wr_data  <= '0;
            wr_strb  <= '0;
            rd_busy  <= 1'b0;
            rd_addr  <= '0;
            rd_wait  <= '0;
            m_bvalid <= 1'b0;
            m_rvalid <= 1'b0;
            m_rdata  <= '0;
            done_r   <= 1'b0;
            cpu_run_q<= 1'b0;
            stage[0] <= '0; stage[1] <= '0; stage[2] <= '0;
        end else begin
            cpu_run_q <= cpu_run;
            if (cpu_run && !cpu_run_q) done_r <= 1'b0;   // rising edge: re-arm
            else if (cpu_trap)         done_r <= 1'b1;   // ebreak / illegal insn

            // ---- write address / data channels -------------------------------
            if (m_awvalid && m_awready) begin
                aw_got  <= 1'b1;
                wr_addr <= m_awaddr;
            end
            if (m_wvalid && m_wready) begin
                w_got   <= 1'b1;
                wr_data <= m_wdata;
                wr_strb <= m_wstrb;
            end

            // ---- perform the write -------------------------------------------
            if (aw_got && w_got && !m_bvalid && !cmd_stalled) begin
                if (is_fw(wr_addr)) begin
                    if (wr_strb[0]) fw_mem[wr_addr[FW_AW+1:2]][ 7: 0] <= wr_data[ 7: 0];
                    if (wr_strb[1]) fw_mem[wr_addr[FW_AW+1:2]][15: 8] <= wr_data[15: 8];
                    if (wr_strb[2]) fw_mem[wr_addr[FW_AW+1:2]][23:16] <= wr_data[23:16];
                    if (wr_strb[3]) fw_mem[wr_addr[FW_AW+1:2]][31:24] <= wr_data[31:24];
                end
                if (wr_is_cmd && !wr_cmd_commit)
                    stage[wr_addr[3:2]] <= wr_data;
                if (is_mmio(wr_addr) && (wr_addr[11:0] == MMIO_DONE))
                    done_r <= 1'b1;

                // The scratchpad write has to be granted; hold the transaction
                // open until it is (the arbiter puts this port below the MXU and
                // the VPU, so a denial is normal, not exceptional).
                if (!wr_is_spad || s_wgnt) begin
                    m_bvalid <= 1'b1;
                    aw_got   <= 1'b0;
                    w_got    <= 1'b0;
                end
            end
            if (m_bvalid && m_bready) m_bvalid <= 1'b0;

            // ---- read channel -------------------------------------------------
            if (m_arvalid && m_arready) begin
                rd_busy <= 1'b1;
                rd_addr <= m_araddr;
                rd_wait <= 2'd1;              // one cycle of memory latency
                fw_q    <= fw_mem[m_araddr[FW_AW+1:2]];
            end else if (rd_busy && !m_rvalid) begin
                if (rd_is_spad) begin
                    // s_re is held until the grant arrives; data lands the cycle
                    // after that.
                    if (rd_wait == 2'd1 && s_rgnt) rd_wait <= 2'd2;
                    else if (rd_wait == 2'd2) begin
                        rd_wait  <= 2'd0;
                        m_rdata  <= s_rdata;
                        m_rvalid <= 1'b1;
                    end
                end else if (rd_wait != 2'd0) begin
                    rd_wait  <= 2'd0;
                    m_rdata  <= is_mmio(rd_addr) ? mmio_rdata : fw_q;
                    m_rvalid <= 1'b1;
                end
            end
            if (m_rvalid && m_rready) begin
                m_rvalid <= 1'b0;
                rd_busy  <= 1'b0;
            end

            // ---- host firmware load (core in reset) ---------------------------
            if (fw_we && !cpu_run) fw_mem[fw_waddr] <= fw_wdata;
        end
    end

endmodule
