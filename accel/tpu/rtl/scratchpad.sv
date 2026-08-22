`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// scratchpad.sv — TPU on-chip working memory (banked BRAM + arbitrated port)
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
//   DMA   (read/write)  DMA engine            — DRAM <-> scratchpad
//
// -----------------------------------------------------------------------------
// WHY THIS IS BANKED (and what changed from the flat behavioural model)
// -----------------------------------------------------------------------------
// The previous version was a flat `logic [7:0] mem [0:2**ADDR_W-1]` with six
// independent read ports and four write ports, each taking an unaligned byte
// window at a runtime address. That is an excellent *behavioural* model and a
// terrible netlist: block RAM has two ports, so `ram_style="block"` cannot be
// honoured, and Vivado falls back to registers (65536 x 8 = 524,288 FFs against
// the A7-35T's 41,600) or to LUTRAM replicated per read port (~3 Mbit against
// 400 Kbit). Either way it does not fit, at any array size. See docs/synth.md §5.
//
// Two changes make it synthesizable:
//
//   1. BYTE-LANE BANKING. Storage is split into NBANK single-byte-wide banks,
//      byte address `a` living in bank `a[OFF_W-1:0]` at row `a[ADDR_W-1:OFF_W]`.
//      An NBANK-byte window at an *arbitrary* byte address then takes exactly one
//      byte from every bank: bank b supplies address `a + ((b - off) mod NBANK)`,
//      which is row `row0` for `b >= off` and `row0 + 1` for `b < off`. So each
//      bank needs only a 1-bit row-select adjustment, and the gathered bytes come
//      back rotated by `off` — undone by one NBANK-byte barrel rotate on the way
//      out (and the mirror rotate on the way in for writes). This is the standard
//      unaligned-access banking trick; it replaces NBANK 64Ki-to-1 byte
//      multiplexers with NBANK dual-port BRAMs plus two log2(NBANK)-stage rotate
//      networks.
//
//   2. PORT ARBITRATION. Each bank is one simple dual-port BRAM: one write port,
//      one read port. The six read requesters are muxed onto the shared read
//      port by fixed priority, the four write requesters onto the shared write
//      port. A read and a write still proceed in the same cycle (separate BRAM
//      ports); what is no longer possible is two reads, or two writes, at once.
//
// -----------------------------------------------------------------------------
// ARBITRATION AND GRANTS (this replaced the exclusivity invariant)
// -----------------------------------------------------------------------------
// This file used to state, and assert every cycle, that at most one read and one
// write were ever requested per clock -- which made the priority mux *exact*
// rather than approximate, and meant no requester ever needed a stall handshake.
// That held only because the scalar unit was issue-and-wait: a dispatch parked it
// in S_WAIT until the unit reported done, so MXU, VPU and DMA activity could not
// overlap.
//
// Per-unit command queues (cmd_mxu.sv / cmd_vpu.sv / cmd_dma.sv) delete that
// premise on purpose -- overlapping the DMA with compute is the whole point of
// them (docs/picorv32_migration.md §9.3). So the mux now genuinely arbitrates,
// and the requesters that can lose take a grant back:
//
//   reads   A > W > C > V > s > DMA        writes   C > V > s > DMA
//
// Both chains are ordered so the requester that CANNOT stall is first. The MXU's
// three ports lead both and are mutually exclusive inside its own FSM, so it is
// never denied and gets no grant wire. The VPU freezes its FSM for a cycle when
// `V_rgnt`/`V_wgnt` is low; the scalar/CPU port re-presents; the DMA parks the
// byte in its skid buffer and pauses the SRAM stream. A denied requester that
// ignores its grant loses the access silently, which is the one failure mode this
// arrangement can have and the reason each stall path is tested end to end.
// -----------------------------------------------------------------------------
// TIMING CONTRACT — unchanged from the flat model
// -----------------------------------------------------------------------------
// **Synchronous read**: address/enable presented one cycle, `*_rdata` valid the
// next. The bank output register *is* the port register — the output rotate is
// combinational, so latency stays at exactly one cycle rather than two. Writes
// complete in one cycle under a per-byte write strobe. Read-during-write to the
// same byte returns the OLD value (read-first), which is what the units and the
// inline TB memory models assume.
//
// Note for hardware: read-first holds because the write and the read are issued
// to the two ports of the same BRAM in the same cycle, and 7-series simple
// dual-port BRAM does not forward a write to the opposite read port. The units
// do not read and write the same byte in one cycle in any case.
//
// RESET. `rst_n` no longer clears the read data — a behavioural change from the
// flat model, and a deliberate one. `bank_dout` *is* the BRAM's read register,
// and a block RAM's primary output latch has no reset input; asking for one
// makes Vivado either add a fabric register stage (silently costing a cycle of
// latency) or abandon block RAM entirely. Nothing needs it: every consumer
// samples `*_rdata` only in the cycle after it asserted its own enable, and
// each unit clears its own state on reset. Storage is likewise not reset, same
// as before. `rst_n` now only clears the captured rotate offset.
//
// Widths: each port gathers/scatters its own byte count starting at its byte
// address (docs/scratchpad.md §4). int8 operands occupy one byte/element, int32
// four; the defaults below describe a 128x128 array (A=128 B activation column,
// C=512 B int32 result row, V=64 B / 512-bit VPU access, S=4 B scalar word).
// Address arithmetic wraps mod DEPTH rather than indexing out of range, same as
// the flat model.
//
// Registers vs BRAM (`MEM_STYLE`). The banking is unconditional; MEM_STYLE only
// picks the primitive each bank is built from:
//   MEM_STYLE = "BRAM" (default) -> ram_style="block":     dense FPGA block RAM.
//   MEM_STYLE = "REG"            -> ram_style="registers": flip-flop bank.
// Both honour the identical timing above. "REG" is only sensible for the small
// depths used in unit testbenches; at ADDR_W=16 it is the flip-flop explosion
// this rewrite exists to avoid.
// -----------------------------------------------------------------------------

module scratchpad #(
    parameter     MEM_STYLE = "BRAM",  // "BRAM" -> block RAM; "REG" -> flip-flops
    parameter int ADDR_W    = 16,      // byte-address width; storage = 2**ADDR_W bytes
    parameter int A_BYTES   = 128,     // A_rd  width (bytes): MXU activation column
    parameter int W_BYTES   = 32,      // W_rd  width (bytes): packed int4 weight row
    parameter int C_BYTES   = 512,     // C_rw  width (bytes): int32 result row
    parameter int V_BYTES   = 64,      // V_rw  width (bytes): VPU SIMD access (512b)
    parameter int S_BYTES   = 4,       // S_rw  width (bytes): scalar word (int32)
    parameter int DMA_BYTES = 64,      // DMA   width (bytes): DMA burst
    parameter     INIT_FILE = ""       // optional $readmemh preload (one byte/line)
) (
    input  logic clk,
    input  logic rst_n,   // see note below: neither storage nor read data is reset

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
    output logic                 V_rgnt,      // this cycle's read was accepted
    output logic                 V_wgnt,      // ...and this cycle's write

    // ---- S_rw : scalar unit single word (shared addr for read + write) ------
    input  logic                 s_re,
    input  logic                 s_we,
    input  logic [ADDR_W-1:0]    s_addr,
    input  logic [S_BYTES*8-1:0] s_wdata,
    output logic [S_BYTES*8-1:0] s_rdata,     // valid the cycle after s_re
    output logic                 s_rgnt,
    output logic                 s_wgnt,

    // ---- DMA : DRAM <-> scratchpad engine (read/modify/write) ---------------
    input  logic                 dma_re,
    input  logic [ADDR_W-1:0]    dma_raddr,
    output logic [DMA_BYTES*8-1:0] dma_rdata,   // valid the cycle after dma_re
    input  logic                 dma_we,
    input  logic [ADDR_W-1:0]    dma_waddr,
    input  logic [DMA_BYTES*8-1:0] dma_wdata,
    input  logic [DMA_BYTES-1:0]   dma_wstrb,   // per-byte write enable
    output logic                   dma_rgnt,
    output logic                   dma_wgnt
);

    // -------------------------------------------------------------------------
    // Geometry.
    //
    // NBANK is the widest port rounded up to a power of two: one access must be
    // satisfiable by taking at most one byte from each bank. Rounding to a power
    // of two makes the bank/row split a bit-slice of the address rather than a
    // divide, and makes the barrel rotate a clean log2 network.
    // -------------------------------------------------------------------------
    function automatic int max_of6(input int a, b, c, d, e, f);
        int m;
        begin
            m = a;
            if (b > m) m = b;
            if (c > m) m = c;
            if (d > m) m = d;
            if (e > m) m = e;
            if (f > m) m = f;
            max_of6 = m;
        end
    endfunction

    localparam int DEPTH = (1 << ADDR_W);
    localparam int MAX_B = max_of6(A_BYTES, W_BYTES, C_BYTES, V_BYTES, S_BYTES, DMA_BYTES);
    localparam int OFF_W = $clog2(MAX_B);     // bank-select bits
    localparam int NBANK = (1 << OFF_W);      // banks (>= MAX_B, power of two)
    localparam int BROWS = DEPTH / NBANK;     // rows per bank
    localparam int ROW_W = ADDR_W - OFF_W;    // row-address width
    localparam int DW    = NBANK * 8;         // shared datapath width (bits)

    // The row-select trick assumes a whole access fits in one byte per bank and
    // that the row index is a clean address slice. Checked at time 0 rather than
    // with a generate-scope $error, which iverilog does not accept.
    initial begin
        if (MAX_B > NBANK)
            $fatal(1, "scratchpad: NBANK (%0d) < widest port (%0d)", NBANK, MAX_B);
        if (DEPTH % NBANK != 0)
            $fatal(1, "scratchpad: DEPTH (%0d) not a multiple of NBANK (%0d)", DEPTH, NBANK);
        if (BROWS < 2)
            $fatal(1, "scratchpad: BROWS (%0d) < 2 — ADDR_W too small for NBANK %0d",
                   BROWS, NBANK);
    end

    // -------------------------------------------------------------------------
    // Barrel rotates.
    //
    //   rotr_b(v, k)[i] = v[(i + k) mod NBANK]   — undoes the gather skew (reads)
    //   rotate left by k == rotr by (NBANK - k)  — applies it (writes)
    //
    // Expressed as "duplicate, then shift right by a variable amount, then take
    // the low half", the canonical rotate idiom: synthesis builds one log-depth
    // barrel shifter. Writing it as explicit per-stage part-selects would be the
    // same hardware but is not portable — the slice bounds involve the loop
    // variable, and iverilog requires part-select bounds to be constant.
    // -------------------------------------------------------------------------
    function automatic logic [DW-1:0] rotr_b(input logic [DW-1:0] v,
                                             input logic [OFF_W-1:0] k);
        logic [2*DW-1:0] d;
        begin
            d      = {v, v};
            rotr_b = DW'(d >> (int'(k) * 8));
        end
    endfunction

    function automatic logic [NBANK-1:0] rotr_1(input logic [NBANK-1:0] v,
                                                input logic [OFF_W-1:0] k);
        logic [2*NBANK-1:0] d;
        begin
            d      = {v, v};
            rotr_1 = NBANK'(d >> int'(k));
        end
    endfunction

    function automatic logic [OFF_W-1:0] neg_off(input logic [OFF_W-1:0] k);
        neg_off = OFF_W'((NBANK - int'(k)) & (NBANK - 1));
    endfunction

    // -------------------------------------------------------------------------
    // Read arbitration. Fixed priority A > W > C > V > S > DMA: the MXU
    // activation feed is the bandwidth-critical port, DMA is background work.
    // Under the exclusivity invariant at most one request is ever asserted, so
    // the ordering never actually arbitrates anything — it only keeps the
    // hardware well defined if the invariant is ever broken.
    // -------------------------------------------------------------------------
    logic              rd_req;
    logic [ADDR_W-1:0] rd_addr;

    always_comb begin
        rd_req = 1'b1;
        if      (A_re)   rd_addr = A_raddr;
        else if (W_re)   rd_addr = W_raddr;
        else if (C_re)   rd_addr = C_raddr;
        else if (V_re)   rd_addr = V_raddr;
        else if (s_re)   rd_addr = s_addr;
        else if (dma_re) rd_addr = dma_raddr;
        else begin
            rd_addr = '0;
            rd_req  = 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Write arbitration. Priority **C > V > s > DMA**: the compute units first,
    // background traffic last. This reverses the original DMA > S > V > C order,
    // and reversing it was free -- under the old exclusivity invariant at most one
    // writer was ever asserted, so the order never arbitrated anything. It matters
    // now: with per-unit command queues the MXU, the VPU and the DMA can all be
    // mid-dispatch at once, and the arbiter genuinely has to decide.
    //
    // The order is chosen so the requester that CANNOT stall wins. The MXU's
    // writeback is a free-running drain of `result_buf` with no handshake, so it
    // is first and is never denied. Everything below takes its `*_wgnt` back and
    // holds: the VPU freezes its FSM for a cycle (vpu.sv), the scalar/CPU port
    // re-presents, and the DMA parks the byte in its skid buffer (dma.sv).
    //
    // The scalar port has no sub-word strobe -- a STORES writes the whole word.
    // -------------------------------------------------------------------------
    logic              wr_req;
    logic [ADDR_W-1:0] wr_addr;
    logic [DW-1:0]     wr_data;
    logic [NBANK-1:0]  wr_strb;

    always_comb begin
        wr_req = 1'b1;
        if (C_we) begin
            wr_addr = C_waddr;
            wr_data = DW'(C_wdata);
            wr_strb = NBANK'(C_wstrb);
        end else if (V_we) begin
            wr_addr = V_waddr;
            wr_data = DW'(V_wdata);
            wr_strb = NBANK'(V_wstrb);
        end else if (s_we) begin
            wr_addr = s_addr;
            wr_data = DW'(s_wdata);
            wr_strb = NBANK'({S_BYTES{1'b1}});
        end else if (dma_we) begin
            wr_addr = dma_waddr;
            wr_data = DW'(dma_wdata);
            wr_strb = NBANK'(dma_wstrb);
        end else begin
            wr_addr = '0;
            wr_data = '0;
            wr_strb = '0;
            wr_req  = 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Grants -- one bit per stallable requester, combinational off the same two
    // priority chains: "your access is happening on this clock edge".
    //
    // The MXU's three ports (A/W/C) are top of both chains and are mutually
    // exclusive inside its own FSM, so they are never denied and have no grant
    // wire. Everyone else must look at theirs.
    // -------------------------------------------------------------------------
    assign V_rgnt   = V_re   && !(A_re || W_re || C_re);
    assign s_rgnt   = s_re   && !(A_re || W_re || C_re || V_re);
    assign dma_rgnt = dma_re && !(A_re || W_re || C_re || V_re || s_re);

    assign V_wgnt   = V_we   && !C_we;
    assign s_wgnt   = s_we   && !(C_we || V_we);
    assign dma_wgnt = dma_we && !(C_we || V_we || s_we);

    // -------------------------------------------------------------------------
    // Address decomposition and skew.
    //
    // For an access at byte address {row0, off}, bank b holds the byte at offset
    // (b - off) mod NBANK, which lives at row0 for b >= off and row0+1 for
    // b < off. row0+1 wraps mod BROWS, matching the flat model's mod-DEPTH wrap.
    // -------------------------------------------------------------------------
    logic [OFF_W-1:0] rd_off, wr_off;
    logic [ROW_W-1:0] rd_row0, wr_row0;

    assign rd_off  = rd_addr[OFF_W-1:0];
    assign rd_row0 = rd_addr[ADDR_W-1:OFF_W];
    assign wr_off  = wr_addr[OFF_W-1:0];
    assign wr_row0 = wr_addr[ADDR_W-1:OFF_W];

    // Write data/strobe pre-rotated into bank order: bank b <- byte (b - off).
    logic [DW-1:0]    wr_data_bank;
    logic [NBANK-1:0] wr_strb_bank;

    assign wr_data_bank = rotr_b(wr_data, neg_off(wr_off));
    assign wr_strb_bank = rotr_1(wr_strb, neg_off(wr_off));

    // -------------------------------------------------------------------------
    // The banks. One simple dual-port RAM each: a write port and a read port,
    // independently addressed, read-first. This is the canonical Vivado SDP
    // inference template — write then read in one clocked process.
    //
    // `bank_dout` is an unpacked array so each element has exactly one driver
    // (its own generate instance); it is flattened into `bank_word` afterwards.
    // -------------------------------------------------------------------------
    logic [7:0] bank_dout [0:NBANK-1];

    // -------------------------------------------------------------------------
    // Simulation backdoor (see bd_peek / bd_poke at the bottom of the file).
    // Declared here because the per-bank hooks live inside the generate below.
    // -------------------------------------------------------------------------
// synthesis translate_off
    logic [ROW_W-1:0] bd_row;
    logic [OFF_W-1:0] bd_bank;
    logic [7:0]       bd_wdata;
    logic [7:0]       bd_rd [0:NBANK-1];
    event             bd_wr_ev;
// synthesis translate_on

    genvar b;
    generate
        for (b = 0; b < NBANK; b++) begin : g_bank

            // Row select: +1 for the banks that wrapped into the next row.
            // Comparison is done on the unsigned offset rather than casting the
            // genvar, so no signedness rules are involved.
            logic [ROW_W-1:0] raddr_b, waddr_b;
            assign raddr_b = rd_row0 + ROW_W'(rd_off > OFF_W'(b));
            assign waddr_b = wr_row0 + ROW_W'(wr_off > OFF_W'(b));

            if (MEM_STYLE == "REG" || MEM_STYLE == "registers" ||
                MEM_STYLE == "distributed") begin : g_style
                (* ram_style = "registers" *) logic [7:0] mem [0:BROWS-1];

                always_ff @(posedge clk) begin
                    if (wr_req && wr_strb_bank[b]) mem[waddr_b] <= wr_data_bank[b*8 +: 8];
                    if (rd_req)                    bank_dout[b]  <= mem[raddr_b];
                end

                // Backdoor hooks: one read mux input per bank, plus a poke that
                // only the addressed bank acts on.
// synthesis translate_off
                assign bd_rd[b] = mem[bd_row];
                always @(bd_wr_ev) if (bd_bank == OFF_W'(b)) mem[bd_row] = bd_wdata;
// synthesis translate_on

                // Preload, elaborated away entirely when INIT_FILE is "" — which
                // it is for every testbench and for the board wrapper. Leaving a
                // DEPTH-entry flat image in the netlist unconditionally is one of
                // the ways to reintroduce the huge inferred memory this rewrite
                // exists to remove. Each bank reads the file and keeps its own
                // stride-NBANK slice; that parses the file NBANK times, which is
                // acceptable for a simulation-only path.
                if (INIT_FILE != "") begin : g_preload
                    logic [7:0] img [0:DEPTH-1];
                    initial begin
                        $readmemh(INIT_FILE, img);
                        for (int r = 0; r < BROWS; r++) mem[r] = img[r*NBANK + b];
                    end
                end
            end else begin : g_style
                (* ram_style = "block" *)     logic [7:0] mem [0:BROWS-1];

                always_ff @(posedge clk) begin
                    if (wr_req && wr_strb_bank[b]) mem[waddr_b] <= wr_data_bank[b*8 +: 8];
                    if (rd_req)                    bank_dout[b]  <= mem[raddr_b];
                end

                // Backdoor hooks: one read mux input per bank, plus a poke that
                // only the addressed bank acts on.
// synthesis translate_off
                assign bd_rd[b] = mem[bd_row];
                always @(bd_wr_ev) if (bd_bank == OFF_W'(b)) mem[bd_row] = bd_wdata;
// synthesis translate_on

                if (INIT_FILE != "") begin : g_preload
                    logic [7:0] img [0:DEPTH-1];
                    initial begin
                        $readmemh(INIT_FILE, img);
                        for (int r = 0; r < BROWS; r++) mem[r] = img[r*NBANK + b];
                    end
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read return path.
    //
    // `bank_dout` is the port register — the only register on the read path, so
    // `*_rdata` is valid exactly one cycle after the enable. The rotate that
    // undoes the gather skew is combinational on the way out, using the offset
    // registered alongside the request.
    //
    // All six `*_rdata` are slices of the same rotated word. Only the granted
    // port's data is meaningful, and under the exclusivity invariant that is the
    // only port whose consumer is looking (see the header).
    // -------------------------------------------------------------------------
    logic [OFF_W-1:0] rd_off_q;

    always_ff @(posedge clk) begin
        if (!rst_n)      rd_off_q <= '0;
        else if (rd_req) rd_off_q <= rd_off;
    end

    logic [DW-1:0] bank_word, rd_word;

    always_comb begin
        for (int i = 0; i < NBANK; i++) bank_word[i*8 +: 8] = bank_dout[i];
    end

    assign rd_word = rotr_b(bank_word, rd_off_q);

    assign A_rdata   = rd_word[A_BYTES*8-1   : 0];
    assign W_rdata   = rd_word[W_BYTES*8-1   : 0];
    assign C_rdata   = rd_word[C_BYTES*8-1   : 0];
    assign V_rdata   = rd_word[V_BYTES*8-1   : 0];
    assign s_rdata   = rd_word[S_BYTES*8-1   : 0];
    assign dma_rdata = rd_word[DMA_BYTES*8-1 : 0];

    // -------------------------------------------------------------------------
    // Exclusivity checks (simulation only).
    //
    // The arbitration above is only *exact* while at most one read and one write
    // are requested per cycle. That holds by construction today; these checks are
    // here so a future change that breaks it fails loudly instead of quietly
    // dropping an access. Summed as integers rather than with $countones for
    // iverilog portability.
    // -------------------------------------------------------------------------
// synthesis translate_off
`ifndef SYNTHESIS
    // Set +VERBOSE_ARB at runtime to trace every contended cycle.
    bit verbose_arb = 0;
    initial if ($test$plusargs("VERBOSE_ARB")) verbose_arb = 1;

    logic [2:0] n_rd, n_wr;
    assign n_rd = 3'(A_re) + 3'(W_re) + 3'(C_re) + 3'(V_re) + 3'(s_re) + 3'(dma_re);
    assign n_wr = 3'(C_we) + 3'(V_we) + 3'(s_we) + 3'(dma_we);

    // -------------------------------------------------------------------------
    // Simulation backdoor: single-byte peek/poke at an absolute byte address,
    // bypassing the ports entirely.
    //
    // Testbenches must not reach into the storage directly — the banked layout
    // means a byte lives at `mem[a >> OFF_W]` of bank `a & (NBANK-1)`, and a
    // generate instance cannot be indexed by a runtime value anyway. These tasks
    // hide that: the address is broadcast to every bank, the read mux is a
    // per-bank continuous assign, and the write is an event only the addressed
    // bank acts on. `#0` lets those settle before the value is used.
    //
    // Used by tb/dma_tb.sv to seed and check scratchpad bytes.
    // -------------------------------------------------------------------------
    task automatic bd_peek(input logic [ADDR_W-1:0] a, output logic [7:0] d);
        bd_row = a[ADDR_W-1:OFF_W];
        #0;
        d = bd_rd[a[OFF_W-1:0]];
    endtask

    task automatic bd_poke(input logic [ADDR_W-1:0] a, input logic [7:0] d);
        bd_bank  = a[OFF_W-1:0];
        bd_row   = a[ADDR_W-1:OFF_W];
        bd_wdata = d;
        ->bd_wr_ev;
        #0;
    endtask

    // Plain `always`, not `always_ff`: this block contains only reporting tasks,
    // and a synthesis-intent process holding $error draws (fair) tool warnings.
    always @(posedge clk) if (rst_n) begin
        // Contention is legal now, so counting requesters is no longer an error
        // condition. What would be an error is a denied requester whose own logic
        // does not treat the denial as a stall -- that access is silently dropped.
        // The units' stall paths are what the end-to-end vector checks exercise;
        // this trace is what you turn on when they stop passing.
        if (verbose_arb && (n_rd > 3'd1 || n_wr > 3'd1))
            $display("[%0t] scratchpad: contention rd=%0d wr=%0d (re A=%b W=%b C=%b V=%b S=%b D=%b / we C=%b V=%b S=%b D=%b)",
                     $time, n_rd, n_wr, A_re, W_re, C_re, V_re, s_re, dma_re,
                     C_we, V_we, s_we, dma_we);
    end
`endif
// synthesis translate_on

endmodule
