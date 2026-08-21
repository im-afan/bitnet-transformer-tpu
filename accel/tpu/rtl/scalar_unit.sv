// -----------------------------------------------------------------------------
// scalar_unit.sv — TPU control processor / ISA sequencer
//
// Implements the block described in accel/tpu/docs/scalar_unit.md: a small
// in-order engine ("microcontroller, not out-of-order core") that fetches a
// program from a dedicated instruction BRAM, performs int32 scalar math, and
// dispatches compute ops to the MXU (mxu.md §7), VPU (vpu.md §6), DMA
// (Read/WriteMemory), and inter-TPU LINK (comms.md §6).
//
// DISPATCH IS NOW A MACRO-OP PUSH, not a wire bundle. Instead of driving a
// unit's operand ports directly and pulsing `start`, this unit packs its config
// registers and the instruction's register operands into one or two 128-bit
// commands and pushes them into that unit's queue (cmd_mxu.sv / cmd_vpu.sv /
// cmd_dma.sv). Nothing else about the machine changed: it still blocks until the
// target unit is idle again before retiring, so the issue-and-wait model, every
// existing .tpu program, iss.py and the golden vectors are all unaffected. This
// is phase 1 of docs/picorv32_migration.md §11 — the command interface gets
// built and validated behind the ISA that already works, before a CPU exists to
// produce commands a different way.
//
// The config register file survives *here*, as the thing being packed. It is no
// longer visible to the units, which is what removes the stale-config hazard:
// what a matmul executes with is the geometry that was packed into its own
// command, not whatever the file happened to hold when the array got round to it.
//
// ONE SHIM WORTH KNOWING ABOUT. `requant`/`dyt`/`tquant` and `matmul.rq` take
// their {m0,n} as a *scratchpad address* in this ISA, and the units now want the
// 16-bit value. So a dispatch that needs one first reads it over the S port
// (S_RQRD/S_RQLATCH below) and packs the literal. That costs two clocks per
// requantizing dispatch and exists only because the producer is this ISA;
// firmware computes {m0,n} in a register and pushes it directly.
//
// Encoding (scalar_unit.md §4): fixed 32-bit instruction
//     [ opcode:6 | dst:8 | src0:8 | src1:8 | flags:2 ]
// The three 8-bit operand fields are interpreted as *register indices* into a
// scalar register file (low REG_AW bits used); the register file holds the
// int32 loop counters, computed scratchpad addresses, and requant scalars the
// unit manipulates (§1 "scalar math ... loop counters, addresses"). Compute
// ops therefore read the *contents* of the named registers to obtain the
// scratchpad/DRAM byte addresses they hand to the target unit.
//
// The doc lists a "minimal set" of scalar ops (AddS/MulS/CmpS/Branch/SetCfg/
// Wait). To make that set usable this module also provides the obvious support
// ops noted inline below: SubS, LoadImm, LoadS/StoreS (the register<->scratchpad
// moves needed to bring computed scales/addresses in and out via the 32-bit
// S_rw port from scratchpad.md §4), an unconditional Jmp, and Halt.
//
// Assumed memory contract: instruction memory and the external scratchpad
// scalar port are *synchronous* — address/enable presented one cycle, data
// valid the next. Writes complete in one cycle.
//
// This file is the scalar unit only; the MXU/VPU/DMA/LINK/scratchpad are
// separate blocks and appear here purely as interface ports.
// -----------------------------------------------------------------------------

module scalar_unit #(
    parameter int XLEN     = 32,  // scalar datapath width
    parameter int REG_AW   = 5,   // register file: 2**REG_AW regs (fields are 8b)
    parameter int IMEM_AW  = 10,  // instruction memory depth = 2**IMEM_AW
    parameter int ADDR_W   = 16,  // scratchpad byte-address width
    // External DRAM is a *wider* space than the scratchpad: the async SRAM chip
    // is 2**19 bytes (tpu_top.sv MEM_ADDR_W, Cmod A7 512K x 8). The DMA engine,
    // the SRAM controller and the UART host have always addressed all of it —
    // only this unit's rdmem/wrmem address was truncated to ADDR_W, which
    // confined a *program* to the low 64 KB of a 512 KB chip while the host
    // could reach every byte of it. Registers are XLEN wide, so carrying the
    // full width costs three extra wires on one port. See
    // accel/tpulang/adder_kernel.md §2 for what it unblocks.
    parameter int MEM_ADDR_W = 19, // external DRAM (SRAM chip) byte-address width
    // Config register file: 2**CFG_AW regs. Widened 4 -> 5 for the DMA's
    // transpose geometry (docs/dma.md §5): 0..14 were assigned and index 15 was
    // the last free slot, which three registers do not fit into. The `dst`
    // instruction field is 8 bits, so the encoding was already wide enough and
    // nothing outside this file's cfg array grows.
    parameter int CFG_AW   = 5,
    // Requant {m0,n} field widths, matching requant.sv / mxu.sv / vpu.sv. Needed
    // here because the packer now carries the literal rather than its address.
    parameter int M0_W     = 12,
    parameter int N_W      = 4
) (
    input  logic                 clk,
    input  logic                 rst_n,

    // ---- Host control / program + config load (used while idle) -------------
    input  logic                 host_run,     // pulse: start program at PC 0
    input  logic [IMEM_AW-1:0]   boot_pc,      // entry point sampled on host_run
    output logic                 busy,         // program executing
    output logic                 done,         // HALT reached
    output logic [IMEM_AW-1:0]   pc_dbg,        // PC, for debug/trace
    // High while blocked in S_WAIT on a dispatched unit's `done`. This is the
    // cost of the issue-and-wait model (§2) made measurable: perf_counters.sv
    // integrates it over a run so control-stall time can be separated from the
    // time the units are actually computing. Debug/telemetry only — nothing in
    // the datapath observes it.
    output logic                 wait_active,

    input  logic                 imem_we,       // host: write instruction word
    input  logic [IMEM_AW-1:0]   imem_waddr,
    input  logic [31:0]          imem_wdata,

    input  logic                 cfg_we,        // host: preset a config register
    input  logic [CFG_AW-1:0]    cfg_waddr,
    input  logic [XLEN-1:0]      cfg_wdata,

    // ---- Scratchpad scalar port (S_rw, 32-bit; scratchpad.md §4) ------------
    output logic                 s_re,
    output logic                 s_we,
    output logic [ADDR_W-1:0]    s_addr,
    output logic [XLEN-1:0]      s_wdata,
    input  logic [XLEN-1:0]      s_rdata,       // valid the cycle after s_re
    // The S port is arbitrated (scratchpad.sv, "grants"), so a request is only an
    // access if its grant comes back. Below the MXU and the VPU in both chains,
    // this unit re-presents until it wins.
    input  logic                 s_rgnt,
    input  logic                 s_wgnt,

    // ---- Macro-op command push (docs/picorv32_migration.md §3) -------------
    // One push port with a unit selector, rather than three. tpu_top muxes
    // `cmd_full` from the selected unit's queue and fans `cmd_we` out to it.
    output logic                 cmd_we,
    output logic [1:0]           cmd_unit,   // U_MXU / U_VPU / U_DMA / U_LINK
    output logic [127:0]         cmd_data,
    input  logic                 cmd_full,   // selected queue full: hold the push
    // Per-unit "queue empty and nothing in flight". This replaces the `done`
    // pulses the old dispatch waited on: with a queue in the way, "the unit has
    // caught up" is the only completion this producer can observe, and it is
    // exactly what issue-and-wait needs.
    input  logic [3:0]           unit_idle
);

    // -------------------------------------------------------------------------
    // Opcode map (6-bit). Compact so opcode space stays < 32 (scalar_unit.md §4,
    // "leaving room for future fused ops").
    // -------------------------------------------------------------------------
    localparam logic [5:0]
        OP_MATMUL  = 6'h00,  // MXU : act=r[src0], weight=r[src1], out=r[dst];
                             //       flags[0]=accumulate, flags[1]=requant,
                             //       requant {M0,N} word addr from cfg[CFG_SCALAR]
        // 0x02, 0x05, 0x0A-0x0F, 0x1B and 0x20 are **retired**: vecmul, gelu,
        // vecemul, square, exp, redmax, redsum, sadd, sdiv and softmax were
        // removed with the VPU datapath that implemented them (see vpu.sv's
        // header). They are left as holes rather than reused so an old binary
        // decodes to nothing rather than to a different op.
        OP_VECDOT  = 6'h01,  // VPU : out=r[dst] = Σ r[src0]·r[src1].
                             //       The shipped model never issues this, but
                             //       it is VOP_VECMATMUL's inner primitive, so
                             //       the datapath is there regardless and the
                             //       opcode costs one decode arm.
        OP_VECADD  = 6'h03,  // VPU : r[dst] = r[src0] + r[src1]
        OP_RELU    = 6'h04,  // VPU : r[dst] = relu(r[src0])
        OP_WRMEM   = 6'h06,  // DMA : scratch r[src0] -> DRAM r[src1]
        OP_RDMEM   = 6'h07,  // DMA : DRAM r[src0] -> scratch r[dst]
        OP_WRNEIGH = 6'h08,  // LINK: DRAM r[src0] -> neighbor DRAM r[src1], dir=flags
        OP_REQUANT = 6'h09,  // VPU : r[dst] = requant(r[src0], {n,m0}=r[src1])
        OP_ADDS    = 6'h10,  // r[dst] = r[src0] + r[src1]
        OP_SUBS    = 6'h11,  // r[dst] = r[src0] - r[src1]   (support op)
        OP_MULS    = 6'h12,  // r[dst] = r[src0] * r[src1]   (low XLEN bits)
        OP_CMPS    = 6'h13,  // flags <- cmp(r[src0], r[src1])
        OP_LIS     = 6'h14,  // r[dst] = zero-extend(imm16)  (support op)
        OP_SETCFG  = 6'h15,  // cfg[dst] = zero-extend(imm16)
        OP_LOADS   = 6'h16,  // r[dst] = scratch[r[src0]]    (support op)
        OP_STORES  = 6'h17,  // scratch[r[src0]] = r[src1]   (support op)
        OP_BRANCH  = 6'h18,  // if cond(flags) pc = imm16
        OP_JMP     = 6'h19,  // pc = imm16                   (support op)
        OP_WAIT    = 6'h1A,  // block on unit flags{0:MXU,1:VPU,2:DMA,3:LINK}
        OP_SETCFGR = 6'h1C,  // cfg[dst] = r[src0]   (register -> config)
        // Past the original 0x00-0x1F block. The field is 6 bits, so 0x22-0x3E
        // are free (0x20 is a retired hole, not reusable); HALT keeps 0x1F.
        OP_DYT     = 6'h21,  // VPU : r[dst] = dyt(r[src0], {n,m0}=r[src1]).
                             //       Same operands and same fixed point as
                             //       REQUANT, symmetric +-127 clip: DyT's
                             //       hardtanh with the output scale pinned to
                             //       1/127 (vpu.sv header, adder_kernel.md §4).
        OP_TQUANT  = 6'h22,  // VPU : r[dst] = tquant(r[src0], {n,m0}=r[src1]).
                             //       Again the same operands and fixed point,
                             //       clipped to +-1 and written **2 bits** wide
                             //       in the MXU's weight encoding, 4 elements
                             //       per byte. This is the op that turns an int8
                             //       activation into a ternary weight operand,
                             //       which is what put attention's Q@K^T and
                             //       P@V on the array instead of on VECMM.
        OP_VECMM   = 6'h1E,  // VPU : macro op — S[t][s] = sum_d src0[t][d]*
                             //       src1[s][d] over cfg vrows x vcols.
                             //       Attention's Q@K^T / P@V: both operands
                             //       are int8 activations, which the MXU
                             //       (ternary weights) cannot do at all.
        OP_MATMUL_T= 6'h1D,  // MXU : as MATMUL, but operands are tiles of a
                             //       larger matrix — strides from cfg arow/
                             //       crow/wcol. Same flags (acc/rq) as MATMUL.
                             //       Separate opcode rather than a flag bit
                             //       because MATMUL's 2-bit flags field is full,
                             //       and because keying tiling off "the stride
                             //       registers are nonzero" would let one
                             //       program's leftover config change the next
                             //       program's plain matmul (config survives
                             //       across runs).
        OP_HALT    = 6'h1F;  // stop, raise `done`

    // OP_SETCFGR is what makes a config value *runtime* rather than compile-time.
    // OP_SETCFG takes an immediate, so every geometry a macro op reads from the
    // config file would otherwise be frozen at assembly time — fine for prefill,
    // fatal for incremental decode, where the attended key count grows by one
    // per step (docs/macro_ops.md §9.2). The config file already has a write
    // port; this only muxes its data source.

    // VPU op selector values driven on vpu_op (must match vpu.sv VOP_* codes).
    // Still 5 bits even though only six codes survive the VPU trim: the values
    // themselves were left where they were so vpu.sv, iss.py and the assembler
    // needed no re-synchronization, and VOP_DYT is 16. Compacting the encoding
    // would save two flops on this bus and cost a four-way rename.
    localparam logic [4:0]
        VOP_DOT = 5'd0,        // vpu.sv-internal: vecmatmul's inner primitive
        VOP_ADD = 5'd1, VOP_RELU = 5'd3, VOP_REQUANT = 5'd10,
        VOP_VECMATMUL = 5'd13, VOP_DYT = 5'd16, VOP_TQUANT = 5'd17;

    // Named config registers (host-, SETCFG- or SETCFGR-written; scalar_unit.md
    // §6). 0..3 are the original set; 4..13 carry the macro ops' geometry
    // (docs/macro_ops.md §3). 14/15 remain unassigned.
    localparam logic [CFG_AW-1:0]
        CFG_TLEN   = 'd0, // MXU token count T -> mxu_t_len
        CFG_VLEN   = 'd1, // VPU vector length -> vpu_vlen
        CFG_LEN    = 'd2, // DMA / WriteNeighbor byte length
        CFG_SCALAR = 'd3, // MXU requant {M0,N} word address -> mxu_scalar_addr
        CFG_KTILES = 'd4, // MXU contraction tiles  = K / ROWS
        CFG_NTILES = 'd5, // MXU output tiles       = N / COLS
        CFG_AROW   = 'd6, // MXU activation row stride, bytes (= K)
        CFG_CROW   = 'd7, // MXU result row stride, bytes (= N*4, or N requantized)
        CFG_WCOL   = 'd8, // MXU weight column stride, bytes (= K*2/8)
        CFG_VSCALAR= 'd9, // **retired, slot reserved.** Was the VPU macro-op
                          //   {m0,n} word address, read only by OP_SOFTMAX,
                          //   which needed it because its three register
                          //   operands were dst/src/tmp with no slot left over.
                          //   Nothing reads it now that softmax is gone, so
                          //   synthesis trims the register — but the *index*
                          //   stays put, because renumbering 10..19 would
                          //   silently repoint every `setcfg` in every existing
                          //   program and in the assembler's cfg table.
        CFG_VROWS  = 'd10,// VPU macro-op row count (vecmatmul: query rows)
        CFG_VCOLS  = 'd11,// vecmatmul key rows — independent of VROWS so decode
                          //   (1 x t+1) and prefill (T x T) are the same op
        CFG_VROW0  = 'd12,// vecmatmul src0 row stride, bytes
        CFG_VROW1  = 'd13,// vecmatmul src1 row stride, bytes
        CFG_VCROW  = 'd14,// vecmatmul dst row stride, bytes (int32).
                          //   Separate from CFG_CROW: that one is the
                          //   MXU's, and a layer has both live at once.
        // DMA transpose geometry (docs/dma.md §5). Read only by RDMEM/WRMEM
        // carrying the `.t` flag; a plain rdmem/wrmem ignores them outright,
        // for the same reason OP_MATMUL ignores the MXU strides.
        CFG_TCOLS  = 'd15,// source row length in elements (the inner counter)
        CFG_TSROW  = 'd16,// source row stride, bytes (0 => dense = TCOLS)
        CFG_TDROW  = 'd17;// destination row stride, bytes (0 => 1)

    // Dispatch-target selector (also the WAIT flags encoding).
    localparam logic [1:0] U_MXU = 2'd0, U_VPU = 2'd1, U_DMA = 2'd2, U_LINK = 2'd3;

    // Branch condition codes carried in the flags field of BRANCH.
    localparam logic [1:0] C_EQ = 2'b00, C_NE = 2'b01, C_LT = 2'b10, C_GE = 2'b11;

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE, S_FETCH, S_DECODE, S_EXEC, S_LOAD,
        S_RQRD,    // fetch the {m0,n} word this dispatch needs as a literal
        S_RQLATCH, // ...it lands here (S-port read latency)
        S_PUSH,    // push this dispatch's 1 or 2 commands into the unit's queue
        S_WAIT, S_HALT
    } state_t;
    state_t state, state_n;

    logic [IMEM_AW-1:0] pc;
    logic [31:0]        ir;          // latched instruction word
    logic [1:0]         sel_unit;    // unit selected by the in-flight dispatch/WAIT

    // Comparison flags (set by CMPS, consumed by BRANCH).
    logic flag_eq, flag_lt;

    // ---- Instruction field decode (on latched ir; continuous) ---------------
    wire [5:0]  opc    = ir[31:26];
    wire [7:0]  f_dst  = ir[25:18];
    wire [7:0]  f_src0 = ir[17:10];
    wire [7:0]  f_src1 = ir[9:2];
    wire [1:0]  flags  = ir[1:0];
    wire [15:0] imm16  = ir[17:2];

    wire [REG_AW-1:0] a_dst  = f_dst [REG_AW-1:0];
    wire [REG_AW-1:0] a_src0 = f_src0[REG_AW-1:0];
    wire [REG_AW-1:0] a_src1 = f_src1[REG_AW-1:0];

    // -------------------------------------------------------------------------
    // Register file (r0 hardwired to 0). Combinational read, synchronous write.
    // -------------------------------------------------------------------------
    logic [XLEN-1:0] rf [0:(1<<REG_AW)-1];
    wire [XLEN-1:0] r_dst  = (a_dst  == '0) ? '0 : rf[a_dst];
    wire [XLEN-1:0] r_src0 = (a_src0 == '0) ? '0 : rf[a_src0];
    wire [XLEN-1:0] r_src1 = (a_src1 == '0) ? '0 : rf[a_src1];

    // Register write control (single write port).
    logic            rf_we;
    logic [REG_AW-1:0] rf_waddr;
    logic [XLEN-1:0]   rf_wdata;

    always_ff @(posedge clk) begin
        if (rf_we && rf_waddr != '0)
            rf[rf_waddr] <= rf_wdata;
    end

    // -------------------------------------------------------------------------
    // Config register file. Host presets while idle; SETCFG writes at runtime.
    //
    // "Idle" here — and for the instruction memory below — means `!busy`, not
    // `state == S_IDLE`. S_IDLE is only ever reached out of reset: once a program
    // HALTs the FSM parks in S_HALT and stays there until the next `host_run`.
    // Gating the host write ports on S_IDLE therefore made every load after the
    // first one a silent no-op, so a second, different program would run the
    // *first* program's instructions until the board was reset. `!busy` is
    // (S_IDLE | S_HALT), and is the same condition uart_interface arbitrates on
    // (`core_busy`), so the host's view of "safe to load" now matches the
    // scalar unit's.
    // -------------------------------------------------------------------------
    // Reset to zero. Previously unreset, which was invisible only because every
    // program set each config register it read: an unset one was X in
    // simulation and whatever the fabric powered up with on hardware. That
    // stopped being harmless once zero acquired a meaning — the MXU reads
    // `a_row`/`c_row`/`w_col` on every dispatch and takes zero as "use the
    // single-tile default" (mxu.sv), so an undefined register poisons the
    // address generators of programs that never mention them.
    logic [XLEN-1:0] cfg [0:(1<<CFG_AW)-1];
    logic            cfg_prog_we;   // SETCFG  write strobe (immediate source)
    logic            cfg_reg_we;    // SETCFGR write strobe (register source)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (int ci = 0; ci < (1<<CFG_AW); ci++) cfg[ci] <= '0;
        else if (cfg_we && !busy)
            cfg[cfg_waddr] <= cfg_wdata;
        else if (cfg_prog_we)
            cfg[a_dst[CFG_AW-1:0]] <= {{(XLEN-16){1'b0}}, imm16};
        else if (cfg_reg_we)
            // SETCFGR: full XLEN from the register file, no extension. The
            // consumers below slice what they need, so a value too wide for a
            // field truncates exactly as the immediate form would.
            cfg[a_dst[CFG_AW-1:0]] <= r_src0;
    end

    // -------------------------------------------------------------------------
    // Instruction memory (BRAM), synchronous read + host write port.
    // -------------------------------------------------------------------------
    logic [31:0] imem [0:(1<<IMEM_AW)-1];
    logic [31:0] imem_rdata;
    always_ff @(posedge clk) begin
        if (imem_we && !busy)          // !busy == S_IDLE | S_HALT; see cfg above
            imem[imem_waddr] <= imem_wdata;
        imem_rdata <= imem[pc];      // valid the cycle after pc is presented
    end

    // -------------------------------------------------------------------------
    // Command packing. Field layouts are the ones cmd_mxu.sv / cmd_vpu.sv /
    // cmd_dma.sv decode; they are duplicated here rather than shared through a
    // package, the same way VOP_* already is (see the note above it).
    //
    //   word0 = cmd_data[31:0]   word1 = [63:32]   word2 = [95:64]   word3 = [127:96]
    // -------------------------------------------------------------------------
    localparam logic [7:0] MXU_GEOM = 8'h01, MXU_MM   = 8'h02,
                           VPU_OPC  = 8'h01, VPU_GEOM = 8'h02,
                           DMA_MOVE = 8'h01;

    // The {m0,n} literal, fetched over the S port for the ops whose operand is
    // an address (see the header's shim note).
    logic [M0_W+N_W-1:0] rq_word_r;

    // Which scratchpad address that fetch uses: cfg scalar for the MXU, the
    // third register operand for the VPU.
    wire [ADDR_W-1:0] rq_fetch_addr =
        ((opc == OP_MATMUL) || (opc == OP_MATMUL_T)) ? cfg[CFG_SCALAR][ADDR_W-1:0]
                                                     : r_src1[ADDR_W-1:0];

    // Does this dispatch need the literal before it can be packed?
    function automatic logic needs_rq(input logic [5:0] o, input logic rq_flag);
        needs_rq = (((o == OP_MATMUL) || (o == OP_MATMUL_T)) && rq_flag) ||
                   (o == OP_REQUANT) || (o == OP_DYT) || (o == OP_TQUANT);
    endfunction

    // Two commands are pushed when the geometry has to travel with the op:
    // every MXU matmul, and the VPU's one macro op.
    function automatic logic [1:0] push_count(input logic [5:0] o);
        unique case (o)
            OP_MATMUL, OP_MATMUL_T, OP_VECMM: push_count = 2'd2;
            OP_WRNEIGH:                       push_count = 2'd0;  // stub, no queue
            default:                          push_count = 2'd1;
        endcase
    endfunction

    logic [1:0] push_idx;      // which command of the burst is being pushed

    // The config file's entire contribution to a command, named once. (Also a
    // tool requirement: a bit-select of an array element inside an always_* block
    // is not something every simulator will take.)
    wire [15:0] c_arow   = cfg[CFG_AROW]  [15:0];
    wire [15:0] c_crow   = cfg[CFG_CROW]  [15:0];
    wire [15:0] c_wcol   = cfg[CFG_WCOL]  [15:0];
    wire [7:0]  c_ktiles = cfg[CFG_KTILES][7:0];
    wire [7:0]  c_ntiles = cfg[CFG_NTILES][7:0];
    wire [5:0]  c_tlen   = cfg[CFG_TLEN]  [5:0];
    wire [9:0]  c_vlen   = cfg[CFG_VLEN]  [9:0];
    wire [15:0] c_vrows  = cfg[CFG_VROWS] [15:0];
    wire [15:0] c_vcols  = cfg[CFG_VCOLS] [15:0];
    wire [15:0] c_vrow0  = cfg[CFG_VROW0] [15:0];
    wire [15:0] c_vrow1  = cfg[CFG_VROW1] [15:0];
    wire [15:0] c_vcrow  = cfg[CFG_VCROW] [15:0];
    wire [15:0] c_len    = cfg[CFG_LEN]   [15:0];
    wire [15:0] c_tcols  = cfg[CFG_TCOLS] [15:0];
    wire [15:0] c_tsrow  = cfg[CFG_TSROW] [15:0];
    wire [15:0] c_tdrow  = cfg[CFG_TDROW] [15:0];

    // Instruction flag bits and register operands, pre-sliced. Same reason as
    // the cfg wires above.
    wire f_acc = flags[0];   // matmul .acc  / dma .t
    wire f_rq  = flags[1];   // matmul .rq
    wire [15:0]           rd16    = r_dst [15:0];
    wire [15:0]           rs0_16  = r_src0[15:0];
    wire [15:0]           rs1_16  = r_src1[15:0];
    wire [MEM_ADDR_W-1:0] rs0_mem = r_src0[MEM_ADDR_W-1:0];
    wire [MEM_ADDR_W-1:0] rs1_mem = r_src1[MEM_ADDR_W-1:0];

    logic [31:0] w0, w1, w2, w3;
    always_comb begin
        w0 = '0; w1 = '0; w2 = '0; w3 = '0;
        unique case (1'b1)
            // ---- MXU ---------------------------------------------------------
            (opc == OP_MATMUL) || (opc == OP_MATMUL_T): begin
                if (push_idx == 2'd0) begin
                    w0 = {c_arow, 8'h00, MXU_GEOM};
                    w1 = {c_wcol, c_crow};
                    w2 = {10'b0, c_tlen,
                          c_ntiles, c_ktiles};
                end else begin
                    w0 = {rd16, 5'b0, (opc == OP_MATMUL_T), f_rq, f_acc,
                          MXU_MM};
                    w1 = {rs1_16, rs0_16};
                    w2 = {16'b0, rq_word_r};
                end
            end
            // ---- VPU ---------------------------------------------------------
            is_vpu_op(opc): begin
                if ((opc == OP_VECMM) && (push_idx == 2'd0)) begin
                    w0 = {c_vrows, 8'h00, VPU_GEOM};
                    w1 = {c_vrow0, c_vcols};
                    w2 = {c_vcrow, c_vrow1};
                end else begin
                    w0 = {rd16, 3'b0, vpu_op_sel, VPU_OPC};
                    w1 = {rs1_16, rs0_16};
                    w2 = {rq_word_r, 6'b0, c_vlen};
                end
            end
            // ---- DMA ---------------------------------------------------------
            (opc == OP_WRMEM) || (opc == OP_RDMEM): begin
                w0 = {((opc == OP_WRMEM) ? rs0_16 : rd16),
                      6'b0, f_acc, (opc == OP_WRMEM), DMA_MOVE};
                w1 = {{(32-MEM_ADDR_W){1'b0}},
                      ((opc == OP_WRMEM) ? rs1_mem
                                         : rs0_mem)};
                w2 = {c_tcols, c_len};
                w3 = {c_tdrow, c_tsrow};
            end
            default: ;
        endcase
    end

    assign cmd_data = {w3, w2, w1, w0};

    // VPU op selector (must match vpu.sv VOP_* codes).
    logic [4:0] vpu_op_sel;
    always_comb begin
        unique case (opc)
            OP_VECMM:   vpu_op_sel = VOP_VECMATMUL;
            OP_VECDOT:  vpu_op_sel = VOP_DOT;
            OP_VECADD:  vpu_op_sel = VOP_ADD;
            OP_RELU:    vpu_op_sel = VOP_RELU;
            OP_REQUANT: vpu_op_sel = VOP_REQUANT;
            OP_DYT:     vpu_op_sel = VOP_DYT;
            OP_TQUANT:  vpu_op_sel = VOP_TQUANT;
            default:    vpu_op_sel = VOP_DOT;
        endcase
    end

    // Every op that dispatches to the VPU (declared first: is_dispatch and
    // dispatch_unit below both call it).
    function automatic logic is_vpu_op(input logic [5:0] o);
        return (o == OP_VECDOT) || (o == OP_VECADD) || (o == OP_RELU) ||
               (o == OP_REQUANT) || (o == OP_VECMM)  || (o == OP_DYT) ||
               (o == OP_TQUANT);
    endfunction

    // Is this opcode a compute/comms dispatch (assert start, then wait on done)?
    function automatic logic is_dispatch(input logic [5:0] o);
        return is_vpu_op(o) || (o == OP_MATMUL) || (o == OP_MATMUL_T) ||
               (o == OP_WRMEM) || (o == OP_RDMEM) || (o == OP_WRNEIGH);
    endfunction

    // Which unit a dispatch opcode targets.
    function automatic logic [1:0] dispatch_unit(input logic [5:0] o);
        unique case (1'b1)
            is_vpu_op(o):                               dispatch_unit = U_VPU;
            (o == OP_MATMUL), (o == OP_MATMUL_T):        dispatch_unit = U_MXU;
            (o == OP_WRMEM), (o == OP_RDMEM):           dispatch_unit = U_DMA;
            (o == OP_WRNEIGH):                          dispatch_unit = U_LINK;
            default:                                    dispatch_unit = U_MXU;
        endcase
    endfunction

    // "The selected unit has caught up." With a queue between the producer and
    // the unit, this is the only completion visible from here — and it is exactly
    // what issue-and-wait needs, so the FSM below is unchanged in shape. The LINK
    // has no queue and no block behind it, so its idle bit is tied high in
    // tpu_top and `wrneigh` stays a completing no-op.
    logic sel_idle;
    assign sel_idle = unit_idle[sel_unit];

    // Command push. One per clock while the queue has room; `push_idx` selects
    // which word of a two-command burst the packer is presenting.
    assign cmd_unit = dispatch_unit(opc);
    assign cmd_we   = (state == S_PUSH) && !cmd_full;

    wire [1:0] n_push   = push_count(opc);
    wire       push_end = (state == S_PUSH) && !cmd_full &&
                          (push_idx + 2'd1 >= n_push);

    // Scratchpad scalar port: LOADS reads, STORES writes, and S_RQRD borrows it
    // for the {m0,n} fetch. The port is arbitrated now, so a read only happened
    // if `s_rgnt` came back — every user below re-presents until it does.
    assign s_re    = ((state == S_EXEC) && (opc == OP_LOADS)) || (state == S_RQRD);
    assign s_we    = (state == S_EXEC) && (opc == OP_STORES);
    assign s_addr  = (state == S_RQRD) ? rq_fetch_addr : r_src0[ADDR_W-1:0];
    assign s_wdata = r_src1;

    // Branch taken?
    logic br_taken;
    always_comb begin
        unique case (flags)
            C_EQ:    br_taken =  flag_eq;
            C_NE:    br_taken = !flag_eq;
            C_LT:    br_taken =  flag_lt;
            C_GE:    br_taken = !flag_lt;
            default: br_taken = 1'b0;
        endcase
    end

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:   if (host_run) state_n = S_FETCH;
            S_FETCH:  state_n = S_DECODE;              // imem read launched
            S_DECODE: state_n = S_EXEC;                // ir latched
            S_EXEC: begin
                if (opc == OP_HALT)          state_n = S_HALT;
                else if (opc == OP_LOADS)  begin if (s_rgnt) state_n = S_LOAD;  end
                else if (opc == OP_STORES) begin if (s_wgnt) state_n = S_FETCH; end
                else if (is_dispatch(opc)) begin
                    if      (push_count(opc) == 2'd0) state_n = S_WAIT;  // LINK stub
                    else if (needs_rq(opc, f_rq))    state_n = S_RQRD;
                    else                              state_n = S_PUSH;
                end
                else if (opc == OP_WAIT)     state_n = S_WAIT;
                else                         state_n = S_FETCH;     // single-cycle op
            end
            S_LOAD:   state_n = S_FETCH;               // s_rdata valid this cycle
            S_RQRD:   if (s_rgnt) state_n = S_RQLATCH;
            S_RQLATCH:state_n = S_PUSH;
            S_PUSH:   if (push_end) state_n = S_WAIT;
            // The queue makes "done" unobservable, so wait on the unit going idle
            // again. For a burst that is still one wait, after the last push.
            S_WAIT:   if (sel_idle) state_n = S_FETCH;
            S_HALT:   if (host_run) state_n = S_FETCH;
            default:  state_n = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Datapath / sequencing (registered)
    // -------------------------------------------------------------------------
    // Single register-file write port, shared by the S_EXEC scalar-op commit
    // and the S_LOAD (LOADS) writeback from s_rdata.
    always_comb begin
        rf_we       = 1'b0;
        rf_waddr    = a_dst;
        rf_wdata    = '0;
        cfg_prog_we = 1'b0;
        cfg_reg_we  = 1'b0;

        if (state == S_EXEC) begin
            unique case (opc)
                OP_ADDS:   begin rf_we = 1'b1; rf_wdata = r_src0 + r_src1; end
                OP_SUBS:   begin rf_we = 1'b1; rf_wdata = r_src0 - r_src1; end
                OP_MULS:   begin rf_we = 1'b1; rf_wdata = r_src0 * r_src1; end
                // ZERO-extended, not sign-extended. `li` almost always loads an
                // address, and registers now feed a 19-bit DRAM address as well
                // as the 16-bit scratchpad one. Under sign extension
                // `li r, 0x8000` puts 0xFFFF8000 in the register: masking that
                // to ADDR_W wraps back to 0x8000, but masking it to MEM_ADDR_W
                // gives 0x78000 — so every DRAM address in [0x8000, 0xFFFF]
                // silently aliased to the top of the chip. Zero extension makes
                // the register hold what was written. A negative constant is
                // `li t, N` then `subs t, r0, t`; nothing in the tree needed one.
                OP_LIS:    begin rf_we = 1'b1; rf_wdata = {{(XLEN-16){1'b0}}, imm16}; end
                OP_SETCFG:  cfg_prog_we = 1'b1;
                OP_SETCFGR: cfg_reg_we  = 1'b1;
                default:   ;   // LOADS commits in S_LOAD; others don't write rf
            endcase
        end else if (state == S_LOAD) begin
            rf_we    = 1'b1;               // LOADS writeback (s_rdata now valid)
            rf_wdata = s_rdata;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            pc       <= '0;
            ir       <= '0;
            sel_unit  <= U_MXU;
            flag_eq   <= 1'b0;
            flag_lt   <= 1'b0;
            push_idx  <= 2'd0;
            rq_word_r <= '0;
        end else begin
            state <= state_n;

            unique case (state)
                S_IDLE:   if (host_run) pc <= boot_pc;
                S_DECODE: ir <= imem_rdata;   // latch fetched word
                S_EXEC: begin
                    sel_unit <= dispatch_unit(opc);
                    unique case (opc)
                        OP_CMPS: begin
                            flag_eq <= (r_src0 == r_src1);
                            flag_lt <= ($signed(r_src0) < $signed(r_src1));
                            pc      <= pc + 1'b1;
                        end
                        OP_BRANCH: pc <= br_taken ? imm16[IMEM_AW-1:0] : pc + 1'b1;
                        OP_JMP:    pc <= imm16[IMEM_AW-1:0];
                        default: begin
                            // LOADS / dispatch / WAIT advance PC when they retire
                            // in S_LOAD / S_WAIT; everything else retires now.
                            // STORES is the exception among the "now" group: the
                            // S port is arbitrated, so it only retires once its
                            // write was actually granted.
                            if (opc == OP_STORES) begin
                                if (s_wgnt) pc <= pc + 1'b1;
                            end else if (opc != OP_LOADS && !is_dispatch(opc) &&
                                         opc != OP_WAIT && opc != OP_HALT)
                                pc <= pc + 1'b1;
                        end
                    endcase
                end
                S_LOAD: pc <= pc + 1'b1;                       // LOADS retires
                // The {m0,n} literal for a requantizing dispatch.
                S_RQLATCH: rq_word_r <= s_rdata[M0_W+N_W-1:0];
                S_PUSH: if (!cmd_full) push_idx <= push_end ? 2'd0 : (push_idx + 2'd1);
                S_WAIT: if (sel_idle) pc <= pc + 1'b1;         // dispatch/WAIT retires
                S_HALT: if (host_run) pc <= boot_pc;
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Status outputs
    // -------------------------------------------------------------------------
    assign busy   = (state != S_IDLE) && (state != S_HALT);
    assign done   = (state == S_HALT);
    assign pc_dbg = pc;
    assign wait_active = (state == S_WAIT);

endmodule
