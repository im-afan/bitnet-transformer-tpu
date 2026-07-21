// -----------------------------------------------------------------------------
// scalar_unit.sv — TPU control processor / ISA sequencer
//
// Implements the block described in accel/tpu/docs/scalar_unit.md: a small
// in-order engine ("microcontroller, not out-of-order core") that fetches a
// program from a dedicated instruction BRAM, performs int32 scalar math, and
// dispatches compute ops to the MXU (mxu.md §7), VPU (vpu.md §6), DMA
// (Read/WriteMemory), and inter-TPU LINK (comms.md §6). Synchronization uses
// the doc's v1 model: issue-and-wait — assert a unit's `start`, then block on
// its `done` before issuing the next dependent op (§2).
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
    parameter int ADDR_W   = 16,  // scratchpad / DRAM byte-address width
    parameter int CFG_AW   = 4    // config register file: 2**CFG_AW regs
) (
    input  logic                 clk,
    input  logic                 rst_n,

    // ---- Host control / program + config load (used while idle) -------------
    input  logic                 host_run,     // pulse: start program at PC 0
    input  logic [IMEM_AW-1:0]   boot_pc,      // entry point sampled on host_run
    output logic                 busy,         // program executing
    output logic                 done,         // HALT reached
    output logic [IMEM_AW-1:0]   pc_dbg,        // PC, for debug/trace

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

    // ---- MXU dispatch (mxu.md §7) -------------------------------------------
    output logic                 mxu_start,
    output logic [ADDR_W-1:0]    mxu_act_addr,
    output logic [ADDR_W-1:0]    mxu_weight_addr,
    output logic [ADDR_W-1:0]    mxu_out_addr,
    output logic [ADDR_W-1:0]    mxu_scalar_addr, // requant {M0,N} word (config reg)
    output logic [5:0]           mxu_t_len,      // token columns (config reg)
    output logic                 mxu_accumulate, // tiling accumulate (instr flag[0])
    output logic                 mxu_requant,    // narrow store int32->int8 (instr flag[1])
    input  logic                 mxu_busy,
    input  logic                 mxu_done,

    // ---- VPU dispatch (vpu.md §6) -------------------------------------------
    output logic                 vpu_start,
    output logic [3:0]           vpu_op,
    output logic [ADDR_W-1:0]    vpu_src0,
    output logic [ADDR_W-1:0]    vpu_src1,
    output logic [ADDR_W-1:0]    vpu_scalar,
    output logic [ADDR_W-1:0]    vpu_dst,
    output logic [9:0]           vpu_vlen,       // vector length (config reg)
    input  logic                 vpu_busy,
    input  logic                 vpu_done,

    // ---- DMA (Read/WriteMemory: DRAM <-> scratchpad) ------------------------
    output logic                 dma_start,
    output logic                 dma_write,      // 1: scratch->DRAM, 0: DRAM->scratch
    output logic [ADDR_W-1:0]    dma_scratch_addr,
    output logic [ADDR_W-1:0]    dma_dram_addr,
    output logic [15:0]          dma_len,        // bytes (config reg)
    input  logic                 dma_busy,
    input  logic                 dma_done,

    // ---- Inter-TPU LINK (WriteNeighbor; comms.md §6) ------------------------
    output logic                 nb_start,
    output logic [1:0]           nb_dir,         // target direction (instr flags)
    output logic [ADDR_W-1:0]    nb_src_addr,    // local DRAM
    output logic [ADDR_W-1:0]    nb_dst_addr,    // neighbor DRAM
    output logic [15:0]          nb_len,         // bytes (config reg)
    input  logic                 nb_done
);

    // -------------------------------------------------------------------------
    // Opcode map (6-bit). Compact so opcode space stays < 32 (scalar_unit.md §4,
    // "leaving room for future fused ops").
    // -------------------------------------------------------------------------
    localparam logic [5:0]
        OP_MATMUL  = 6'h00,  // MXU : act=r[src0], weight=r[src1], out=r[dst];
                             //       flags[0]=accumulate, flags[1]=requant,
                             //       requant {M0,N} word addr from cfg[CFG_SCALAR]
        OP_VECDOT  = 6'h01,  // VPU : out=r[dst] = Σ r[src0]·r[src1]
        OP_VECMUL  = 6'h02,  // VPU : r[dst] = r[src0] × scalar r[src1]
        OP_VECADD  = 6'h03,  // VPU : r[dst] = r[src0] + r[src1]
        OP_RELU    = 6'h04,  // VPU : r[dst] = relu(r[src0])
        OP_GELU    = 6'h05,  // VPU : r[dst] = gelu(r[src0])
        OP_WRMEM   = 6'h06,  // DMA : scratch r[src0] -> DRAM r[src1]
        OP_RDMEM   = 6'h07,  // DMA : DRAM r[src0] -> scratch r[dst]
        OP_WRNEIGH = 6'h08,  // LINK: DRAM r[src0] -> neighbor DRAM r[src1], dir=flags
        OP_REQUANT = 6'h09,  // VPU : r[dst] = requant(r[src0], {n,m0}=r[src1])
        OP_VECEMUL = 6'h0A,  // VPU : r[dst] = r[src0] (.) r[src1]   (elementwise mul)
        OP_SQUARE  = 6'h0B,  // VPU : r[dst] = r[src0]^2             (LayerNorm var)
        OP_EXP     = 6'h0C,  // VPU : r[dst] = exp_lut(r[src0])      (softmax)
        OP_REDMAX  = 6'h0D,  // VPU : r[dst] = max_i r[src0][i]      (softmax/stats)
        OP_REDSUM  = 6'h0E,  // VPU : r[dst] = Σ_i r[src0][i]       (softmax/LN mean)
        OP_SADD    = 6'h0F,  // VPU : r[dst] = r[src0] + scalar r[src1]  (x-max / x-mean)
        OP_SDIV    = 6'h1B,  // VPU : r[dst] = r[src0] / scalar r[src1]  (Q15; softmax/LN)
        OP_ADDS    = 6'h10,  // r[dst] = r[src0] + r[src1]
        OP_SUBS    = 6'h11,  // r[dst] = r[src0] - r[src1]   (support op)
        OP_MULS    = 6'h12,  // r[dst] = r[src0] * r[src1]   (low XLEN bits)
        OP_CMPS    = 6'h13,  // flags <- cmp(r[src0], r[src1])
        OP_LIS     = 6'h14,  // r[dst] = sign-extend(imm16)  (support op)
        OP_SETCFG  = 6'h15,  // cfg[dst] = zero-extend(imm16)
        OP_LOADS   = 6'h16,  // r[dst] = scratch[r[src0]]    (support op)
        OP_STORES  = 6'h17,  // scratch[r[src0]] = r[src1]   (support op)
        OP_BRANCH  = 6'h18,  // if cond(flags) pc = imm16
        OP_JMP     = 6'h19,  // pc = imm16                   (support op)
        OP_WAIT    = 6'h1A,  // block on unit flags{0:MXU,1:VPU,2:DMA,3:LINK}
        OP_HALT    = 6'h1F;  // stop, raise `done`

    // VPU op selector values driven on vpu_op (must match vpu.sv VOP_* codes).
    localparam logic [3:0]
        VOP_DOT = 4'd0, VOP_ADD = 4'd1, VOP_MUL = 4'd2, VOP_RELU = 4'd3, VOP_GELU = 4'd4,
        VOP_SQUARE = 4'd5, VOP_EXP = 4'd6, VOP_REDUCEMAX = 4'd7, VOP_REDUCESUM = 4'd8,
        VOP_ELEMENT_MUL = 4'd9, VOP_REQUANT = 4'd10, VOP_SCALAR_ADD = 4'd11,
        VOP_SCALAR_DIV = 4'd12;

    // Named config registers (host- or SETCFG-written; scalar_unit.md §6).
    localparam logic [CFG_AW-1:0]
        CFG_TLEN   = 'd0, // MXU token count T -> mxu_t_len
        CFG_VLEN   = 'd1, // VPU vector length -> vpu_vlen
        CFG_LEN    = 'd2, // DMA / WriteNeighbor byte length
        CFG_SCALAR = 'd3; // MXU requant {M0,N} word address -> mxu_scalar_addr

    // Dispatch-target selector (also the WAIT flags encoding).
    localparam logic [1:0] U_MXU = 2'd0, U_VPU = 2'd1, U_DMA = 2'd2, U_LINK = 2'd3;

    // Branch condition codes carried in the flags field of BRANCH.
    localparam logic [1:0] C_EQ = 2'b00, C_NE = 2'b01, C_LT = 2'b10, C_GE = 2'b11;

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE, S_FETCH, S_DECODE, S_EXEC, S_LOAD, S_WAIT, S_HALT
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
    // -------------------------------------------------------------------------
    logic [XLEN-1:0] cfg [0:(1<<CFG_AW)-1];
    logic            cfg_prog_we;   // SETCFG write strobe
    always_ff @(posedge clk) begin
        if (cfg_we && state == S_IDLE)
            cfg[cfg_waddr] <= cfg_wdata;
        else if (cfg_prog_we)
            cfg[a_dst[CFG_AW-1:0]] <= {{(XLEN-16){1'b0}}, imm16};
    end

    assign mxu_t_len      = cfg[CFG_TLEN][5:0];
    assign mxu_scalar_addr = cfg[CFG_SCALAR][ADDR_W-1:0];
    assign vpu_vlen       = cfg[CFG_VLEN][9:0];
    assign dma_len        = cfg[CFG_LEN][15:0];
    assign nb_len         = cfg[CFG_LEN][15:0];

    // -------------------------------------------------------------------------
    // Instruction memory (BRAM), synchronous read + host write port.
    // -------------------------------------------------------------------------
    logic [31:0] imem [0:(1<<IMEM_AW)-1];
    logic [31:0] imem_rdata;
    always_ff @(posedge clk) begin
        if (imem_we && state == S_IDLE)
            imem[imem_waddr] <= imem_wdata;
        imem_rdata <= imem[pc];      // valid the cycle after pc is presented
    end

    // -------------------------------------------------------------------------
    // Dispatch operand routing (combinational; qualified by *_start pulses).
    // -------------------------------------------------------------------------
    assign mxu_act_addr    = r_src0[ADDR_W-1:0];
    assign mxu_weight_addr = r_src1[ADDR_W-1:0];
    assign mxu_out_addr    = r_dst [ADDR_W-1:0];
    assign mxu_accumulate  = flags[0];
    assign mxu_requant     = flags[1];

    assign vpu_src0   = r_src0[ADDR_W-1:0];
    assign vpu_src1   = r_src1[ADDR_W-1:0];
    assign vpu_scalar = r_src1[ADDR_W-1:0];   // VECMUL/SADD scalar, SDIV divisor,
                                              // REQUANT {n,m0} addr
    assign vpu_dst    = r_dst [ADDR_W-1:0];
    always_comb begin
        unique case (opc)
            OP_VECDOT:  vpu_op = VOP_DOT;
            OP_VECADD:  vpu_op = VOP_ADD;
            OP_VECMUL:  vpu_op = VOP_MUL;
            OP_RELU:    vpu_op = VOP_RELU;
            OP_GELU:    vpu_op = VOP_GELU;
            OP_SQUARE:  vpu_op = VOP_SQUARE;
            OP_EXP:     vpu_op = VOP_EXP;
            OP_REDMAX:  vpu_op = VOP_REDUCEMAX;
            OP_REDSUM:  vpu_op = VOP_REDUCESUM;
            OP_VECEMUL: vpu_op = VOP_ELEMENT_MUL;
            OP_REQUANT: vpu_op = VOP_REQUANT;   // in=r[src0], {n,m0}=r[src1], out=r[dst]
            OP_SADD:    vpu_op = VOP_SCALAR_ADD; // in=r[src0], scalar=r[src1], out=r[dst]
            OP_SDIV:    vpu_op = VOP_SCALAR_DIV; // in=r[src0], divisor=r[src1], out=r[dst]
            default:    vpu_op = VOP_DOT;
        endcase
    end

    // WRMEM: scratch r[src0] -> DRAM r[src1];  RDMEM: DRAM r[src0] -> scratch r[dst]
    assign dma_write        = (opc == OP_WRMEM);
    assign dma_scratch_addr = (opc == OP_WRMEM) ? r_src0[ADDR_W-1:0] : r_dst [ADDR_W-1:0];
    assign dma_dram_addr    = (opc == OP_WRMEM) ? r_src1[ADDR_W-1:0] : r_src0[ADDR_W-1:0];

    assign nb_dir      = flags;
    assign nb_src_addr = r_src0[ADDR_W-1:0];
    assign nb_dst_addr = r_src1[ADDR_W-1:0];

    // Every op that dispatches to the VPU (declared first: is_dispatch and
    // dispatch_unit below both call it).
    function automatic logic is_vpu_op(input logic [5:0] o);
        return (o == OP_VECDOT)  || (o == OP_VECMUL)  || (o == OP_VECADD)   ||
               (o == OP_RELU)    || (o == OP_GELU)    || (o == OP_SQUARE)   ||
               (o == OP_EXP)     || (o == OP_REDMAX)  || (o == OP_REDSUM)   ||
               (o == OP_VECEMUL) || (o == OP_REQUANT) || (o == OP_SADD)     ||
               (o == OP_SDIV);
    endfunction

    // Is this opcode a compute/comms dispatch (assert start, then wait on done)?
    function automatic logic is_dispatch(input logic [5:0] o);
        return is_vpu_op(o) || (o == OP_MATMUL) ||
               (o == OP_WRMEM) || (o == OP_RDMEM) || (o == OP_WRNEIGH);
    endfunction

    // Which unit a dispatch opcode targets.
    function automatic logic [1:0] dispatch_unit(input logic [5:0] o);
        unique case (1'b1)
            is_vpu_op(o):                               dispatch_unit = U_VPU;
            (o == OP_MATMUL):                           dispatch_unit = U_MXU;
            (o == OP_WRMEM), (o == OP_RDMEM):           dispatch_unit = U_DMA;
            (o == OP_WRNEIGH):                          dispatch_unit = U_LINK;
            default:                                    dispatch_unit = U_MXU;
        endcase
    endfunction

    // done for the currently selected unit.
    logic sel_done;
    always_comb begin
        unique case (sel_unit)
            U_MXU:  sel_done = mxu_done;
            U_VPU:  sel_done = vpu_done;
            U_DMA:  sel_done = dma_done;
            U_LINK: sel_done = nb_done;
            default:sel_done = 1'b1;
        endcase
    end

    // start pulses: asserted only in S_EXEC, only for the matching dispatch op.
    assign mxu_start = (state == S_EXEC) && (opc == OP_MATMUL);
    assign vpu_start = (state == S_EXEC) && is_vpu_op(opc);
    assign dma_start = (state == S_EXEC) && ((opc == OP_WRMEM) || (opc == OP_RDMEM));
    assign nb_start  = (state == S_EXEC) && (opc == OP_WRNEIGH);

    // Scratchpad scalar port: LOADS reads, STORES writes, both in S_EXEC.
    assign s_re    = (state == S_EXEC) && (opc == OP_LOADS);
    assign s_we    = (state == S_EXEC) && (opc == OP_STORES);
    assign s_addr  = r_src0[ADDR_W-1:0];
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
                else if (opc == OP_LOADS)    state_n = S_LOAD;      // await s_rdata
                else if (is_dispatch(opc))   state_n = S_WAIT;      // await done
                else if (opc == OP_WAIT)     state_n = S_WAIT;
                else                         state_n = S_FETCH;     // single-cycle op
            end
            S_LOAD:   state_n = S_FETCH;               // s_rdata valid this cycle
            S_WAIT:   if (sel_done) state_n = S_FETCH;
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

        if (state == S_EXEC) begin
            unique case (opc)
                OP_ADDS:   begin rf_we = 1'b1; rf_wdata = r_src0 + r_src1; end
                OP_SUBS:   begin rf_we = 1'b1; rf_wdata = r_src0 - r_src1; end
                OP_MULS:   begin rf_we = 1'b1; rf_wdata = r_src0 * r_src1; end
                OP_LIS:    begin rf_we = 1'b1; rf_wdata = {{(XLEN-16){imm16[15]}}, imm16}; end
                OP_SETCFG: cfg_prog_we = 1'b1;
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
            sel_unit <= U_MXU;
            flag_eq  <= 1'b0;
            flag_lt  <= 1'b0;
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
                            if (opc != OP_LOADS && !is_dispatch(opc) &&
                                opc != OP_WAIT && opc != OP_HALT)
                                pc <= pc + 1'b1;
                        end
                    endcase
                end
                S_LOAD: pc <= pc + 1'b1;                       // LOADS retires
                S_WAIT: if (sel_done) pc <= pc + 1'b1;         // dispatch/WAIT retires
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

endmodule
