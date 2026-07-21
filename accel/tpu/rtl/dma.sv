`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// dma.sv — DMA engine (DRAM <-> scratchpad)
//
// Moves `dma_len` bytes between the external SRAM ("DRAM", via sram.sv) and the
// on-chip scratchpad (scratchpad.sv), driven by the scalar unit's Read/WriteMemory
// dispatch. See accel/tpu/docs/dma.md for the full plan and per-state notes.
//
//   dma_write = 0  -> FILL : DRAM  -> scratchpad  (ReadMemory)
//   dma_write = 1  -> SPILL: scratchpad -> DRAM   (WriteMemory)
//
// v1 is deliberately byte-serial: the SRAM is one byte / multi-cycle per access
// and is the bottleneck, so we move one byte at a time and use only lane 0 of the
// wide scratchpad port (masked by a one-hot write strobe). Line-buffered/burst
// mode is a later optimization (docs/dma.md §6).
//
// Handshake (issue-and-wait, same as MXU/VPU): `dma_start` is a one-cycle pulse
// and the scalar unit holds the operands stable until `dma_done`, so addresses
// are generated combinationally from the counter and DONE->IDLE cannot re-trigger.
//
// Outputs are decoded combinationally from the (registered) state, giving clean
// single-cycle `sram_start` pulses with no output-lag. Only `state`, `counter`,
// and the one-byte holding register `data` are sequential. `sram_busy` is unused:
// completion is signalled by `sram_done`.
// -----------------------------------------------------------------------------

module dma #(
    parameter integer MEM_ADDR_W        = 19,   // external SRAM (DRAM) byte address
    parameter integer SCRATCHPAD_ADDR_W = 16,   // scratchpad byte address
    parameter integer MEM_DATA_W        = 8,    // SRAM data width (bits) = 1 byte
    parameter integer SCRATCHPAD_BYTES  = 64    // scratchpad DMA port width in bytes
) (
    input  logic clk,
    input  logic rst_n,

    // ---- SRAM controller interface (one byte per transaction) ---------------
    output logic                    sram_start,
    output logic                    sram_we,
    output logic [MEM_ADDR_W-1:0]   sram_addr,
    output logic [MEM_DATA_W-1:0]   sram_din,
    input  logic [MEM_DATA_W-1:0]   sram_dout,
    input  logic                    sram_busy,   // unused: we wait on sram_done
    input  logic                    sram_done,

    // ---- Scratchpad DMA port (sync read valid next cycle; byte-strobed write) -
    output logic                          scratchpad_re,
    output logic [SCRATCHPAD_ADDR_W-1:0]  scratchpad_raddr,
    input  logic [SCRATCHPAD_BYTES*8-1:0] scratchpad_rdata,   // valid the cycle after re
    output logic                          scratchpad_we,
    output logic [SCRATCHPAD_ADDR_W-1:0]  scratchpad_waddr,
    output logic [SCRATCHPAD_BYTES*8-1:0] scratchpad_wdata,
    output logic [SCRATCHPAD_BYTES-1:0]   scratchpad_wstrb,   // per-byte write enable

    // ---- Scalar unit dispatch (issue-and-wait) ------------------------------
    input  logic                          dma_start,
    input  logic                          dma_write,          // 1: scratch->DRAM, 0: DRAM->scratch
    input  logic [SCRATCHPAD_ADDR_W-1:0]  dma_scratch_addr,
    input  logic [MEM_ADDR_W-1:0]         dma_dram_addr,
    input  logic [15:0]                   dma_len,            // bytes
    output logic                          dma_busy,
    output logic                          dma_done
);

    // State encoding (all distinct). Fill path: SRAM read -> scratchpad write.
    // Spill path: scratchpad read -> SRAM write. STEP advances the byte counter.
    localparam [3:0]
        IDLE                 = 4'd0,
        READ_SRAM            = 4'd1,   // fill : issue SRAM read
        READ_SRAM_WAIT       = 4'd2,   // fill : wait sram_done, capture byte
        WRITE_SCRATCHPAD     = 4'd3,   // fill : 1-cycle scratchpad byte write
        READ_SCRATCHPAD      = 4'd4,   // spill: issue scratchpad read
        READ_SCRATCHPAD_WAIT = 4'd5,   // spill: capture byte (rdata valid this cycle)
        WRITE_SRAM           = 4'd6,   // spill: issue SRAM write
        WRITE_SRAM_WAIT      = 4'd7,   // spill: wait sram_done
        STEP                 = 4'd8,   // i++ ; loop or finish
        DONE                 = 4'd9;   // one-cycle dma_done pulse

    reg [3:0]                   state, state_n;
    reg [SCRATCHPAD_ADDR_W-1:0] counter;   // index of the byte being moved
    reg [MEM_DATA_W-1:0]        data;      // one-byte holding register

    // -------------------------------------------------------------------------
    // Address generation. Inputs are held stable by the scalar unit for the whole
    // transfer, so these can track the counter combinationally. Scratchpad math is
    // mod 2**SCRATCHPAD_ADDR_W (matches scratchpad.sv's wrap); the SRAM offset is
    // zero-extended to the wider DRAM address.
    // -------------------------------------------------------------------------
    assign scratchpad_raddr = dma_scratch_addr + counter;
    assign scratchpad_waddr = dma_scratch_addr + counter;
    assign sram_addr        = dma_dram_addr + MEM_ADDR_W'(counter);

    // -------------------------------------------------------------------------
    // Next-state logic (default: hold).
    // -------------------------------------------------------------------------
    always_comb begin
        state_n = state;
        case (state)
            IDLE: if (dma_start) begin
                if      (dma_len == 16'd0) state_n = DONE;            // empty transfer
                else if (dma_write)        state_n = READ_SCRATCHPAD; // spill
                else                       state_n = READ_SRAM;       // fill
            end
            READ_SRAM:            state_n = READ_SRAM_WAIT;
            READ_SRAM_WAIT:       if (sram_done) state_n = WRITE_SCRATCHPAD;
            WRITE_SCRATCHPAD:     state_n = STEP;
            READ_SCRATCHPAD:      state_n = READ_SCRATCHPAD_WAIT;
            READ_SCRATCHPAD_WAIT: state_n = WRITE_SRAM;
            WRITE_SRAM:           state_n = WRITE_SRAM_WAIT;
            WRITE_SRAM_WAIT:      if (sram_done) state_n = STEP;
            STEP: begin
                if (counter + 1 >= dma_len) state_n = DONE;
                else state_n = dma_write ? READ_SCRATCHPAD : READ_SRAM;
            end
            DONE:                 state_n = IDLE;
            default:              state_n = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Output decode (combinational Moore outputs from the registered state).
    // -------------------------------------------------------------------------
    always_comb begin
        sram_start       = 1'b0;
        sram_we          = 1'b0;
        sram_din         = data;
        scratchpad_re    = 1'b0;
        scratchpad_we    = 1'b0;
        scratchpad_wdata = { {(SCRATCHPAD_BYTES-1){8'b0}}, data };  // byte in lane 0
        scratchpad_wstrb = '0;
        dma_busy         = (state != IDLE) && (state != DONE);
        dma_done         = (state == DONE);

        case (state)
            READ_SRAM:        begin sram_start = 1'b1; sram_we = 1'b0; end
            WRITE_SCRATCHPAD: begin scratchpad_we = 1'b1; scratchpad_wstrb = SCRATCHPAD_BYTES'(1); end
            READ_SCRATCHPAD:  begin scratchpad_re = 1'b1; end
            WRITE_SRAM:       begin sram_start = 1'b1; sram_we = 1'b1; end
            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential state: FSM register, byte counter, and the holding register.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            counter <= '0;
            data    <= '0;
        end else begin
            state <= state_n;
            case (state)
                IDLE:                 counter <= '0;
                READ_SRAM_WAIT:       if (sram_done) data <= sram_dout;
                READ_SCRATCHPAD_WAIT: data <= scratchpad_rdata[MEM_DATA_W-1:0];  // lane 0
                STEP:                 counter <= counter + 1'b1;
                default: ;
            endcase
        end
    end
endmodule
