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
// ---- Ranges, not bytes ------------------------------------------------------
//
// v1 was byte-serial: one `sram_start` per byte, four DMA states around each,
// ~8 clocks per byte. The SRAM controller now takes a *range* (start address,
// byte count, address stride) and streams it at one byte per clock for reads and
// one per two clocks for writes, so this engine's job changed shape: instead of
// sequencing an access it now feeds or drains a stream.
//
//   FILL   `sram_dout_valid` pulses once per byte; each pulse is written
//          straight into the scratchpad, which takes a byte-strobed write every
//          clock. Nothing buffers, and nothing has to be aligned.
//   SPILL  the scratchpad read is issued one clock ahead of the byte being
//          offered, which is exactly the one-clock turnaround the controller's
//          `din_ready` is designed around, so the write stream never starves.
//
// The one structural constraint is that a range is contiguous in *address
// space*, and a transposed transfer is not: it is a set of rows. So the engine
// issues **one range per source row**, which in linear mode is one range for the
// whole transfer (`tcols` defaults to `dma_len`). The row loop is the same 2-D
// walk transpose mode already needed; only the range dispatch is new.
//
// Handshake (issue-and-wait, same as MXU/VPU): `dma_start` is a one-cycle pulse
// and the scalar unit holds the operands stable until `dma_done`, so addresses
// are generated combinationally from the counters and DONE->IDLE cannot
// re-trigger.
//
// ---- Transpose mode (`dma_transpose`, docs/dma.md §5) -----------------------
//
// The byte *sequence* is unchanged; only the two address generators differ. The
// engine walks the transfer as a 2-D loop and applies opposite orders to the two
// sides:
//
//     source      is read  ROW-MAJOR    : src + r*tsrow + c
//     destination is written TRANSPOSED : dst + c*tdrow + r
//
// for c in 0..tcols-1 (the inner counter) and r advancing on each wrap, stopping
// after `dma_len` bytes. "Source" is direction-relative and needs no separate
// encoding: on a fill it is DRAM and on a spill it is the scratchpad, which is
// exactly how the instruction reads. One convention covers both, because
// transposing on the way out of a [R][C] tensor and transposing on the way in to
// a [C][R] one are the same permutation.
//
// That is also what sets the range parameters on the DRAM side:
//
//     fill.t   DRAM is the source, read row-major -> stride 1, one range per row
//     spill.t  DRAM is the destination, written transposed -> the inner counter
//              steps it by `tdrow`, so the range carries stride = tdrow
//
// Either way the DRAM side is one range per row and the scratchpad side takes
// the scattered access, which costs nothing: it is single-cycle random access.
//
// `tsrow` exists so the source can be a column slice of a wider tensor (V is the
// last d columns of a fused [T][3d] QKV block, and nothing else needs to move to
// get at it). `tdrow` is a free stride rather than a hard-wired R so the result
// can land inside a wider destination.
//
// Zero means "not set" for all three, and each falls back to the value that makes
// the mode degenerate rather than collapse to address 0 (the mxu.sv stride
// convention, for the reason in docs/macro_ops.md §4.0):
//
//     tcols == 0  ->  dma_len   (one row: a strided scatter/gather, not a fault)
//     tsrow == 0  ->  tcols     (dense source rows)
//     tdrow == 0  ->  1         (all three unset == a plain linear copy)
//
// Addresses accumulate in SCRATCHPAD_ADDR_W bits on both sides, so a transposed
// transfer addresses one 64 KB DRAM window — the same window the scalar unit's
// 16-bit DRAM address can name in the first place.
// -----------------------------------------------------------------------------

module dma #(
    parameter integer MEM_ADDR_W        = 19,   // external SRAM (DRAM) byte address
    parameter integer SCRATCHPAD_ADDR_W = 16,   // scratchpad byte address
    parameter integer MEM_DATA_W        = 8,    // SRAM data width (bits) = 1 byte
    parameter integer SCRATCHPAD_BYTES  = 64    // scratchpad DMA port width in bytes
) (
    input  logic clk,
    input  logic rst_n,

    // ---- SRAM controller interface (one range per row) ----------------------
    output logic                    sram_start,
    output logic                    sram_we,
    output logic [MEM_ADDR_W-1:0]   sram_addr,    // first byte of the range
    output logic [15:0]             sram_len,     // bytes in the range
    output logic [15:0]             sram_stride,  // address step per byte
    output logic [MEM_DATA_W-1:0]   sram_din,
    output logic                    sram_din_valid,
    input  logic                    sram_din_ready,
    input  logic [MEM_DATA_W-1:0]   sram_dout,
    input  logic                    sram_dout_valid,
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
    // Transpose mode (instruction flag + cfg geometry; see header and §5 of the doc)
    input  logic                          dma_transpose,      // 1: transposed addressing
    input  logic [15:0]                   dma_tcols,          // source row length, elements
    input  logic [15:0]                   dma_tsrow,          // source row stride, bytes
    input  logic [15:0]                   dma_tdrow,          // destination row stride, bytes
    output logic                          dma_busy,
    output logic                          dma_done
);

    // State encoding. ISSUE hands one row's range to the controller; the stream
    // states then move that row's bytes, one per handshake.
    localparam [2:0]
        IDLE       = 3'd0,
        ISSUE      = 3'd1,   // one clock: `sram_start` for this row's range
        FILL       = 3'd2,   // fill : drain sram_dout into the scratchpad
        SPILL_RD   = 3'd3,   // spill: address the scratchpad for the next byte
        SPILL_SEND = 3'd4,   // spill: offer that byte to the write stream
        SPILL_END  = 3'd5,   // spill: wait for the row's last write to commit
        DONE       = 3'd6;   // one-cycle dma_done pulse

    reg [2:0]                   state, state_n;
    reg [SCRATCHPAD_ADDR_W-1:0] counter;   // index of the byte being moved

    // Transpose-mode 2-D position. `col`/`row` are the source coordinates;
    // `src_row_off` (= row*tsrow) and `dst_col_off` (= col*tdrow) are carried as
    // running sums so neither address generator needs a multiplier.
    reg [15:0]                  col, row;
    reg [SCRATCHPAD_ADDR_W-1:0] src_row_off, dst_col_off;

    // Spill holding register. The scratchpad's contract is that read data is
    // valid *the cycle after* `re` — it says nothing about the cycle after that,
    // so a byte the write stream did not take on its first offer is kept here
    // rather than re-read or assumed still present. With a two-clock write beat
    // the offer is always taken immediately and this never fills; it does at
    // CLOCKS_PER_ACCESS > 0, where the beat is longer than the fetch.
    reg [MEM_DATA_W-1:0]        cap;
    reg                         have_cap;

    // Effective geometry: zero means "not set" (see the header).
    wire [15:0] tcols_eff = (dma_tcols == 16'd0) ? dma_len   : dma_tcols;
    wire [15:0] tsrow_eff = (dma_tsrow == 16'd0) ? tcols_eff : dma_tsrow;
    wire [15:0] tdrow_eff = (dma_tdrow == 16'd0) ? 16'd1     : dma_tdrow;
    wire        row_last  = (col + 16'd1 >= tcols_eff);   // this byte ends a source row
    wire        all_last  = (counter + 1 >= dma_len);     // ...and this one ends the transfer

    // -------------------------------------------------------------------------
    // Address generation. Inputs are held stable by the scalar unit for the whole
    // transfer, so these can track the counters combinationally. Scratchpad math
    // is mod 2**SCRATCHPAD_ADDR_W (matches scratchpad.sv's wrap); the SRAM offset
    // is zero-extended to the wider DRAM address.
    //
    // Linear mode drives both sides from `counter`. Transpose mode gives the
    // row-major offset to whichever side is the source (`dma_write` == spill ==
    // scratchpad is the source) and the transposed offset to the other.
    //
    // The DRAM side is only ever sampled in ISSUE, i.e. at `col == 0`, where it
    // is this row's base address; the controller generates the rest of the row
    // itself from `sram_stride`.
    // -------------------------------------------------------------------------
    wire [SCRATCHPAD_ADDR_W-1:0] lin_off = src_row_off + SCRATCHPAD_ADDR_W'(col);
    wire [SCRATCHPAD_ADDR_W-1:0] tr_off  = dst_col_off + SCRATCHPAD_ADDR_W'(row);

    wire [SCRATCHPAD_ADDR_W-1:0] scratch_off =
        !dma_transpose ? counter : (dma_write ? lin_off : tr_off);
    wire [SCRATCHPAD_ADDR_W-1:0] dram_off =
        !dma_transpose ? counter : (dma_write ? tr_off  : lin_off);

    assign scratchpad_raddr = dma_scratch_addr + scratch_off;
    assign scratchpad_waddr = dma_scratch_addr + scratch_off;
    assign sram_addr        = dma_dram_addr + MEM_ADDR_W'(dram_off);

    // Range geometry. A row is `tcols` bytes, clipped by whatever is left of
    // `dma_len` — a transfer that is not a whole number of rows stops part-way
    // through the last one, which is the documented behaviour and is why the
    // clip is here and not in the caller.
    wire [15:0] remaining = dma_len - 16'(counter);
    assign sram_len    = (remaining < tcols_eff) ? remaining : tcols_eff;
    assign sram_stride = (dma_transpose && dma_write) ? tdrow_eff : 16'd1;

    // A byte moves on either stream's handshake.
    wire adv = (state == FILL       && sram_dout_valid) ||
               (state == SPILL_SEND && sram_din_ready);

    // -------------------------------------------------------------------------
    // Next-state logic (default: hold).
    // -------------------------------------------------------------------------
    always_comb begin
        state_n = state;
        case (state)
            IDLE: if (dma_start) begin
                if (dma_len == 16'd0) state_n = DONE;    // empty transfer
                else                  state_n = ISSUE;
            end
            ISSUE:  state_n = dma_write ? SPILL_RD : FILL;
            // `sram_done` lands on the same clock as the range's last
            // `sram_dout_valid`, so `counter` is still the last byte's index.
            FILL:   if (sram_done) state_n = all_last ? DONE : ISSUE;
            SPILL_RD:   state_n = SPILL_SEND;
            SPILL_SEND: if (sram_din_ready)
                            state_n = (row_last || all_last) ? SPILL_END : SPILL_RD;
            // The last byte of the row has been handed over but not committed.
            SPILL_END:  if (sram_done)
                            state_n = (counter >= dma_len) ? DONE : ISSUE;
            DONE:   state_n = IDLE;
            default: state_n = IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // Output decode (combinational Moore outputs from the registered state).
    // -------------------------------------------------------------------------
    always_comb begin
        sram_start       = (state == ISSUE);
        sram_we          = dma_write;
        sram_din         = have_cap ? cap : scratchpad_rdata[MEM_DATA_W-1:0];
        sram_din_valid   = (state == SPILL_SEND);
        scratchpad_re    = (state == SPILL_RD);
        scratchpad_we    = (state == FILL) && sram_dout_valid;
        scratchpad_wdata = { {(SCRATCHPAD_BYTES-1){8'b0}}, sram_dout };  // byte in lane 0
        scratchpad_wstrb = scratchpad_we ? SCRATCHPAD_BYTES'(1) : '0;
        dma_busy         = (state != IDLE) && (state != DONE);
        dma_done         = (state == DONE);
    end

    // -------------------------------------------------------------------------
    // Sequential state: FSM register, the 2-D walk, and the spill holding
    // register.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            counter     <= '0;
            col         <= '0;
            row         <= '0;
            src_row_off <= '0;
            dst_col_off <= '0;
            cap         <= '0;
            have_cap    <= 1'b0;
        end else begin
            state <= state_n;

            if (state == IDLE) begin
                counter     <= '0;
                col         <= '0;
                row         <= '0;
                src_row_off <= '0;
                dst_col_off <= '0;
                have_cap    <= 1'b0;
            end

            // 2-D walk. Only `counter` decides when the transfer ends, so a
            // `dma_len` that is not a whole number of rows simply stops part-way
            // through the last one.
            if (adv) begin
                counter <= counter + 1'b1;
                if (row_last) begin
                    col         <= '0;
                    row         <= row + 16'd1;
                    src_row_off <= src_row_off + SCRATCHPAD_ADDR_W'(tsrow_eff);
                    dst_col_off <= '0;
                end else begin
                    col         <= col + 16'd1;
                    dst_col_off <= dst_col_off + SCRATCHPAD_ADDR_W'(tdrow_eff);
                end
            end

            if (state == SPILL_SEND) begin
                if (sram_din_ready) have_cap <= 1'b0;
                else if (!have_cap) begin
                    cap      <= scratchpad_rdata[MEM_DATA_W-1:0];
                    have_cap <= 1'b1;
                end
            end
        end
    end
endmodule
