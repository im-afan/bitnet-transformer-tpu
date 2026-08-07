/*
On-chip block-RAM controller — a drop-in stand-in for sram_controller (rtl/sram.sv).

Same user-side contract, clock for clock: start/we/addr/din latched on `start`,
`busy` high from the accept cycle, a one-cycle `done` strobe, `dout` valid with
it. The FSM below is sram_controller's FSM with the chip pins removed and the
storage moved inside the FPGA; CLOCKS_PER_ACCESS is honoured even though nothing
here needs the wait, because the *point* is that the two behave identically.

That is not decoration. This block exists so boards/cmod_a7_bram can run the UART
command protocol with the external memory taken out of the picture (see
rtl/uart_bram.sv). If it completed a transfer in one clock instead of five, every
turnaround in uart_interface would shift and the image would no longer be a
controlled comparison against boards/cmod_a7_mem — it would just be a different
design that also happens to work. Keep the latency equal.

Two things genuinely differ from the SRAM, both unavoidable:

  * Capacity. ADDR_W is the *protocol* address width (19 bits, 512 KiB) and
    DEPTH_W is what is actually built. 2**19 bytes will not fit in an
    Artix-7 35T (1800 Kb of block RAM total), so addresses above 2**DEPTH_W
    alias down onto the implemented window — addr[DEPTH_W-1:0] is the index and
    the high bits are discarded. Deterministic, so a write and a read of the same
    address still agree, but two addresses 2**DEPTH_W apart are the same byte.
    `aliased` goes sticky-high the first time it happens so this is a visible
    event rather than a silent one; see rtl/uart_bram.sv for what it drives.
  * There is no bidirectional bus, no output enable and no chip select, so
    nothing here can be defeated by bus contention, drive strength, or the 30
    bank-14 pins switching. Removing exactly that is the reason the image exists.

Storage comes up all-zero, matching a configured BRAM (the simulation-only
initial block below makes the two agree; hardware needs no help).
*/

module bram_controller #(
    parameter integer CLOCKS_PER_ACCESS = 2,   // held equal to sram_controller's
    parameter integer ADDR_W  = 19,            // protocol address width
    parameter integer DATA_W  = 8,
    parameter integer DEPTH_W = 16             // implemented: 2**DEPTH_W words
) (
    input  logic clk,
    input  logic rst_n,

    // user signals — identical to sram_controller's
    input  logic start,
    input  logic we,
    input  logic [ADDR_W-1:0] addr,
    input  logic [DATA_W-1:0] din,
    output logic [DATA_W-1:0] dout,
    output logic busy,
    output logic done,

    // sticky: an access addressed a byte above the implemented window and was
    // folded back into it. Cleared only by reset.
    output logic aliased
);
    localparam integer DEPTH = 1 << DEPTH_W;

    initial begin
        if (DEPTH_W > ADDR_W)
            $fatal(1, "bram_controller: DEPTH_W (%0d) > ADDR_W (%0d)",
                   DEPTH_W, ADDR_W);
    end

    localparam [1:0] IDLE = 2'b00, WRITE = 2'b01, READ = 2'b10, WAIT = 2'b11;

    reg [1:0] state, state_n;
    reg [7:0] counter;

    // latched request, so the caller need only hold inputs during `start`
    reg [ADDR_W-1:0] addr_q;
    reg [DATA_W-1:0] din_q;
    reg              wr_q;      // 1 = latched op is a write

    // Is this request outside the implemented window? Written as a generate
    // rather than a ternary because addr[ADDR_W-1:DEPTH_W] is an illegal empty
    // range when the two are equal, and an elaboration error does not care that
    // the branch is unreachable.
    wire addr_above_window;
    generate
        if (DEPTH_W < ADDR_W) begin : g_alias_check
            assign addr_above_window = |addr[ADDR_W-1:DEPTH_W];
        end else begin : g_alias_check
            assign addr_above_window = 1'b0;
        end
    endgenerate

    // =========================================================================
    // The memory. One address, mutually exclusive read/write enables — the
    // single-port no-change template Vivado infers as a RAMB. DEPTH_W = 16 is
    // 16 RAMB36 tiles (4 KiB each in x9 mode) of the 35T's 50.
    // =========================================================================
    (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [DATA_W-1:0] q;

    wire [DEPTH_W-1:0] index = addr_q[DEPTH_W-1:0];   // the alias, made explicit
    wire               en_wr = (state == WRITE);
    wire               en_rd = (state == READ);

    always_ff @(posedge clk) begin
        if (en_wr) mem[index] <= din_q;
        if (en_rd) q          <= mem[index];
    end

    // Configuration leaves block RAM zeroed; give the simulator the same start.
    // Synthesis-only initialization of a 2**16-entry array would otherwise be
    // carried through the netlist as an INIT string for no gain.
// synthesis translate_off
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
        q = '0;
    end
// synthesis translate_on

    // =========================================================================
    // FSM — sram_controller's, unchanged apart from the chip pins.
    //
    // `q` is captured at the end of the READ cycle, so it is valid throughout
    // WAIT, exactly where sram_controller samples the asynchronous chip bus.
    // `dout` therefore lands on the same clock edge in both controllers.
    // =========================================================================
    always_comb begin
        state_n = state;
        case (state)
            IDLE:  if (start) state_n = we ? WRITE : READ;
            WRITE: state_n = WAIT;
            READ:  state_n = WAIT;
            WAIT:  if (counter >= CLOCKS_PER_ACCESS) state_n = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            counter  <= 8'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
            dout     <= '0;
            addr_q   <= '0;
            din_q    <= '0;
            wr_q     <= 1'b0;
            aliased  <= 1'b0;
        end else begin
            state <= state_n;
            done  <= 1'b0;  // one-cycle strobe, default low

            case (state)
                IDLE: begin
                    counter <= 8'b0;
                    busy    <= start;   // accept the request this cycle
                    if (start) begin
                        addr_q <= addr;
                        din_q  <= din;
                        wr_q   <= we;
                        if (addr_above_window) aliased <= 1'b1;  // sticky
                    end
                end
                WRITE: begin
                    busy <= 1'b1;       // the write commits on this edge
                end
                READ: begin
                    busy <= 1'b1;       // mem[index] is captured into q here
                end
                WAIT: begin
                    busy    <= 1'b1;
                    counter <= counter + 1'b1;
                    if (!wr_q) dout <= q;
                    if (state_n == IDLE) begin
                        done <= 1'b1;
                        busy <= 1'b0;
                    end
                end
            endcase
        end
    end
endmodule
