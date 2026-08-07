// -----------------------------------------------------------------------------
// uart_memory.sv — UART command link + external SRAM, with the TPU removed.
//
// tpu_top minus the machine: scalar unit, MXU, VPU, scratchpad and the DMA
// engine are gone, and with them the SRAM arbitration mux (there is only one
// requester left, so `mem_* = uart_mem_*` unconditionally). What remains is
// exactly the path host/test_uart_link.py exercises — uart_receiver ->
// uart_interface -> sram_controller -> the chip pins, and back — instantiated
// from the same unmodified files the production image uses, at the same pinout.
//
// This is the middle rung of the ladder in docs/uart_selftest.md:
//
//   cmod_a7        everything. The image that fails.
//   cmod_a7_mem    this one: the protocol and the memory, no core.
//   cmod_a7_bram   the protocol only — same FSM, on-chip block RAM behind it.
//   cmod_a7_echo   neither. Just the wire framing. Runs clean for 5 minutes.
//
// So it splits the remaining search space in half. If test_uart_link.py still
// corrupts against this image, the fault is in uart_interface or
// sram_controller — nothing else is left. If it runs clean, the fault needs the
// core present, which points at the arbitration mux, `core_busy`, or the DMA
// engine's half of the shared controller, none of which the echo image could
// say anything about.
//
// rtl/uart_bram.sv is the next cut after this one, and the only one that
// separates the two suspects this image leaves: identical protocol, identical
// uart_interface, bram_controller in place of sram_controller.
//
// Two things behave differently to tpu_top, both deliberate and both visible
// only to commands this rig is not meant to be driven with:
//
//   * 'I' (write IMEM) and 'G' (go) are still decoded, still range-checked and
//     still ACK'd, because uart_interface is instantiated unmodified — but
//     there is no instruction memory and no core, so the write lands nowhere
//     and the run never starts. Use 'R'/'W' only.
//   * `core_busy` is tied low, so nothing is ever NAK'd for arbitration. That
//     is the point: it removes the core's claim on the controller from the
//     experiment entirely.
//
// The status outputs are instrumentation, not decoration — see `collision`.
// -----------------------------------------------------------------------------

module uart_memory #(
    // ---- UART host link ------------------------------------------------------
    parameter int UART_CPB        = 104,  // 12 MHz / 115200 baud
    parameter int UART_RX_TIMEOUT = 0,    // inter-byte abort (clocks; 0 = off)

    // ---- External async SRAM (Cmod A7 cellular RAM: 512K x 8) ----------------
    parameter int MEM_ADDR_W      = 19,
    parameter int MEM_DATA_W      = 8,
    parameter int SRAM_CPA        = 2,    // sram_controller CLOCKS_PER_ACCESS

    // ---- Instruction memory address width -----------------------------------
    //   No IMEM exists in this image; this only has to match what the host's
    //   'I' frames would be range-checked against, so that a command the rig
    //   rejects is rejected for the same reason the real image rejects it.
    parameter int IMEM_AW         = 10,

    // ---- Status LED timing ---------------------------------------------------
    parameter int ACT_W           = 19,   // activity stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W            = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    // ---- USB-serial link -----------------------------------------------------
    input  logic uart_rx,
    output logic uart_tx,

    // ---- External SRAM chip pins --------------------------------------------
    //   Same signals, same widths and (via constraints/cmod_a7.xdc) the same
    //   package pins as tpu_top. The 30 bank-14 pins switch here exactly as they
    //   do in the production image.
    output logic [MEM_ADDR_W-1:0] sram_addr,
    inout  wire  [MEM_DATA_W-1:0] sram_data,   // inout must be a net, not a var
    output logic                  sram_we,
    output logic                  sram_ce,
    output logic                  sram_oen,

    // ---- Status, for the board wrapper's LEDs -------------------------------
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag `collision`
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic collision     // sticky: see below
);

    // =========================================================================
    // sram_controller user side.
    //
    // In tpu_top these nets are the output of a mux between the DMA engine and
    // the UART host. Here the UART host is the only driver, so the mux is gone
    // rather than tied off — a mux with one input is a thing that can be wrong.
    // =========================================================================
    logic                   mem_start, mem_we, mem_busy, mem_done;
    logic [MEM_ADDR_W-1:0]  mem_addr;
    logic [MEM_DATA_W-1:0]  mem_din, mem_dout;

    // ---- UART host interface -------------------------------------------------
    logic [7:0]             uart_rx_data;
    logic                   uart_rx_valid;
    logic                   uart_tx_start, uart_tx_busy;
    logic [7:0]             uart_tx_data;
    logic                   uart_host_busy;

    // IMEM write port and run trigger. Decoded and driven by uart_interface, and
    // then dropped: there is no scalar unit in this image. Declared rather than
    // left as unnamed open ports so the dead end is explicit in the netlist.
    logic                   uart_imem_we;
    logic [IMEM_AW-1:0]     uart_imem_waddr;
    logic [31:0]            uart_imem_wdata;
    logic                   uart_run_start;
    logic [IMEM_AW-1:0]     uart_run_pc;
    logic                   uart_rx_overrun;

    // =========================================================================
    // SRAM controller — chip-side pins go straight to the top level.
    // =========================================================================
    sram_controller #(
        .CLOCKS_PER_ACCESS (SRAM_CPA),
        .ADDR_W            (MEM_ADDR_W),
        .DATA_W            (MEM_DATA_W)
    ) u_sram (
        .clk   (clk),
        .rst_n (rst_n),

        // user side ← UART host (sole owner)
        .start (mem_start),
        .we    (mem_we),
        .addr  (mem_addr),
        .din   (mem_din),
        .dout  (mem_dout),
        .busy  (mem_busy),
        .done  (mem_done),

        // chip side → top-level pins
        .sram_addr (sram_addr),
        .sram_data (sram_data),
        .sram_we   (sram_we),
        .sram_ce   (sram_ce),
        .sram_oen  (sram_oen)
    );

    // =========================================================================
    // UART host link — the production blocks, untouched.
    // =========================================================================
    uart_receiver #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_rx (
        .clk     (clk),
        .rst_n   (rst_n),
        .uart_rx (uart_rx),
        .data    (uart_rx_data),
        .valid   (uart_rx_valid)
    );

    uart_transmitter #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_tx (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (uart_tx_start),
        .data    (uart_tx_data),
        .uart_tx (uart_tx),
        .busy    (uart_tx_busy)
    );

    uart_interface #(
        .ADDR_W     (MEM_ADDR_W),
        .LENGTH_W   (16),
        .IMEM_AW    (IMEM_AW),
        .RX_TIMEOUT (UART_RX_TIMEOUT)
    ) u_uart (
        .clk   (clk),
        .rst_n (rst_n),

        // No core to arbitrate against: nothing is ever NAK'd for being busy.
        .core_busy (1'b0),

        // receiver / transmitter
        .data_in           (uart_rx_data),
        .receiver_valid    (uart_rx_valid),
        .transmitter_start (uart_tx_start),
        .data_out          (uart_tx_data),
        .transmitter_busy  (uart_tx_busy),

        // sram_controller user side (sole owner here)
        .sram_start (mem_start),
        .sram_we    (mem_we),
        .sram_addr  (mem_addr),
        .sram_din   (mem_din),
        .sram_dout  (mem_dout),
        .sram_busy  (mem_busy),
        .sram_done  (mem_done),

        // instruction-memory write port → nowhere (no scalar unit)
        .imem_we    (uart_imem_we),
        .imem_waddr (uart_imem_waddr),
        .imem_wdata (uart_imem_wdata),

        // run trigger → nowhere (no scalar unit)
        .run_start  (uart_run_start),
        .run_pc     (uart_run_pc),

        .host_busy  (uart_host_busy),
        .rx_overrun (uart_rx_overrun)
    );

    // =========================================================================
    // Instrumentation.
    //
    // `rx_byte` is derived exactly as uart_interface.sv:116 derives it — the
    // rising edge of the receiver's `valid`, which is a level held from STOP
    // until half way through the next start bit. Counting it here means the
    // status logic sees precisely the bytes the command FSM sees, no more.
    //
    // `collision` is the one output worth watching. It sets on either of two
    // things, both of which mean "the host got ahead of the device":
    //
    //   * a byte landed while the device was mid-transmit. On this link the host
    //     is the sole master and waits for each reply before sending the next
    //     command, so it should never happen. Since uart_interface grew its RX
    //     holding register this is survivable rather than fatal — the byte is
    //     kept, not dropped — so it is now information about the *host*, not a
    //     device fault on its own.
    //   * uart_interface's `rx_overrun`: a byte arrived with the holding
    //     register still full, so one really was lost. That is the fault.
    //
    // Either way the evidence is a one-clock event that a 4 kB block will have
    // buried by the time the host notices, so it is caught here, on an LED, with
    // no host involvement.
    //
    // Sticky until reset, like uart_echo's `overflow`: the interesting event is
    // "did this ever happen", and a one-clock pulse 40 minutes into a soak is
    // not something anyone is watching for.
    // =========================================================================
    logic uart_rx_valid_prev;
    wire  rx_byte = uart_rx_valid & ~uart_rx_valid_prev;

    // The transmitter asserts `busy` the cycle *after* it accepts `start`, so
    // the strobe has to be OR'd in or the first cycle of every frame would look
    // idle.
    wire  tx_active = uart_tx_busy | uart_tx_start;

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_rx_valid_prev <= 1'b0;
            act_cnt            <= '0;
            hb_cnt             <= '0;
            collision          <= 1'b0;
        end else begin
            uart_rx_valid_prev <= uart_rx_valid;
            hb_cnt             <= hb_cnt + 1'b1;

            if (uart_rx_overrun) collision <= 1'b1;   // sticky until reset

            if (rx_byte) begin
                act_cnt <= '1;
                if (tx_active) collision <= 1'b1;   // sticky until reset
            end else if (act_cnt != 0) begin
                act_cnt <= act_cnt - 1'b1;
            end
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
