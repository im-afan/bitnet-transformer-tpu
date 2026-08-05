## =============================================================================
## cmod_a7_echo.xdc — constraints for cmod_a7_echo_top (UART echo self-test).
##
## The same physical board as cmod_a7.xdc, but only the six pins this image has.
## cmod_a7.xdc is not reusable here: it constrains 31 SRAM ports that
## cmod_a7_echo_top does not declare, and set_property against an empty
## get_ports result is an error, not a warning.
##
## Pin assignments are identical to cmod_a7.xdc for every port they share, so
## the clock and the UART reach the receiver through the same package pins and
## the same I/O standard in both images. That is a precondition for the self-test
## being evidence about the production image at all.
##
## Port names here must match boards/cmod_a7_echo/cmod_a7_echo_top.sv.
## =============================================================================

## ---- 12 MHz system clock ----------------------------------------------------
set_property -dict { PACKAGE_PIN L17   IOSTANDARD LVCMOS33 } [get_ports { sysclk }]; #IO_L12P_T1_MRCC_14 Sch=gclk
create_clock -add -name sys_clk_pin -period 83.333 -waveform {0 41.666} [get_ports { sysclk }];

## ---- LEDs: led[0] = heartbeat / FIFO-overflow flag, led[1] = RX activity ----
set_property -dict { PACKAGE_PIN A17   IOSTANDARD LVCMOS33 } [get_ports { led[0] }]; #IO_L12N_T1_MRCC_16 Sch=led[1]
set_property -dict { PACKAGE_PIN C16   IOSTANDARD LVCMOS33 } [get_ports { led[1] }]; #IO_L13P_T2_MRCC_16 Sch=led[2]

## ---- Buttons: btn[0] = manual reset (active high), btn[1] reserved ----------
set_property -dict { PACKAGE_PIN A18   IOSTANDARD LVCMOS33 } [get_ports { btn[0] }]; #IO_L19N_T3_VREF_16 Sch=btn[0]
set_property -dict { PACKAGE_PIN B18   IOSTANDARD LVCMOS33 } [get_ports { btn[1] }]; #IO_L19P_T3_16 Sch=btn[1]

## ---- UART (on-board FT2232 USB-serial bridge) -------------------------------
## Names are from the host's point of view: uart_txd_in is the host transmitting
## into the FPGA, uart_rxd_out is the FPGA transmitting to the host.
set_property -dict { PACKAGE_PIN J18   IOSTANDARD LVCMOS33 } [get_ports { uart_rxd_out }]; #IO_L7N_T1_D10_14 Sch=uart_rxd_out
set_property -dict { PACKAGE_PIN J17   IOSTANDARD LVCMOS33 } [get_ports { uart_txd_in  }]; #IO_L7P_T1_D09_14 Sch=uart_txd_in

## =============================================================================
## Timing exceptions
## =============================================================================

## Asynchronous inputs, exactly as in cmod_a7.xdc. btn[0] goes through a two-flop
## synchronizer in the wrapper; uart_txd_in is oversampled by uart_receiver.
## Neither has any timing relationship to sysclk.
set_false_path -from [get_ports { btn[*] }]
set_false_path -from [get_ports { uart_txd_in }]

## =============================================================================
## Configuration
## =============================================================================
## CFGBVS/CONFIG_VOLTAGE are mandatory on 7-series (DRC NSTD-1 / UCIO-1 escalate
## to errors without them). SPI x4 matches how the Cmod A7 boots from its QSPI
## flash.
set_property CFGBVS VCCO                       [current_design]
set_property CONFIG_VOLTAGE 3.3                [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4   [current_design]
