## =============================================================================
## cmod_a7.xdc — active constraints for cmod_a7_top on the Digilent Cmod A7-35T.
##
## Only the pins the design actually uses. `constraings-cmod.xdc` alongside this
## file is Digilent's unmodified master (everything commented out) and is kept
## purely as a pin reference — it is NOT read by the build.
##
## Port names here must match synth/vivado/boards/cmod_a7/cmod_a7_top.sv.
## =============================================================================

## ---- 12 MHz system clock ----------------------------------------------------
## The board's only oscillator. It feeds the core directly (no MMCM), so this is
## the design's one and only clock. 1/12 MHz = 83.333 ns.
set_property -dict { PACKAGE_PIN L17   IOSTANDARD LVCMOS33 } [get_ports { sysclk }]; #IO_L12P_T1_MRCC_14 Sch=gclk
create_clock -add -name sys_clk_pin -period 83.333 -waveform {0 41.666} [get_ports { sysclk }];

## ---- LEDs: led[0] = busy, led[1] = done -------------------------------------
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

## ---- Cellular RAM — the external "DRAM" behind the DMA engine ---------------
## 512K x 8 async SRAM. MEM_ADDR_W = 19, MEM_DATA_W = 8 in board.tcl must match
## the 19 address / 8 data lines below.
set_property -dict { PACKAGE_PIN M18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[0]  }]; #IO_L11P_T1_SRCC_14 Sch=sram-a[0]
set_property -dict { PACKAGE_PIN M19   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[1]  }]; #IO_L11N_T1_SRCC_14 Sch=sram-a[1]
set_property -dict { PACKAGE_PIN K17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[2]  }]; #IO_L12N_T1_MRCC_14 Sch=sram-a[2]
set_property -dict { PACKAGE_PIN N17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[3]  }]; #IO_L13P_T2_MRCC_14 Sch=sram-a[3]
set_property -dict { PACKAGE_PIN P17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[4]  }]; #IO_L13N_T2_MRCC_14 Sch=sram-a[4]
set_property -dict { PACKAGE_PIN P18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[5]  }]; #IO_L14P_T2_SRCC_14 Sch=sram-a[5]
set_property -dict { PACKAGE_PIN R18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[6]  }]; #IO_L14N_T2_SRCC_14 Sch=sram-a[6]
set_property -dict { PACKAGE_PIN W19   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[7]  }]; #IO_L16N_T2_A15_D31_14 Sch=sram-a[7]
set_property -dict { PACKAGE_PIN U19   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[8]  }]; #IO_L15P_T2_DQS_RDWR_B_14 Sch=sram-a[8]
set_property -dict { PACKAGE_PIN V19   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[9]  }]; #IO_L15N_T2_DQS_DOUT_CSO_B_14 Sch=sram-a[9]
set_property -dict { PACKAGE_PIN W18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[10] }]; #IO_L16P_T2_CSI_B_14 Sch=sram-a[10]
set_property -dict { PACKAGE_PIN T17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[11] }]; #IO_L17P_T2_A14_D30_14 Sch=sram-a[11]
set_property -dict { PACKAGE_PIN T18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[12] }]; #IO_L17N_T2_A13_D29_14 Sch=sram-a[12]
set_property -dict { PACKAGE_PIN U17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[13] }]; #IO_L18P_T2_A12_D28_14 Sch=sram-a[13]
set_property -dict { PACKAGE_PIN U18   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[14] }]; #IO_L18N_T2_A11_D27_14 Sch=sram-a[14]
set_property -dict { PACKAGE_PIN V16   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[15] }]; #IO_L19P_T3_A10_D26_14 Sch=sram-a[15]
set_property -dict { PACKAGE_PIN W16   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[16] }]; #IO_L20P_T3_A08_D24_14 Sch=sram-a[16]
set_property -dict { PACKAGE_PIN W17   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[17] }]; #IO_L20N_T3_A07_D23_14 Sch=sram-a[17]
set_property -dict { PACKAGE_PIN V15   IOSTANDARD LVCMOS33 } [get_ports { sram_addr[18] }]; #IO_L21P_T3_DQS_14 Sch=sram-a[18]

set_property -dict { PACKAGE_PIN W15   IOSTANDARD LVCMOS33 } [get_ports { sram_data[0]  }]; #IO_L21N_T3_DQS_A06_D22_14 Sch=sram-dq[0]
set_property -dict { PACKAGE_PIN W13   IOSTANDARD LVCMOS33 } [get_ports { sram_data[1]  }]; #IO_L22P_T3_A05_D21_14 Sch=sram-dq[1]
set_property -dict { PACKAGE_PIN W14   IOSTANDARD LVCMOS33 } [get_ports { sram_data[2]  }]; #IO_L22N_T3_A04_D20_14 Sch=sram-dq[2]
set_property -dict { PACKAGE_PIN U15   IOSTANDARD LVCMOS33 } [get_ports { sram_data[3]  }]; #IO_L23P_T3_A03_D19_14 Sch=sram-dq[3]
set_property -dict { PACKAGE_PIN U16   IOSTANDARD LVCMOS33 } [get_ports { sram_data[4]  }]; #IO_L23N_T3_A02_D18_14 Sch=sram-dq[4]
set_property -dict { PACKAGE_PIN V13   IOSTANDARD LVCMOS33 } [get_ports { sram_data[5]  }]; #IO_L24P_T3_A01_D17_14 Sch=sram-dq[5]
set_property -dict { PACKAGE_PIN V14   IOSTANDARD LVCMOS33 } [get_ports { sram_data[6]  }]; #IO_L24N_T3_A00_D16_14 Sch=sram-dq[6]
set_property -dict { PACKAGE_PIN U14   IOSTANDARD LVCMOS33 } [get_ports { sram_data[7]  }]; #IO_25_14 Sch=sram-dq[7]

set_property -dict { PACKAGE_PIN P19   IOSTANDARD LVCMOS33 } [get_ports { sram_oen }]; #IO_L10P_T1_D14_14 Sch=sram-oe
set_property -dict { PACKAGE_PIN R19   IOSTANDARD LVCMOS33 } [get_ports { sram_we  }]; #IO_L10N_T1_D15_14 Sch=sram-we
set_property -dict { PACKAGE_PIN N19   IOSTANDARD LVCMOS33 } [get_ports { sram_ce  }]; #IO_L9N_T1_DQS_D13_14 Sch=sram-ce

## =============================================================================
## Timing exceptions
## =============================================================================

## Asynchronous inputs. btn[0] goes through a two-flop synchronizer in the
## wrapper; uart_txd_in is oversampled by uart_receiver. Neither has any timing
## relationship to sysclk.
set_false_path -from [get_ports { btn[*] }]
set_false_path -from [get_ports { uart_txd_in }]

## External async SRAM. The chip is asynchronous, so there is no I/O clock to
## constrain against and set_input_delay/set_output_delay have no meaningful
## reference. The access timing is instead guaranteed by sram_controller's
## CLOCKS_PER_ACCESS: at 12 MHz each phase is held 2 x 83.3 ns = 167 ns, against
## a part rated for 10-55 ns access. That is >3x margin, so declaring these
## paths unconstrained is safe *at this clock rate*.
##
## If you ever add an MMCM and raise the core clock, revisit this: at 100 MHz a
## 2-cycle access is only 20 ns and these false paths would be hiding a real
## violation. Replace them with set_input_delay/set_output_delay against a
## virtual clock, or raise SRAM_CPA.
set_false_path -to   [get_ports { sram_addr[*] sram_data[*] sram_we sram_ce sram_oen }]
set_false_path -from [get_ports { sram_data[*] }]

## =============================================================================
## Configuration
## =============================================================================
## CFGBVS/CONFIG_VOLTAGE are mandatory on 7-series (DRC NSTD-1 / UCIO-1 escalate
## to errors without them). SPI x4 matches how the Cmod A7 boots from its QSPI
## flash and is what mode=mcs writes.
set_property CFGBVS VCCO                       [current_design]
set_property CONFIG_VOLTAGE 3.3                [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4   [current_design]
