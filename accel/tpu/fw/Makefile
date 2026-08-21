# Firmware for the PicoRV32 command producer (rtl/cpu_subsys.sv).
#
#   make                  build matmul.hex (the default kernel)
#   make PROG=foo         build foo.hex from foo.c
#   make dis              disassemble the ELF
#   make run PORT=COM5    build, then load + run + check on the board
#   make clean
#
# Needs a bare-metal RISC-V gcc on PATH; the prefix is autodetected, override it
# with CROSS= (e.g. CROSS=riscv-none-elf-).
#
# -march MUST match how cpu_subsys.sv builds the core: COMPRESSED_ISA(1) and
# ENABLE_FAST_MUL(1) but ENABLE_DIV(0), so `mul` is fine and `div`/`rem` would
# decode as an illegal instruction and trap. That is what rv32ic_zmmul says.
# A gcc older than 12 has no zmmul — use -march=rv32ic there and keep division
# and multiplication out of the kernel (libgcc is not linked).

CROSS ?= $(shell for p in riscv-none-elf riscv64-elf riscv64-unknown-elf riscv32-unknown-elf; do \
             command -v $$p-gcc >/dev/null 2>&1 && { echo $$p-; exit 0; }; \
           done)

GOALS := $(or $(MAKECMDGOALS),all)
ifeq ($(CROSS),)
ifneq ($(filter-out clean,$(GOALS)),)
$(error no RISC-V cross gcc found on PATH — install one (brew install riscv64-elf-gcc, \
or the xPack riscv-none-elf-gcc) or pass CROSS=<prefix>-)
endif
endif

CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

ARCH    := -march=rv32ic_zmmul -mabi=ilp32
CFLAGS  := $(ARCH) -Os -ffreestanding -nostdlib -fno-builtin -Wall -Wextra \
           -ffunction-sections -fdata-sections -MMD -MP

# Problem shape. Empty = the kernel's own #defines. Make does not track a
# changed -D, so `make clean` first when sweeping (../tb/run_fw_sweep.sh does).
#   make M=8 KTILES=16 NTILES=16
M      ?=
KTILES ?=
NTILES ?=
CFLAGS += $(if $(M),-DM=$(M)) $(if $(KTILES),-DKTILES=$(KTILES)) \
          $(if $(NTILES),-DNTILES=$(NTILES))
LDFLAGS := $(ARCH) -nostdlib -Wl,-T,link.ld -Wl,--gc-sections

PROG ?= matmul
OBJS := start.o $(PROG).o

PORT ?=
PORTARG := $(if $(PORT),-p $(PORT))

.PHONY: all dis size run clean
.PRECIOUS: %.elf %.bin

all: $(PROG).hex

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROG).elf: $(OBJS) link.ld
	$(CC) $(LDFLAGS) -Wl,-Map,$(@:.elf=.map) -o $@ $(OBJS)
	$(SIZE) $@

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.hex: %.bin
	python bin2hex.py $< $@

dis: $(PROG).elf
	$(OBJDUMP) -d $<

size: $(PROG).elf
	$(SIZE) -A $<

# The host end: load the firmware over 'I', stage the operands, run, read back.
run: $(PROG).hex
	python ../host/run_fw_matmul.py --fw $(PROG).hex $(PORTARG)

clean:
	rm -f *.o *.d *.elf *.bin *.hex *.map

-include $(OBJS:.o=.d)
