# Firmware for the PicoRV32 command producer (rtl/cpu_subsys.sv).
#
#   make                  build matmul.hex (the default kernel)
#   make PROG=foo         build foo.hex from foo.c
#   make dis              disassemble the ELF
#   make run PORT=COM5    build, then load + run + check on the board
#   make trace            build + run the kernel NATIVELY, printing its command
#                         trace (docs/picorv32_migration.md §8.1). Needs only the
#                         host cc — no RISC-V toolchain, no board, no simulator.
#   make clean
#
# Needs a bare-metal RISC-V gcc on PATH; the prefix is autodetected, override it
# with CROSS= (e.g. CROSS=riscv-none-elf-). `make trace` is the exception: it
# uses HOSTCC and works with no cross toolchain installed at all.
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
# `trace` and `clean` are host-only, so neither may hard-fail on a missing cross
# compiler — that is the point of the native trace path.
ifneq ($(filter-out clean trace,$(GOALS)),)
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

.PHONY: all dis size run trace clean
.PRECIOUS: %.elf %.bin %.trace

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

# ---- native command trace (docs/picorv32_migration.md §8.1) ------------------
# The same .c, the same builders out of tpu.h, compiled by the host compiler
# with the two MMIO primitives swapped for a trace emitter. This is the producer
# that replaces the RV32IM interpreter, and the reason it cannot drift from the
# firmware is that nothing but those two primitives differs between the builds.
HOSTCC   ?= cc
TRACEFLAGS := -DTPU_TRACE -I. -Wall -Wextra -O1 \
              $(if $(M),-DM=$(M)) $(if $(KTILES),-DKTILES=$(KTILES)) \
              $(if $(NTILES),-DNTILES=$(NTILES))

%.trace: %.c mock/tpu_trace.c tpu.h
	$(HOSTCC) $(TRACEFLAGS) -o $@ $< mock/tpu_trace.c

trace: $(PROG).trace
	./$(PROG).trace

clean:
	rm -f *.o *.d *.elf *.bin *.hex *.map *.trace

-include $(OBJS:.o=.d)
