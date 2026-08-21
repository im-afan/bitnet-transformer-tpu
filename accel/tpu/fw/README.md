# Firmware (PicoRV32 command producer)

C kernels for the CPU in [`../rtl/cpu_subsys.sv`](../rtl/cpu_subsys.sv) — the
second producer of the 128-bit macro-ops the MXU/VPU/DMA queues consume
(`../docs/picorv32_migration.md` §4). The scalar unit and its `.tpu` programs are
untouched; both paths push into the same queues.

| File | Contents |
| --- | --- |
| `tpu.h` | the MMIO aperture and one builder per command. No abstraction — the fields are packed exactly as `cmd_mxu.sv` / `cmd_dma.sv` decode them |
| `matmul.c` | `C[8x16] = A[8x32] @ W[32x16]`, DMA in, one `matmul_t`, DMA out |
| `matmul_loop.c` | the same product with the 4x2 tile grid walked in C — 8 single-tile dispatches, `.acc` across the contraction — instead of by the array |
| `start.S` | reset entry: `gp`/`sp`, zero `.bss`, `main`, then raise `done` |
| `link.ld` | the 16 KB firmware RAM at address 0 |
| `bin2hex.py` | `.bin` → one 32-bit word per line, for `'I'` and for `$readmemh` |

## Build

Needs a bare-metal RISC-V gcc — `brew install riscv64-elf-gcc` (what these were
built with: 16.2.0, binutils 2.47), or the xPack `riscv-none-elf-gcc`. The
Makefile autodetects the prefix; override with `CROSS=`. Nothing else: the
newlib the formula installs is never linked.

Current sizes, all text, no `.data`/`.bss`: `matmul` 248 bytes (62 words),
`matmul_loop` 352 (88), against 16 KB of firmware RAM.

```bash
make -C accel/tpu/fw            # -> matmul.hex
make -C accel/tpu/fw dis        # disassembly
make -C accel/tpu/fw PROG=foo   # foo.c instead
```

`-march=rv32ic_zmmul -mabi=ilp32` matches how `cpu_subsys.sv` parameterizes the
core: compressed on, fast multiplier on, **divider off** — a `div` or `rem`
would decode as an illegal instruction and trap, so plain `rv32imc` is wrong
here. Nothing is linked (`-nostdlib`, no libgcc), so a kernel that needs `/` or
`%` needs the RTL to enable the divider first. With a gcc older than 12 (no
`zmmul`), use `-march=rv32ic` and keep multiplication out of the kernel too.

## Simulate it

`../tb/fw_matmul_tb.sv` loads the image through `cpu_subsys.sv`'s `FW_INIT`
(`$readmemh`, no UART), seeds DRAM, releases the CPU and checks the int32 C:

```bash
cd accel/tpu/tb
make fw                     # ../fw/matmul.hex
make fw FWPROG=matmul_loop  # the software tile loop, same expectations
```

Both pass, 129 checks: `matmul` halts after 2 077 clocks (MXU 289, DMA 1 420,
368 with no unit busy), `matmul_loop` after 2 454 (MXU 392, same DMA, 642 idle).
The 103 extra MXU clocks are the int32 partials round-tripping through the
scratchpad between contraction tiles, which is exactly what the hardware tile
loop exists to delete. Breakdown in
[`../docs/picorv32_migration.md`](../docs/picorv32_migration.md) §9.6.

Both take the shape from the build, so `make fwsweep` walks a range of them and
tabulates what each costs the CPU:

```bash
make -C accel/tpu/fw M=8 KTILES=16 NTILES=16   # one shape (make clean first)
cd accel/tpu/tb && make fwsweep                # the whole sweep, ~40 s
```

`matmul.c` issues 5 commands at every shape and its CPU cost is flat at ~380
clocks across a 205x range of array work; `matmul_loop.c` pays ~85 clocks per
dispatch, of which only `max(0, 85 - MXU-clocks-per-tile)` is exposed — nothing
at all once M is past ~24. Full table in §9.7.

## Run it on the board

```bash
make -C accel/tpu/fw run PORT=COM5
make -C accel/tpu/fw run PROG=matmul_loop PORT=COM5   # the software tile loop
# or: python accel/tpu/host/run_fw_matmul.py -p COM5
python accel/tpu/host/run_fw_matmul.py --dry-run   # operands + reference, no board
```

[`../host/run_fw_matmul.py`](../host/run_fw_matmul.py) loads the image over
`'I'`, writes A and W into DRAM, releases the core with `'G'`, then reads the
int32 result back and checks it against a Python matmul. Both commands take the
firmware RAM by setting bit 12 of their word address (`tpu_uart.FW_BASE`); the
CPU always resets to firmware address 0.

`matmul.c` and `matmul_loop.c` share the problem, the layout and the result
address, so the same script checks either — pass `--fw matmul_loop.hex`, or let
`make run PROG=` do it. The pair is the firmware version of
[`tiled_matmul_hw.tpu`](../../tpulang/examples/tiled_matmul_hw.tpu) vs
[`tiled_matmul.tpu`](../../tpulang/examples/tiled_matmul.tpu): one `matmul_t`
against 8 single-tile dispatches with `.acc` across the contraction.

The board must be running `board=cmod_a7` and the bitstream must be new enough
to contain `cpu_subsys.sv`.

## Two things that are software's problem now

- **Cross-unit ordering.** Each unit has its own queue, so a `matmul` will start
  on top of a DMA that has not finished. `tpu_wait(unit)` — retired caught up
  with issued — is the fence.
- **Queue-ordered geometry.** `MXU_GEOM` sticks until the next one, but only
  within the MXU's own command stream, so it cannot be corrupted by another unit
  or an earlier program the way the old `cfg` registers could.

Flow control is *not* software's problem: a full queue withholds the write
response on the fourth word and the CPU stalls inside the store.
