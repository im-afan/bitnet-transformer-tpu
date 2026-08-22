# Firmware (PicoRV32 command producer)

C kernels for the CPU in [`../rtl/cpu_subsys.sv`](../rtl/cpu_subsys.sv) — the
second producer of the 128-bit macro-ops the MXU/VPU/DMA queues consume
(`../docs/picorv32_migration.md` §4). The scalar unit and its `.tpu` programs are
untouched; both paths push into the same queues.

| File | Contents |
| --- | --- |
| `tpu.h` | the MMIO aperture and one builder per command. No abstraction — the fields are packed exactly as `cmd_mxu.sv` / `cmd_vpu.sv` / `cmd_dma.sv` decode them |
| `matmul.c` | `C[8x16] = A[8x32] @ W[32x16]`, DMA in, one `matmul_t`, DMA out |
| `matmul_loop.c` | the same product with the 4x2 tile grid walked in C — 8 single-tile dispatches, `.acc` across the contraction — instead of by the array |
| `ffn.c` | the feed-forward block, `X@W1 → relu → requant → @W2`. The first kernel to issue a **VPU** command |
| `mha.c` | one head of ReLU attention: all three DMA modes including the transposing spill, plus the `quant4` pack that turns an activation into a weight operand |
| `adder.c` | **the whole shipped model** — four transformer layers and the output head, 518 commands, one run |
| `adder_rq.h` | `adder.c`'s 16 requant `{m0,n}` words per layer. The checked-in copy is tuned for the synthetic operands; a real checkpoint overrides it (below) |
| `mock/tpu_trace.c` | the host-side `tpu_push`/`tpu_wait`, so `-DTPU_TRACE` turns any kernel here into its own command-trace producer |
| `start.S` | reset entry: `gp`/`sp`, zero `.bss`, `main`, then raise `done` |
| `link.ld` | the 16 KB firmware RAM at address 0 |
| `bin2hex.py` | `.bin` → one 32-bit word per line, for `'I'` and for `$readmemh` |

## Build

Needs a bare-metal RISC-V gcc — `brew install riscv64-elf-gcc` (what these were
built with: 16.2.0, binutils 2.47), or the xPack `riscv-none-elf-gcc`. The
Makefile autodetects the prefix; override with `CROSS=`. Nothing else: the
newlib the formula installs is never linked.

Current sizes, all text, no `.data`/`.bss`: `matmul` 248 bytes (62 words),
`matmul_loop` 352 (88), `adder` 1544 (386), against 16 KB of firmware RAM. The
whole four-layer model is 1.5 KB of RISC-V because the per-layer body is a loop
over a base register, not unrolled — depth costs no instructions at all.

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

## `adder.c` — the whole model

`model/transformer.py::adder_int4_vanilla` (`d=64`, `f=256`, `layers=4`,
`q_heads=kv_heads=4`, `head_dim=16`, `vocab=13`, int4 weights *and* int4
activations, no bias) as one program: **518 commands, 1544 bytes of firmware,
439 917 clocks.**

```bash
cd accel/tpu/tb && make fw FWPROG=adder     # ~3 min, 526 879 checks, 0 errors
```

```
counters: run=439916 mxu=206361 mload=33440 vpu=84352 dma=131168
          idlec=18035 qfull=0 ovlap=0
command trace: 518 commands, 518 expected
```

`idlec` — clocks with no unit busy at all — is **4.1%** of the run, which is what
the CPU costs as a command producer on a real workload. `ovlap` is 0 because the
kernel fences after every cross-unit dependency; nothing is overlapped yet.

Two layout facts are the whole design, and both are the *opposite* of the
retired ternary kernel's, because weights went row-major:

- **`Q @ K^T` needs the transpose.** Its weight is `K^T[h][s]`, so row `h` must
  be contiguous over `s` — that is K column-major, and K leaves its projection
  row-major. It goes out through the DMA's `.t` spill and back in, as **bytes**,
  because a packed int4 nibble is half of one; then `quant4` packs it.
- **`P @ V` does not.** It contracts over keys, so its weight is `V[s][h]`, row
  `s` contiguous over `h` — exactly how V left its projection. Free.

Everything else worth knowing:

| | |
| --- | --- |
| scratchpad | one 8 KB weight window refilled 4x per layer, everything else resident; top byte used is `0xDFFF` of 64 KB |
| DRAM | 106 KB — `X0`, the mask, `W_fc`, the logits, and `0x2000 + L*0x6000` per layer |
| the mask | int8 `0`/`-8`. S is already int4, so `S-8 <= -1` whatever `s_s` is and ReLU takes a masked entry to **exactly** zero |
| `vlen` | 10 bits, so no VPU pass exceeds 1023 elements; every tensor here is a multiple of 512 and the one int32 temp stays 2 KB |
| the head | 13 logits padded to a whole 16-wide tile, so the second output tile cannot land on the next token's row. Read 13 of every 16 int32 words back |
| the host | the token embedding (no gather in the ISA) and the argmax (nothing returns an index). Nothing else |

**The requant table is a compile-time input.** The `{m0,n}` word is a literal in
the macro-op, so unlike the retired scalar ISA there is no path by which the
device could fetch it from memory — the 16 words per layer have to be in the
image. `adder_rq.h` is the checked-in default and is tuned for the *synthetic*
operands `../../tpulang/fw_vectors.py` stages, so `make fw FWPROG=adder` is a
self-contained datapath regression with no `.pt` involved. A real checkpoint
goes through [`../../tpulang/adder_export.py`](../../tpulang/adder_export.py),
which derives the table from the model's learned `ActQuant` scales, compiles the
same `adder.c` against it, and runs the trace on `iss.py`:

```bash
python accel/tpulang/adder_export.py -n 256          # accuracy on the addition task
python accel/tpulang/adder_export.py --dump-rq -n 0  # just the 16 words per layer
```

Measured on `model/saved/int4_d64_f256_l4.pt`, 256 problems: **100.00%
exact-sequence, 100.00% token** — identical to the PyTorch QAT model it came
from, with 0 of 4352 scored argmax positions differing.

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
