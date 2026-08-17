# Synthesis & deployment (AMD/Xilinx Vivado)

How to get `rtl/` onto a **Digilent Cmod A7-35T** (`xc7a35t-cpg236-1`) and talk to it
from the host driver in `host/`.

The build is **non-project batch mode**: no Vivado project is version controlled, no
`.xpr` is ever opened. `synth/vivado/build.tcl` is the entire recipe, and everything it
produces lands in `synth/build/<board>/`, which is gitignored. A clean `git clone` plus
one command reproduces the bitstream.

> **Vivado version: 2024.2** (win64, build 5239630) — the version the checked-in
> `synth/build/cmod_a7/` results were produced with. Tcl commands and default synthesis
> strategies drift between releases; pinning the version is the difference between "it
> builds" and "it built once, for one person".

---

## 1. Quickstart

From anywhere (the script resolves its own paths):

```bash
cd accel/tpu/synth/vivado

# 1. Does it even elaborate?  (seconds)
vivado -mode batch -source build.tcl -tclargs mode=rtl

# 2. How big is it?  (minutes)
vivado -mode batch -source build.tcl -tclargs mode=synth

# 3. Full build to a bitstream
vivado -mode batch -source build.tcl

# 4. Flash the connected board
vivado -mode batch -source build.tcl -tclargs mode=program
```

Section 5 has the sizing expectations, including which blocks dominate and what to turn
if the design comes back over budget.

`mode=help` prints every argument.

### On Windows

`vivado` is `vivado.bat`, and cmd.exe's batch argument parser treats `=` as a delimiter
(like space or comma), so `-tclargs mode=rtl` reaches Tcl as the two words `mode` `rtl`.
`build.tcl` accepts that split form as well, so all three of these work:

```powershell
vivado -mode batch -source build.tcl -tclargs mode=rtl      # split by cmd, handled
vivado -mode batch -source build.tcl -tclargs "mode=rtl"    # quoted, arrives intact
vivado -mode batch -source build.tcl -tclargs mode rtl      # explicit space form
```

Vivado must be on `PATH` — run from the "Vivado HLx Command Prompt", or invoke
`C:\Xilinx\Vivado\<ver>\bin\vivado.bat` by full path.

---

## 2. What each file does

| File | Role |
| --- | --- |
| `synth/vivado/build.tcl` | The flow. Board-agnostic; parses args, dispatches on `mode=`. |
| `synth/vivado/sources.tcl` | The RTL file list — declared **once**, kept in the same order as `tb/Makefile`'s `UART_RTL` so the sim and synth source sets stay visibly in sync. |
| `synth/vivado/boards/cmod_a7/board.tcl` | Everything target-specific: part, top module, clock, geometry generics, flash part. |
| `synth/vivado/boards/cmod_a7/cmod_a7_top.sv` | Board wrapper. Ties off `tpu_top`'s parallel host port and brings out only clock/reset/LEDs/UART/SRAM. |
| `constraints/cmod_a7.xdc` | Pin assignment + the 12 MHz clock + timing exceptions. The **only** xdc the build reads. |
| `constraints/constraings-cmod.xdc` | Digilent's unmodified master, fully commented out. Pin reference only; not read by the build. |

To add a second board, copy `boards/cmod_a7/` and run `board=<name>`. Nothing in
`build.tcl` needs to change.

---

## 3. Modes

| `mode=` | Does | Use when |
| --- | --- | --- |
| `rtl` | `synth_design -rtl` — elaboration only, no netlist | Always run this first. Catches port/width/parameter mistakes in seconds. |
| `ooc module=<name>` | Out-of-context synth of one module | "How big is `mxu` on its own?" Isolates one block from the rest of the design. |
| `synth` | Full synthesis + utilization/timing reports | Area check before committing to a 20-minute place-and-route. |
| `impl` | synth + opt/place/`phys_opt`/route + reports | Timing closure work. |
| `bit` *(default)* | everything, ending in `<top>.bit` | Producing a bitstream. |
| `mcs` | `bit`, then a `.mcs` QSPI image | Making the design persist across power cycles. |
| `program` | Downloads an existing `.bit`. **Builds nothing.** | Re-flashing. Kept separate so re-flashing can't silently rebuild with different arguments than the `.bit` was made with. |

Reports always go to `synth/build/<board>/reports/`. The script prints a utilization
summary and an explicit **WNS/WHS verdict** to the log, because a design that misses
timing still writes a perfectly valid-looking bitstream — that check is the only thing
between you and a board that behaves nondeterministically.

---

## 4. Configuration

All overrides are `key=value` after `-tclargs`, order independent. Unknown keys are a
hard error, so typos fail loudly instead of being silently ignored.

```bash
vivado -mode batch -source build.tcl -tclargs mode=synth rows=4 cols=4 addr_w=13
vivado -mode batch -source build.tcl -tclargs mode=ooc module=vpu
vivado -mode batch -source build.tcl -tclargs part=xc7a15t-cpg236-1     # Cmod A7-15T
```

Defaults live in `boards/cmod_a7/board.tcl`:

| Knob | Default | Note |
| --- | --- | --- |
| `rows` / `cols` | `8` / `8` | **Not** `tpu_top`'s 128×128 defaults — see below. |
| `addr_w` | `16` | Scratchpad byte-address width → 64 KB. |
| `vpu_bytes` | `32` | 256-bit VPU port. **Not** `tpu_top`'s own default of 64. |
| `mem_addr_w` | `19` | Cmod A7 cellular SRAM is 512K×8. |
| `clk_mhz` | `12` | The board's only oscillator. |
| `cpb` | *derived* | `clk_mhz * 1e6 / baud` = **104**, not the 868 in `uart_host.md`. |

### Why 8×8 and not 128×128

`tpu_top`'s parameter defaults are 128×128, but `tb/tpu_top_uart_tb.sv` and
`tpulang/gen_vectors.py` are both written against **8×8**. Building at 8×8 means the
bitstream accepts exactly the programs and golden vectors that the end-to-end simulation
already passes. It is also the only size in the right order of magnitude for this part —
a 128×128 array is 16384 PEs each holding a 32-bit partial sum, i.e. ~524,000 flip-flops
against the A7-35T's 41,600.

### Why 12 MHz and not 100 MHz

The Cmod A7 has one 12 MHz oscillator. Feeding it straight to the core avoids an MMCM,
the Clocking Wizard, and any `.xci` IP in the repo — the constraint file is then the
entire timing setup and closure at an 83 ns period is essentially free. The cost is that
`UART_CPB` becomes 104 rather than the 868 that `uart_host.md` quotes for a 100 MHz core.
**The host side is unaffected** — `tpu_uart.py` still talks 115200 baud; only the on-chip
divisor changes, and `board.tcl` derives it from `clk_mhz` so the two cannot drift apart.

If you later add an MMCM, also revisit the SRAM false paths in the xdc — see the comment
there. At 12 MHz they are safe; at 100 MHz they would be hiding a real violation.

---

## 5. Sizing: what will and won't fit

> ### ⚠️ It does not currently fit. `mode=bit` fails in DRC before the placer runs.
>
> ```
> ERROR: [DRC UTLZ-1] LUT as Logic over-utilized in Top Level Design.
> This design requires 21657 of such cell types but only 20800 ... are available.
> ```
>
> **857 LUTs over (4.1%)**, at the default `board=cmod_a7 mode=bit`. This is a
> regression from the work between the post-route numbers below and today —
> `perf_counters.sv` and macro-op phases 0–5 (`matmul_t`'s hardware tile loop in
> particular) — not from the VPU trim, which moved it the *other* way:
>
> | `mode=bit` DRC, same board and geometry | LUT as Logic required |
> | --- | --- |
> | before the VPU trim | 26032 (5232 over) |
> | after the VPU trim | **21657** (857 over) |
>
> So the trim recovered 4375 LUTs and closed 84% of the gap. **The MXU is now where
> the remaining LUTs are**, not the VPU — post-synth it is 14476 LUTs against the
> VPU's 5162. `mode=synth` and `mode=ooc` still complete, so the design is
> measurable; only `impl`/`bit` are blocked.
>
> Options, cheapest first: build the `xc7a35t` at `rows=4 cols=4` (the geometry is a
> build argument, but the vectors and programs assume 8×8); revisit `matmul_t`'s tile
> loop; or move to a larger part (`part=xc7a50t-cpg236-1` is pin-compatible on some
> boards — unverified here).

**The numbers below are the last post-route measurement (Aug 8), when it did fit and
routed with timing met.** They are kept as the reference point the regression is measured
against, not as the current state. Geometry `boards/cmod_a7` (8×8, `addr_w=16`) on the
`xc7a35t-cpg236-1`:

| Resource | Used | Available | % |
| --- | --- | --- | --- |
| Slice LUTs | 19,227 | 20,800 | **92%** |
| Slice registers (FF) | 19,919 | 41,600 | 48% |
| Block RAM tiles | 33 (64×RAMB18 + 1×RAMB36) | 50 | 66% |
| DSP48E1 | 67 | 90 | 74% |

Timing: **WNS +35.9 ns** against the 83.3 ns (12 MHz) period, WHS +0.035 ns, zero failing
endpoints — "All user specified timing constraints are met." The clock is slow enough that
timing was never going to be the constraint; **LUTs are** — 92% then, over 100% now.

Per block (`report_utilization -hierarchical`, post-route):

| Block | LUTs | FFs | BRAM | DSP |
| --- | --- | --- | --- | --- |
| `mxu` | 10,845 | 19,029 | — | 16 |
| `vpu` | 4,514 | 438 | — | 48 |
| `scratchpad` | 2,269 | 10 | 64×RAMB18 | — |
| `dma` | 784 | 28 | — | — |
| `scalar_unit` | 542 | 65 | 1×RAMB36 | 3 |
| `uart_interface` + rx + tx | 263 | 287 | — | — |
| `sram_controller` | 16 | 51 | — | — |

**`mxu` dominates**, as predicted: 10.8k LUTs and 19k of the design's 19.9k flip-flops —
not the 8×8 array itself (~2k FFs of partial sums) but `resbuf[0:63][0:7]` of int32, which
is ~16k FFs on its own. `TOK_W` is the knob if area gets tight, and with LUTs at 92% it is
the first one to reach for before growing the array.

The estimates in the rest of this section were made before the first successful build;
they are kept because the *reasoning* is what matters when you change the geometry. Where
they differ from the table above, the table is measured and wins — notably `scratchpad`
came in at 2.3k LUTs against a 6–7k guess, and its BRAM prediction of 64×RAMB18 was exact.

**`scratchpad.sv` is now banked.** It used to be a flat 64 KB byte array with six
independent read ports and four write ports, each taking an unaligned byte window at a
runtime address — which does not fit any part in this family. Block RAM has two ports,
so `ram_style="block"` could not be honoured; Vivado fell back to registers (65536 × 8 =
**524,288 FFs** vs 41,600 available, ~12× over) or to LUTRAM replicated per read port
(~3 Mbit vs 400 Kbit), on top of 64 parallel 64 Ki-to-1 byte multiplexers for the `V`
and `DMA` ports.

The rewrite does the two things `docs/scratchpad.md` sketched:

- **Byte-lane banking.** `NBANK = 64` banks of 1024 × 8 bit, byte address `a` in bank
  `a[5:0]` at row `a[15:6]`. An unaligned 64-byte window takes one byte from each bank,
  needing only a `+1` row select on the banks that wrapped, plus a barrel rotate to
  unskew. Each bank is a genuine simple-dual-port BRAM.
- **Port arbitration.** Six readers muxed onto one read port, four writers onto one
  write port. Free, because the units are already mutually exclusive in time (MXU
  drives `A`/`W`/`C` from different FSM states; the scalar unit is issue-and-wait). No
  stall handshake, no changes to `mxu.sv` / `vpu.sv` / `scalar_unit.sv` / `dma.sv`, and
  the 1-cycle read latency is unchanged.

Rough expectation at 8×8, `addr_w=16` — **verify with `mode=ooc module=scratchpad`
rather than trusting these numbers**:

| | estimate | of A7-35T |
| --- | --- | --- |
| Block RAM | ~64 × RAMB18 | ~64% of 100 |
| LUTs (rotate networks) | ~6–7k | ~30% of 20,800 |

Storage efficiency is deliberately poor — 512 Kbit of data occupying ~1150 Kbit of
primitives — because each bank is only 8 Kbit deep and a RAMB18 is 18 Kbit. That is the
price of byte granularity at arbitrary alignment. The two 64-byte barrel rotates are the
LUT cost, and they are the part to attack if area gets tight: they exist purely to
support unaligned windows, so constraining the ISA's operand addresses to 64-byte
alignment would delete both.

**Suggested order of attack:**

1. `mode=rtl` — confirm the wrapper and generics elaborate.
2. `mode=ooc module=mxu`, then `scratchpad`, `vpu`, `dma`, `sram_controller`,
   `uart_interface`, `scalar_unit` — get real per-block numbers.
3. `mode=synth` on the whole design — check the totals against the sum.
4. `mode=impl`, read the WNS/WHS verdict, then `mode=bit`.

If area does come back over budget, the knobs in rough order of payoff are: `TOK_W` in
`mxu` (the `resbuf[0:63][0:7]` int32 buffer is ~16k FFs on its own), 64-byte alignment to
drop the rotates, then `addr_w` — though note `gen_vectors.py` puts the requant word at
`0x7000`, so `addr_w` cannot go below 15 without breaking the existing programs.

---

## 6. Bring-up after flashing

The wrapper exposes no parallel host port, so everything happens over the serial link —
the same path `tb/tpu_top_uart_tb.sv` validates.

LEDs give you the first signal: `led[0]` = core busy, `led[1]` = program halted.
`btn[0]` is an active-high manual reset.

```bash
cd accel/tpu

# Assemble a program and generate its golden vectors
python ../tpulang/gen_vectors.py -p ../tpulang/examples/relu_layer.tpu -o tb/vectors_uart

# Preload inputs into external SRAM, load the program, run it
python host/tpu_uart.py -p COM4 write 0x0 --file tb/vectors_uart/tpu_spad_in.bin
python host/tpu_uart.py -p COM4 load tb/vectors_uart/tpu_prog.hex --go

# Read results back and diff against tpu_spad_exp.hex
python host/tpu_uart.py -p COM4 read 0x0 256
```

Use the port the FT2232 bridge enumerates as (`COM<n>` on Windows, `/dev/ttyUSB*` on
Linux). Baud stays at the default 115200.

One known functional gap on hardware, matching current simulation behaviour rather than
being introduced by this flow:

- `nb_*` (inter-TPU LINK) is stubbed in `tpu_top`; a LINK op is a completing no-op.

The design no longer reads **any** `$readmemh` file by default. The VPU's activation LUTs
(`GELU_INIT` / `EXP_INIT` → `rtl/luts/{gelu,exp}_lut.hex`) went with the `gelu` / `exp`
instructions ([vpu.md §Removed ops](vpu.md#removed-ops)), and `luts.py` and `rtl/luts/`
were deleted. `SPAD_INIT` (an optional scratchpad preload) is the only such path left and
is empty.

`read_design` still cds into `accel/tpu` before reading the RTL, and that is still worth
keeping if you set `SPAD_INIT`: a `$readmemh` that misses only *warns* and leaves the
memory zero-filled, which looks exactly like logic that computes zero.

---

## 7. Why no `.xpr` in git

`.xpr`, `.runs/`, `.cache/`, `.sim/`, `.hw/`, `.ip_user_files/` are all regenerable,
version-sensitive, and merge badly — an `.xpr` is XML full of absolute paths and a Vivado
version stamp. Committing one means every collaborator inherits one machine's local
state, and a diff on it is unreviewable.

The reviewable surface is instead four small text files: `build.tcl`, `sources.tcl`,
`board.tcl`, and the `.xdc`. `.gitignore` at the repo root covers the rest, including the
debris Vivado drops into whatever directory it was launched from.

If you want to keep a working bitstream around, attach it to a GitHub Release rather than
committing it — `*.bit` and `*.mcs` are ignored on purpose.
