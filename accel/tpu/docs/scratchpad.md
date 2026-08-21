# Scratchpad Memory

On-chip working memory for the TPU, built from FPGA BRAM. Every compute unit (MXU,
VPU, scalar unit) reads and writes the scratchpad; DRAM is only touched through DMA.
See [README](README.md) §1 for the high-level role.

## 1. Purpose

- Hold the **working set** for the layer currently executing: the ternary weight tile
  loaded into the MXU, the int8 activation tile being multiplied, and the int32
  partial/result tiles.
- Provide enough read bandwidth to feed **one full activation column per clock** into
  the MXU (the bandwidth constraint from the overview).
- Serve as the shared operand/result space for the ISA — every math instruction names
  scratchpad addresses, never DRAM addresses (§ Scalar Unit).

DRAM is the backing store for full model weights (~288 KB for the ternary adder, see
sizing below); the scratchpad streams weights in per-layer via `ReadMemory` DMA.

## 2. Datatypes and word formats

| Region       | Element        | Encoding                                    |
| ------------ | -------------- | ------------------------------------------- |
| Activations  | int8           | signed two's-complement                     |
| Weights      | ternary        | 2-bit: `00`=0, `01`=+1, `11`=−1 (`10` unused)|
| Accumulators | int32          | MXU/VPU results before requantization       |
| Scalars      | int32          | scalar-unit operands                         |

Requantization (int32 → int8) happens on the store path out of the MXU/VPU, not in
the scratchpad; the scratchpad stores whatever width the writer produces.

## 3. Address space and banking

The scratchpad is **byte-addressed** but physically organized as parallel banks so a
whole column can be read in one cycle.

- **Activation banking.** A contraction dimension of up to `d = 128` int8 elements =
  **1024 bits** must be read per clock. Organize activations across `NBANK_A = 16`
  BRAM blocks, each 64 bits wide, addressed by the same column index. One read returns
  the full 128-element column; the array's staggered feed (see [mxu.md](mxu.md)) then
  skews it in time.
- **Weight banking.** Ternary weights load into PE registers once per tile and stay
  stationary, so weight reads are not on the per-clock critical path. Weights are
  packed 2-bit and streamed in over `LOAD_LATENCY` cycles while the previous tile
  drains.
- **Result banking.** MXU results are int32; a `COLS`-wide result row is
  `COLS × 32` bits. Give results their own bank group so a result write and an
  activation read can proceed in the same cycle (no structural hazard between
  consecutive matmul tiles).

### Suggested map (parameters, not final)

```
0x0000               activation tile A     (T × d int8)
0x1000               weight tile W         (d × COLS ternary, 2-bit packed)
0x3000               result / accumulator  (T × COLS int32)
0x6000               VPU scratch / vectors
0x7000               instruction operands, norm stats, softmax temporaries
```

Base offsets are `PARAM`s resolved at synthesis from the array size and context length.

## 4. Ports

| Port    | Width      | Users                        | Notes                                  |
| ------- | ---------- | ---------------------------- | -------------------------------------- |
| `A_rd`  | 1024 bit   | MXU activation feed          | one column/clock, banked               |
| `W_rd`  | 256 bit    | MXU weight load              | 2-bit packed, active during tile load  |
| `C_wr`  | COLS×32    | MXU result store             | int32 result row                       |
| `V_rw`  | 512 bit    | VPU SIMD lanes               | read/modify/write for pointwise ops    |
| `S_rw`  | 32 bit     | scalar unit                  | single-element scalar access           |
| `DMA`   | bus width  | DMA engine (`Read/WriteMem`) | DRAM ⇄ scratchpad, background          |

Arbitration is static where possible: the MXU owns `A_rd`/`C_wr` for the duration of a
`Matmul`, the VPU owns `V_rw`, and DMA is scheduled into idle cycles by the scalar unit.
Genuinely conflicting accesses to the same bank group stall the lower-priority requester.

## 5. Sizing for the target model

Per-layer weight footprint (ternary, 2-bit packed), `d=128`, `f=512`:

| Tensor      | Shape     | Trits   | Packed   |
| ----------- | --------- | ------- | -------- |
| Wq/Wk/Wv/Wo | 128 × 128 | 16 384  | 4 KB ea  |
| FF fc1      | 128 × 512 | 65 536  | 16 KB    |
| FF fc2      | 512 × 128 | 65 536  | 16 KB    |
| **Layer**   |           |         | **48 KB**|
| **6 layers**|           |         | **288 KB**|

Activation working set is tiny: `T × d` int8 with `T ≤ 32` ⇒ `32 × 128 = 4 KB`. The
scratchpad therefore needs to hold **one layer of weights + a couple activation tiles**
(~56 KB) at once; the full 288 KB lives in DRAM and is double-buffered per layer so DMA
of layer *n+1*'s weights overlaps compute of layer *n*.

## 6. RTL implementation (`rtl/scratchpad.sv`)

`scratchpad.sv` is the single owner of the on-chip storage. It exposes the typed
ports from §4 with exactly the signal names the compute units already drive, so a
top-level wires them straight through:

| Port  | Direction | Driven by                                   |
| ----- | --------- | ------------------------------------------- |
| `A_rd`| read      | `mxu.sv` `A_re / A_raddr / A_rdata`         |
| `W_rd`| read      | `mxu.sv` `W_re / W_raddr / W_rdata`         |
| `C_rw`| read/write| `mxu.sv` `C_re… / C_we…` (result + accumulate readback) |
| `V_rw`| read/write| `vpu.sv` `V_re… / V_we…`                    |
| `S_rw`| read/write| `scalar_unit.sv` `s_re / s_we / s_addr…` (shared read/write address) |
| `DMA` | read/write| DMA engine (future block)                   |

**Contract.** Byte-addressed; each port gathers/scatters its own byte width
(parameters `A_BYTES`, `W_BYTES`, `C_BYTES`, `V_BYTES`, `S_BYTES`, `DMA_BYTES`,
defaulting to a 128×128 array). Reads are **synchronous** — `*_rdata` valid the
cycle after `*_re`. Writes are single-cycle under a per-byte strobe. A read and
write to the same byte in one cycle returns the **old** value (read-first). A
window running off the top of the memory wraps mod `2**ADDR_W`.

**One read and one write per cycle.** The six read ports are muxed onto a single
banked-BRAM read port (priority `A > W > C > V > S > DMA`), the four write ports
onto a single write port (`DMA > S > V > C`, reproducing the old flat model's
last-writer-wins order). A read and a write still proceed together — they are the
two ports of the same BRAM — but two simultaneous reads, or two simultaneous
writes, are not supported.

`W_re` and `C_re` from mutually exclusive FSM states (`S_LOAD` / `S_RUN` /
`S_WB_*`), and the VPU likewise reads and writes in different states. **What is no
longer true is that the *units* are mutually exclusive with each other.** That
held only while the scalar unit was issue-and-wait -- a dispatch parked it in
`S_WAIT` until the unit reported done, so the MXU, the VPU and the DMA could not
overlap -- and per-unit command queues (`cmd_*.sv`) exist precisely to break it.

So the priority mux now genuinely arbitrates, and every requester that can lose
takes a grant back: `V_rgnt`/`V_wgnt`, `s_rgnt`/`s_wgnt`, `dma_rgnt`/`dma_wgnt`.
Both chains are ordered so the requester that *cannot* stall is first --

    reads   A > W > C > V > s > DMA          writes   C > V > s > DMA

-- which puts all three MXU ports at the top, since its writeback drains
`result_buf` with no handshake at all. The VPU freezes its FSM for a clock when
denied, the scalar/CPU port re-presents, and the DMA parks the byte in a skid
buffer and pauses the SRAM read stream (`dma.sv`, `sram.sv`'s `dout_ready`).
See [picorv32_migration.md](picorv32_migration.md) §5.

One visible consequence: `*_rdata` are slices of a single shared output register,
so a read on any port updates all of them. Each consumer samples one cycle after
its own enable with nothing else reading, so behaviour is unchanged — but the
ports are no longer independently held.

**Banking.** Storage is split into `NBANK` one-byte-wide banks (`NBANK` = widest
port rounded up to a power of two), byte address `a` living in bank
`a[OFF_W-1:0]` at row `a[ADDR_W-1:OFF_W]`. An unaligned `NBANK`-byte window then
takes exactly one byte from every bank: bank `b` needs row `row0` when `b >= off`
and `row0 + 1` when `b < off`, so the per-bank address is a 1-bit adjustment
rather than an arbitrary index. The gathered bytes come back rotated by `off`,
undone by one barrel rotate on the read path (and the mirror rotate applied on
the write path). That replaces `NBANK` 64 Ki-to-1 byte multiplexers with `NBANK`
dual-port BRAMs plus two log₂(`NBANK`)-stage rotate networks — the difference
between "does not fit on any part" and "fits in about two thirds of an A7-35T's
block RAM". See [synth.md](synth.md) §5.

**Registers vs BRAM.** Banking is unconditional; `MEM_STYLE` only picks the
primitive each bank is built from, with identical functional/timing behaviour
either way:

| `MEM_STYLE` | `ram_style` hint | Inferred primitive            | Use when                     |
| ----------- | ---------------- | ----------------------------- | ---------------------------- |
| `"BRAM"` (default) | `block`   | FPGA block RAM                | anything board-sized         |
| `"REG"`     | `registers`      | flip-flops                    | shallow unit-test depths only |

At `ADDR_W=16`, `"REG"` is the flip-flop explosion the banking exists to avoid;
it is there so `scratchpad_tb.sv` can cross-check the two elaborations.

Verify with `make TEST=scratchpad sim` (`tb/scratchpad_tb.sv` instantiates the
`"BRAM"` and `"REG"` variants side by side and checks both against a byte-array
reference, covering strobed writes, cross-port coherence, an unaligned sweep over
every byte offset in a bank row, address wrap, back-to-back reads on different
ports, and read-first).

## 7. Open questions

- BRAM primitive width on the chosen board (36 Kb vs 18 Kb blocks) sets the real
  `NBANK_A`; revisit once a board is fixed in `constraints/`.
- Whether ternary weights are stored 2-bit packed in BRAM or expanded to a sign+zero
  pair at load time (trades BRAM for routing).
- Whether to keep KV-cache tokens in scratchpad or spill to DRAM once `T` approaches the
  context limit.
