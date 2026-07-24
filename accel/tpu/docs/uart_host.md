# UART Host–Memory Interface

A minimal serial link that lets a host PC **read and write the external SRAM**
("DRAM", `sram.sv`) directly, with no involvement from the scalar unit or DMA
engine. Its purpose is bring-up / debug / preload: push a program image and input
tensors into DRAM before a run, and read back results afterward, over a single
UART.

This is the host-side counterpart to `comms.md` (which is *device-to-device*).
Here the host is the master and the FPGA is a pure memory slave.

## 1. Scope (v1)

- **Two operations only: read and write a byte range of SRAM.** No register access,
  no run/halt control, no scratchpad access. Those can layer on later as extra
  command codes.
- **Byte-serial**, matching everything downstream: the SRAM is one byte / multi-cycle
  per access (`sram.sv`), and the DMA engine is already byte-serial (`dma.md`). There
  is no point buffering wider than the SRAM can drain.
- **Host is the sole master.** The FPGA never initiates a frame; it only answers.
- Reuses the existing `uart_receiver.sv` and `uart_transmitter.sv` unchanged
  (8N1, 115200 baud, `CLK_PER_BIT = 868` @ 100 MHz).

Out of scope for v1: CRC/checksums, flow control beyond the natural UART back-pressure,
multi-bank addressing, and any access while the core is running (see §6).

## 2. Physical layer

| Property   | Value                                             |
| ---------- | ------------------------------------------------- |
| Encoding   | UART 8N1 (1 start, 8 data LSB-first, 1 stop)      |
| Baud       | 115200 (parameter `CLK_PER_BIT`, default 868)     |
| Flow ctrl  | none at the wire; framing is command-length-based |
| Throughput | ~11.5 KB/s → full 512 KB SRAM ≈ 45 s              |

The device decodes one byte at a time as `uart_receiver` raises `valid`, and emits
reply bytes through `uart_transmitter` (pulse `start` while `!busy`).

## 3. Address / data sizes

The SRAM is **512 K × 8** (ISSI IS61/64WV5128): a **19-bit** byte address and an
**8-bit** data word.

- **Address** is carried as **3 bytes = 24 bits, big-endian (MSB first)**. Only
  `addr[18:0]` are used; the top 5 bits must be 0. Any frame with non-zero top
  5 bits, or whose `addr + len` would exceed `2^19`, is rejected (NAK).
- **Length** is carried as **2 bytes = 16 bits, big-endian**, so a single command
  moves 1..65535 bytes. `len == 0` is rejected. Larger transfers are issued as
  back-to-back commands by the host.
- **Data** is one byte per SRAM word, sent/received in ascending address order; the
  interface auto-increments the address internally, so the host sends the base
  address only once.

Big-endian is chosen so a frame reads left-to-right the way the address is written
(`0x01_23_45`), and to match the flit field ordering in `comms.md`.

## 4. Frame formats

### 4.1 Command frame (host → device)

```
byte:   0      1     2     3      4     5      6 .. 6+len-1
      +-----+-----+-----+-----+-----+-----+---------------------+
write | CMD | A2  | A1  | A0  | L1  | L0  |  data[0 .. len-1]   |
      +-----+-----+-----+-----+-----+-----+---------------------+

byte:   0      1     2     3      4     5
      +-----+-----+-----+-----+-----+-----+
read  | CMD | A2  | A1  | A0  | L1  | L0  |
      +-----+-----+-----+-----+-----+-----+
```

| Field    | Bytes | Meaning                                          |
| -------- | ----- | ------------------------------------------------ |
| `CMD`    | 1     | `'W'` = 0x57 write, `'R'` = 0x52 read            |
| `A2..A0` | 3     | 24-bit address, MSB first; `addr[18:0]` used     |
| `L1..L0` | 2     | 16-bit byte count, MSB first; `1 ≤ len ≤ 65535`  |
| `data`   | len   | **write only**: payload bytes, ascending address |

The 6-byte header is fixed-length, so the FSM always knows exactly how many bytes to
expect before the command byte matters — the only branch is "does a data phase
follow" (write) or not (read).

### 4.2 Response (device → host)

- **Write:** after the last byte is committed to SRAM, the device sends **one status
  byte**: `ACK` = 0x06 on success, `NAK` = 0x15 on a rejected command.
- **Read:** the device streams **exactly `len` data bytes** in ascending address
  order, and **nothing else** — no header, no trailing status. A rejected read (bad
  address/len) returns a single `NAK` = 0x15 and *no* data; the host detects this by
  the short reply plus its own timeout.

Header-less read replies mean the host, which already knows `len`, can simply
`read(len)` from its serial port. The write reply is a single byte the host waits on
to serialize back-to-back commands.

### 4.3 Status byte values

| Name  | Value | Meaning                                         |
| ----- | ----- | ----------------------------------------------- |
| `ACK` | 0x06  | write completed, all bytes committed            |
| `NAK` | 0x15  | rejected: bad `CMD`, `len == 0`, or range error |

## 5. Device FSM (behavioral plan)

Single command channel, one op in flight.

```
IDLE
  └─ on valid byte → latch CMD
       CMD == 'R' or 'W' → RX_ADDR
       else              → send NAK, back to IDLE   (resync, see §7)

RX_ADDR   collect 3 bytes → addr[23:0]
RX_LEN    collect 2 bytes → len[15:0]
  └─ validate: addr[23:19] == 0, addr + len ≤ 2^19, len != 0
       fail → send NAK, IDLE

CMD == 'W':  WR_RX  →  WR_MEM  (loop len times)  →  send ACK  →  IDLE
CMD == 'R':  RD_MEM →  RD_TX   (loop len times)  →  IDLE
```

- **WR_RX** waits for a data byte (`valid`); **WR_MEM** issues one SRAM write
  (`start`, `we=1`, `addr`, `din`), waits `done`, increments `addr`, decrements `len`.
- **RD_MEM** issues one SRAM read (`start`, `we=0`, `addr`), waits `done`, latches
  `dout`; **RD_TX** pulses the transmitter `start` with that byte and waits for
  `!busy`, then increments `addr`, decrements `len`.
- Loop exits when `len` reaches 0.

No pipelining in v1: at 868 clocks/bit a UART byte takes ~8680 clocks while an SRAM
access takes ~4, so the UART is ~2000× slower and the SRAM is never the bottleneck.
A single holding byte in each direction suffices — no elastic buffering, no FIFO.

## 6. Integration & SRAM arbitration

In `tpu_top.sv` the `sram_controller` user port is currently owned by the DMA engine
(`u_dma` → `u_sram`). The UART interface needs that same port, so exactly one of
{DMA, UART host} may drive it at a time.

**v1 policy: the UART host only touches SRAM while the core is idle** (`busy == 0`,
no program running). Recommended wiring:

- A 2:1 mux on the `sram_controller` user side, selected by "core idle". When idle the
  UART FSM drives `start`/`we`/`addr`/`din` and observes `dout`/`busy`/`done`; while a
  program runs, the DMA engine drives it as today.
- A command arriving mid-run is answered with `NAK` (see open questions).

This keeps the host link a pure bring-up/debug path and avoids needing a coherency
story between host writes and in-flight compute.

## 7. Framing, errors, resync

There is **no delimiter byte** — framing is purely the fixed 6-byte header plus the
known `len`. If host and device desynchronize (host aborts mid-frame, line glitch),
they must be able to realign:

- **Inter-byte timeout.** If the FSM is mid-frame and no byte arrives for a
  programmable window (a few character times), it aborts the command and returns to
  `IDLE`. Parameter `RX_TIMEOUT` in clocks; `0` disables.
- **Unknown `CMD`** → `NAK` and return to `IDLE`, so a stray byte costs one bad
  command rather than permanent desync.
- The host resynchronizes by pausing longer than `RX_TIMEOUT` before issuing a fresh
  command.

Data integrity relies on the UART itself (short, direct link). A per-frame CRC and an
explicit `SYNC` preamble are the natural v2 additions if the link proves noisy.

## 8. Proposed module

`rtl/uart_host.sv` — instantiates `uart_receiver` + `uart_transmitter` plus the command
FSM; exposes the raw serial pins and an `sram_controller` **user-side** port (to be
muxed in `tpu_top`).

| Signal          | Dir | Width        | Meaning                                |
| --------------- | --- | ------------ | -------------------------------------- |
| `clk`, `rst_n`  | in  | 1            | 100 MHz core clock, active-low reset   |
| `uart_rx`       | in  | 1            | serial in from host                    |
| `uart_tx`       | out | 1            | serial out to host                     |
| `sram_start`    | out | 1            | SRAM controller `start`                |
| `sram_we`       | out | 1            | 1 = write                              |
| `sram_addr`     | out | `MEM_ADDR_W` | 19-bit SRAM byte address               |
| `sram_din`      | out | `MEM_DATA_W` | write data byte                        |
| `sram_dout`     | in  | `MEM_DATA_W` | read data byte                         |
| `sram_busy`     | in  | 1            | SRAM controller busy                   |
| `sram_done`     | in  | 1            | SRAM access complete strobe            |
| `host_busy`     | out | 1            | a command is in progress (for the mux) |

Parameters: `CLK_PER_BIT` (passed to the UART blocks), `MEM_ADDR_W = 19`,
`MEM_DATA_W = 8`, `RX_TIMEOUT`.

## 9. Example exchanges

Write 3 bytes `AA BB CC` to address `0x00100`:

```
host → dev:  57  00 01 00  00 03  AA BB CC
dev  → host: 06                              (ACK)
```

Read 4 bytes from address `0x00100`:

```
host → dev:  52  00 01 00  00 04
dev  → host: AA BB CC DD                     (4 data bytes, no framing)
```

Bad command byte `0x99`:

```
host → dev:  99 ...
dev  → host: 15                              (NAK, FSM back to IDLE)
```

## 10. Test plan (`tb/uart_host_tb.sv`)

Drive `uart_rx` with the same bit-banging task style as `uart_receiver_tb.sv`, and
decode `uart_tx` with a matching sampler task. Instantiate `uart_host` against a
behavioral SRAM model (or the real `sram_controller` plus a simple array model) and
check:

1. Write N bytes, read them back, compare — the round-trip is the core test.
2. Single-byte write/read (`len == 1`) and a multi-byte burst crossing a 256-byte
   boundary (address carry).
3. `len == 0`, top-5-bits-set address, and `addr + len` overflow each return `NAK`
   and leave memory untouched.
4. Unknown command byte returns `NAK` and the FSM accepts a valid command next.
5. Mid-frame abort plus `RX_TIMEOUT` elapse, then a fresh command decodes correctly.
6. Back-to-back commands with no idle gap between them.

## 11. Open questions

- **Mid-run commands:** NAK immediately, or block the host until the core is idle?
  NAK is simpler and keeps the host in control; blocking is friendlier but needs a
  host-side timeout story.
- **Read error signalling:** a header-less read reply gives the host no in-band error
  channel (it infers failure from a short/absent reply). If that proves fragile,
  prefix read replies with a 1-byte status and make replies framed.
- **Verify-on-write:** optionally read back each byte after writing and fold a mismatch
  into `NAK`, trading throughput for a self-checking preload.
- **Cross-command auto-increment:** keep every command self-contained (current plan)
  vs. a "continue from last address" mode to shave header bytes on large sequential
  transfers.
- **Baud:** 115200 makes a full-SRAM preload ~45 s. Worth raising (921600 → ~6 s) once
  the link is proven, since `CLK_PER_BIT` is already a parameter.
