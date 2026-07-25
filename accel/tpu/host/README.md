# Host tools

Host-PC side of the TPU's serial link. Device side is
[`rtl/uart_interface.sv`](../rtl/uart_interface.sv); the protocol is
[`docs/uart_host.md`](../docs/uart_host.md).

`tpu_uart.py` — driver for all four commands the interface implements. Needs
`pyserial`; no other dependency and no imports from the rest of the repo, so it
runs standalone (`python accel/tpu/host/tpu_uart.py ...`) rather than as a
`-m` package module.

## Commands

| CMD       | Method                    | Frame                                  | Reply            |
| --------- | ------------------------- | -------------------------------------- | ---------------- |
| `R` 0x52  | `read_mem(addr, len)`     | `CMD A2 A1 A0 L1 L0`                   | `len` data bytes |
| `W` 0x57  | `write_mem(addr, data)`   | `CMD A2 A1 A0 L1 L0` + `data[len]`     | ACK / NAK        |
| `I` 0x49  | `load_program(waddr, ws)` | `CMD A2 A1 A0 L1 L0` + `data[len]`     | ACK / NAK        |
| `G` 0x47  | `go(pc)`                  | `CMD A2 A1 A0`                         | ACK / NAK        |

Address is 3 bytes big-endian, length 2 bytes big-endian **in bytes**. `R`/`W`
address 19-bit SRAM bytes; `I` addresses 10-bit instruction *word* indices and
its length must be a multiple of 4 (words packed MSB first); `G` has no length
or payload and pulses the scalar unit's run trigger with `run_pc = addr`.

Transfers longer than the 16-bit length field are split into back-to-back
commands automatically.

## Library

```python
from tpu_uart import TPUUart

with TPUUart("/dev/ttyUSB0") as tpu:          # 115200 8N1 (CLK_PER_BIT = 868 @ 100 MHz)
    tpu.write_mem(0x1000, input_tensor)       # preload DRAM
    tpu.load_program(0, [0x54000010, ...])    # or a packed big-endian bytes object
    tpu.go(0)                                 # start at pc 0
    result = tpu.read_mem(0x2000, 256)        # once the program has halted
```

Failures raise: `ValueError` for a frame the device would reject (caught before
anything is sent), `NakError` when the device rejects one anyway, `ReplyTimeout`
when the expected bytes never arrive.

## CLI

```bash
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 write 0x1000 --hex deadbeef
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 write 0x1000 --file acts.bin
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 read  0x1000 64        # hex dump
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 read  0x1000 64 -o out.bin
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 load ../tb/vectors/tpu_prog.hex --go
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 go 0
```

`load` takes the `$readmemh`-style program format used by `tb/vectors/` — one
32-bit hex word per line, `//` comments allowed.

## Two things to know

**Client-side validation is load-bearing, not belt-and-braces.** The FSM
validates a frame right after the 6-byte header and, on failure, sends NAK and
drops back to IDLE *without* consuming the data phase — so payload bytes
already in flight get re-decoded as command bytes. The driver applies the RTL's
exact address/length rules before sending, so an invalid frame never goes out;
a NAK on a write is therefore treated as a desync (link drained, error raised),
not as a routine rejection.

**The core has priority.** Any command arriving while the scalar unit is running
is NAK'd and touches nothing, so preload and readback only work while the core is
idle. There is no status-read command, so `go()` returns once the launch is
ACK'd and this link cannot poll `busy`/`done` — wait out-of-band (a known
run time, or an LED/pin) before reading results back.

## Not covered

- `resync()` drains the link and idles it long enough to trip the device's
  `RX_TIMEOUT` mid-frame abort. That parameter defaults to `0` (disabled) in
  `uart_interface.sv`; with it disabled, a device stuck mid-frame needs a reset.
- A rejected read with `len == 1` is undetectable — a lone `NAK` (0x15) is
  indistinguishable from one data byte of value 0x15. `len > 1` is detected.
- No CRC and no `SYNC` preamble; integrity relies on the UART itself, per
  `docs/uart_host.md` §7.
