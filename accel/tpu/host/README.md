# Host tools

Host-PC side of the TPU's serial link. Device side is
[`rtl/uart_interface.sv`](../rtl/uart_interface.sv); the protocol is
[`docs/uart_host.md`](../docs/uart_host.md).

`tpu_uart.py` — driver for all four commands the interface implements. Needs
`pyserial`; no other dependency and no imports from the rest of the repo, so it
runs standalone (`python accel/tpu/host/tpu_uart.py ...`) rather than as a
`-m` package module.

`run_program.py` — the end-to-end runner built on that driver: assembles a
tpulang program, preloads it and its tensors, runs it, reads the results back
and checks them. See [Running a program](#running-a-program) below.

`test_uart_link.py` — self-checking tests for the link itself, no program and no
toolchain involved. See [Testing the link](#testing-the-link) below.

`uart_echo.py` — one layer below that: a 64-byte block loopback check against a
separate bitstream that contains nothing but the UART blocks. See
[Testing the UART alone](#testing-the-uart-alone) below.

## Commands

| CMD       | Method                    | Frame                                  | Reply            |
| --------- | ------------------------- | -------------------------------------- | ---------------- |
| `R` 0x52  | `read_mem(addr, len)`     | `CMD A2 A1 A0 L1 L0`                   | `len` data bytes |
| `W` 0x57  | `write_mem(addr, data)`   | `CMD A2 A1 A0 L1 L0` + `data[len]`     | ACK / NAK        |
| `I` 0x49  | `load_program(waddr, ws)` | `CMD A2 A1 A0 L1 L0` + `data[len]`     | ACK / NAK        |
| `G` 0x47  | `go(pc)`                  | `CMD A2 A1 A0`                         | ACK / NAK        |
| `T` 0x54  | `read_timer()`            | `CMD`                                  | 4 counter bytes  |

Address is 3 bytes big-endian, length 2 bytes big-endian **in bytes**. `R`/`W`
address 19-bit SRAM bytes; `I` addresses 10-bit instruction *word* indices and
its length must be a multiple of 4 (words packed MSB first); `G` has no length
or payload and pulses the scalar unit's run trigger with `run_pc = addr`.

`T` is one byte and cannot fail. It returns the scalar unit's run-length counter
(`rtl/cycle_timer.sv`) as 4 bytes MSB first: core clocks the last run took, or
how far the current one has got, measured on the FPGA between `busy` going high
and going low. Divide by the device clock (12 MHz on the Cmod A7) for seconds.
It is the **only** command answered while the core is running — see
[Two things to know](#two-things-to-know) — because it touches no memory. A
count that keeps rising means the core is still working; a count that has
stopped does *not* by itself mean the run finished (it also never moves if the
run never started), so completion is still inferred from a command that stops
being NAK'd.

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
python accel/tpu/host/tpu_uart.py -p /dev/ttyUSB0 timer      # clocks of the last run
```

`load` takes the `$readmemh`-style program format used by `tb/vectors/` — one
32-bit hex word per line, `//` comments allowed.

## Seeing the bytes

Both link failure modes (a desync, a NAK you didn't expect) are byte-level, so
the driver can log every byte in both directions — `-T/--trace` for a live
hexdump on stderr, `--trace-file FILE` to capture it instead:

```bash
python accel/tpu/host/tpu_uart.py -p COM5 -T write 0x1000 --hex deadbeef
```

```
   0.0000  --           open COM5 @ 115200 8N1, reply timeout 2.0s
   0.0013  TX      10B  write_mem[0x00001000+4]
           0000  57 00 10 00 00 04 de ad be ef                     |W.........|
   0.0498  RX       1B  write_mem[0x00001000+4]
           0000  06                                                |.|
   0.0498  --           write_mem[0x00001000+4]  ACK
   0.0499  --           close COM5
uart trace: 5 events, TX 10 B, RX 1 B, 0.050 s
```

`DROP` rows are bytes the driver found sitting in the input buffer and threw
away — the tell-tale of a previous command desyncing. From Python:

```python
from tpu_uart import TPUUart, UartTrace

with TPUUart("COM5", trace=True) as tpu:      # live to stderr
    ...

tpu = TPUUart("COM5", trace=UartTrace(sink=None))   # record quietly
try:
    tpu.load_program(0, words)
finally:
    tpu.dump_trace()                          # ...print it only on failure
```

`trace=` also takes any file object or `callable(line)` (e.g. `logging.info`).
`tpu.trace.enabled` flips recording at runtime, `tpu.trace.events` is the raw
list of `TraceEvent`s. The proxy sits on `tpu.ser` itself, so raw frame pokes
like the ones in `test_uart_link.py` show up in the trace too.

## Running a program

`run_program.py` is the whole flow in one command — the hardware counterpart of
`tb/tpu_top_uart_tb.sv`:

```bash
python accel/tpu/host/run_program.py -p COM5                 # tiled_matmul.tpu
python accel/tpu/host/run_program.py -p COM5 --verify-inputs # read the tensors back too
python accel/tpu/host/run_program.py --program ../../tpulang/examples/relu_layer.tpu -p COM5
python accel/tpu/host/run_program.py --dry-run               # toolchain only, no board
python accel/tpu/host/run_program.py -p COM5 --no-torch      # skip the PyTorch check
```

1. assemble the `.tpu` source (`tpulang/assembler.py`);
2. build the DRAM input image and the golden output image with the *same*
   builders `tpulang/gen_vectors.py` uses for the testbench vectors — host and
   simulation cannot drift on layout;
3. `I` the program into IMEM, `W` the tensors into DRAM, `G` at PC 0;
4. wait for the core to go idle, `T` how many core clocks the run took, `R` the
   outputs back, compare to the golden image (and, for the tiled matmul, print
   the [8x32] @ [32x16] result matrix);
5. compare the same bytes against an independent **PyTorch reference** —
   [Checking the kernel](#checking-the-kernel-against-pytorch) below.

Defaults to `tpulang/examples/tiled_matmul.tpu`; any example works, since the
image builder is chosen from the program's own `.equ` constants exactly as
`gen_vectors.py` chooses it. It needs the toolchain on disk (it imports
`assembler` / `iss` / `gen_vectors` by path), unlike `tpu_uart.py` itself.

### Checking the kernel against PyTorch

Step 4's golden image comes out of `iss.py`, so on its own it only proves the
FPGA agrees with the simulator the rest of the toolchain is built on — if the
kernel and the ISS are wrong the same way, it still passes.
[`tpulang/torch_ref.py`](../../tpulang/torch_ref.py) is the second, independent
opinion: it decodes tensors out of the bytes actually written to and read back
from the board, recomputes the kernel in plain PyTorch, and compares element by
element. It never calls the ISS, and it takes the DRAM layout from the program's
own `.equ` symbol table rather than restating it.

The comparison is **exact** — these kernels are integer end to end (int8
operands, int32 accumulate, fixed-point requant), so there is no tolerance to
allow. Output looks like:

```
torch   : tiled_matmul vs PyTorch, over the device output
  [OK  ] C = requant(A @ W)  [8x32] @ [32x16]  int8  (8, 16)  0/128 differ, max|diff| = 0
PASSED: the device output matches the PyTorch reference
```

| Program            | Reference                                                  |
| ------------------ | ---------------------------------------------------------- |
| `relu_layer.tpu`   | `Y = requant(A @ W)` int8, `Z = relu(Y)` int32             |
| `vector_add.tpu`   | `C = A + B`, int8 operands widened to int32                |
| `tiled_matmul.tpu` | `C = requant(A @ W)`, tiles reassembled into one `[M,K]@[K,N]` |
| `vpu_matmul.tpu`   | `S32 = Q @ K^T` int32, `S8 = requant(S32)` int8 — attention scores |
| `vecmatmul.tpu`    | the same score block as one `vecmatmul` macro op            |
| `transpose_dma.tpu`| `Vᵀ` / `Qᵀ` out of a fused `[T][3D]` block                   |
| `highmem_dma.tpu`  | linear + transposed spills above the low 64 KB of DRAM      |
| `adder_model.tpu`  | the whole model: per-layer `X`, `Vᵀ`, `A`, and the logits   |

Every example now has a reference — `torch_ref.UNSUPPORTED` is empty. (Its one
entry was `softmax_row.tpu`, deleted along with the `exp`/`redsum`/`sdiv`
instructions it was written against; see
[docs/vpu.md §Removed ops](../docs/vpu.md#removed-ops).)

A program with no reference, or a host without `torch`, prints a `skipped` line
and leaves the golden-image verdict standing. `--no-torch` turns the stage off.

To score the real checkpoint problem-by-problem rather than check one program's
bytes, use [`run_adder.py`](run_adder.py) instead.

The same references run without a board, against the ISS instead of the device —
useful to confirm the reference itself before spending a board run:

```bash
python accel/tpulang/torch_ref.py                          # every example with a reference
python accel/tpulang/torch_ref.py -p examples/relu_layer.tpu
python accel/tpu/host/run_program.py --dry-run             # same check, via the runner
```

**How it waits for the run.** There is no status command, but there is the
arbitration rule below: while the core runs, everything is NAK'd. So the runner
probes with a harmless 2-byte read — NAK means "still running", data means
"idle, results are readable". `--run-timeout` bounds the wait; on the Cmod A7 the
same state is on `led[1]` (`done`). Port defaults to the one FTDI device found
(`--port` if there are several).

Geometry must match the bitstream: `boards/cmod_a7/board.tcl` builds the 8x8
array these programs and vectors are written against.

## Testing the link

`test_uart_link.py` answers "is the link and the SRAM good?" separately from "is
the core computing the right answer?" — run it first when a board misbehaves.
Every test writes its own expected data and checks the readback, so a run is
pass/fail with nothing to eyeball.

```bash
python accel/tpu/host/test_uart_link.py -p COM5              # ~10 s
python accel/tpu/host/test_uart_link.py -p COM5 --slow       # + the >64 KiB transfer
python accel/tpu/host/test_uart_link.py -p COM5 --length 16384 --base 0x10000
python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
python accel/tpu/host/test_uart_link.py --offline            # driver checks, no board
```

| Test                       | What it catches                                            |
| -------------------------- | ---------------------------------------------------------- |
| `sram_roundtrip`           | `W` then `R` a block, six data patterns — the core check    |
| `sram_isolation`           | one command's write landing in another's region             |
| `sram_address_bus`         | shorted/open address lines (one byte per one-hot address)   |
| `sram_short_frames`        | off-by-one in the FSM's `idx + 1 == len` termination        |
| `link_rejects_bad_command` | unknown command byte → NAK, and the FSM returns to IDLE     |
| `link_rejects_out_of_range`| the device's VALIDATE rules, with the host's check bypassed |
| `link_timer_command`       | `T` replies with 4 bytes, holds still while the core is idle |
| `sram_long_transfer`       | (`--slow`) the driver's >64 KiB frame splitting             |
| `host_validation`          | (offline) host frame rules still a superset of the RTL's    |
| `frame_encoding`           | (offline) header/word endianness — no readback can catch it |

Two things it assumes. **It overwrites SRAM**, all of it (`sram_address_bus`
touches the top of the space) — nothing on the board survives a run, which is
fine because `run_program.py` reloads its tensors every time. **The core must be
idle**, since a running program NAKs every command; the suite checks that up
front and stops with a message instead of reporting 20 bogus failures.

Not covered: anything requiring a program to be running (the `core_busy`
arbitration path, `I` + `G` end to end) — that is `run_program.py`'s job, and
`tb/tpu_top_uart_tb.sv`'s in simulation.

## Testing the UART alone

`test_uart_link.py` still exercises seven blocks at once: a bad byte there could
come from `uart_receiver`, `uart_transmitter`, `uart_interface`, the SRAM
controller, the arbitration mux, the cable or the host. `uart_echo.py` deletes
four of those by talking to a **different bitstream** — `cmod_a7_echo`, which
contains the two UART blocks, a 64-byte register file and nothing else. Full
details and the design rationale are in
[`docs/uart_selftest.md`](../docs/uart_selftest.md).

```bash
# build and flash the echo image (once)
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_echo mode=bit
vivado -mode batch -source synth/vivado/build.tcl -tclargs board=cmod_a7_echo mode=program

python accel/tpu/host/uart_echo.py -p COM5                # 30-second run
python accel/tpu/host/uart_echo.py -p COM5 --minutes 30   # soak until it breaks
python accel/tpu/host/uart_echo.py -p COM5 --baud 117000  # +1.6% sampling-margin check
python accel/tpu/host/uart_echo.py --offline              # check the forensics, no board
```

The device is store-and-forward: it buffers **64 bytes** in registers, sends
those 64 back, and repeats. The block length is the entire protocol, so
`--block` must match the synthesised `BLOCK_LEN` — the host checks that at
startup and says so rather than leaving you to infer it from a byte diff. It
first walks the device's block counter to a known position (`resync`), since a
half-finished block from an interrupted run would otherwise make every
subsequent exchange short.

The exchange is half duplex by construction, so unlike the earlier streaming
echo it does not test both directions live at once; what it does test is the
**turnaround** — burst in, gap, burst out, next block's first byte arriving right
behind the reply — which is the shape of the real protocol's traffic. A byte that
arrives while the reply is going out is dropped and latched on `led[0]`.

On a mismatch it reports whether the received stream is the sent stream
shifted by a whole **bit** (a mis-framed byte), by a whole **byte** (a lost or
invented frame), or by neither (a single mis-sampled bit) — three different bugs
that a byte-by-byte diff renders identically.

`--baud` is the cheap experiment: 8N1 tolerates about ±5% of bit-period error, so
sweeping the host a few percent either side of 115200 measures where the receiver
actually falls over. Asymmetric margin ⇒ the sample point is off-centre and the
fix is arithmetic. Symmetric and wide, but still failing at 0% ⇒ noise or
metastability.

This image has no SRAM and no core, so **reflash `cmod_a7` before running
anything else** — `test_uart_link.py` and `run_program.py` will time out against
it (it says nothing at all until 64 bytes have piled up, and then replies with
those bytes rather than obeying any command).

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
idle. `T` is the one exception: it reads a counter, contends over nothing, and is
answered mid-run. There is still no `busy`/`done` status command, so `go()`
returns once the launch is ACK'd and completion is inferred out-of-band (a known
run time, an LED/pin, or a command that stops being NAK'd) before reading results
back. `read_timer()` tells you how long the core has been running, not whether it
stopped.

## Not covered

- `resync()` drains the link and idles it long enough to trip the device's
  `RX_TIMEOUT` mid-frame abort. That parameter defaults to `0` (disabled) in
  `uart_interface.sv`; with it disabled, a device stuck mid-frame needs a reset.
- A rejected read with `len == 1` is undetectable — a lone `NAK` (0x15) is
  indistinguishable from one data byte of value 0x15. `len > 1` is detected.
- No CRC and no `SYNC` preamble; integrity relies on the UART itself, per
  `docs/uart_host.md` §7.
