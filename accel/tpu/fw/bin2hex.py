#!/usr/bin/env python3
"""bin2hex.py — flat firmware .bin -> one 32-bit little-endian word per line.

The firmware RAM (``rtl/cpu_subsys.sv``) is 32 bits wide and word-addressed, and
both of its loaders want words rather than bytes: the UART ``'I'`` command
(``host/tpu_uart.py`` ``load_program``) and ``$readmemh`` through the ``FW_INIT``
parameter. ``//`` comments are accepted by both, so the header line is safe.

    python bin2hex.py matmul.bin matmul.hex
"""

import os
import sys


def main(argv):
    if len(argv) != 3:
        print(f"usage: {os.path.basename(argv[0])} <in.bin> <out.hex>",
              file=sys.stderr)
        return 2

    with open(argv[1], "rb") as fh:
        data = fh.read()
    data += b"\x00" * (-len(data) % 4)      # whole words only

    with open(argv[2], "w") as fh:
        fh.write(f"// {os.path.basename(argv[1])}: {len(data) // 4} words "
                 f"({len(data)} bytes), word 0 at firmware address 0\n")
        for off in range(0, len(data), 4):
            fh.write(f"{int.from_bytes(data[off:off + 4], 'little'):08x}\n")

    print(f"{argv[2]}: {len(data) // 4} words ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
