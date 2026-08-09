#!/usr/bin/env python3
"""luts.py — the VPU's GELU / EXP activation tables.

``vpu.sv`` implements ``gelu`` and ``exp`` as 256-entry int8 ROMs indexed by the
*raw operand byte*::

    res32 = gelu_rom[idx8]        idx8 = the int8 operand, read unsigned

There is no arithmetic in the hardware at all — **the table is the op's
numerics**. That has one consequence worth stating up front, because it is the
whole contract of this file:

    A lookup table can only be built if the meaning of an int8 input byte is
    fixed in advance. So it is: the tables below are generated at a *canonical
    input scale*, pinned here and burned into the bitstream. `gelu`/`exp` do not
    take an operand scale, they **assume** one.

An operand reaches a table in its input scale by an ordinary ``requant`` the
compiler inserts ahead of the op; getting there is the compiler's job, not the
hardware's. Feed `exp` a buffer at some other scale and it will not fail, it
will quietly evaluate ``exp`` of the wrong number.

Canonical scales
----------------

Index ``u`` (0..255) is the operand byte, so the real number it stands for is
``x = s8(u) * in_scale``, and the entry stored is
``clip_int8(round(f(x) / out_scale))``.

===== ============== ========== =========== ==============================
table x range         in_scale   out_scale   notes
===== ============== ========== =========== ==============================
gelu  [-8, +7.9375]  1/16       1/16        same scale in and out, so the op
                                            is scale-preserving: whatever
                                            requant put an activation into
                                            gelu's scale for is still valid
                                            for the result. Saturation is
                                            not reachable (|gelu(x)| <= |x|),
                                            and the dip at x ~ -0.75 survives
                                            as -3/16.
exp   [-8, 0]        1/16       1/127       softmax evaluates exp on
                                            ``x - max`` <= 0, so only the
                                            negative half is real. exp(0) = 1
                                            -> +127 puts the peak at int8's
                                            top; exp(-8) = 3.4e-4 -> 0 is the
                                            floor, and entries below
                                            x = -5.54 are 0 (an underflow
                                            the format cannot avoid: those
                                            weights are < 0.4% of the peak).
                                            **Positive indices all saturate
                                            at 127** — they are unreachable
                                            after ``x - max``, and a program
                                            that reaches them has forgotten
                                            the subtract.
===== ============== ========== =========== ==============================

Both input scales are 1/16 = 2**-4, so a compiler emitting the requant ahead of
an activation is choosing ``{m0, n}`` against a power of two.

``gelu`` is the exact (erf) form, ``0.5x(1 + erf(x/sqrt2))`` — that is what
``nn.GELU()`` / ``F.gelu`` default to in ``model/transformer.py``, so the table
tracks the golden reference rather than the tanh approximation.

Rounding is round-half-up (``floor(v + 0.5)``), which is what ``requant8``'s
``(v*m0 + (1 << (n-1))) >> n`` does everywhere else in the datapath — halves go
toward +inf, not away from zero.

Outputs
-------

``$readmemh`` files under ``accel/tpu/rtl/luts/``, one byte per line, loaded by
``vpu.sv``'s ``GELU_INIT`` / ``EXP_INIT`` parameters. The same two lists are
importable as :data:`GELU_LUT` / :data:`EXP_LUT` for ``iss.py``, which is how
the simulator stays bit-exact with the ROM contents.

    python luts.py                 # (re)generate both .hex files
    python luts.py --check         # verify the checked-in files still match
    python luts.py --dump gelu     # print the table as x -> f(x), int8
"""

from __future__ import annotations

import argparse
import math
import os

# --- canonical scales — the contract above; changing one invalidates every -----
# --- bitstream and every compiled program that feeds these ops ----------------
GELU_IN_SCALE = 1 / 16
GELU_OUT_SCALE = 1 / 16
EXP_IN_SCALE = 1 / 16
EXP_OUT_SCALE = 1 / 127

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "tpu", "rtl", "luts"))
GELU_HEX = "gelu_lut.hex"
EXP_HEX = "exp_lut.hex"


def clip8(v: int) -> int:
    return 127 if v > 127 else (-128 if v < -128 else v)


def round_half_up(v: float) -> int:
    """Round to nearest, halves toward +inf — matching requant8's shift."""
    return math.floor(v + 0.5)


def s8(u: int) -> int:
    """The signed value of operand byte ``u`` — the table's index is unsigned."""
    return u - 0x100 if u >= 0x80 else u


def gelu(x: float) -> float:
    """Exact GELU, x * Phi(x). ``nn.GELU()``'s default (approximate='none')."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_table(fn, in_scale: float, out_scale: float) -> list[int]:
    """Quantize ``fn`` over all 256 operand bytes into a 256 x int8 table."""
    return [clip8(round_half_up(fn(s8(u) * in_scale) / out_scale))
            for u in range(256)]


GELU_LUT = build_table(gelu, GELU_IN_SCALE, GELU_OUT_SCALE)
EXP_LUT = build_table(math.exp, EXP_IN_SCALE, EXP_OUT_SCALE)

TABLES = {
    "gelu": (GELU_LUT, GELU_HEX, gelu, GELU_IN_SCALE, GELU_OUT_SCALE),
    "exp": (EXP_LUT, EXP_HEX, math.exp, EXP_IN_SCALE, EXP_OUT_SCALE),
}


# --- $readmemh emit / verify --------------------------------------------------

def fmt_scale(s: float) -> str:
    """`1/16`, not `0.0625` — these are all unit fractions by construction."""
    return f"1/{round(1 / s)}"


def hex_lines(name: str) -> list[str]:
    """The exact file body for ``name``, header comment included.

    Plain ASCII: this file is parsed by iverilog and by Vivado's ``$readmemh``,
    and neither is worth trusting with a stray non-ASCII byte in a comment.
    Deterministic (no timestamps), so a regenerated file is byte-identical unless
    the numerics actually changed and ``--check`` can be a plain comparison.
    """
    table, _fname, _fn, in_scale, out_scale = TABLES[name]
    lines = [
        f"// {name} activation LUT for vpu.sv: 256 x int8, indexed by the raw",
        f"// operand byte (unsigned). in_scale = {fmt_scale(in_scale)}, "
        f"out_scale = {fmt_scale(out_scale)}:",
        f"//     entry[u] = clip_int8(round({name}(s8(u) * in_scale) / out_scale))",
        "// Generated by accel/tpulang/luts.py -- do not edit; run that instead.",
    ]
    lines += [f"{v & 0xFF:02x}" for v in table]
    return lines


def write_hex(name: str, out_dir: str) -> str:
    path = os.path.join(out_dir, TABLES[name][1])
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(hex_lines(name)) + "\n")
    return path


def check_hex(name: str, out_dir: str) -> tuple[str, bool]:
    path = os.path.join(out_dir, TABLES[name][1])
    want = "\n".join(hex_lines(name)) + "\n"
    try:
        with open(path, encoding="ascii", newline="") as f:
            got = f.read().replace("\r\n", "\n")
    except (FileNotFoundError, UnicodeDecodeError):
        return path, False
    return path, got == want


def dump(name: str) -> None:
    """Print the table as (index, input, exact value, stored int8)."""
    table, _fname, fn, in_scale, out_scale = TABLES[name]
    print(f"  idx   int8      x     {name}(x)   stored   error")
    for u in range(256):
        q = s8(u)
        x = q * in_scale
        y = fn(x)
        stored = table[u]
        print(f"  {u:3d}  {q:5d}  {x:+7.4f}  {y:+9.5f}   {stored:5d}   "
              f"{stored * out_scale - y:+.5f}")


# --- sanity checks on the tables themselves -----------------------------------

def selftest() -> list[str]:
    """Properties the tables must have. Returns a list of failure messages."""
    bad = []

    # exp is monotone in the real input — walk the indices in signed order
    # (128..255 = -128..-1, then 0..127). gelu is not (it has the dip).
    order = list(range(0x80, 0x100)) + list(range(0x00, 0x80))
    seq = [EXP_LUT[u] for u in order]
    if any(b < a for a, b in zip(seq, seq[1:])):
        bad.append("exp table is not monotone non-decreasing")

    # Anchors that pin the format.
    if EXP_LUT[0x00] != 127:
        bad.append(f"exp(0) should store 127, got {EXP_LUT[0x00]}")
    if EXP_LUT[0x80] != 0:
        bad.append(f"exp(-8) should store 0, got {EXP_LUT[0x80]}")
    if GELU_LUT[0x00] != 0:
        bad.append(f"gelu(0) should store 0, got {GELU_LUT[0x00]}")
    if GELU_LUT[0x7F] != 127:
        bad.append(f"gelu(+7.9375) should store 127, got {GELU_LUT[0x7F]}")
    if GELU_LUT[0x80] != 0:
        bad.append(f"gelu(-8) should store 0, got {GELU_LUT[0x80]}")
    if min(GELU_LUT) != -3:
        bad.append(f"gelu's dip should quantize to -3, got {min(GELU_LUT)}")

    # gelu is scale-preserving only if it never saturates: |gelu(x)| <= |x| and
    # in_scale == out_scale, so a clip here would mean a broken scale choice.
    for u in range(256):
        x = s8(u) * GELU_IN_SCALE
        if abs(gelu(x) / GELU_OUT_SCALE) > 127.5:
            bad.append(f"gelu saturates at index {u} (x={x})")
            break

    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out-dir", default=OUT_DIR,
                    help="where the .hex files go (default: accel/tpu/rtl/luts)")
    ap.add_argument("--check", action="store_true",
                    help="verify the existing files match; write nothing")
    ap.add_argument("--dump", choices=sorted(TABLES),
                    help="print one table as x -> f(x) and exit")
    args = ap.parse_args(argv)

    if args.dump:
        dump(args.dump)
        return 0

    failures = selftest()
    for msg in failures:
        print(f"SELFTEST FAILED: {msg}")
    if failures:
        return 1

    if args.check:
        stale = 0
        for name in sorted(TABLES):
            path, ok = check_hex(name, args.out_dir)
            print(f"  [{'OK  ' if ok else 'STALE'}] {path}")
            stale += not ok
        if stale:
            print(f"{stale} file(s) out of date — run: python luts.py")
        return 1 if stale else 0

    os.makedirs(args.out_dir, exist_ok=True)
    for name in sorted(TABLES):
        table = TABLES[name][0]
        path = write_hex(name, args.out_dir)
        print(f"{name:>4}: 256 x int8, range [{min(table)}, {max(table)}] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
