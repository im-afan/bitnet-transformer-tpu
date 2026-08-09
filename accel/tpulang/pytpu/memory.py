#!/usr/bin/env python3
"""memory.py — tensor descriptors and the bump allocators that place them.

Two ideas, both small:

:class:`Tensor`
    A named region with a shape, a dtype, an address, and a **compile-time**
    quantization ``scale`` (``real = int * scale``). The scale never exists in
    TPU memory — it is what :mod:`quant` uses to pick the ``{m0, n}`` word for
    each rescale site, exactly as ``README.md`` §5 describes.

:class:`Arena`
    A bump allocator with a hard limit. Every allocation is named, so a memory
    map can be printed and an overflow names the tensor that broke the budget
    instead of silently wrapping — scratchpad addresses wrap mod 2**16 in both
    the RTL and the ISS, so an unchecked overflow is a corrupted result, not a
    crash.

**The identical-address convention.** Composed primitives read their inputs from
DRAM and write their outputs to DRAM (see PLAN.md §4.4), and every tensor uses
the *same* address in DRAM and in the scratchpad, so a fill/spill is ``rdmem a,
a`` / ``wrmem a, a``. One :class:`Arena` therefore covers both spaces; a second,
disjoint arena hands out scratchpad-only working buffers that no DMA ever
touches.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from template import Hex                                        # noqa: E402

# Element sizes. 'ternary' is 2 bits, so a packed tensor is prod(shape)//4 bytes
# (see quant.pack_ternary for the tile-major layout the MXU expects).
DTYPES = {"int8": 1, "int32": 4, "ternary": 0.25}


class MemoryError_(Exception):
    """An arena overflowed, or a tensor was described inconsistently."""


@dataclass
class Tensor:
    """A named region of TPU memory with a compile-time quantization scale."""

    name: str
    shape: tuple
    dtype: str
    addr: int
    scale: float = 1.0
    data: list | None = None          # host bytes, set by the composer

    @property
    def nelem(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return nbytes_of(self.shape, self.dtype)

    @property
    def a(self) -> Hex:
        """The address, formatted as hex for template substitution."""
        return Hex(self.addr)

    @property
    def end(self) -> int:
        return self.addr + self.nbytes

    def __repr__(self) -> str:
        return (f"Tensor({self.name} {self.dtype}{list(self.shape)} "
                f"@0x{self.addr:04x} +{self.nbytes}B scale={self.scale:g})")


def nbytes_of(shape: tuple, dtype: str) -> int:
    if dtype not in DTYPES:
        raise MemoryError_(f"unknown dtype '{dtype}' (use {sorted(DTYPES)})")
    n = 1
    for s in shape:
        n *= s
    if dtype == "ternary":
        if n % 4:
            raise MemoryError_(
                f"ternary tensor {shape} has {n} elements, not a multiple of 4 — "
                f"2-bit packing needs whole bytes"
            )
        return n // 4
    return n * DTYPES[dtype]


@dataclass
class Arena:
    """A named bump allocator over ``[base, limit)``."""

    name: str
    base: int
    limit: int
    align: int = 4
    _next: int = field(init=False)
    tensors: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self._next = self.base

    @property
    def used(self) -> int:
        return self._next - self.base

    @property
    def free(self) -> int:
        return self.limit - self._next

    def alloc(self, name: str, nbytes: int, align: int | None = None) -> int:
        a = align or self.align
        addr = (self._next + a - 1) // a * a
        if addr + nbytes > self.limit:
            raise MemoryError_(
                f"{self.name} arena overflow allocating '{name}' ({nbytes} B) at "
                f"0x{addr:04x}: limit is 0x{self.limit:04x}, "
                f"{self.limit - addr} B free"
            )
        self._next = addr + nbytes
        return addr

    def tensor(self, name: str, shape, dtype: str = "int8",
               scale: float = 1.0, align: int | None = None) -> Tensor:
        shape = tuple(shape)
        t = Tensor(name, shape, dtype,
                   self.alloc(name, nbytes_of(shape, dtype), align), scale)
        self.tensors.append(t)
        return t

    def buffer(self, name: str, nbytes: int, align: int | None = None) -> Hex:
        """A raw, untyped working buffer. Returns the address, hex-formatted."""
        addr = self.alloc(name, nbytes, align)
        self.tensors.append(Tensor(name, (nbytes,), "int8", addr))
        return Hex(addr)

    def map(self) -> str:
        """A printable memory map of everything allocated here."""
        lines = [f"{self.name} arena  0x{self.base:04x}..0x{self.limit:04x}  "
                 f"({self.used} B used, {self.free} B free)",
                 "  addr    bytes  dtype    shape                name"]
        for t in self.tensors:
            lines.append(f"  0x{t.addr:04x}  {t.nbytes:6d}  {t.dtype:<7s}  "
                         f"{str(list(t.shape)):<18s}   {t.name}")
        return "\n".join(lines)
