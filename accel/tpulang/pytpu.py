#!/usr/bin/env python3
"""pytpu — a staged Python emitter for tpulang.  See pytpu.md for the design.

In one paragraph: every value here is either **build-time** (a Python ``int`` or
:class:`Sym` — folded into an immediate, free) or **run-time** (a :class:`Reg` —
one instruction per operation).  Python's own ``for`` is therefore the *unrolled*
loop; ``with p.loop(n)`` emits a real ``cmps``/``bge``/``jmp`` loop whose
instruction cost does not grow with ``n``.  That distinction is the whole idea:
instruction memory is 1024 words, so anything shaped like a tensor loop has to
stay a loop.

The output is ordinary tpulang text, so ``assembler.py``, ``iss.py``,
``gen_vectors.py``, ``torch_ref.py`` and ``host/run_program.py`` all work on a
generated program unchanged — emit your geometry with :meth:`Program.equ` and
those tools' ``.equ``-keyed dispatch finds it exactly as it finds a hand-written
one.

    from pytpu import Program
    p = Program("relu layer")
    N = p.equ("N", 64)                 # -> `.equ N 64`; an int you can range()
    p.cfg(vlen=N, len=N)
    p.rdmem(ACT, ACT)
    p.relu(OUT, ACT)
    p.wrmem(OUT, OUT)
    p.halt()
    p.save("examples/out/relu.tpu")
    words = p.assemble()               # raises if it does not fit in IMEM

**Register discipline** — the one thing to know.  A chained expression allocates
one register per operation and never reuses them, so write address arithmetic
with the in-place operators, the way the hand-written examples do:

    arow = mt * ASTRIDE_M              # one temporary
    arow += A                          # mutates it; no second temporary

``+=`` on a *protected* register (a ``p.const`` cache entry, a loop counter)
falls back to allocating a fresh one, so it can never corrupt either.
Temporaries created inside a ``with p.loop(...)`` block are reclaimed when it
ends; the matching rule is *do not use a Reg outside the block that made it*.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager

# assembler/iss/gen_vectors import each other flatly; join them rather than
# turning tpulang into a package (same dance as torch_ref.py).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import assembler                                              # noqa: E402

IMEM_WORDS = 1024        # scalar_unit.sv IMEM_AW = 10
NUM_REGS = assembler.NUM_REGS


class TpuError(Exception):
    """A pytpu build-time error (bad operand, out of registers, too big)."""


class Sym(int):
    """A named build-time constant: an ``int`` that renders as its .equ name.

    Arithmetic on a Sym yields a plain int — the name is lost and the value
    folds, which is what you want for `T * ROWS`.  Use `p.equ` for anything that
    should keep a name in the generated source.
    """

    def __new__(cls, name: str, value: int):
        s = super().__new__(cls, value)
        s.name = name
        return s

    def __repr__(self) -> str:
        return self.name


class Reg:
    """A run-time value living in a scalar register.  Operators emit code."""

    def __init__(self, prog: "Program", idx: int, name: str, protected: bool = False):
        self.p, self.idx, self.name, self.protected = prog, idx, name, protected

    def __repr__(self) -> str:
        return self.name

    def _bin(self, mnem: str, other, swap: bool = False, inplace: bool = False):
        # Build-time identities: these are what make a peeled `arow + 0*ATILE`
        # cost nothing, so peeling a loop iteration in Python stays free.
        if not isinstance(other, (Reg, str)):
            v = int(other)
            if mnem == "muls" and v == 1:
                return self
            if mnem == "muls" and v == 0:
                return self.p.zero
            if v == 0 and (mnem == "adds" or not swap):     # x±0, 0+x  (not 0-x)
                return self
        a, b = (other, self) if swap else (self, other)
        return self.p._arith(mnem, a, b, self if inplace and not self.protected else None)

    def __add__(self, o):   return self._bin("adds", o)
    def __radd__(self, o):  return self._bin("adds", o, swap=True)
    def __sub__(self, o):   return self._bin("subs", o)
    def __rsub__(self, o):  return self._bin("subs", o, swap=True)
    def __mul__(self, o):   return self._bin("muls", o)
    def __rmul__(self, o):  return self._bin("muls", o, swap=True)
    def __iadd__(self, o):  return self._bin("adds", o, inplace=True)
    def __isub__(self, o):  return self._bin("subs", o, inplace=True)
    def __imul__(self, o):  return self._bin("muls", o, inplace=True)


class Program:
    """Accumulates tpulang lines.  `render()` is the source; `assemble()` the words."""

    def __init__(self, title: str = ""):
        self.title = title
        self._equ: list[tuple[str, int]] = []      # .equ NAME value
        self._aliases: list[tuple[str, int]] = []  # .reg NAME rN
        self._pro: list[str] = []                  # prologue: hoisted `li`
        self._body: list[str] = []
        self._free = list(range(1, NUM_REGS))      # r0 is the hardwired zero
        self._permanent: set[int] = set()          # survive scope exit
        self._consts: dict[str, Reg] = {}
        self._taken: set[str] = set()              # one flat symbol namespace
        self._nlabel = 0
        self._depth = 0
        self.zero = Reg(self, 0, "r0", protected=True)
        self._consts["0"] = self.zero

    # ---- symbols and values --------------------------------------------------
    def equ(self, name: str, value: int) -> Sym:
        """Define a build-time constant, emitted as `.equ NAME value`."""
        self._claim(name)
        self._equ.append((name, int(value)))
        return Sym(name, int(value))

    def const(self, value, name: str | None = None) -> Reg:
        """A register holding `value` (int, Sym or expression string), cached.

        The `li` is hoisted into the prologue: a constant first used inside a
        loop body must not re-execute its `li` every iteration, nor be undefined
        when the loop runs zero times.
        """
        e = self._expr(value)
        if e in self._consts:
            return self._consts[e]
        r = self._alloc(name or f"k_{e}", permanent=True, protected=True)
        self._pro.append(f"    li      {r}, {e}")
        self._consts[e] = r
        return r

    def reg(self, name: str = "t", init=None) -> Reg:
        """A fresh named register, optionally initialised with `li`."""
        r = self._alloc(name)
        if init is not None:
            self._body.append(f"{self._pad}li      {r}, {self._expr(init)}")
        return r

    def _expr(self, v) -> str:
        if isinstance(v, Sym):
            return v.name
        if isinstance(v, str):
            return v
        v = int(v)
        return f"0x{v:x}" if v >= 0x100 else str(v)

    def _val(self, x) -> Reg:
        return x if isinstance(x, Reg) else self.const(x)

    def _claim(self, name: str) -> str:
        if name in self._taken:
            raise TpuError(f"duplicate symbol '{name}' (labels, .equ and .reg share one namespace)")
        self._taken.add(name)
        return name

    def _uniq(self, base: str) -> str:
        base = re.sub(r"\W", "_", str(base)) or "t"
        name, k = base, 0
        while name in self._taken:
            k += 1
            name = f"{base}{k}"
        self._taken.add(name)
        return name

    # ---- registers -----------------------------------------------------------
    def _alloc(self, name: str, permanent: bool = False, protected: bool = False) -> Reg:
        if not self._free:
            recent = ", ".join(n for n, _ in self._aliases[-6:])
            raise TpuError(
                f"out of scalar registers (r1..r{NUM_REGS - 1}); last allocated: {recent}. "
                "Hoist constants, or use += to mutate temporaries in place.")
        idx = self._free.pop(0)
        alias = self._uniq(name)
        self._aliases.append((alias, idx))
        if permanent:
            self._permanent.add(idx)
        return Reg(self, idx, alias, protected)

    @contextmanager
    def scope(self):
        """Reclaim every non-permanent register allocated inside the block."""
        saved = list(self._free)
        try:
            yield
        finally:
            self._free = [i for i in saved if i not in self._permanent]

    def _arith(self, mnem: str, a, b, dst: Reg | None = None) -> Reg:
        ra, rb = self._val(a), self._val(b)
        out = dst if dst is not None else self._alloc("t")
        self._ins(mnem, out, ra, rb)
        return out

    # ---- emission ------------------------------------------------------------
    @property
    def _pad(self) -> str:
        return "    " + "  " * self._depth

    def _ins(self, mnem: str, *ops) -> None:
        self._body.append(f"{self._pad}{mnem:<8}" + ", ".join(str(o) for o in ops))

    def emit(self, line: str = "") -> None:
        """Append a raw line (comment, or an instruction pytpu does not wrap)."""
        self._body.append(line)

    def comment(self, text: str) -> None:
        self._body.append(f"{self._pad}; {text}")

    # ---- config, MXU, control ------------------------------------------------
    def cfg(self, **kw) -> None:
        """setcfg for each keyword: tlen / vlen / len / scalar (immediates, not registers)."""
        for k, v in kw.items():
            if k not in assembler.CFG_NAMES:
                raise TpuError(f"unknown cfg '{k}' (use {'/'.join(assembler.CFG_NAMES)})")
            self._body.append(f"{self._pad}setcfg  {k}, {self._expr(v)}")

    def matmul(self, out, act, wgt, acc: bool = False, rq: bool = False) -> None:
        self._ins("matmul" + (".acc" if acc else "") + (".rq" if rq else ""),
                  self._val(out), self._val(act), self._val(wgt))

    def loads(self, addr, name: str = "v") -> Reg:
        """Read one int32 out of the scratchpad into a *new* register."""
        dst = self._alloc(name)
        self._ins("loads", dst, self._val(addr))
        return dst

    def wait(self, unit: str) -> None:
        if unit not in assembler.UNIT_CODES:
            raise TpuError(f"unknown unit '{unit}' (use {'/'.join(assembler.UNIT_CODES)})")
        self._body.append(f"{self._pad}wait    {unit}")

    def halt(self) -> None:
        self._body.append(f"{self._pad}halt")

    @contextmanager
    def loop(self, start, stop=None, name: str = "i"):
        """A run-time loop: `loop(n)` or, range-style, `loop(start, stop)`.

        Costs 4 words plus the counter, whatever the trip count.  For the
        *unrolled* form just use Python's own `for i in range(n)`.
        """
        if stop is None:
            start, stop = 0, start
        n, self._nlabel = self._nlabel, self._nlabel + 1
        top, end = self._uniq(f"L{n}_top"), self._uniq(f"L{n}_end")
        with self.scope():
            i = self._alloc(name, protected=True)
            bound, one = self.const(stop), self.const(1)
            self._body.append(f"{self._pad}li      {i}, {self._expr(start)}")
            self._body.append(f"{top}:")
            self._ins("cmps", i, bound)
            self._body.append(f"{self._pad}bge     {end}")
            self._depth += 1
            try:
                yield i
            finally:
                self._depth -= 1
                self._ins("adds", i, i, one)
                self._body.append(f"{self._pad}jmp     {top}")
                self._body.append(f"{end}:")

    # ---- output --------------------------------------------------------------
    def render(self) -> str:
        """The tpulang source: header, .equ, .reg, hoisted `li`, then the body."""
        out: list[str] = []
        if self.title:
            out += [f"; {self.title}",
                    "; generated by pytpu.py — edit the generator, not this file", ""]
        if self._equ:
            out += [f".equ {n:<10} {self._expr(v)}" for n, v in self._equ] + [""]
        if self._aliases:
            out += [f".reg {n:<10} r{i}" for n, i in self._aliases] + [""]
        if self._pro:
            out += self._pro + [""]
        return "\n".join(out + self._body) + "\n"

    def __str__(self) -> str:
        return self.render()

    def assemble(self) -> list[int]:
        words = assembler.assemble(self.render())
        if len(words) > IMEM_WORDS:
            raise TpuError(f"program is {len(words)} words; IMEM holds {IMEM_WORDS}. "
                           "Turn an unrolled Python `for` into a `with p.loop(...)`.")
        return words

    def save(self, path: str) -> str:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
        return path


# --- the remaining ops, generated from a table rather than hand-written -------
# Every operand is a register holding a scratchpad/DRAM byte address, so they all
# have the same shape: coerce each argument through _val and emit.  Mnemonics in
# assembler.SPECS but absent here are deliberate omissions: `branch`/`jmp` (use
# p.loop), `li`/`setcfg` (use p.const/p.cfg), `cmps` (no run-time `if` in v1),
# `wrneigh` (LINK is a no-op in RTL).
_OPS = {
    "vecdot": 3, "vecmul": 3, "vecadd": 3, "vecemul": 3,
    "sadd": 3, "sdiv": 3, "requant": 3,
    "relu": 2, "gelu": 2, "exp": 2, "square": 2, "redmax": 2, "redsum": 2,
    "rdmem": 2, "wrmem": 2, "stores": 2,
}


def _make_op(mnem: str, arity: int):
    def op(self, *args):
        if len(args) != arity:
            raise TpuError(f"'{mnem}' takes {arity} operands, got {len(args)}")
        self._ins(mnem, *[self._val(a) for a in args])
    op.__name__ = mnem
    op.__doc__ = f"emit `{mnem}` ({arity} address operands)"
    return op


for _mnem, _arity in _OPS.items():
    setattr(Program, _mnem, _make_op(_mnem, _arity))
