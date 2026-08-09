#!/usr/bin/env python3
"""template.py — render a parameterized tpulang primitive.

A primitive lives in ``lib/*.tpu`` as ordinary, readable tpulang with two
additions. Both are chosen so that a mistake is a hard error rather than a
program that assembles into something subtly wrong.

``{{NAME}}`` — parameter substitution
    Textual, before assembly. Declared in the template's header comment with
    ``@param``/``@default``, so the template is self-describing and
    ``primitives.py`` does not restate the list. A ``{{NAME}}`` with no value, or
    a value with no ``{{NAME}}``, is an error — a typo must not silently take a
    default.

``$name`` — instance-local symbol
    tpulang has one flat namespace for ``.equ`` names, ``.reg`` aliases and
    labels, and redefining one is an ``AsmError`` (assembler.py ``_directive``).
    Two instantiations of the same primitive would therefore collide on every
    symbol, so every symbol written ``$name`` is rewritten to ``<prefix>_name``.

    ``$`` is the sigil because it is a *syntax error* inside the assembler's
    expression evaluator — an unmangled one cannot survive into a program that
    assembles, which is exactly the failure mode we want.

Header metadata (a comment block, so the raw template is still valid tpulang
apart from its placeholders)::

    ; @primitive matmul_ternary
    ; @param   M   token rows of A (multiple of TT)
    ; @default TT  8
    ; @clobbers r1-r26, cfg tlen/len/scalar
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

_PARAM_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_LOCAL_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_HDR_RE = re.compile(r"^\s*;\s*@(primitive|param|default|clobbers)\s+(.*)$")


class TemplateError(Exception):
    """A user-facing error in a primitive template or in a call to one."""


class Hex(int):
    """An int that renders as ``0x1234``.

    Addresses read better in hex in the generated source, and the generated
    source is meant to be read — it is the artifact this whole package produces.
    """

    def __str__(self) -> str:
        return f"0x{int(self):04x}"


def fmt(value) -> str:
    """Render a parameter value into the source text."""
    if isinstance(value, Hex):
        return str(value)
    if isinstance(value, bool):  # before int: bool is an int subclass
        raise TemplateError(f"bool parameter {value!r}: pass an int or a string")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise TemplateError(f"parameter value {value!r} is not an int or a string")


@dataclass
class Template:
    """One primitive: its source text plus the header metadata it declares."""

    name: str
    path: str
    text: str
    params: dict = field(default_factory=dict)     # name -> description
    defaults: dict = field(default_factory=dict)   # name -> value
    clobbers: str = ""

    @property
    def used(self) -> set:
        """Every ``{{NAME}}`` the body actually references."""
        return set(_PARAM_RE.findall(self.text))

    def render(self, prefix: str, params: dict) -> str:
        """Substitute ``params`` and mangle ``$locals`` with ``prefix``."""
        values = dict(self.defaults)
        values.update(params)

        declared = set(self.params) | set(self.defaults)
        unknown = set(params) - declared
        if unknown:
            raise TemplateError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                f"this template declares {sorted(declared)}"
            )
        missing = self.used - set(values)
        if missing:
            raise TemplateError(
                f"{self.name}: no value for {sorted(missing)} "
                f"(declare a @default or pass it)"
            )
        unused = declared - self.used
        if unused:
            raise TemplateError(
                f"{self.name}: parameter(s) {sorted(unused)} are declared in the "
                f"header but never used in the body — the header has drifted"
            )

        out = _PARAM_RE.sub(lambda m: fmt(values[m.group(1)]), self.text)
        out = _LOCAL_RE.sub(lambda m: f"{prefix}_{m.group(1)}", out)

        # Belt and braces: neither sigil may reach the assembler.
        leftover = _PARAM_RE.search(out)
        if leftover:
            raise TemplateError(f"{self.name}: unsubstituted {leftover.group(0)}")
        if "$" in out:
            raise TemplateError(
                f"{self.name}: a bare '$' survived rendering — a local symbol "
                f"must be written $name with no space"
            )
        return out


def parse(text: str, name: str, path: str = "<string>") -> Template:
    """Parse a template's header metadata out of its leading comment block."""
    tpl = Template(name=name, path=path, text=text)
    for raw in text.splitlines():
        m = _HDR_RE.match(raw)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind == "primitive":
            tpl.name = rest
        elif kind == "clobbers":
            tpl.clobbers = rest
        elif kind == "param":
            key, _, desc = rest.partition(" ")
            tpl.params[key] = desc.strip()
        else:  # default
            key, _, val = rest.partition(" ")
            tpl.defaults[key] = int(val.strip(), 0)
    return tpl


_CACHE: dict = {}


def load(name: str) -> Template:
    """Load ``lib/<name>.tpu`` (cached)."""
    if name not in _CACHE:
        path = os.path.join(LIB_DIR, f"{name}.tpu")
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise TemplateError(f"cannot read primitive '{name}': {exc}") from None
        _CACHE[name] = parse(text, name, path)
    return _CACHE[name]


def render(name: str, prefix: str, **params) -> str:
    """Load and render ``lib/<name>.tpu`` in one call."""
    return load(name).render(prefix, params)
