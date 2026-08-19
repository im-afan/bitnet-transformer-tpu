#!/usr/bin/env python3
"""run_adder.py — score the real adder checkpoint on the FPGA, problem by problem.

``run_program.py`` answers "does the board compute what the ISS computes", on a
*synthetic* image built by ``gen_vectors``. That is the right question for a
kernel and the wrong one for a model: it says nothing about whether the thing on
the board still does addition. This script asks the other question. It stages the
**real checkpoint** — the same DRAM image ``accel/tpulang/adder_export.py`` hands
the ISS, produced by the same :func:`adder_export.stage_dram` — runs one problem
per ``G``, reads the int32 logits back, argmaxes on the host, and scores exact
sequence and per-token accuracy over N problems.

Everything that is not the transport is shared with the simulated path, so a
disagreement between ``--dry-run`` and a board run localizes to the hardware
rather than to a second copy of the staging logic. That mattered: the pipeline
has drifted from the model twice before (``adder_export.py``'s docstring).

**What moves over the wire, and when.** The weights are ~50 KB and identical for
every problem, so they are written once, before the loop. Per problem only the
[T][D] input (4096 B) goes down and the [T][VPAD] int32 logits (2048 B) come
back — about 5.8 KB, or ~0.5 s at 115200 baud including the header tax. The
weights are *not* rewritten between problems, which also means a run is a
standing test that SRAM holds them across ``G`` boundaries.

The host keeps the embedding lookup and the final argmax (the ISA has no gather,
and ``redmax`` returns a maximum rather than its index) — the same split
``adder_kernel.md`` §1 defines, so the device's share is unchanged.

    python accel/tpu/host/run_adder.py --dry-run -n 8      # ISS, no board
    python accel/tpu/host/run_adder.py -p COM5 -n 64
    python accel/tpu/host/run_adder.py -p COM5 -n 8 --compare-iss
    python accel/tpu/host/run_adder.py -p COM5 -n 64 --no-show   # progress only

By default every problem prints as it finishes � the expression, the expected
answer and what the device answered, in human digit order (the dataset stores
numbers reversed). ``--no-show`` falls back to the periodic progress counter,
which is what you want when N is large.

The device's performance counters are read after every problem (``--no-timing``
to skip the extra round trip), so each row carries that problem's *core* run
time and the summary at the end reports mean/min/max plus unit occupancy. The
counters reset at each ``G``, so one read per problem measures one forward pass
and nothing accumulates.

``--compare-iss`` runs each problem in *both* the ISS and the board and reports
the first differing logit, which is the diagnostic to reach for when the accuracy
comes out below the simulated number.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TPULANG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tpulang"))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, TPULANG_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                 # noqa: E402

import adder_export as ax                                    # noqa: E402
import gen_vectors as gv                                     # noqa: E402
import model.numbers_data as numbers_data                    # noqa: E402
import model.quant as quant                                  # noqa: E402
from iss import TPU                                          # noqa: E402
from run_program import (                                    # noqa: E402
    contiguous_runs, probe_idle, report_run_time, wait_until_idle,
)
from tpu_uart import ProtocolError, TPUUart                  # noqa: E402

T = ax.T
BOOT_PC = 0


# =============================================================================
# Display. The dataset writes every number least-significant digit first, so a
# raw token dump is unreadable � everything here goes through
# ``numbers_data.unreverse_expression`` to get human digit order, the same way
# ``model/tests/test_inference.py`` prints its samples.
# =============================================================================
def decode_answer(ids) -> str:
    """A row of answer-region token ids -> the number it spells, human order.

    Pads are dropped by ``detokenize``; a prediction that emits a pad *between*
    digits therefore shows up as a shorter string rather than as a hole, which
    is what we want � it is wrong either way and the length says so.
    """
    return numbers_data.unreverse_expression(
        numbers_data.detokenize([int(t) for t in ids]))


def report_timing(counters: list[dict], clk_mhz: float) -> None:
    """Per-problem run time and unit occupancy, from the device's counters.

    ``run`` is the scalar unit's busy interval, so this is the cost of one
    forward pass on the hardware with the UART excluded — the wall-clock loop
    time is dominated by the link and says nothing about the design. The other
    counters are printed as a fraction of ``run``; they **overlap** and do not
    partition it (``swait`` covers nearly all of ``mxu``/``vpu``/``dma`` under
    issue-and-wait, ``mload`` is a subset of ``mxu``, and ``vmm`` is a subset of
    ``vpu``).

    ``vmm`` gets its own line because it is the one counter whose interesting
    denominator is not ``run``: it answers "is the VPU's share `vecmatmul`, or
    is it the pointwise ops?", and that ratio is what says where VPU work is
    worth optimizing. On the current kernel it should read **0.0%** — ternary
    K/V and a ternary output head put every matmul in the model on the MXU, so
    nothing issues `vecmatmul` at all (``adder_kernel.md`` §2.5, §2.6). A
    nonzero ``vmm`` here means the device is running an older image than the
    one this script assembled.
    """
    if not counters:
        return
    runs = [c["run"] for c in counters]
    lo, hi = min(runs), max(runs)
    mean = sum(runs) / len(runs)
    us = lambda cyc: cyc / clk_mhz                       # clocks -> microseconds

    print(f"\nscalar unit active time over {len(runs)} run(s) @ {clk_mhz:g} MHz:")
    print(f"  per problem   mean {us(mean) / 1e3:8.3f} ms   "
          f"min {us(lo) / 1e3:.3f} ms   max {us(hi) / 1e3:.3f} ms")
    print(f"                mean {mean:.0f} core clocks "
          f"({1e6 / us(mean):.1f} problems/s if the link were free)")

    units = [k for k in counters[0] if k != "run"]
    total = sum(runs)
    if total:
        share = "  ".join(f"{k} {100.0 * sum(c[k] for c in counters) / total:.1f}%"
                          for k in units)
        print(f"  unit busy     {share}")

        vpu = sum(c.get("vpu", 0) for c in counters)
        vmm = sum(c.get("vmm", 0) for c in counters)
        if vpu:
            print(f"  vpu split     vecmatmul {100.0 * vmm / total:.1f}% of run "
                  f"= {100.0 * vmm / vpu:.1f}% of VPU time; "
                  f"other vpu ops {100.0 * (vpu - vmm) / total:.1f}% of run")


def decode_prompt(expr: str) -> str:
    """``'321+54=861'`` (generator order) -> ``'123 + 45'``, the question only."""
    lhs = numbers_data.unreverse_expression(expr).split("=")[0]
    return lhs.replace("+", " + ")


def new_tpu() -> TPU:
    return TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W,
               mem_addr_w=gv.MEM_ADDR_W, m0_w=gv.M0_W, n_w=gv.N_W)


# =============================================================================
# Staging, split into the part that is written once and the part that is not.
# =============================================================================
def stage(qm, model, consts: dict, X0: torch.Tensor) -> TPU:
    """A fresh ISS instance with the full DRAM image for one problem."""
    tpu = new_tpu()
    ax.stage_dram(tpu, consts, qm, X0)
    return tpu


def split_image(tpu: TPU, consts: dict) -> tuple[dict, dict]:
    """The staged DRAM, split into (constant, per-problem) ``{addr: byte}``.

    The split is by address, from the program's own ``.equ`` table: ``D_XIN`` is
    the only input a new problem changes, so everything else — mask, ``fc``,
    every layer's packed weights and requant words — is written once. Reading
    the bounds out of ``consts`` rather than hardcoding them means a layout
    change in the kernel moves this too.

    Both halves are dense over the regions the kernel reads. Sending only the
    nonzero bytes would be wrong, not merely different: a trit of 0 packs to a
    0 byte, and SRAM is not assumed to start cleared, so a dropped zero leaves
    whatever the previous run put there.
    """
    xin = (consts["D_XIN"], T * consts["D"])
    dense = lambda lo, n: {a: tpu.dram[tpu._d(a)] for a in range(lo, lo + n)}

    weights: dict = {}
    for lo, n in region_map(consts):
        weights.update(dense(lo, n))
    return weights, dense(*xin)


def region_map(c: dict) -> list[tuple[int, int]]:
    """(base, length) of every DRAM block the kernel *reads* but never rewrites."""
    regions = [(c["D_MASK"], T * T), (c["D_WFC"], c["B_WFC"])]
    for li in range(c["LAYERS"]):
        base = c["D_L0"] + li * c["LSTRIDE"]
        regions.append((base + c["O_WQKV"], c["D3"] * c["WCOL"]))
        regions.append((base + c["O_WO"], c["D"] * c["WCOL"]))
        regions.append((base + c["O_W1"], c["D"] * c["WCOL"]))
        regions.append((base + c["O_W2"], c["D"] * c["WCOL"]))
        regions.append((base + c["O_RQW"], c["B_RQW"]))
    return regions


# =============================================================================
# The two backends. Same inputs, same outputs, different machine.
# =============================================================================
class IssBackend:
    """The ISS, driven through the same interface as the board."""

    name = "ISS"

    def __init__(self, words, consts):
        self.words, self.consts = words, consts
        self.weights: dict = {}
        self.counters: list[dict] = []      # the ISS has no clock; stays empty

    def load_weights(self, weights: dict) -> None:
        self.weights = weights

    def run_problem(self, inputs: dict) -> torch.Tensor:
        tpu = new_tpu()
        for a, b in self.weights.items():
            tpu.dram[tpu._d(a)] = b
        for a, b in inputs.items():
            tpu.dram[tpu._d(a)] = b
        tpu.run(self.words, max_steps=2_000_000)
        return ax.read_logits(tpu, self.consts, self.consts["VOCAB"])


class BoardBackend:
    """The FPGA over UART. Weights once, then one G per problem."""

    name = "FPGA"

    def __init__(self, uart: TPUUart, words, consts, args):
        self.u, self.words, self.consts, self.args = uart, words, consts, args
        self.counters: list[dict] = []
        self._timer_warned = False

    def read_counters(self) -> dict | None:
        """The device's per-run counters for the problem that just halted.

        Safe to call once per problem: the counters reset at each ``G`` and
        freeze at the halt, so what comes back describes *this* run and nothing
        accumulates across problems (``tpu_uart.read_counters``). ``run`` is the
        scalar unit's busy interval — 'G' to HALT — which is the honest cost of
        one forward pass; the wall clock is mostly UART.

        Non-fatal, and warns only once: 'T' is newer than the other commands and
        a bitstream flashed before it existed NAKs it, which should not cost a
        scoring run. See ``run_program.report_run_time``.
        """
        try:
            ctr = self.u.read_counters()
        except (ProtocolError, OSError) as exc:
            if not self._timer_warned:
                print(f"  timer   : unavailable — {exc}")
                print("            (bitstream predates the 'T' command? "
                      "reflash board=cmod_a7)")
                self._timer_warned = True
            return None
        self.counters.append(ctr)
        return ctr

    def _idle(self) -> None:
        if not probe_idle(self.u):
            wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)

    def load_weights(self, weights: dict) -> None:
        self._idle()
        self.u.load_program(0, self.words)
        runs = contiguous_runs(weights)
        for addr, blob in runs:
            self.u.write_mem(addr, blob)
        print(f"loaded  : {len(self.words)} instruction words, "
              f"{len(weights)} weight bytes in {len(runs)} frame(s)")
        if self.args.verify_inputs:
            bad = sum(1 for addr, blob in runs
                      for i, b in enumerate(self.u.read_mem(addr, len(blob)))
                      if b != blob[i])
            if bad:
                raise SystemExit(f"weight readback FAILED: {bad} byte(s) differ")
            print(f"verify  : all {len(weights)} weight bytes read back identical")

    def run_problem(self, inputs: dict) -> torch.Tensor:
        for addr, blob in contiguous_runs(inputs):
            self.u.write_mem(addr, blob)
        self.u.go(BOOT_PC)
        wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)
        if self.args.timing:
            self.read_counters()

        # The device writes VPAD words per row, not VOCAB: the head's output is
        # padded up to a whole MXU output tile (the program's section 8), so the
        # row stride and the column count are different numbers.
        vpad = self.consts["VPAD"]
        base, n = self.consts["D_LOG"], T * vpad * 4
        raw = self.u.read_mem(base, n)
        out = []
        for t in range(T):
            row = []
            for v in range(self.consts["VOCAB"]):
                o = (t * vpad + v) * 4
                u = int.from_bytes(raw[o:o + 4], "little")
                row.append(u - (1 << 32) if u >= (1 << 31) else u)
            out.append(row)
        return torch.tensor(out, dtype=torch.int64)


# =============================================================================
# Driver.
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("-t", "--timeout", type=float, default=2.0)
    ap.add_argument("-n", "--problems", type=int, default=32)
    ap.add_argument("-c", "--calib", type=int, default=128,
                    help="problems used to derive the scales (disjoint from the eval set)")
    ap.add_argument("--ckpt", default=ax.CKPT)
    ap.add_argument("--program",
                    default=os.path.join(TPULANG_DIR, "examples", "adder_model.tpu"))
    ap.add_argument("--dyt-scale", choices=["fixed", "calibrated"], default="fixed")
    ap.add_argument("--scales", choices=["auto", "qat", "calib"], default="auto",
                    help="activation scales: the checkpoint's learned LSQ scales, "
                         "absmax calibration, or auto")
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--run-timeout", type=float, default=30.0)
    ap.add_argument("--clk-mhz", type=float, default=12.0)
    ap.add_argument("--verify-inputs", action="store_true",
                    help="read the weights back after writing them (bring-up check)")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                    help="print the problem, the expected answer and the device's "
                         "answer for every problem (--no-show: progress only)")
    ap.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True,
                    help="read the device's performance counters after every "
                         "problem and report the per-problem run time")
    ap.add_argument("--compare-iss", action="store_true",
                    help="also run every problem in the ISS and diff the logits")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the ISS as the backend; never open the serial port")
    args = ap.parse_args(argv)

    # ---- model -> the device's integers ------------------------------------
    print(f"checkpoint: {os.path.relpath(args.ckpt, ROOT)}")
    m = ax.load_model(args.ckpt)
    print(f"  d={m.d} f={m.f} layers={len(m.layers)} q_heads={m.q_heads} "
          f"head_dim={m.head_dim}")

    words, consts = gv.assemble_program(args.program)
    if consts["LAYERS"] != len(m.layers):
        raise SystemExit(
            f"{os.path.basename(args.program)} is built for LAYERS={consts['LAYERS']} "
            f"but the checkpoint has {len(m.layers)} — fix the kernel's .equ LAYERS")
    print(f"program   : {os.path.basename(args.program)}  ({len(words)} words)")

    _, tok_c, _ = quant.make_batch(args.calib, 99, T)
    qm = quant.prepare(m, tok_c, dyt_fixed=(args.dyt_scale == "fixed"),
                       use_qat={"auto": None, "qat": True, "calib": False}[args.scales])

    exprs, tok_e, _ = quant.make_batch(args.problems, 1234, T)
    X0 = quant.quantize_input(m, tok_e, qm.s_x0)             # [N, T, D] int8
    tgt = tok_e[:, quant.ANS:]

    # ---- one staging pass to get the constant image ------------------------
    weights, _ = split_image(stage(qm, m, consts, X0[0]), consts)
    print(f"image     : {len(weights)} constant bytes (weights, mask, fc, requant words)")

    # ---- backends ----------------------------------------------------------
    iss = IssBackend(words, consts)
    iss.load_weights(weights)

    uart = None
    if args.dry_run:
        dev = iss
        print("backend   : ISS (--dry-run; nothing is sent to a board)")
    else:
        from run_program import autodetect_port
        port = args.port or autodetect_port()
        uart = TPUUart(port, args.baud, args.timeout).__enter__()
        dev = BoardBackend(uart, words, consts, args)
        dev.load_weights(weights)
        print(f"backend   : FPGA on {port} @ {args.baud} baud")

    # ---- the loop ----------------------------------------------------------
    print(f"\nscoring {args.problems} problems on the {dev.name} "
          f"(answer region [{quant.ANS}, {T})):")
    preds, ref_preds, first_diff = [], [], None
    wrong = 0
    if args.show:
        print(f"  {'#':>4s}  {'problem':<16s} {'expected':>12s} {'device':>12s}"
              f"  {'':9s} {'run':>10s}")
    t0 = time.monotonic()
    try:
        for i in range(args.problems):
            _, inputs = split_image(stage(qm, m, consts, X0[i]), consts)
            logits = dev.run_problem(inputs)
            pred = logits.argmax(-1)[quant.ANS - 1:-1]
            preds.append(pred)

            ok = bool((pred == tgt[i]).all())
            wrong += not ok
            if args.show:
                # The device's own clocks for this problem, if it kept any.
                ms = (f"{dev.counters[-1]['run'] / args.clk_mhz / 1e3:.2f} ms"
                      if len(dev.counters) == i + 1 else "")
                print(f"  {i:>4d}  {decode_prompt(exprs[i]):<16s} "
                      f"{decode_answer(tgt[i]):>12s} {decode_answer(pred):>12s}"
                      f"  {'' if ok else '<-- WRONG':9s} {ms:>10s}")

            if args.compare_iss and dev is not iss:
                want = iss.run_problem(inputs)
                if first_diff is None and bool((logits != want).any()):
                    t, v = (logits != want).nonzero()[0].tolist()
                    first_diff = (i, t, v, int(logits[t, v]), int(want[t, v]))
            if not args.show and ((i + 1) % 8 == 0 or i + 1 == args.problems):
                print(f"  {i + 1}/{args.problems} problems "
                      f"({time.monotonic() - t0:.1f}s)  {wrong} wrong")
    finally:
        if uart is not None:
            # Only fall back to the one-shot report when the loop collected
            # nothing — otherwise report_timing below covers every run, and
            # this would just print the last problem's counters twice.
            if isinstance(dev, BoardBackend) and not dev.counters:
                try:
                    report_run_time(uart, args.clk_mhz)
                except (ProtocolError, OSError):
                    pass
            uart.__exit__(None, None, None)

    if args.show:
        print(f"\n  {args.problems - wrong}/{args.problems} exact "
              f"({time.monotonic() - t0:.1f}s wall clock)")
    report_timing(dev.counters, args.clk_mhz)

    seq, tok = quant.score(torch.stack(preds), tgt)

    # The simulated number the board is being held to. Same checkpoint, same
    # scales, same problems — so a gap here is the hardware and nothing else.
    ref = quant.int_forward(qm, X0)
    ref_seq, ref_tok = quant.score(quant.predictions(ref), tgt)

    print(f"\naccuracy over {args.problems} problems:")
    print(f"  {'':22s} {'exact-sequence':>15s} {'token':>9s}")
    print(f"  {dev.name:22s} {seq * 100:14.2f}% {tok * 100:8.2f}%")
    print(f"  {'quant.int_forward':22s} {ref_seq * 100:14.2f}% {ref_tok * 100:8.2f}%")

    if first_diff is not None:
        i, t, v, got, want = first_diff
        print(f"\n  !! problem {i} logit[{t}][{v}]: device {got}  ISS {want}")
        return 1
    if (seq, tok) != (ref_seq, ref_tok):
        print("\n  !! the device did not reproduce the simulated accuracy — "
              "rerun with --compare-iss to localize")
        return 1
    print(f"\nPASSED: the {dev.name} reproduces the simulated integer model exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
