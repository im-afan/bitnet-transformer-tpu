#!/usr/bin/env python3
"""adder_export.py — take the real ternary adder checkpoint all the way to integers.

The last leg of [`adder_kernel.md`](adder_kernel.md) §7: everything else proves
`examples/adder_model.tpu` computes what the model *specifies* (the ISS matches an
independent PyTorch implementation, and the RTL matches the ISS, byte for byte).
None of it says the fully-integer model still does addition. That is what this
measures.

**Where the arithmetic lives.** Calibration, the requant/DyT words and the
integer forward are all `model/quant.py` — this module only *stages* them into
the kernel's DRAM layout and runs it. That split is deliberate: the pipeline has
to follow the model as it changes (it has already gained a double residual and
DyT since the first version of this script), and two copies of it drifted apart
once already. Here `int_forward` is imported, not reimplemented, so "the ISS
agrees with the reference" and "the reference is what quant.py benchmarks" are
the same statement.

Three stages:

  1. **prepare**  — `model/quant.py` calibrates on a set disjoint from the
     evaluation set and derives the 14 `{m0,n}` words per layer (§4 of the doc).
     Two of the fourteen are consumed by `dyt` rather than `requant`; the word
     format is identical, so DRAM staging does not care which.
  2. **evaluate** — float baseline vs the integer pipeline over N problems,
     plus the saturation rates §8 says are the diagnostic.
  3. **iss-check** — pack one real problem into the kernel's DRAM layout, run
     `examples/adder_model.tpu` in the ISS, and compare the device's int32
     logits against `quant.int_forward`'s. When that passes, the accuracy
     numbers above describe the actual kernel rather than a model of it.

    python accel/tpulang/adder_export.py                    # 256 problems
    python accel/tpulang/adder_export.py -n 512 --iss-check
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model.quant as quant                                     # noqa: E402
import model.transformer as transformer                         # noqa: E402

T = 32                            # train.py --max_tokens; the kernel's cfg tlen
CKPT = os.path.join(ROOT, "model", "saved", "ternary_mha.pt")

# The requant slots, in the order adder_model.tpu reads them out of its layer
# block. Taken from quant.py rather than restated so the two cannot disagree
# about the order — which would be a silent, catastrophic mis-staging.
SLOTS = quant.SLOTS


def load_model(path: str = CKPT) -> transformer.Model:
    m = transformer.adder_ternary_vanilla()
    state = torch.load(path, map_location="cpu")
    # strict=False: checkpoints predating the act_scale buffers list them as
    # missing. They are unused here — every scale is derived by quant.prepare.
    missing, unexpected = m.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    m.eval()
    return m


# =============================================================================
# Staging: a QuantModel -> the kernel's DRAM image.
# =============================================================================
def stage_dram(tpu, consts: dict, qm: quant.QuantModel, X0: torch.Tensor) -> None:
    """Write everything `examples/adder_model.tpu` reads into `tpu.dram`.

    `X0` is one `[T, D]` int8 sequence — the host's embedding + positional
    encoding, already at layer 0's input scale. Addresses come from the
    program's own `.equ` table, so a layout change moves both sides at once.
    """
    D, VOCAB, WCOL = consts["D"], consts["VOCAB"], consts["WCOL"]

    def put(addr: int, byte: int) -> None:
        tpu.dram[tpu._d(addr)] = byte & 0xFF

    for t in range(T):
        for d in range(D):
            put(consts["D_XIN"] + t * D + d, int(X0[t, d]))
        for s in range(T):
            put(consts["D_MASK"] + t * T + s, int(qm.mask[t, s]))

    # `Model.fc` is a plain nn.Linear, so the device runs it on the VPU with
    # int8 weights. quant.QLinear holds every weight as [in, out]; the kernel's
    # vecmatmul contracts src1 over its own rows, so it wants [VOCAB][D].
    wfc = qm.fc.w.t().contiguous()
    for v in range(VOCAB):
        for d in range(D):
            put(consts["D_WFC"] + v * D + d, int(wfc[v, d]))

    def pack(base: int, tmat: torch.Tensor) -> None:
        """[K][N] trits -> column-major 2-bit packed (00=0, 01=+1, 11=-1)."""
        K, N = tmat.shape
        for n in range(N):
            colint = 0
            for k in range(K):
                t = int(tmat[k, n])
                colint |= (0 if t == 0 else (1 if t > 0 else 0b11)) << (2 * k)
            for b in range(WCOL):
                put(base + n * WCOL + b, (colint >> (8 * b)) & 0xFF)

    for li, lay in enumerate(qm.layers):
        base = consts["D_L0"] + li * consts["LSTRIDE"]
        w = lay["w"]
        pack(base + consts["O_WQKV"],
             torch.cat([w["Wq"].w, w["Wk"].w, w["Wv"].w], dim=1))
        pack(base + consts["O_WO"], w["Wo"].w)
        pack(base + consts["O_W1"], w["W1"].w)
        pack(base + consts["O_W2"], w["W2"].w)
        for i, k in enumerate(SLOTS):
            m0, n = lay["words"][k]
            word = (n << quant.M0_W) | m0
            for b in range(4):
                put(base + consts["O_RQW"] + 4 * i + b, (word >> (8 * b)) & 0xFF)


def read_logits(tpu, consts: dict, vocab: int) -> torch.Tensor:
    """The kernel's result: `[T, VOCAB]` int32, little-endian, from D_LOG."""
    out = []
    for t in range(T):
        row = []
        for v in range(vocab):
            a = consts["D_LOG"] + t * (vocab * 4) + v * 4
            u = sum(tpu.dram[tpu._d(a + b)] << (8 * b) for b in range(4))
            row.append(u - (1 << 32) if u >= (1 << 31) else u)
        out.append(row)
    return torch.tensor(out, dtype=torch.int64)


# =============================================================================
# Cross-check: the real kernel, on real weights, in the ISS.
# =============================================================================
def iss_check(model, qm: quant.QuantModel, tokens_1: torch.Tensor,
              program: str) -> int:
    """Run one real problem through `examples/adder_model.tpu` in the ISS.

    This is what makes the accuracy numbers statements about the *kernel*
    rather than about `model/quant.py`.
    """
    import gen_vectors as gv
    from iss import TPU

    print("\nISS cross-check on the real checkpoint:")
    words, consts = gv.assemble_program(program)
    print(f"  {os.path.basename(program)}: {len(words)} instruction words")
    tpu = TPU(rows=gv.ROWS, cols=gv.COLS, addr_w=gv.ADDR_W,
              mem_addr_w=gv.MEM_ADDR_W, m0_w=gv.M0_W, n_w=gv.N_W)

    X0 = quant.quantize_input(model, tokens_1, qm.s_x0)      # [1, T, D]
    stage_dram(tpu, consts, qm, X0[0])
    tpu.run(words, max_steps=2_000_000)

    got = read_logits(tpu, consts, model.vocab_size)
    want = quant.int_forward(qm, X0)[0]

    bad = int((got != want).sum())
    print(f"  logits [{T}x{model.vocab_size}] int32: {bad}/{got.numel()} differ")
    if bad:
        where = (got != want).nonzero()[:4]
        for t, v in where.tolist():
            print(f"    [{t}][{v}]: device {int(got[t, v]):8d}  "
                  f"reference {int(want[t, v]):8d}")
        print("  !! the ISS and model/quant.py disagree — the accuracy numbers "
              "above do not describe the kernel")
        return 1
    print("  exact — the accuracy numbers above are the kernel's")
    return 0


# =============================================================================
# Driver.
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--problems", type=int, default=256)
    ap.add_argument("-c", "--calib", type=int, default=128,
                    help="problems in the calibration set (disjoint from the eval set)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--program", default=os.path.join(HERE, "examples", "adder_model.tpu"))
    ap.add_argument("--dyt-scale", choices=["fixed", "calibrated"], default="fixed",
                    help="fixed pins a DyT output to 1/127, so `dyt`'s clip IS the "
                         "hardtanh; calibrated uses absmax/127 instead")
    ap.add_argument("--words", action="store_true", help="print the requant words")
    ap.add_argument("--iss-check", action="store_true",
                    help="also run one problem through the ISS on the real kernel")
    args = ap.parse_args(argv)

    print(f"checkpoint: {os.path.relpath(args.ckpt, ROOT)}")
    m = load_model(args.ckpt)
    print(f"  d={m.d} f={m.f} layers={len(m.layers)} q_heads={m.q_heads} "
          f"kv_heads={m.kv_heads} head_dim={m.head_dim}")
    alphas = [(quant.dyt_alpha(l.norm1), quant.dyt_alpha(l.norm2)) for l in m.layers]
    print("  DyT alpha per layer: " +
          "  ".join(f"L{i}=({a1:.3f}, {a2:.3f})" for i, (a1, a2) in enumerate(alphas)))

    # The kernel is fixed at T=32 (cfg tlen) and the model's answer slice is
    # pinned to EQUALS_POS, so the benchmark geometry is not a free parameter.
    rep = quant.benchmark(m, n_problems=args.problems, n_calib=args.calib,
                          max_tokens=T, dyt_fixed=(args.dyt_scale == "fixed"))

    if args.words:
        quant.report_words(rep.qm)
    quant.report_clips(rep)

    print(f"\naccuracy over {rep.n_problems} problems (T={T}):")
    print(f"  {'':22s} {'exact-sequence':>15s} {'token':>9s}")
    print(f"  {'float (quant off)':22s} {rep.float_seq * 100:14.2f}% {rep.float_tok * 100:8.2f}%")
    print(f"  {'int8 activations':22s} {rep.int_seq * 100:14.2f}% {rep.int_tok * 100:8.2f}%")
    print(f"  the two agree on {rep.agreement * 100:.2f}% of sequences")

    if args.iss_check:
        # Same seed as quant.benchmark's evaluation set, so the problem the ISS
        # runs is one of the problems that was just scored.
        _, tok_e, _ = quant.make_batch(args.problems, 1234, T)
        return iss_check(m, rep.qm, tok_e[0:1], args.program)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
