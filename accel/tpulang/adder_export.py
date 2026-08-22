#!/usr/bin/env python3
"""adder_export.py — a real checkpoint through the firmware adder kernel.

Takes `model/saved/*.pt` (an `adder_int4_vanilla` QAT checkpoint), turns it into
the integers `accel/tpu/fw/adder.c` expects, runs that kernel's command trace on
`iss.py`, and scores the result on the addition task. Nothing here is an
estimate: the ISS is bit-exact with the RTL (`make fw FWPROG=adder` checks the
two against each other byte for byte), so the accuracy this prints is what the
device scores, not a model of it.

    python accel/tpulang/adder_export.py -n 256
    python accel/tpulang/adder_export.py -n 64 --emit-rq accel/tpu/fw/adder_rq_ckpt.h

Three things happen, in order:

1. **Derive.** Every `Int4Linear` weight becomes int4 codes plus one scale;
   every `ActQuant` gives up its LSQ-learned scale. The two combine into the 16
   `{m0,n}` words per layer that `adder.c`'s enum indexes — the table below is
   the only place those multipliers are written down.
2. **Build the trace.** The rq table is a *compile-time* input to the kernel:
   the requant word is a literal in the macro-op, so unlike the retired scalar
   ISA there is no path by which the device could fetch it from memory. So this
   writes a header and compiles `adder.c` natively against it (`-DTPU_TRACE`,
   host cc, no cross toolchain and no board) to get the command trace.
3. **Run.** Weights, the causal mask and the output head are staged into DRAM
   once — they are read-only, exactly as they would stay resident in the board's
   SRAM across forwards — and then each problem stages only its embedded `X0`
   and re-runs the same 518 commands.

The host owns the two ends, structurally: the token embedding (the ISA has no
gather) and the argmax over 13 logits (nothing returns an index).

WHERE THE MULTIPLIERS COME FROM. Every tensor carries a compile-time scale;
integer `v` means real `v * s`. A requant's multiplier is
`s_in * s_weight / s_out`, and the constraints that are *not* free choices:

- `vecadd` adds two int8 operands at one scale, so a residual add pins its
  addend's output scale. `RQ_O` and `RQ_XO` land on `s_x`, `RQ_F` on `s_x1`.
  The model expresses that by *sharing the ActQuant instance* (`q_o is q_xo is
  x_quant`), which is why the pinning is asserted here rather than assumed.
- `1/sqrt(head_dim)` is not an op; it folds into `RQ_S`.
- DyT is the `dyt` instruction, not an extra pass: with the output scale pinned
  to 1/7 the saturating narrow the residual add already needed *is* the
  hardtanh, with `alpha` folded into its multiplier.
- The `{1,0}` identities (`RQ_KP`, `RQ_VP`, `RQ_ID`, `RQ_P`, `RQ_XO`, `RQ_HR`)
  are identities by construction, not by tuning: their input is already on the
  output's grid and scale.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import model.numbers_data as numbers_data  # noqa: E402
import model.transformer as transformer  # noqa: E402
from fw_vectors import put_rowmajor_i4  # noqa: E402
from iss import TPU, parse_trace  # noqa: E402

FW_DIR = os.path.join(REPO, "accel", "tpu", "fw")

# ---- geometry. Must agree with fw/adder.c, and is checked against the model. --
T, D, DFF, NH, LAYERS = 32, 64, 256, 4, 4
DH = D // NH
VOCAB, VPAD = 13, 16
ROWS = COLS = 8

# ---- DRAM map (fw/adder.c) ---------------------------------------------------
DR_X, DR_MASK, DR_WFC, DR_LOG = 0x00000, 0x00800, 0x01400, 0x01800
DR_LAYER, DR_LSTEP = 0x02000, 0x06000
LW_QKV, LW_O, LW_1, LW_2 = 0x0000, 0x1800, 0x2000, 0x4000

# ---- the 16 requant sites, in fw/adder.c's enum order ------------------------
RQ_NAMES = ["Q", "K", "V", "KP", "VP", "S", "ID", "P", "A", "O", "XO", "X1",
            "H", "HR", "F", "X2"]
RQ_N = len(RQ_NAMES)
RQ_IDX = {n: i for i, n in enumerate(RQ_NAMES)}

M0_W, N_W = 12, 4
Q4_MIN, Q4_MAX = -8, 7


# =============================================================================
# Fixed point.
# =============================================================================
def fixed_point(m: float, what: str) -> tuple:
    """The `{m0, n}` pair closest to real multiplier `m`.

    `requant` computes `clip((acc*m0 + 2**(n-1)) >> n)`, i.e. a multiplier
    `m0/2**n` with `m0 < 4096` and `n <= 15`. Take the **largest** `n` that
    keeps `m0` in range — every extra shift is another bit of precision on a
    multiplier that is usually much smaller than 1.
    """
    if not (m > 0):
        raise SystemExit(f"{what}: multiplier {m} is not positive — the hardware "
                         f"m0 is unsigned, so a negative scale (or a negative "
                         f"DyT alpha) cannot be represented")
    best = None
    for n in range(1 << N_W):
        m0 = math.floor(m * (1 << n) + 0.5)      # round-half-up, like the RTL
        if 0 < m0 < (1 << M0_W):
            best = (m0, n)
    if best is None:
        if m >= 1:
            print(f"  WARNING {what}: m = {m:.4g} exceeds the representable "
                  f"4095, clamping", file=sys.stderr)
            return ((1 << M0_W) - 1, 0)
        print(f"  WARNING {what}: m = {m:.4g} underflows m0 = 0 at n = 15; "
              f"this tensor will be all zeros", file=sys.stderr)
        return (0, 0)
    return best


def rq_word(m: float, what: str) -> int:
    m0, n = fixed_point(m, what)
    return (n << M0_W) | m0


RQ_ONE = (0 << M0_W) | 1


# =============================================================================
# The checkpoint -> integers.
# =============================================================================
def int4_weight(w: torch.Tensor) -> tuple:
    """`Int4Linear`'s own quantization: codes in [-8, 7] and one scale.

    Mirrors `Int4Linear.forward` exactly — `absmax/7`, clamped at eps=1e-5, and
    `RoundClip`. `w` is `[in_dim, out_dim]` (the transpose of `nn.Linear`), which
    is already the row-major orientation the array reads a weight in, so nothing
    is transposed on the way to DRAM.
    """
    scale = float((w.detach().abs().max() / transformer.INT4_QMAX).clamp_min(1e-5))
    codes = (w.detach() / scale).round().clamp(Q4_MIN, Q4_MAX).to(torch.int64)
    return codes, scale


def act_scale(q: transformer.ActQuant) -> float:
    """The site's scale as `ActQuant.forward` uses it — magnitude, floored.

    LSQ can drive a learned scale negative; the forward takes `.abs()`, so
    reading `.scale` raw would put a sign into a multiplier that has none.
    """
    return float(q.scale.detach().abs().clamp_min(1e-8))


def derive(model) -> tuple:
    """Return `(rq_table, weights)`: the 16 words per layer and every int4 block.

    `weights` is a dict of plain int64 tensors keyed by the DRAM block they go
    to; staging is a separate step so `--emit-rq` can skip it.
    """
    rq_table, weights = [], {}
    s_x = act_scale(model.q_embed)               # layer 0 enters on the embedding

    for L, lay in enumerate(model.layers):
        att = lay.attention
        s_q, s_k, s_v = (act_scale(att.q_q), act_scale(att.q_k), act_scale(att.q_v))
        s_s, s_a = act_scale(att.q_s), act_scale(att.q_a)
        s_x1, s_x2, s_h, s_f = (act_scale(lay.q_x1), act_scale(lay.q_x2),
                                act_scale(lay.q_h), act_scale(lay.q_f))
        alpha1, alpha2 = float(lay.norm1.alpha), float(lay.norm2.alpha)

        # The three pinnings the ISA forces. They hold in the model because the
        # sites *share an ActQuant instance*; if a retrain stops sharing them,
        # the kernel silently adds two tensors on different scales, so check.
        for site, pinned, name in ((att.q_o, s_x, "RQ_O"), (att.q_xo, s_x, "RQ_XO"),
                                   (lay.q_hr, s_h, "RQ_HR"), (att.q_p, s_s, "RQ_P")):
            got = act_scale(site)
            if abs(got - pinned) > 1e-9 * max(1.0, pinned):
                raise SystemExit(
                    f"layer {L}: {name}'s site is at {got:.6g} but the residual/"
                    f"identity it feeds needs {pinned:.6g}. vecadd takes two int8 "
                    f"operands at ONE scale — this kernel cannot express it.")
        if abs(s_f - s_x1) > 1e-9 * max(1.0, s_x1):
            raise SystemExit(f"layer {L}: RQ_F is at {s_f:.6g} but the second "
                             f"residual add needs s_x1 = {s_x1:.6g}")

        wq, a_q = int4_weight(att.Wq.w)
        wk, a_k = int4_weight(att.Wk.w)
        wv, a_v = int4_weight(att.Wv.w)
        wo, a_o = int4_weight(att.Wo.w)
        w1, a_1 = int4_weight(lay.ff[0].w)
        w2, a_2 = int4_weight(lay.ff[2].w)

        weights[(L, "qkv")] = torch.cat([wq, wk, wv], dim=1)   # [D][3D]
        weights[(L, "o")] = wo
        weights[(L, "w1")] = w1
        weights[(L, "w2")] = w2

        row = [0] * RQ_N
        row[RQ_IDX["Q"]] = rq_word(s_x * a_q / s_q, f"L{L} RQ_Q")
        row[RQ_IDX["K"]] = rq_word(s_x * a_k / s_k, f"L{L} RQ_K")
        row[RQ_IDX["V"]] = rq_word(s_x * a_v / s_v, f"L{L} RQ_V")
        # K and V are already int4 — whatever requant produced them clipped to
        # [-8, 7] — so the packs are pure repacks.
        row[RQ_IDX["KP"]] = row[RQ_IDX["VP"]] = RQ_ONE
        row[RQ_IDX["S"]] = rq_word(s_q * s_k / (math.sqrt(DH) * s_s), f"L{L} RQ_S")
        row[RQ_IDX["ID"]] = row[RQ_IDX["P"]] = RQ_ONE
        row[RQ_IDX["A"]] = rq_word(s_s * s_v / s_a, f"L{L} RQ_A")
        row[RQ_IDX["O"]] = rq_word(s_a * a_o / s_x, f"L{L} RQ_O")
        row[RQ_IDX["XO"]] = RQ_ONE
        row[RQ_IDX["X1"]] = rq_word(alpha1 * s_x / s_x1, f"L{L} RQ_X1")
        row[RQ_IDX["H"]] = rq_word(s_x1 * a_1 / s_h, f"L{L} RQ_H")
        row[RQ_IDX["HR"]] = RQ_ONE
        row[RQ_IDX["F"]] = rq_word(s_h * a_2 / s_x1, f"L{L} RQ_F")
        row[RQ_IDX["X2"]] = rq_word(alpha2 * s_x1 / s_x2, f"L{L} RQ_X2")
        rq_table.append(row)

        s_x = s_x2      # every later layer enters on the previous DyT, i.e. 1/7

    wfc, _ = int4_weight(model.fc.w)                           # [D][VOCAB]
    pad = torch.zeros(D, VPAD - VOCAB, dtype=torch.int64)
    weights["fc"] = torch.cat([wfc, pad], dim=1)               # [D][VPAD]
    return rq_table, weights


# =============================================================================
# The rq header, and the command trace built against it.
# =============================================================================
def write_rq_header(path: str, rq_table: list, note: str) -> None:
    # One object-like macro spanning several physical lines, so every line but
    # the last needs a continuation — a multi-line #define that forgets them
    # fails inside adder.c's initializer rather than here.
    rows = []
    for L, row in enumerate(rq_table):
        cells = ", ".join(f"0x{w:04x}u" for w in row)
        rows.append(f"    /* layer {L} */ {{ {cells} }}")
    body = ", \\\n".join(rows)
    with open(path, "w") as f:
        f.write(f"""/* Generated by accel/tpulang/adder_export.py — do not edit.
 *
 * {note}
 *
 * 16 {{m0,n}} words per layer, in fw/adder.c's enum order:
 *   {', '.join(RQ_NAMES)}
 */
#ifndef ADDER_RQ_GENERATED
#define ADDER_RQ_GENERATED

#define ADDER_RQ_INIT {{ \\
{body} \\
}}

#endif
""")


def build_trace(rq_header: str, workdir: str) -> list:
    """Compile `adder.c` natively against `rq_header` and return its trace.

    The same source the RISC-V image is built from, with tpu.h's two MMIO
    primitives swapped for the trace emitter — so the commands here cannot drift
    from the ones the firmware issues (docs/picorv32_migration.md §8.1).
    """
    exe = os.path.join(workdir, "adder.trace")
    cc = os.environ.get("HOSTCC", "cc")
    cmd = [cc, "-DTPU_TRACE", f'-DADDER_RQ_H="{os.path.abspath(rq_header)}"',
           "-I", FW_DIR, "-O1", "-o", exe,
           os.path.join(FW_DIR, "adder.c"),
           os.path.join(FW_DIR, "mock", "tpu_trace.c")]
    subprocess.run(cmd, check=True)
    out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    return parse_trace(out)


# =============================================================================
# DRAM staging.
# =============================================================================
def stage_static(tpu: TPU, weights: dict) -> None:
    """Everything that does not change per problem: weights, mask, output head.

    Read-only for the whole run, which is the point — on the board this is one
    ~9 s upload and then every forward stages only the 2 KB `X0`.
    """
    img: dict = {}
    for t in range(T):
        for s in range(T):
            # 0 where s <= t, -8 above. S is int4, so S-8 <= -1 for every value
            # in range and ReLU takes a masked entry to exactly zero.
            img[DR_MASK + t * T + s] = (0 if s <= t else -8) & 0xFF

    fc = weights["fc"]
    put_rowmajor_i4(img, DR_WFC, D, VPAD, lambda r, c: int(fc[r][c]))

    for L in range(LAYERS):
        base = DR_LAYER + L * DR_LSTEP
        qkv, wo = weights[(L, "qkv")], weights[(L, "o")]
        w1, w2 = weights[(L, "w1")], weights[(L, "w2")]
        put_rowmajor_i4(img, base + LW_QKV, D, 3 * D, lambda r, c: int(qkv[r][c]))
        put_rowmajor_i4(img, base + LW_O, D, D, lambda r, c: int(wo[r][c]))
        put_rowmajor_i4(img, base + LW_1, D, DFF, lambda r, c: int(w1[r][c]))
        put_rowmajor_i4(img, base + LW_2, DFF, D, lambda r, c: int(w2[r][c]))

    for addr, byte in img.items():
        tpu.dram[addr] = byte


def stage_input(tpu: TPU, x0: torch.Tensor) -> None:
    """`X0[T][D]` int8, row-major. `x0` is already int4 codes."""
    for t in range(T):
        row = x0[t]
        for d in range(D):
            tpu.dram[DR_X + t * D + d] = int(row[d]) & 0xFF


def read_logits(tpu: TPU) -> torch.Tensor:
    """`[T][VOCAB]` int32, out of the `[T][VPAD]` block the head writes.

    The row stride is VPAD and the column count is VOCAB; those are different
    numbers (§the head's padding) and conflating them is the obvious way to get
    a plausible wrong answer.
    """
    out = torch.zeros(T, VOCAB, dtype=torch.int64)
    for t in range(T):
        for j in range(VOCAB):
            a = DR_LOG + (t * VPAD + j) * 4
            v = sum(tpu.dram[a + b] << (8 * b) for b in range(4))
            out[t][j] = v - (1 << 32) if v >= (1 << 31) else v
    return out


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="model/saved/int4_d64_f256_l4.pt")
    ap.add_argument("-n", "--problems", type=int, default=64,
                    help="addition problems to score (each is one ~2 s ISS run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-rq", metavar="PATH",
                    help="write the requant header here and stop")
    ap.add_argument("--dump-rq", action="store_true",
                    help="print the 16 words per layer as {m0,n}")
    args = ap.parse_args()

    model = transformer.adder_int4_vanilla()
    state = torch.load(os.path.join(REPO, args.model_path)
                       if not os.path.isabs(args.model_path) else args.model_path,
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if (model.d, model.f, len(model.layers), model.q_heads, model.head_dim) != \
            (D, DFF, LAYERS, NH, DH):
        raise SystemExit(
            f"fw/adder.c is compiled for d={D} f={DFF} layers={LAYERS} "
            f"heads={NH} head_dim={DH}; this checkpoint is d={model.d} "
            f"f={model.f} layers={len(model.layers)} heads={model.q_heads} "
            f"head_dim={model.head_dim}")

    rq_table, weights = derive(model)

    if args.dump_rq or args.emit_rq:
        for L, row in enumerate(rq_table):
            cells = " ".join(f"{n}={w & 0xFFF}/2^{w >> M0_W}"
                             for n, w in zip(RQ_NAMES, row))
            print(f"layer {L}: {cells}")
    if args.emit_rq:
        write_rq_header(args.emit_rq, rq_table, f"from {args.model_path}")
        print(f"wrote {args.emit_rq}")
        return 0
    if args.problems == 0:      # `--dump-rq -n 0`: derive and stop
        return 0

    workdir = tempfile.mkdtemp(prefix="adder_export_")
    header = os.path.join(workdir, "adder_rq_ckpt.h")
    write_rq_header(header, rq_table, f"from {args.model_path}")
    records = build_trace(header, workdir)
    n_cmds = sum(1 for r in records if r[0] == "CMD")

    tpu = TPU(rows=ROWS, cols=COLS)
    stage_static(tpu, weights)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    exprs, tokens, masks = numbers_data.create_addition_batch(args.problems, T)
    tok = torch.tensor(tokens)
    attn_mask = torch.stack(masks)

    with torch.no_grad():
        ref_logits = model(tok, attn_mask)
    ref_pred = ref_logits.argmax(-1)

    # The host's whole share of the front end: embed, then quantize onto the
    # site the model itself learned. There is no positional encoding.
    s_x0 = act_scale(model.q_embed)
    with torch.no_grad():
        embedded = model.embedding(tok)
    x0 = (embedded / s_x0).round().clamp(Q4_MIN, Q4_MAX).to(torch.int64)

    ans = numbers_data.EQUALS_POS
    dev_ok_seq = dev_ok_tok = ref_ok_seq = ref_ok_tok = 0
    n_tok = 0
    # Argmax agreement is reported twice on purpose. The scored window is the
    # only part `train.py`'s loss ever saw; positions before it predict the
    # *prompt*, are untrained, and sit on near-tied logits, so they diverge on
    # nothing more than torch's round-half-to-even against the hardware's
    # round-half-up. Counting those in one number would hide the number that
    # means something.
    argmax_diff_ans = argmax_diff_all = 0

    print(f"kernel: {n_cmds} commands, {args.problems} problems, "
          f"{args.model_path}")
    for i in range(args.problems):
        stage_input(tpu, x0[i])
        tpu.run_trace(records)
        dev_logits = read_logits(tpu)
        dev_pred = dev_logits.argmax(-1)

        target = tok[i, ans:]
        d = dev_pred[ans - 1:-1]
        r = ref_pred[i, ans - 1:-1]
        argmax_diff_all += int((dev_pred != ref_pred[i]).sum())
        argmax_diff_ans += int((d != r).sum())
        dev_ok_tok += int((d == target).sum())
        ref_ok_tok += int((r == target).sum())
        dev_ok_seq += int(bool((d == target).all()))
        ref_ok_seq += int(bool((r == target).all()))
        n_tok += target.numel()

        if i < 4:
            # Digits are least-significant first on the wire; unreverse to read.
            got = numbers_data.detokenize([int(v) for v in d])
            print(f"  {numbers_data.unreverse_expression(exprs[i]).rstrip('N'):24s}"
                  f"  device says {numbers_data.unreverse_expression(got)}")
        if (i + 1) % 16 == 0 or i + 1 == args.problems:
            print(f"  {i + 1:4d}/{args.problems}  device {100 * dev_ok_seq / (i + 1):6.2f}% "
                  f"exact-seq  model {100 * ref_ok_seq / (i + 1):6.2f}%  "
                  f"argmax differs {argmax_diff_ans} in-window, "
                  f"{argmax_diff_all} overall", flush=True)

    n = args.problems
    print()
    print(f"{'':22s} {'exact-sequence':>16s} {'token':>10s}")
    print(f"{'QAT model (PyTorch)':22s} {100 * ref_ok_seq / n:15.2f}% "
          f"{100 * ref_ok_tok / n_tok:9.2f}%")
    print(f"{'TPU kernel (ISS)':22s} {100 * dev_ok_seq / n:15.2f}% "
          f"{100 * dev_ok_tok / n_tok:9.2f}%")
    print()
    print(f"argmax vs. the model it came from: {argmax_diff_ans} of {n_tok} "
          f"scored positions differ, {argmax_diff_all} of {n * T} overall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
