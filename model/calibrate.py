"""Calibrate a saved ternary model for symmetric per-tensor int8 activations.

Loads a trained ternary checkpoint, pushes a batch of synthetic addition
expressions through it while every ``TernaryLinear`` records the absmax of the
activations entering its matmul, then sets one symmetric int8 scalar per layer
(``scale = absmax / 127``) and writes out a calibrated checkpoint.

Run from the repo root:

    python -m model.calibrate \
        --model-path model/saved/colab_ternary_mha.pt \
        --out model/saved/colab_ternary_mha_int8act.pt
"""

import argparse

import torch

import model.numbers_data as numbers_data
import model.transformer as transformer

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def make_calibration_batches(n_batches, batch_size, max_tokens, max_digits):
    """Pre-generate calibration batches as (tokens, attn_mask) tensor pairs."""
    batches = []
    for _ in range(n_batches):
        _, tokens, attn_mask = numbers_data.create_addition_batch(
            batch_size, max_tokens, max_digits=max_digits
        )
        tokens = torch.tensor(tokens).to(device)
        attn_mask = torch.stack(attn_mask).to(device)
        batches.append((tokens, attn_mask))
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/saved/colab_ternary_mha_small.pt")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the calibrated checkpoint "
        "(default: <model-path stem>_int8act.pt)",
    )
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-digits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        stem = args.model_path[:-3] if args.model_path.endswith(".pt") else args.model_path
        out_path = f"{stem}_int8act.pt"

    torch.manual_seed(args.seed)

    model = transformer.adder_ternary_vanilla()
    state = torch.load(args.model_path, map_location="cpu")
    # strict=False: the checkpoint predates the act_scale buffers, so they show
    # up as `missing` and keep their NaN init until calibration overwrites them.
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    model = model.to(device)

    batches = make_calibration_batches(
        args.batches, args.batch_size, args.max_tokens, args.max_digits
    )

    def run_forward(m):
        for tokens, attn_mask in batches:
            m(tokens, attn_mask)

    scales = transformer.calibrate_activations(model, run_forward)

    print(f"calibrated {len(scales)} TernaryLinear layers on "
          f"{args.batches} x {args.batch_size} examples")
    for name, scale in scales.items():
        print(f"  {name:<40} act_scale={scale:.6g}")

    torch.save(model.state_dict(), out_path)
    print(f"saved calibrated checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
