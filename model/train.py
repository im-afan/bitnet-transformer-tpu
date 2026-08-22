import argparse
import datetime
import os

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import model.numbers_data as numbers_data
import model.transformer as transformer
from model.transformer import Model

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def train(
    model: Model,
    optim,
    epochs=100,
    batches=10000,
    mini_batch_size=32,
    batch_size=32,
    max_tokens=32,
    max_digits=5,
    save_freq=100,
    grad_clip=1.0,
):
    model = model.to(device)

    steps_per_batch = batch_size // mini_batch_size

    steps = 0
    avg_loss = 0
    batch_steps = 0
    saved_models = []

    # Gradient-norm bookkeeping, reported alongside the loss. clip_grad_norm_
    # returns the norm *before* clipping, which is the diagnostic that matters
    # when chasing a blow-up: the mean says whether the run is drifting, the max
    # says whether a single batch spiked.
    gnorm_sum = 0.0
    gnorm_max = 0.0
    gnorm_count = 0
    clipped = 0

    for i in range(epochs):
        for j in range(batches):
            batch, tokens, attn_mask = numbers_data.create_addition_batch(
                mini_batch_size, max_tokens, max_digits=max_digits
            )
            tokens = torch.tensor(tokens).to(device)
            # print(attn_mask)
            attn_mask = torch.stack(attn_mask).to(device)

            pred = model(tokens, attn_mask)  # (B, T, vocab)

            ans_pos = numbers_data.EQUALS_POS
            pred = pred[:, ans_pos - 1 : -1].flatten(0, 1)
            y = tokens[:, ans_pos:].flatten(0, 1)
            loss = F.cross_entropy(pred, y)
            avg_loss += loss.item()

            loss.backward()

            steps += 1
            batch_steps += 1
            if batch_steps % steps_per_batch == 0:
                # Clip the fully accumulated gradient, i.e. immediately before the
                # step. Clipping each micro-batch instead would bound the partial
                # sums separately and give a different (smaller) effective bound.
                if grad_clip:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    ).item()
                    gnorm_sum += gnorm
                    gnorm_max = max(gnorm_max, gnorm)
                    gnorm_count += 1
                    clipped += gnorm > grad_clip

                optim.step()
                optim.zero_grad()

            if steps % save_freq == 0:
                avg_loss /= save_freq
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"model/saved/test_model_{timestamp}.pt"
                torch.save(model.state_dict(), filepath)

                saved_models.append(filepath)
                if len(saved_models) > 3:
                    old_model = saved_models.pop(0)
                    if os.path.exists(old_model):
                        os.remove(old_model)

                msg = f"Epoch {i} training loss: {avg_loss}"
                if gnorm_count:
                    msg += (
                        f"  grad_norm mean {gnorm_sum / gnorm_count:.3f}"
                        f" max {gnorm_max:.3f}"
                        f"  clipped {clipped}/{gnorm_count}"
                    )
                print(msg)

                avg_loss = 0
                gnorm_sum = gnorm_max = 0.0
                gnorm_count = clipped = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=["vanilla", "gqa", "int4_vanilla"],
        default="vanilla",
        help="Which adder architecture to use",
    )
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--max_tokens", type=int, default=32, help="Context length (sequence length)"
    )
    parser.add_argument(
        "--max_digits",
        type=int,
        default=5,
        help="Max digits per operand in addition expressions",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max global grad norm, applied to the accumulated gradient just "
        "before each optimizer step. 0 disables clipping.",
    )
    args = parser.parse_args()

    vocab_size = len(numbers_data.VOCAB)

    if args.arch == "gqa":
        model = transformer.adder_gqa()
    elif args.arch == "vanilla":
        model = transformer.adder_vanilla()
    # elif args.arch == "int4_vanilla":
    else:
        model = transformer.adder_int4_vanilla()

    optim = Adam(model.parameters(), lr=1e-3)
    train(
        model,
        optim,
        save_freq=args.save_freq,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_digits=args.max_digits,
        grad_clip=args.grad_clip,
    )
