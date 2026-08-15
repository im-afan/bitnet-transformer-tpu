import random
from typing import List, Tuple
import torch

VOCAB = {str(i): i for i in range(10)}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
PAD_TOKEN = 'N'
PAD_ID = 12
VOCAB.update({'+': 10, '=': 11, 'N': PAD_ID})
MAX_INTEGER = 99999
MIN_TOKEN_LENGTH = 5  # e.g. "0+0=0"
EQUALS_POS = 15

# Every number in a generated expression is written least-significant digit
# first: "123+45=168" is emitted as "321+54=861".
#
# This is for learnability, and the answer is the half that matters. Addition
# carries propagate from the ones digit upward, which is the direction an
# autoregressive model *cannot* look: emitting the answer most-significant
# first asks the model to know every carry before it writes the first digit.
# Reversed, answer digit k depends only on operand digits 0..k and the carry
# out of digit k-1 — the token it just emitted.
#
# The second, smaller win is positional. The answer is left-aligned at
# EQUALS_POS and padded on the right, so with the digits reversed, position
# EQUALS_POS+k is *always* the 10^k place. Unreversed it is a different place
# value for every answer length, so the model has to learn the alignment
# separately at each magnitude.
#
# Operands are reversed for the same alignment reason (position 0 is now always
# the left operand's ones digit). The right operand still begins at a
# length-dependent offset after '+'; fixing that would mean padding the
# operands to a fixed width, which moves '+' and is a larger change.
REVERSE_DIGITS = True


def _digits(n: int) -> str:
    """Render an integer in the generator's digit order."""
    s = str(n)
    return s[::-1] if REVERSE_DIGITS else s

def tokenize(expression: str, max_tokens: int = None):
    """Convert an expression like '123+45=168' into token ids."""
    token_ids = [VOCAB[ch] for ch in expression]
    if max_tokens is None:
        return token_ids
    if len(token_ids) > max_tokens:
        raise ValueError(f"Expression {expression!r} has {len(token_ids)} tokens, exceeds max_tokens={max_tokens}")
    token_ids = token_ids + [PAD_ID] * (max_tokens - len(token_ids))

    mask = torch.tensor([-1e9 if token_ids[i] == PAD_ID else 0 for i in range(max_tokens)])
    mask = mask.repeat(max_tokens, 1)
    mask = mask + mask.T

    return token_ids, mask


def detokenize(token_ids: List[int]) -> str:
    """Convert token ids back into a string expression, in generator order.

    Digits stay least-significant first — this is the inverse of ``tokenize``,
    not a display function. Use :func:`unreverse_expression` to read it.
    """
    return ''.join(INV_VOCAB[token_id] for token_id in token_ids if token_id != PAD_ID)


def unreverse_expression(expr: str) -> str:
    """Restore human digit order for display: ``321+54=861`` -> ``123+45=168``.

    Tolerant of malformed input, because model samples are not guaranteed to
    parse: anything that is not a run of digits is passed through untouched.
    """
    if not REVERSE_DIGITS:
        return expr

    out = []
    run = ''
    for ch in expr:
        if ch.isdigit():
            run += ch
        else:
            out.append(run[::-1])
            run = ''
            out.append(ch)
    out.append(run[::-1])
    return ''.join(out)


def _sample_number(max_digits: int) -> int:
    """Sample an integer where each digit-length 1..max_digits is equally likely."""
    n_digits = random.randint(1, max_digits)
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
    hi = 10 ** n_digits - 1
    return random.randint(lo, hi)


def generate_addition_expression(max_digits: int = 5, max_length: int = 32) -> str:
    """Generate a random addition expression where each operand length is equally likely.

    Digits are least-significant first (see REVERSE_DIGITS): 123+45=168 is
    generated as ``321+54NNNNNNNN861NN...``. Lengths are unchanged, so
    EQUALS_POS and the ``max_digits <= 6`` limit still hold.
    """
    left = _sample_number(max_digits)
    right = _sample_number(max_digits)
    expr = f"{_digits(left)}+{_digits(right)}"
    expr += PAD_TOKEN * (EQUALS_POS - len(expr) - 1)
    expr += f"={_digits(left + right)}"
    expr += PAD_TOKEN * (max_length - len(expr))
    return expr


def create_addition_batch(batch_size: int, max_tokens: int, max_digits: int = 5) -> Tuple[List[str], List[List[int]]]:
    """Create a batch of addition expressions and corresponding token id sequences."""
    if max_tokens < MIN_TOKEN_LENGTH:
        raise ValueError(f"max_tokens must be at least {MIN_TOKEN_LENGTH}")

    expressions: List[str] = []
    token_batches: List[List[int]] = []
    attention_masks = []

    for _ in range(batch_size):
        while True:
            expr = generate_addition_expression(max_digits=max_digits, max_length=max_tokens)
            if len(expr) <= max_tokens:
                break
        expressions.append(expr)
        token_ids, mask = tokenize(expr, max_tokens=max_tokens)
        token_batches.append(token_ids)
        attention_masks.append(mask)

    return expressions, token_batches, attention_masks
