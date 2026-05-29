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
    """Convert token ids back into a string expression."""
    return ''.join(INV_VOCAB[token_id] for token_id in token_ids if token_id != PAD_ID)


def generate_addition_expression(max_digits: int = 5, max_length: int = 32) -> str:
    """Generate a random addition expression using integers up to max_digits digits."""
    left = random.randint(0, 10**max_digits - 1)
    right = random.randint(0, 10**max_digits - 1)
    expr = f"{left}+{right}"
    expr += PAD_TOKEN * (EQUALS_POS - len(expr) - 1)
    expr += f"={left+right}" 
    expr += PAD_TOKEN * (max_length - len(expr))
    return expr


def create_addition_batch(batch_size: int, max_tokens: int) -> Tuple[List[str], List[List[int]]]:
    """Create a batch of addition expressions and corresponding token id sequences."""
    if max_tokens < MIN_TOKEN_LENGTH:
        raise ValueError(f"max_tokens must be at least {MIN_TOKEN_LENGTH}")

    expressions: List[str] = []
    token_batches: List[List[int]] = []
    attention_masks = []

    for _ in range(batch_size):
        while True:
            expr = generate_addition_expression(max_digits=5, max_length=max_tokens)
            if len(expr) <= max_tokens:
                break
        expressions.append(expr)
        token_ids, mask = tokenize(expr, max_tokens=max_tokens)
        token_batches.append(token_ids)
        attention_masks.append(mask)

    return expressions, token_batches, attention_masks
