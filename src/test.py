import transformer
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token


model = Model(tokenizer.vocab_size, 1024, 2)
model.load_state_dict(torch.load("saved/test_model.pt", weights_only=False))
model.eval()

text = "The quick brown fox jumps"

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)
input_tokens = inputs["input_ids"]

probs = model(input_tokens)

print(probs)
print(tokenizer.decode(model.predict_token(probs)))