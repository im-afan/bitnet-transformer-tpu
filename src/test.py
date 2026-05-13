import transformer
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model

tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-10k")

if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token


model = Model(tokenizer.vocab_size, embedding_dim=128, n_transformers=4, heads=4)
model.load_state_dict(torch.load("saved/test_model_20260513_042742(2).pt", map_location="cpu", weights_only=True))
model.eval()

text = "One day "

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