import transformer
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-10k")

if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token


model = Model(tokenizer.vocab_size, embedding_dim=384, n_transformers=8, heads=12, d_ff=512)
model.load_state_dict(torch.load("saved/test_model_20260514_154722.pt", map_location="cpu", weights_only=True))
model.eval()

model = model.to(device)

text = "It was"

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)
input_tokens = inputs["input_ids"].to(device)

model_out = model.inference(input_tokens, max_len=128, temperature=5)

print(model_out)
print(tokenizer.decode(model_out))