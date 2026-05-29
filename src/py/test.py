import torch
from transformer import Model
import numbers_data

# device = torch.device("cpu")
# if(torch.cuda.is_available()):
#     device = torch.device("cuda")
# elif(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
#     device = torch.device("mps")

model = Model(len(numbers_data.VOCAB), d=128, f=512, layers=6, q_heads=8, kv_heads=8)
model.load_state_dict(torch.load("saved/good_model_colab.pt", map_location="cpu", weights_only=True))
model.eval()
# model = model.to(device)

ans_pos = numbers_data.EQUALS_POS

batch, tokens, attn_mask = numbers_data.create_addition_batch(1, 32)
attn_mask = torch.stack(attn_mask)
x = torch.tensor(tokens)
y = torch.tensor(tokens)[:, ans_pos:]
pred = model(x, attn_mask)

print("expr: ", batch, len(batch[0]))
print("expected: ", y)
# print("logits: ", pred)

print("prediction: ", model.sample_pred_best(pred)[:, ans_pos-1:-1])