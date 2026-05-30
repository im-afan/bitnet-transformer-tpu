import torch
import transformer
from transformer import Model
import numbers_data

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")

# choose architecture
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', choices=['vanilla', 'gqa'], default='vanilla', help='Which adder architecture to use')
parser.add_argument('--model-path', type=str, default='saved/really_good_model_colab.pt')
args = parser.parse_args()

if args.arch == 'gqa':
	model = transformer.adder_gqa()
else:
	model = transformer.adder_vanilla()

state = torch.load(args.model_path, map_location="cpu")
model.load_state_dict(state)
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