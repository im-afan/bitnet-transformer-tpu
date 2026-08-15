import torch
import model.transformer as transformer
from model.transformer import Model
import model.numbers_data as numbers_data
import time

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")

def time_func(f, samples, *args, **kwargs):
	mean = 0	
	for i in range(samples):
		start = time.time()
		f(*args, **kwargs)
		end = time.time()
		mean += end - start
	mean /= samples
	return mean 

# choose architecture
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', choices=['vanilla', 'gqa', 'ternary'], default='ternary', help='Which adder architecture to use')
parser.add_argument('--model-path', type=str, default='saved/colab_ternary_mha_small.pt')
args = parser.parse_args()

if args.arch == 'gqa':
	model = transformer.adder_gqa()
elif args.arch == 'vanilla':
	model = transformer.adder_vanilla()
else:
	model = transformer.adder_ternary_vanilla()

state = torch.load(args.model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()
model = model.to(device)


ans_pos = numbers_data.EQUALS_POS

batch, tokens, attn_mask = numbers_data.create_addition_batch(1, 64)
attn_mask = torch.stack(attn_mask).to(device)
x = torch.tensor(tokens).to(device)
y = torch.tensor(tokens)[:, ans_pos:].to(device)
pred = model(x, attn_mask)

print("expr: ", batch, len(batch[0]))
print("expr (readable): ", [numbers_data.unreverse_expression(e) for e in batch])
print("expected: ", y)
print("prediction: ", model.sample_pred_best(pred)[:, ans_pos-1:-1])