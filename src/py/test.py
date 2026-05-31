import torch
import transformer
from transformer import Model
import numbers_data
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
parser.add_argument('--arch', choices=['vanilla', 'gqa'], default='vanilla', help='Which adder architecture to use')
parser.add_argument('--model-path', type=str, default='saved/colab_vanilla_mha.pt')
parser.add_argument('--custom-attention', type=bool, default=True)
args = parser.parse_args()

if args.arch == 'gqa':
	model = transformer.adder_gqa()
else:
	model = transformer.adder_vanilla()

state = torch.load(args.model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()
model = model.to(device)
# model.use_custom_attention = args.custom_attention


ans_pos = numbers_data.EQUALS_POS

batch, tokens, attn_mask = numbers_data.create_addition_batch(32, 64)
attn_mask = torch.stack(attn_mask).to(device)
x = torch.tensor(tokens).to(device)
y = torch.tensor(tokens)[:, ans_pos:].to(device)
pred = model(x, attn_mask, use_custom_attention=args.custom_attention)

print(f"using custom attention: {args.custom_attention}")
print("expr: ", batch, len(batch[0]))
print("expected: ", y)
print("prediction: ", model.sample_pred_best(pred)[:, ans_pos-1:-1])

# print("native attention: ", time_func(model.forward, 10, x, attn_mask, use_custom_attention=False))
# print("custom cuda attention: ", time_func(model.forward, 10, x, attn_mask, use_custom_attention=True))