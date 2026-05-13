import argparse
import datetime
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model
from datasets import load_dataset
from data import Tokenize

parser = argparse.ArgumentParser(description='Train Transformer')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--mini_batch_size', type=int, default=16, help='Mini batch size')
parser.add_argument('--num_transformers', type=int, default=4, help='Number of transformer layers')
args = parser.parse_args()

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")
elif(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
    device = torch.device("mps")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

context_size = 128
# train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
train_dataset = load_dataset("roneneldan/TinyStories", name="default", split="train")
train_dataset = train_dataset.with_format("torch")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
transform = Tokenize()

# for batch in train_dataloader:
#     # print(len(batch['text']))
#     tokenized = transform(batch)
#     print(tokenized['text'])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token
pad_token_idx = tokenizer.vocab_size-1


model = Model(tokenizer.vocab_size, args.embed_dim, args.num_transformers, args.num_heads)

steps_per_batch = args.batch_size // args.mini_batch_size

epochs = 5
optim = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss(ignore_index=pad_token_idx)

model = model.to(device)
# train_dataloader = train_dataloader.to(device) 

avg_loss = 0
steps = 0
batch_steps = 0

for i in range(epochs):
    for batch in train_dataloader:
        tokens = transform(batch)['text']

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        x = x.to(device)
        y = y.to(device)


        pred = model(x)
        pred = pred.flatten(0, 1)
        y = y.flatten(0, 1)
        loss = loss_fn(pred, y)
        avg_loss += loss

        loss.backward()
        
        steps += 1
        batch_steps += 1
        if(batch_steps % steps_per_batch):
            optim.step()
            optim.zero_grad()

        if(steps % 20 == 0):
            avg_loss /= 20
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f"./saved/test_model_{timestamp}.pt")
            print(f"Epoch {i} training loss: {avg_loss}")
            avg_loss = 0
  
        