from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model
from datasets import load_dataset
from data import Tokenize

context_size = 32
train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
train_dataset = train_dataset.with_format("torch")
train_dataloader = DataLoader(train_dataset, batch_size=16)
transform = Tokenize()

# for batch in train_dataloader:
#     # print(len(batch['text']))
#     tokenized = transform(batch)
#     print(tokenized['text'])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token
pad_token_idx = tokenizer.vocab_size-1


model = Model(tokenizer.vocab_size, 256, 4)

epochs = 1
optim = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss(ignore_index=pad_token_idx)

for i in range(epochs):
    for batch in train_dataloader:
        tokens = transform(batch)['text']

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        optim.zero_grad()

        pred = model(x)
        pred = pred.flatten(0, 1)
        y = y.flatten(0, 1)
        # print(pred.shape, y.shape)
        loss = loss_fn(pred, y)

        loss.backward()
        optim.step()

        torch.save(model.state_dict(), "./saved/test_model.pt")
        print(f"Epoch {i} training loss: {loss}")

        