import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from transformer import Model

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if(tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token


model = Model(tokenizer.vocab_size, 768, 12)

epochs = 1000
optim = Adam(model.parameters(), lr=0.1)
loss_fn = CrossEntropyLoss()

for i in range(epochs):
    text = "The quick brown fox jumps"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_tokens = inputs["input_ids"]
    x = input_tokens[:, :-1].repeat_interleave(32, dim=0).to(dtype=torch.int32)
    y = input_tokens[:, -1].repeat_interleave(32, dim=0).long()
    # print(y.shape)
    y = F.one_hot(y, num_classes=tokenizer.vocab_size).float()
    print(y.shape)

    optim.zero_grad()

    pred = model(x)
    loss = loss_fn(pred, y)

    print(tokenizer.decode(model.predict_token(pred)[0]))
    print(f"Epoch {i} training loss: {loss}")

    loss.backward()
    optim.step()

    torch.save(model.state_dict(), "./saved/test_model.pt")

    