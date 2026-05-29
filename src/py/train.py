import argparse
import datetime
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformer import Model
import transformer
import numbers_data 

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")
elif(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
    device = torch.device("mps")

def train(
        model: Model,
        optim,
        epochs = 100,
        batches = 10000,
        mini_batch_size = 32,
        batch_size = 32,
        max_tokens = 32,
        save_freq = 100
    ):
    model = model.to(device)

    steps_per_batch = batch_size // mini_batch_size

    steps = 0
    avg_loss = 0
    batch_steps = 0
    saved_models = []

    for i in range(epochs):
        for j in range(batches):
            batch, tokens, attn_mask = numbers_data.create_addition_batch(mini_batch_size, max_tokens)
            tokens = torch.tensor(tokens).to(device)
            # print(attn_mask)
            attn_mask = torch.stack(attn_mask).to(device)

            pred = model(tokens, attn_mask) # (B, T, vocab)

            ans_pos = numbers_data.EQUALS_POS
            pred = pred[:, ans_pos-1:-1].flatten(0, 1)
            y = tokens[:, ans_pos:].flatten(0, 1)
            loss = F.cross_entropy(pred, y)
            avg_loss += loss.item()

            loss.backward()
            
            steps += 1
            batch_steps += 1
            if(batch_steps % steps_per_batch == 0):
                optim.step()
                optim.zero_grad()

            if(steps % save_freq == 0):
                avg_loss /= save_freq
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"./saved/test_model_{timestamp}.pt"
                torch.save(model.state_dict(), filepath)
                
                saved_models.append(filepath)
                if len(saved_models) > 3:
                    old_model = saved_models.pop(0)
                    if os.path.exists(old_model):
                        os.remove(old_model)
                        
                print(f"Epoch {i} training loss: {avg_loss}")
                avg_loss = 0
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['vanilla', 'gqa'], default='vanilla', help='Which adder architecture to use')
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    vocab_size = len(numbers_data.VOCAB)
    if args.arch == 'gqa':
        model = transformer.adder_gqa()
    else:
        model = transformer.adder_vanilla()

    optim = Adam(model.parameters(), lr=1e-3)
    train(model, optim, save_freq=args.save_freq, mini_batch_size=args.mini_batch_size, batch_size=args.batch_size)