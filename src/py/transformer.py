import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import numbers_data

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")
elif(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
    device = torch.device("mps")

class MultiHeadAttention(nn.Module):
    def __init__(self, d, q_heads, kv_heads, head_dim):
        super().__init__()

        assert(q_heads % kv_heads == 0)

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d = d
        self.heads_per_q = q_heads // kv_heads
        self.head_dim = head_dim

        self.Wq = nn.Linear(d, q_heads * self.head_dim) 
        self.Wk = nn.Linear(d, kv_heads * self.head_dim)
        self.Wv = nn.Linear(d, kv_heads * self.head_dim)
        self.Wo = nn.Linear(q_heads * self.head_dim, d)
    
    def forward(self, X, attn_mask):
        batch_size = X.shape[0]
        n_tokens = X.shape[1]

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        Q = torch.reshape(Q, [batch_size, n_tokens, self.kv_heads, self.heads_per_q, self.head_dim])
        K = torch.reshape(K, [batch_size, n_tokens, self.kv_heads, self.head_dim])
        V = torch.reshape(V, [batch_size, n_tokens, self.kv_heads, self.head_dim])

        mask = torch.triu(torch.ones([n_tokens, n_tokens]) * -1e9, diagonal=1).reshape([1, n_tokens, n_tokens, 1, 1])
        # attn_mask = attn_mask.repeat(1, 1, n_tokens)
        # print(attn_mask.shape, n_tokens)
        # attn_mask = attn_mask + attn_mask.T
        # print(attn_mask.shape)
        mask = mask.to(device)# + attn_mask.reshape([batch_size, n_tokens, n_tokens, 1, 1])

        attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(self.head_dim)
        attention_scores = F.softmax(attention_scores + mask, dim=2) # btskg
        A = torch.einsum("btskg,bskh->btkgh", attention_scores, V).reshape([batch_size, n_tokens, self.q_heads * self.head_dim])

        O = self.Wo(A) 
        return O + X

class Transformer(nn.Module):
    def __init__(self, d, f, q_heads, kv_heads, head_dim):
        super().__init__()

        self.attention = MultiHeadAttention(d, q_heads, kv_heads, head_dim);
        self.norm1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, f)
        self.fc2 = nn.Linear(f, d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, attn_mask):
        X = self.norm1(X + self.dropout(self.attention(X, attn_mask)))
        X_ff = self.dropout(self.fc1(X))
        X_ff = F.gelu(X_ff)
        X_ff = self.dropout(self.fc2(X_ff))

        return self.norm2(X + X_ff)

class Model(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d=128, 
        f=256,
        layers=8, 
        q_heads=8, 
        kv_heads=8,
        head_dim=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = d

        if(head_dim == None):
            head_dim = d // q_heads

        self.d = d
        self.f = f 
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.layers = layers


        self.embedding = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([
            Transformer(self.d, self.f, self.q_heads, self.kv_heads, self.head_dim)
            for i in range(layers)
        ])
        self.fc = nn.Linear(self.d, self.vocab_size)
        self.dropout = nn.Dropout(p=0.1)

    def positional_encoding(self, n_tokens):
        device = next(self.parameters()).device
        div = torch.exp(torch.arange(0, self.d, 2) * math.log(1/10000) * 1/self.d)
        pos_tensor = torch.arange(n_tokens).unsqueeze(1)

        pe = torch.zeros((1, n_tokens, self.d))
        pe[:, :, 0::2] = torch.sin(pos_tensor * div) 
        pe[:, :, 1::2] = torch.cos(pos_tensor * div) 

        return pe.to(device)

    def forward(self, inputs, attn_mask):
        pe = self.positional_encoding(inputs.shape[1])
        X = self.dropout(self.embedding(inputs) + pe)
        for layer in self.layers:
            X = layer(X, attn_mask)

        return self.fc(X) 
    
    def sample_pred(self, logits):
        dist = Categorical(logits=logits)
        return dist.sample()
    
    def sample_pred_best(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)
    

def adder_vanilla():
    model = Model(len(numbers_data.VOCAB), d=128, f=512, layers=6, q_heads=8, kv_heads=8)
    return model

def adder_gqa():
    model = Model(len(numbers_data.VOCAB), d=128, f=512, layers=6, q_heads=8, kv_heads=2)
    return model