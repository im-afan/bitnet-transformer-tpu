import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads

        self.Wq = nn.ModuleList([nn.Linear(d_model, d_k) for i in range(heads)])
        self.Wk = nn.ModuleList([nn.Linear(d_model, d_k) for i in range(heads)])
        self.Wv = nn.ModuleList([nn.Linear(d_model, d_v) for i in range(heads)])
        self.Wo = nn.Linear(heads * d_v, d_model)

        self.softmax = nn.Softmax(dim=2)

    def scaled_dot_product_attention(self, Q, K, V):
        K_T = torch.transpose(K, dim0=1, dim1=2)
        return self.softmax(Q @ K_T / self.d_k ** 0.5) @ V

    def forward(self, X):
        head_outputs = []
        for i in range(self.heads):
            attention = self.scaled_dot_product_attention(
                self.Wq[i](X),
                self.Wk[i](X),
                self.Wv[i](X)
            )
            head_outputs.append(attention)

        head_outputs = torch.cat(head_outputs, dim=2) 
        return self.Wo(head_outputs)

class Transformer(nn.Module):
    def __init__(self, d_model, d_k, d_v, heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, d_k, d_v, heads);
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        X = X + self.attention(X)
        X_ff = self.fc1(X)
        X_ff = self.relu(X_ff)
        X_ff = self.fc2(X_ff)

        return X + X_ff

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_transformers):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.d_model = embedding_dim
        self.heads = 12
        self.d_k = self.d_model // self.heads # does not necessarily have to be like this
        self.d_v = self.d_model // self.heads
        self.d_ff = 1024

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformers = []

        for i in range(n_transformers):
            self.transformers.append(
                Transformer(self.d_model, self.d_k, self.d_v, self.heads, self.d_ff))

        self.fc = nn.Linear(self.d_model, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def positional_encoding(self, n_tokens):
        div = torch.exp(torch.arange(0, self.embedding_dim, 2) * math.log(1/10000) * 1/self.embedding_dim)
        pos_tensor = torch.arange(n_tokens).unsqueeze(1)

        pe = torch.zeros((1, n_tokens, self.embedding_dim))
        pe[:, :, 0::2] = torch.sin(pos_tensor * div) 
        pe[:, :, 1::2] = torch.cos(pos_tensor * div) 

        return pe

    def forward(self, inputs):
        pe = self.positional_encoding(inputs.shape[1])
        X = self.embedding(inputs) + pe
        for transformer in self.transformers:
            X = transformer(X)

        prediction = self.fc(X[:, -1, :])
        return self.softmax(prediction)

    def predict_token(self, prediction):
        distribution = Categorical(probs=prediction)
        return distribution.sample()







        