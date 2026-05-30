import torch
import math
import torch.nn.functional as F

mha_cuda = torch.load(name='mha_cuda', sources=["../cuda/kernels.cuh", "bindings.cpp"])

def test():
    n = 8
    k = 8
    h = 64
    b = 16
    t = 16
    s = 16
    g = n // k

    Q = torch.randn([b, t, k, g, h])
    K = torch.randn([b, s, k, h])
    V = torch.randn([b, s, k, h])

    attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(h)
    attention_scores = F.softmax(attention_scores, dim=2) # btskg
    A = torch.einsum("btskg,bskh->btkgh", attention_scores, V)

    A_cuda = mha_cuda.mha_cuda(Q, K, V)

    max_diff = (A - A_cuda).max()

    assert(max_diff < 1e-2)

