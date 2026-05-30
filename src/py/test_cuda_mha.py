import torch
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load

mha_cuda = load(name='mha_cuda', sources=["../cuda/kernels.cu", "../cuda/bindings.cpp"], verbose=True)

def test(n, k, h, b, t, s): 
    g = n // k

    Q = torch.randn([b, t, k, g, h])
    K = torch.randn([b, s, k, h])
    V = torch.randn([b, s, k, h])

    attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(h)
    attention_scores = F.softmax(attention_scores, dim=2) # btskg
    A = torch.einsum("btskg,bskh->btkgh", attention_scores, V).reshape([b, t, n, h])

    A_cuda, attention_scores_cuda = mha_cuda.mha_custom(Q, K, V)

    # print(A, A_cuda, A - A_cuda)
    # print(attention_scores.shape, attention_scores_cuda.shape)
    # print(attention_scores, attention_scores_cuda)
    # print(attention_scores - attention_scores_cuda)
    max_diff_atn_score = (attention_scores - attention_scores_cuda).max().abs()
    max_diff = (A - A_cuda).max().abs()

    print(f"max diff attention score: {max_diff_atn_score}, max_diff: {max_diff}")

    assert(max_diff < 1e-2)

if __name__ == '__main__':
    test(n=16, k=16, h=64, b=32, t=512, s=512)
    test(n=16, k=2, h=64, b=32, t=128, s=128)