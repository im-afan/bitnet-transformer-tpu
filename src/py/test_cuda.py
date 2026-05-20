import torch
import unittest
import math
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import transformer_cuda

# print("compiling custom kernel")
# transformer_cuda = load(
#     name="transformer_cuda",
#     sources=["../cuda/bindings_kernel.cu", "../cuda/bindings.cpp"],
#     verbose=True
# )

class TestCUDABindings(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        self.device = torch.device('cuda')

    def test_linear(self):
        if transformer_cuda is None:
            self.skipTest("transformer_cuda not installed")
            
        batch_size = 4
        embedding_dim = 16
        
        # Initialize inputs
        embedding = torch.randn(batch_size, embedding_dim, device=self.device)
        W = torch.randn(embedding_dim, embedding_dim, device=self.device)
        B = torch.randn(embedding_dim, device=self.device)

        # PyTorch reference
        # The CUDA matmul_batch computes y = xA + b
        expected_output = torch.matmul(embedding, W) + B

        # CUDA implementation
        cuda_output = transformer_cuda.linear(embedding, W, B)

        print(expected_output, cuda_output)

        torch.testing.assert_close(cuda_output, expected_output, rtol=1e-3, atol=1e-3)

    def test_attention_softmax(self):
        if(transformer_cuda is None):
            self.skipTest("transformer_cuda not installed")

        batch_size = 4
        n_tokens = 16
        scale = math.sqrt(384)
        kq_row = torch.randn(batch_size, n_tokens, device=self.device)

        expected_output = F.softmax(kq_row / scale, dim=1)
        cuda_output = transformer_cuda.attention_softmax(kq_row, scale)

        print("softmax test", expected_output, cuda_output)

        torch.testing.assert_close(cuda_output, expected_output, rtol=1e-3, atol=1e-3)

    # def test_multi_head_attention(self):
    #     if transformer_cuda is None:
    #         self.skipTest("transformer_cuda not installed")

    #     batch_size = 2
    #     n_tokens = 8
    #     embedding_dim = 64
    #     n_heads = 4
    #     max_tokens = 32

    #     # Inputs
    #     embedding = torch.randn(batch_size, embedding_dim, device=self.device)
    #     W_K = torch.randn(embedding_dim, embedding_dim, device=self.device)
    #     W_Q = torch.randn(embedding_dim, embedding_dim, device=self.device)
    #     W_V = torch.randn(embedding_dim, embedding_dim, device=self.device)
    #     W_O = torch.randn(embedding_dim, embedding_dim, device=self.device)
    #     B_K = torch.randn(embedding_dim, device=self.device)
    #     B_Q = torch.randn(embedding_dim, device=self.device)
    #     B_V = torch.randn(embedding_dim, device=self.device)
    #     B_O = torch.randn(embedding_dim, device=self.device)

    #     # CUDA allocates cache linearly. In the kernel:
    #     # K_cache + i * batch_size * (embedding_dim / n_heads) * max_tokens
    #     K_cache = torch.zeros(n_heads, batch_size, max_tokens, embedding_dim // n_heads, device=self.device)
    #     V_cache = torch.zeros(n_heads, batch_size, max_tokens, embedding_dim // n_heads, device=self.device)

    #     # Run CUDA kernel
    #     cuda_output = transformer_cuda.multi_head_attention(
    #         embedding, K_cache, V_cache,
    #         W_K, W_Q, W_V, W_O,
    #         B_K, B_Q, B_V, B_O,
    #         n_tokens, n_heads, max_tokens
    #     )

    #     # PyTorch reference
    #     head_dim = embedding_dim // n_heads
        
    #     # 1. Compute Q, K, V
    #     K = torch.matmul(embedding, W_K) + B_K
    #     V = torch.matmul(embedding, W_V) + B_V
    #     Q = torch.matmul(embedding, W_Q) + B_Q

    #     # 2. Reshape into heads: (batch_size, n_heads, head_dim)
    #     K = K.view(batch_size, n_heads, head_dim).transpose(0, 1) # (n_heads, batch_size, head_dim)
    #     V = V.view(batch_size, n_heads, head_dim).transpose(0, 1)
    #     Q = Q.view(batch_size, n_heads, head_dim).transpose(0, 1)

    #     # CUDA writes the new token to the first position in the cache
    #     K_cache[:, :, :1, :] = K.unsqueeze(2)
    #     V_cache[:, :, :1, :] = V.unsqueeze(2)

    #     attention_output = []
    #     for i in range(n_heads):
    #         # Q_i: (batch_size, head_dim) -> (batch_size, head_dim, 1)
    #         Q_i = Q[i].unsqueeze(2)
    #         # K_i: (batch_size, n_tokens, head_dim)
    #         K_i = K_cache[i, :, :n_tokens, :]
    #         # V_i: (batch_size, n_tokens, head_dim)
    #         V_i = V_cache[i, :, :n_tokens, :]

    #         # scores = K_i * Q_i
    #         # K_i is (batch_size, n_tokens, head_dim)
    #         # Q_i is (batch_size, head_dim, 1)
    #         # Result: (batch_size, n_tokens, 1) -> (batch_size, 1, n_tokens)
    #         scores = torch.bmm(K_i, Q_i).transpose(1, 2) / math.sqrt(head_dim)
            
    #         # Softmax over n_tokens
    #         attn = torch.softmax(scores, dim=-1) # (batch_size, 1, n_tokens)

    #         # out = attn * V_i
    #         # attn is (batch_size, 1, n_tokens)
    #         # V_i is (batch_size, n_tokens, head_dim)
    #         out = torch.bmm(attn, V_i) # (batch_size, 1, head_dim)
    #         attention_output.append(out.squeeze(1))

    #     # Concatenate heads
    #     concat_out = torch.cat(attention_output, dim=1) # (batch_size, embedding_dim)

    #     # Final linear layer
    #     expected_output = torch.matmul(concat_out, W_O) + B_O
        
    #     # Residual connection
    #     expected_output = expected_output + embedding

    #     torch.testing.assert_close(cuda_output, expected_output, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    unittest.main()
