#include <cmath>
#include <tuple>
#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_head_attention_phase1_aten(
    at::Tensor embedding,
    at::Tensor W_K,
    at::Tensor W_Q,
    at::Tensor W_V,
    at::Tensor B_K,
    at::Tensor B_Q,
    at::Tensor B_V,
    at::Tensor K_cache,
    at::Tensor V_cache,
    int n_tokens,
    int batch_size,
    int embedding_dim,
    int n_heads,
    int max_tokens) {

    TORCH_CHECK(embedding.dim() == 2, "embedding must be 2D");
    TORCH_CHECK(embedding.size(0) == batch_size, "batch_size mismatch");
    TORCH_CHECK(embedding.size(1) == embedding_dim, "embedding_dim mismatch");
    TORCH_CHECK(W_K.sizes() == at::IntArrayRef({embedding_dim, embedding_dim}), "W_K must be [embedding_dim, embedding_dim]");
    TORCH_CHECK(W_Q.sizes() == at::IntArrayRef({embedding_dim, embedding_dim}), "W_Q must be [embedding_dim, embedding_dim]");
    TORCH_CHECK(W_V.sizes() == at::IntArrayRef({embedding_dim, embedding_dim}), "W_V must be [embedding_dim, embedding_dim]");
    TORCH_CHECK(B_K.numel() == embedding_dim, "B_K must be length embedding_dim");
    TORCH_CHECK(B_Q.numel() == embedding_dim, "B_Q must be length embedding_dim");
    TORCH_CHECK(B_V.numel() == embedding_dim, "B_V must be length embedding_dim");
    TORCH_CHECK(K_cache.dim() == 4, "K_cache must be 4D");
    TORCH_CHECK(V_cache.dim() == 4, "V_cache must be 4D");
    TORCH_CHECK(K_cache.size(0) == batch_size, "K_cache batch_size mismatch");
    TORCH_CHECK(V_cache.size(0) == batch_size, "V_cache batch_size mismatch");
    TORCH_CHECK(K_cache.size(1) == n_heads, "K_cache n_heads mismatch");
    TORCH_CHECK(V_cache.size(1) == n_heads, "V_cache n_heads mismatch");
    TORCH_CHECK(K_cache.size(2) == max_tokens, "K_cache max_tokens mismatch");
    TORCH_CHECK(V_cache.size(2) == max_tokens, "V_cache max_tokens mismatch");
    TORCH_CHECK(embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads");
    TORCH_CHECK(n_tokens >= 0 && n_tokens < max_tokens, "n_tokens must be within [0, max_tokens)");

    int head_dim = embedding_dim / n_heads;
    auto device = embedding.device();

    auto K_proj = at::matmul(embedding, W_K) + B_K;
    auto Q_proj = at::matmul(embedding, W_Q) + B_Q;
    auto V_proj = at::matmul(embedding, W_V) + B_V;

    auto K = K_proj.view({batch_size, n_heads, head_dim});
    auto Q = Q_proj.view({batch_size, n_heads, head_dim});
    auto V = V_proj.view({batch_size, n_heads, head_dim});

    auto K_cache_out = K_cache.clone();
    auto V_cache_out = V_cache.clone();
    K_cache_out.select(2, n_tokens).copy_(K);
    V_cache_out.select(2, n_tokens).copy_(V);

    auto attention_scores = at::zeros({batch_size, n_heads, max_tokens}, embedding.options());
    if (n_tokens > 0) {
      auto K_active = K_cache_out.index({at::indexing::Slice(),
                                        at::indexing::Slice(),
                                        at::indexing::Slice(0, n_tokens),
                                        at::indexing::Slice()});
      auto Q_expanded = Q.unsqueeze(2);
      auto dot = (Q_expanded * K_active).sum(-1);
      attention_scores.slice(2, 0, n_tokens).copy_(dot);
    }

    return std::make_tuple(K_cache_out, V_cache_out, attention_scores);
}
