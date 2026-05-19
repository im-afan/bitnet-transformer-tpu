#include <torch/extension.h>
#include "core.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor linear_forward(
    torch::Tensor embedding,
    torch::Tensor W,
    torch::Tensor B) {
    
    CHECK_INPUT(embedding);
    CHECK_INPUT(W);
    CHECK_INPUT(B);

    int batch_size = embedding.size(0);
    int embedding_dim = embedding.size(1);

    auto output = torch::empty({batch_size, embedding_dim}, embedding.options());

    linear(embedding.data_ptr<float>(),
           W.data_ptr<float>(),
           B.data_ptr<float>(),
           output.data_ptr<float>(),
           embedding_dim,
           batch_size);

    return output;
}

torch::Tensor multi_head_attention_forward(
    torch::Tensor embedding,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor W_K,
    torch::Tensor W_Q,
    torch::Tensor W_V,
    torch::Tensor W_O,
    torch::Tensor B_K,
    torch::Tensor B_Q,
    torch::Tensor B_V,
    torch::Tensor B_O,
    int n_tokens,
    int n_heads,
    int max_tokens) {
    
    CHECK_INPUT(embedding);
    CHECK_INPUT(K_cache);
    CHECK_INPUT(V_cache);
    CHECK_INPUT(W_K);
    CHECK_INPUT(W_Q);
    CHECK_INPUT(W_V);
    CHECK_INPUT(W_O);
    CHECK_INPUT(B_K);
    CHECK_INPUT(B_Q);
    CHECK_INPUT(B_V);
    CHECK_INPUT(B_O);

    int batch_size = embedding.size(0);
    int embedding_dim = embedding.size(1);
    
    auto output = torch::empty({batch_size, embedding_dim}, embedding.options());

    multi_head_attention(
        embedding.data_ptr<float>(),
        K_cache.data_ptr<float>(),
        V_cache.data_ptr<float>(),
        W_K.data_ptr<float>(),
        W_Q.data_ptr<float>(),
        W_V.data_ptr<float>(),
        W_O.data_ptr<float>(),
        B_K.data_ptr<float>(),
        B_Q.data_ptr<float>(),
        B_V.data_ptr<float>(),
        B_O.data_ptr<float>(),
        output.data_ptr<float>(),
        n_tokens,
        batch_size,
        embedding_dim,
        n_heads,
        max_tokens
    );

    return output;
}
