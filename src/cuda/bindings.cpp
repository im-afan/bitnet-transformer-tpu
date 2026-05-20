#include <torch/extension.h>

torch::Tensor linear_forward(torch::Tensor embedding, torch::Tensor W,
                             torch::Tensor B);

torch::Tensor attention_softmax(torch::Tensor kq, float scale);

// torch::Tensor multi_head_attention_forward(
//     torch::Tensor embedding, torch::Tensor K_cache, torch::Tensor V_cache,
//     torch::Tensor W_K, torch::Tensor W_Q, torch::Tensor W_V, torch::Tensor
//     W_O, torch::Tensor B_K, torch::Tensor B_Q, torch::Tensor B_V,
//     torch::Tensor B_O, int n_tokens, int n_heads, int max_tokens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear_forward, "Linear forward (CUDA)");
  //   m.def("multi_head_attention", &multi_head_attention_forward,
  //   "Multi-Head Attention forward (CUDA)");
  m.def("attention_softmax", &attention_softmax,
        "Attention Softmax Function (CUDA)");
}
