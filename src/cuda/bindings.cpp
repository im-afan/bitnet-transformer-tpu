#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> mha_custom(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_custom", torch::wrap_pybind_function(mha_custom), "mha_custom");
}