#include <stdio.h>
#include <ATen/ATen.h>
#include "kernels.cuh"

const int BLOCK_SIZE = 256;

void test_softmax_scale(int batch_size = 64, int n_tokens = 1024) {
    at::Tensor x = at::randn({batch_size, n_tokens});
    at::Tensor expected = at::softmax(x, -1);

    at::Tensor x_gpu = x.to(at::kCUDA);
    at::Tensor output_gpu = at::empty_like(x_gpu);

    int num_blocks = batch_size;
    softmax_scale<<<num_blocks, BLOCK_SIZE>>>(x_gpu.data_ptr<float>(), output_gpu.data_ptr<float>(), 1.0f, batch_size, n_tokens);
    cudaDeviceSynchronize();
    at::Tensor output = output_gpu.to(at::kCPU);

    float max_diff = (output - expected).abs().max().item<float>();
    printf("Softmax test max difference: %f\n", max_diff);
}

void test_layernorm(int batch_size = 64, int embedding_dim = 1024) {
    at::Tensor x = at::randn({batch_size, embedding_dim});
    at::Tensor gamma = at::randn({embedding_dim});
    at::Tensor beta = at::randn({embedding_dim});
    float epsilon = 1e-5;
    at::Tensor expected = at::layer_norm(x, {embedding_dim}, gamma, beta, epsilon);

    at::Tensor x_gpu = x.to(at::kCUDA);
    at::Tensor gamma_gpu = gamma.to(at::kCUDA);
    at::Tensor beta_gpu = beta.to(at::kCUDA);
    at::Tensor output_gpu = at::empty_like(x_gpu);

    int num_blocks = batch_size;
    layer_norm_kernel<<<num_blocks, BLOCK_SIZE>>>(x_gpu.data_ptr<float>(), output_gpu.data_ptr<float>(),
      gamma_gpu.data_ptr<float>(), beta_gpu.data_ptr<float>(), 
      batch_size, embedding_dim, epsilon);
    cudaDeviceSynchronize();
    at::Tensor output = output_gpu.to(at::kCPU);

    // printf("Input mean: %f, variance: %f\n", x.mean().item<float>(), x.var().item<float>());
    // std::cout << "Input: " << x << std::endl;
    // std::cout << "Output: " << output << std::endl;
    // std::cout << "Expected: " << expected << std::endl;

    float max_diff = (output - expected).abs().max().item<float>();
    printf("LayerNorm test max difference: %f\n", max_diff);
}

int main(int argc, char* argv[]) {
    test_softmax_scale(64, 1024);
    test_layernorm(64, 1024);

    return 0;
}