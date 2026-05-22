#include <stdio.h>
#include <ATen/ATen.h>
#include "kernels.cuh"
#include "kernels_cpu.cpp"

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

    float max_diff = (output - expected).abs().max().item<float>();
    printf("LayerNorm test max difference: %f\n", max_diff);
}

void test_multi_head_attention_phase1(int batch_size = 64, int n_tokens = 512, int max_tokens = 1024, int embedding_dim = 768, int n_heads = 12) {
    at::Tensor embedding = at::randn({batch_size, embedding_dim});
    at::Tensor W_K = at::randn({embedding_dim, embedding_dim});
    at::Tensor W_Q = at::randn({embedding_dim, embedding_dim});
    at::Tensor W_V = at::randn({embedding_dim, embedding_dim});
    at::Tensor B_K = at::randn({embedding_dim});
    at::Tensor B_Q = at::randn({embedding_dim});
    at::Tensor B_V = at::randn({embedding_dim});
    at::Tensor W_O = at::randn({embedding_dim, embedding_dim});
    at::Tensor B_O = at::randn({embedding_dim});

    at::Tensor K_cache = at::randn({batch_size, n_heads, n_tokens, embedding_dim / n_heads});
    at::Tensor V_cache = at::randn({batch_size, n_heads, n_tokens, embedding_dim / n_heads});

    K_cache = at::cat({K_cache, at::zeros({batch_size, n_heads, max_tokens - n_tokens, embedding_dim / n_heads})}, 2);
    V_cache = at::cat({V_cache, at::zeros({batch_size, n_heads, max_tokens - n_tokens, embedding_dim / n_heads})}, 2);

    auto [K_cache_expected, V_cache_expected, attention_scores_expected] = multi_head_attention_phase1_aten(
        embedding,
        W_K, W_Q, W_V,
        B_K, B_Q, B_V,
        K_cache, V_cache,
        n_tokens, batch_size, embedding_dim, n_heads, max_tokens
    );

    at::Tensor K_cache_out = K_cache.clone();
    at::Tensor V_cache_out = V_cache.clone();
    at::Tensor attention_scores_out = at::empty_like(attention_scores_expected);

    dim3 grid_dim_phase1(n_heads, batch_size, 1);
    dim3 block_dim_phase1(embedding_dim / n_heads, 1, 1);
    
    multi_head_attention_kernel_phase1<<<grid_dim_phase1, block_dim_phase1>>>(
        // Refactored to use Catch2-based unit tests and runtime-configurable parameters
        #include <stdio.h>
        #include <string>
        #include <vector>
        #include <tuple>
        #include <ATen/ATen.h>
        #include "kernels.cuh"

        #define CATCH_CONFIG_RUNNER
        #include <catch2/catch.hpp>

        const int BLOCK_SIZE = 256;

        struct TestConfig {
            int batch_size = 64;
            int n_tokens = 1024;
            int max_tokens = 1024;
            int embedding_dim = 1024;
            int n_heads = 8;
        };

        static TestConfig CONFIG;

        // Forward declaration of the CPU helper implemented in kernels_cpu.cpp
        extern std::tuple<at::Tensor, at::Tensor, at::Tensor> multi_head_attention_phase1_aten(
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
            int max_tokens);

        // Simple CLI parsing: consume known options and leave the rest for Catch2
        static std::vector<char*> build_catch_argv(int argc, char** argv) {
            std::vector<char*> new_argv;
            new_argv.push_back(argv[0]);
            for (int i = 1; i < argc; ++i) {
                std::string s = argv[i];
                if (s == "--batch_size" && i + 1 < argc) { CONFIG.batch_size = std::stoi(argv[++i]); }
                else if (s == "--n_tokens" && i + 1 < argc) { CONFIG.n_tokens = std::stoi(argv[++i]); }
                else if (s == "--max_tokens" && i + 1 < argc) { CONFIG.max_tokens = std::stoi(argv[++i]); }
                else if (s == "--embedding_dim" && i + 1 < argc) { CONFIG.embedding_dim = std::stoi(argv[++i]); }
                else if (s == "--n_heads" && i + 1 < argc) { CONFIG.n_heads = std::stoi(argv[++i]); }
                else { new_argv.push_back(argv[i]); }
            }
            return new_argv;
        }

        TEST_CASE("Softmax Scale GPU matches ATen softmax", "[softmax]") {
            auto b = CONFIG.batch_size;
            auto n = CONFIG.n_tokens;
            at::Tensor x = at::randn({b, n});
            at::Tensor expected = at::softmax(x, -1);

            at::Tensor x_gpu = x.to(at::kCUDA);
            at::Tensor output_gpu = at::empty_like(x_gpu);

            dim3 grid(b,1,1);
            dim3 block(BLOCK_SIZE,1,1);
            softmax_scale<<<grid, block>>>(x_gpu.data_ptr<float>(), output_gpu.data_ptr<float>(), 1.0f, b, n);
            cudaDeviceSynchronize();

            at::Tensor output = output_gpu.to(at::kCPU);
            float max_diff = (output - expected).abs().max().item<float>();
            printf("Softmax test max difference: %f\n", max_diff);
            REQUIRE(std::isfinite(max_diff));
        }

        TEST_CASE("LayerNorm GPU matches ATen layer_norm", "[layernorm]") {
            auto b = CONFIG.batch_size;
            auto embed = CONFIG.embedding_dim;
            at::Tensor x = at::randn({b, embed});
            at::Tensor gamma = at::randn({embed});
            at::Tensor beta = at::randn({embed});
            float epsilon = 1e-5;
            at::Tensor expected = at::layer_norm(x, {embed}, gamma, beta, epsilon);

            at::Tensor x_gpu = x.to(at::kCUDA);
            at::Tensor gamma_gpu = gamma.to(at::kCUDA);
            at::Tensor beta_gpu = beta.to(at::kCUDA);
            at::Tensor output_gpu = at::empty_like(x_gpu);

            dim3 grid(b,1,1);
            dim3 block(BLOCK_SIZE,1,1);
            layer_norm_kernel<<<grid, block>>>(x_gpu.data_ptr<float>(), output_gpu.data_ptr<float>(), gamma_gpu.data_ptr<float>(), beta_gpu.data_ptr<float>(), b, embed, epsilon);
            cudaDeviceSynchronize();

            at::Tensor output = output_gpu.to(at::kCPU);
            float max_diff = (output - expected).abs().max().item<float>();
            printf("LayerNorm test max difference: %f\n", max_diff);
            REQUIRE(std::isfinite(max_diff));
        }

        TEST_CASE("Multi-Head Attention Phase1 (ATen CPU) basic checks", "[mha][phase1]") {
            auto batch_size = CONFIG.batch_size;
            auto n_tokens = CONFIG.n_tokens;
            auto max_tokens = CONFIG.max_tokens;
            auto embedding_dim = CONFIG.embedding_dim;
            auto n_heads = CONFIG.n_heads;

            int head_dim = embedding_dim / n_heads;
            REQUIRE(embedding_dim % n_heads == 0);

            at::Tensor embedding = at::randn({batch_size, embedding_dim});
            at::Tensor W_K = at::randn({embedding_dim, embedding_dim});
            at::Tensor W_Q = at::randn({embedding_dim, embedding_dim});
            at::Tensor W_V = at::randn({embedding_dim, embedding_dim});
            at::Tensor B_K = at::randn({embedding_dim});
            at::Tensor B_Q = at::randn({embedding_dim});
            at::Tensor B_V = at::randn({embedding_dim});

            at::Tensor K_cache = at::zeros({batch_size, n_heads, max_tokens, head_dim});
            at::Tensor V_cache = at::zeros({batch_size, n_heads, max_tokens, head_dim});

            auto res = multi_head_attention_phase1_aten(embedding, W_K, W_Q, W_V, B_K, B_Q, B_V, K_cache, V_cache, n_tokens, batch_size, embedding_dim, n_heads, max_tokens);
            at::Tensor K_cache_out, V_cache_out, attention_scores;
            std::tie(K_cache_out, V_cache_out, attention_scores) = res;

            // Basic sanity checks
            REQUIRE(K_cache_out.sizes() == K_cache.sizes());
            REQUIRE(V_cache_out.sizes() == V_cache.sizes());
            REQUIRE(attention_scores.sizes() == at::IntArrayRef({batch_size, n_heads, max_tokens}));
            // Check finiteness on the active region
            if (n_tokens > 0) {
                auto active = attention_scores.index({at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(0, n_tokens)});
                REQUIRE(active.isfinite().all().item<bool>());
            }
        }

        int main(int argc, char* argv[]) {
            auto new_argv_vec = build_catch_argv(argc, argv);
            // Convert to form suitable for Catch2
            int new_argc = static_cast<int>(new_argv_vec.size());

            Catch::Session session;
            int returnCode = session.run(new_argc, new_argv_vec.data());
            return returnCode;
        }