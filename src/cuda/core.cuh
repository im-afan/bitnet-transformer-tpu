#include "kernels.cuh"
#include <driver_types.h>

__host__ void matmul_batch(const float *A, const float *B, float *C, int M,
                           int N, int K, int batch_size) {
  // assume n_tokens = 1 during inference
  dim3 threadsPerBlock(16, 16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
  matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K,
                                                    batch_size);
  cudaDeviceSynchronize();
}

__host__ void attention_softmax(const float *embedding, float *output,
                                float scale, int batch_size, int n_tokens) {
  float *sums;
  cudaMalloc((void **)&sums, batch_size * sizeof(float));

  int threadsPerBlock = 256;
  int blocksPerGrid =
      (batch_size * n_tokens + threadsPerBlock - 1) / threadsPerBlock;
  softmax_scale<<<blocksPerGrid, threadsPerBlock>>>(
      embedding, output, sums, scale, batch_size, n_tokens);
  cudaDeviceSynchronize();
}

// __host__ void
// multi_head_attention(const float *embedding, float *K_cache, float *V_cache,
//                      const float *W_K, const float *W_Q, const float *W_V,
//                      const float *W_O, const float *B_K, const float *B_Q,
//                      const float *B_V, const float *B_O, float *output,
//                      int n_tokens, int batch_size, int embedding_dim,
//                      int n_heads, int max_tokens) {
//   float *K;
//   float *Q;
//   float *V;
//   float *attention;
//   float *attention_row;
//   float *KQ_row;
//   cudaMalloc((void **)&K, batch_size * embedding_dim);
//   cudaMalloc((void **)&Q, batch_size * embedding_dim);
//   cudaMalloc((void **)&attention, batch_size * embedding_dim * n_heads);
//   cudaMalloc((void **)&attention_row, batch_size * embedding_dim);
//   cudaMalloc((void **)&KQ_row, batch_size * n_tokens);

//   dim3 add_batch_threads_per_block(16, 16);
//   dim3 add_batch_blocks_per_grid((embedding_dim + 15) / 16,
//                                  (batch_size + 15) / 16);

//   matmul_batch(embedding, W_K, K, 1, embedding_dim, embedding_dim,
//   batch_size); add_batch_kernel<<<add_batch_blocks_per_grid,
//   add_batch_threads_per_block>>>(
//       K, B_K, K, embedding_dim, batch_size);
//   cudaDeviceSynchronize();
//   matmul_batch(embedding, W_V, V, 1, embedding_dim, embedding_dim,
//   batch_size); add_batch_kernel<<<add_batch_blocks_per_grid,
//   add_batch_threads_per_block>>>(
//       V, B_V, V, embedding_dim, batch_size);
//   cudaDeviceSynchronize();

//   // store K_cache and V_cache heads-first
//   // each head stored will be B x T x (d/h)
//   for (int i = 0; i < n_heads; i++) {
//     for (int j = 0; j < batch_size; j++) {
//       float *K_ptr = K + i * batch_size * embedding_dim / n_heads +
//                      j * embedding_dim / n_heads;
//       float *V_ptr = V + i * batch_size * embedding_dim / n_heads +
//                      j * embedding_dim / n_heads;
//       float *K_cache_ptr =
//           K_cache + i * batch_size * embedding_dim / n_heads * max_tokens +
//           j * embedding_dim / n_heads;
//       float *V_cache_ptr =
//           V_cache + i * batch_size * embedding_dim / n_heads * max_tokens +
//           j * embedding_dim / n_heads;

//       cudaMemcpy(K_cache_ptr, K_ptr, embedding_dim / n_heads,
//                  cudaMemcpyDeviceToDevice);
//       cudaMemcpy(V_cache_ptr, V_ptr, embedding_dim / n_heads,
//                  cudaMemcpyDeviceToDevice);
//     }
//   }

//   matmul_batch(embedding, W_Q, Q, 1, embedding_dim, embedding_dim,
//   batch_size); add_batch_kernel<<<add_batch_blocks_per_grid,
//   add_batch_threads_per_block>>>(
//       Q, B_Q, Q, embedding_dim, batch_size);

//   float attention_scale = sqrtf((float)embedding_dim / n_heads);

//   for (int i = 0; i < n_heads; i++) {
//     float *K_cache_ptr =
//         K_cache + i * batch_size * embedding_dim / n_heads * max_tokens;
//     float *V_cache_ptr =
//         V_cache + i * batch_size * embedding_dim / n_heads * max_tokens;

//     matmul_batch(K_cache_ptr, Q, KQ_row, n_tokens, 1, embedding_dim /
//     n_heads,
//                  batch_size);
//     attention_softmax(KQ_row, batch_size, n_tokens, attention_scale);
//     matmul_batch(KQ_row, V_cache_ptr, attention_row, 1, embedding_dim /
//     n_heads,
//                  n_tokens, batch_size);

//     for (int j = 0; j < batch_size; j++) {
//       float *attention_ptr = attention + i * n_tokens * n_heads + j *
//       n_heads; float *attention_row_ptr = attention_row + j * n_tokens;
//       cudaMemcpy(attention_ptr, attention_row_ptr, n_tokens,
//                  cudaMemcpyDeviceToDevice);
//     }
//   }

//   matmul_batch(attention, W_O, output, 1, embedding_dim, embedding_dim,
//                batch_size);
//   add_batch_kernel<<<add_batch_blocks_per_grid,
//   add_batch_threads_per_block>>>(
//       output, B_O, output, embedding_dim, batch_size);
//   dim3 add_threads(256);
//   dim3 add_blocks((batch_size * embedding_dim + 255) / 256);
//   add_kernel<<<add_blocks, add_threads>>>(output, embedding, output,
//                                           batch_size * embedding_dim);
//   cudaDeviceSynchronize();
// }

__host__ void linear(const float *embedding, const float *W, const float *B,
                     float *output, int embedding_dim, int batch_size) {
  matmul_batch(embedding, W, output, 1, embedding_dim, embedding_dim,
               batch_size);
  dim3 add_batch_threads_per_block(16, 16);
  dim3 add_batch_blocks_per_grid((embedding_dim + 15) / 16,
                                 (batch_size + 15) / 16);

  add_batch_kernel<<<add_batch_blocks_per_grid, add_batch_threads_per_block>>>(
      output, B, output, embedding_dim, batch_size);
  cudaDeviceSynchronize();
}