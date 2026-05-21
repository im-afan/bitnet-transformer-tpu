__global__ void matmul_kernel(const float* A, const float* B, float* C, int M,
  int N, int K, int batch_size) {
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < batch_size && row < M && col < N) {
    float sum = 0;
    for (int k = 0; k < K; k++) {
      sum += A[batch * M * K + row * K + k] * B[batch * K * N + k * N + col];
    }
    C[batch * M * N + row * N + col] = sum;
  }
}

__global__ void add_batch_kernel(const float* A, const float* B, float* C,
  int n, int batch_size) {
  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < batch_size && idx < n) {
    C[idx + batch * n] = A[idx + batch * n] + B[idx];
  }
}

__global__ void add_kernel(const float* A, const float* B, float* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void exp_kernel(float* x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = exp(x[idx]);
  }
}

__global__ void mult_kernel(float* x, float* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = x[idx] * a[0];
  }
}

// RELU
__global__ void relu_kernel(float* x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (x[idx] < 0) {
      x[idx] = 0;
    }
  }
}

__device__ void reduction(float* smem) {
  for (int i = 128; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      smem[threadIdx.x] += smem[threadIdx.x + i];
    }
    __syncthreads();
  }
}

__global__ void layer_norm_kernel(float* x, float* output, float* gamma,
  float* beta, int batch_size, int embedding_dim,
  float epsilon) {
  // 1 block = 1 batch
  __shared__ float mean_smem[256];
  __shared__ float var_smem[256];
  int batch_idx = blockIdx.x;

  if (batch_idx < batch_size) {
    if (threadIdx.x < 256) {
      mean_smem[threadIdx.x] = 0;
      var_smem[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
      mean_smem[idx % 256] += x[batch_idx * embedding_dim + idx];
    }
    __syncthreads();
    reduction(mean_smem);
    if (threadIdx.x == 0) {
      mean_smem[0] /= embedding_dim;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
      float x_i = x[batch_idx * embedding_dim + idx];
      var_smem[idx % 256] += (x_i - mean_smem[0]) * (x_i - mean_smem[0]);
    }
    __syncthreads();
    reduction(var_smem);
    if (threadIdx.x == 0) {
      var_smem[0] /= embedding_dim;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
      float x_i = x[batch_idx * embedding_dim + idx];
      float gamma_i = gamma[idx];
      float beta_i = beta[idx];
      output[batch_idx * embedding_dim + idx] =
        (x_i - mean_smem[0]) / sqrt(var_smem[0] + epsilon) * gamma_i + beta_i;
    }
  }
}

__global__ void softmax_scale(const float* x, float* output, float scale,
  int batch_size, int n_tokens) {
  // 1 block = 1 batch
  __shared__ float smem[256];
  int batch_idx = blockIdx.x;

  if (batch_idx < batch_size) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      smem[idx % 256] += exp(x[batch_idx * n_tokens + idx]);
    }
    __syncthreads();

    reduction(smem);

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      output[batch_idx * n_tokens + idx] = exp(x[batch_idx * n_tokens + idx]) / smem[0];
    }
  }
}

// W_K, W_Q, W_V: embedding_dim x embedding_dim <=> n_heads x (embedding_dim / n_heads) x embedding_dim
// B_K, B_Q, B_V: embedding_dim x 1 <=> n_heads x (embedding_dim / n_heads)
// K, Q, V = W_K, W_Q, W_V * embedding + B_K, B_Q, B_V => batch_size x embedding_dim
// W_O: embedding_dim x embedding_dim
// B_O: embedding_dim
// K_cache, V_cache: batch_size x n_heads x max_tokens x (embedding_dim / n_heads)
// embedding: batch_size x embedding_dim
// output: pre-softmax attention scores, batch_size x n_heads x max_tokens
__global__ void multi_head_attention_kernel_phase1(const float* embedding, const float* W_K, const float* W_Q,
  const float* W_V, const float* B_K, const float* B_Q,
  const float* B_V, float* K_cache, float* V_cache,
  float* output, int n_tokens, int batch_size, int embedding_dim, int n_heads,
  int max_tokens) {

  __shared__ float K_smem[64]; // = embedding_dim / n_heads
  __shared__ float V_smem[64];
  __shared__ float Q_smem[64];
  __shared__ float dot_smem[64];

  int head_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  if (batch_idx < batch_size && head_idx < n_heads) {
    int head_offset = head_idx * (embedding_dim / n_heads);
    int idx_in_vec = threadIdx.x % (embedding_dim / n_heads);
    int vec_idx = threadIdx.x / (embedding_dim / n_heads);

    // todo: do only one of K, V, Q in a thread depending on thread idx
    // would not cause divergence because all threads in a warp do the same thing
    if(threadIdx.x < embedding_dim / n_heads) {
      // 1: compute Q, K, V
      K_smem[idx_in_vec] = B_K[head_offset + idx_in_vec];
      V_smem[idx_in_vec] = B_V[head_offset + idx_in_vec];
      Q_smem[idx_in_vec] = B_Q[head_offset + idx_in_vec];
      __syncthreads();

      int real_idx = head_offset + idx_in_vec;
      for (int i = 0; i < embedding_dim; i++) {
          K_smem[idx_in_vec] += W_K[i * embedding_dim + real_idx] * embedding[batch_idx * embedding_dim + i];
          V_smem[idx_in_vec] += W_V[i * embedding_dim + real_idx] * embedding[batch_idx * embedding_dim + i];
          Q_smem[idx_in_vec] += W_Q[i * embedding_dim + real_idx] * embedding[batch_idx * embedding_dim + i];
      }

      __syncthreads();

      // 2: write K, V to cache
      K_cache[batch_idx * n_heads * max_tokens * (embedding_dim / n_heads)
        + head_idx * max_tokens * (embedding_dim / n_heads) 
        + n_tokens * (embedding_dim / n_heads) + idx_in_vec] = K_smem[idx_in_vec];
      V_cache[batch_idx * n_heads * max_tokens * (embedding_dim / n_heads)
        + head_idx * max_tokens * (embedding_dim / n_heads) 
        + n_tokens * (embedding_dim / n_heads) + idx_in_vec] = V_smem[idx_in_vec];
      
      // 3: compute attention
      for(int i = 0; i < n_tokens; i++) {
        dot_smem[idx_in_vec] = Q_smem[idx_in_vec] * 
            K_cache[batch_idx * n_heads * max_tokens * (embedding_dim / n_heads)
              + head_idx * max_tokens * (embedding_dim / n_heads)
              + i * (embedding_dim / n_heads) + idx_in_vec];
        __syncthreads();
        reduction(dot_smem);
        
        if(threadIdx.x == 0) {
          output[batch_idx * n_heads * max_tokens + head_idx * max_tokens + i]
            += dot_smem[0];
        }
        __syncthreads();
      }
    }
  }
}

__global__ void multi_head_attention_kernel_phase2(const float* embedding, const float* attention_scores, 
  const float* W_O, const float* B_O, float* V_cache,
  float* output, int n_tokens, int batch_size, int embedding_dim, int n_heads,
  int max_tokens) {
  __shared__ float O_smem[64]; // = embedding_dim / n_heads

  int head_idx = blockIdx.x;
  int batch_idx = blockIdx.y; 

  if(head_idx < n_heads && batch_idx < batch_size) {
    for(int i = 0; i < n_tokens; i++) {
      float attn_score = attention_scores[batch_idx * n_heads * max_tokens + head_idx * max_tokens + i];
      O_smem[threadIdx.x] += attn_score * V_cache[batch_idx * n_heads * max_tokens * (embedding_dim / n_heads)
        + head_idx * max_tokens * (embedding_dim / n_heads) 
        + i * (embedding_dim / n_heads) + threadIdx.x];
    } 
    __syncthreads();

    int real_idx = head_idx * (embedding_dim / n_heads) + threadIdx.x;
    float output_i = B_O[real_idx] + embedding[batch_idx * embedding_dim + real_idx];
    for(int i = 0; i < embedding_dim; i++) {
      output_i += W_O[i * embedding_dim + real_idx] * O_smem[threadIdx.x];
    }
    output[batch_idx * n_heads * max_tokens + head_idx * max_tokens + threadIdx.x] = output_i;
  }
}

__host__ void multi_head_attention(const float* embedding, const float* W_K, const float* W_Q,
  const float* W_V, const float* W_O, const float* B_K, const float* B_Q,
  const float* B_V, const float* B_O, float* K_cache, float* V_cache,
  float* output, int n_tokens, int batch_size, int embedding_dim, int n_heads,
  int max_tokens) {
  dim3 blockDim1(embedding_dim / n_heads, 1, 1);
  dim3 gridDim1(n_heads, batch_size, 1);
  multi_head_attention_kernel_phase1<<<gridDim1, blockDim1>>>(embedding, W_K, W_Q, W_V, 
    B_K, B_Q, B_V, K_cache, V_cache, output, n_tokens, 
    batch_size, embedding_dim, n_heads, max_tokens);
  cudaDeviceSynchronize();
  
  dim3 blockDimSoftmax(256, 1, 1);
  dim3 gridDimSoftmax(batch_size * n_heads, 1, 1);
  softmax_scale<<<gridDimSoftmax, blockDimSoftmax>>>(output, output, 
    1.0f / sqrt(embedding_dim / n_heads), batch_size * n_heads, n_tokens);
  cudaDeviceSynchronize();

  dim3 blockDim2(embedding_dim / n_heads, 1, 1);
  dim3 gridDim2(n_heads, batch_size, 1);
  multi_head_attention_kernel_phase2<<<gridDim2, blockDim2>>>(embedding, output, W_O, B_O, 
    V_cache, output, n_tokens, batch_size, embedding_dim, n_heads, max_tokens);
  cudaDeviceSynchronize();
}
