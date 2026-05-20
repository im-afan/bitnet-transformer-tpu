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

__global__ void layer_norm_kernel(float* x, float* output, float gamma,
  float beta, int batch_size, int embedding_dim,
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

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      mean_smem[idx % 256] += x[batch_idx * n_tokens + idx];
    }
    __syncthreads();
    for (int i = 128; i > 1; i >>= 1) {
      if (idx < i) {
        mean_smem[idx] += mean_smem[idx + i]
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      mean_smem[0] /= embedding_dim;
    }

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      var_smem[idx % 256] +=
        pow(x[batch_idx * n_tokens + idx] - mean_smem[0], 2);
    }
    __syncthreads();
    for (int i = 128; i > 1; i >>= 1) {
      if (idx < i) {
        var_smem[idx] += var_smem[idx + i]
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      var_smem[0] /= embedding_dim;
    }

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      float x_i = output[batch_idx * n_tokens + idx];
      output[batch_idx * n_tokens + idx] =
        (x_i - mean_smem[0]) / sqrt(var_smem[0] + epsilon) * gamma + beta;
    }
  }
}

__global__ void softmax_scale_kernel(const float* x, float* output, float scale,
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

    for (int i = 128; i > 1; i >>= 1) {
      if (idx < i) {
        smem[idx] += smem[idx + i]
      }
      __syncthreads();
    }

    for (int idx = threadIdx.x; idx < n_tokens; idx += blockDim.x) {
      output[idx] = exp(x[batch_idx * n_tokens] + idx) / smem[0];
    }
  }
}
