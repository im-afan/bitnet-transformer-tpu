__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
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

__global__ void add_batch_kernel(const float *A, const float *B, float *C,
                                 int n, int batch_size) {
  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < batch_size && idx < n) {
    C[idx + batch * n] = A[idx + batch * n] + B[idx];
  }
}

__global__ void add_kernel(const float *A, const float *B, float *C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void exp_kernel(float *x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = exp(x[idx]);
  }
}

__global__ void mult_kernel(float *x, float *a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = x[idx] * a[0];
  }
}

// RELU
__global__ void relu_kernel(float *x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (x[idx] < 0) {
      x[idx] = 0;
    }
  }
}

// LAYER NORM OPS
__global__ void layer_norm_kernel(float *x, float *gamma, float *beta,
                                  float *mean, float *variance, float *output,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] =
        (x[idx] - mean[idx]) / sqrt(variance[idx] + 1e-5) * gamma[idx] +
        beta[idx];
  }
}

__global__ void sum_kernel(float *x, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    atomicAdd(output, x[idx]);
  }
}

__global__ void variance_kernel(float *x, float *mean, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(output, (x[idx] - mean[0]) * (x[idx] - mean[0]));
  }
  if (idx == 0) {
    output[0] /= n;
  }
}

__global__ void softmax_scale(const float *x, float *output, float *sum,
                              float scale, int batch_size, int n_tokens) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = idx / n_tokens;

  if (idx < batch_idx * n_tokens) {
    output[idx] = exp(x[idx] * scale);
    atomicAdd(sum + batch_idx, output[idx]);
    __syncthreads();
    output[idx] = output[idx] / sum[batch_idx];
  }
}
