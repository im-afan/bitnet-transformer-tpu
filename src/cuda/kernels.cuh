#include <torch/extension.h>

__global__ void slow_mha(const float* Q, const float* K, const float* V, float* atn_scores, float* output, int b, int t, int s, int head_dim, int q_heads, int kv_heads) {
    // Q: [b, t, kv_heads, q_heads // kv_heads, head_dim] 
    // K, V: [b, s, kv_heads, head_dim]
    // atn_scores: softmax(Q*K_T); einsum: btkgh * bskh -> btskg <=> btsn
    // output: atn_scores * V => btnh
    __shared__ float softmax_mx[16][16];
    __shared__ float softmax_sum[16][16];

    int g = q_heads / kv_heads;
    float scale = sqrtf((float) head_dim);
    
    int batch_idx = blockIdx.x; 
    int q_head_idx = blockIdx.y;
    int kv_head_idx = q_head_idx / g;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    if(batch_idx < b && q_head_idx < q_heads && kv_head_idx < kv_heads) {
        // 1: calculate attention scores
        // all steps will eventually be fused in FlashAttention
        // iterate through each column (each query token)
        for(int x = tid_x; x < t; x += blockDim.x) {
            // iterate through each element in column (each kv token)
            for(int y = tid_y; y < s; y += blockDim.y) {
                int atn_idx = batch_idx * t * s * q_heads + x * s * q_heads + y * q_heads + q_head_idx;
                atn_scores[atn_idx] = 0;
                for(int i = 0; i < head_dim; i++) {
                    int Q_idx = batch_idx * t * q_heads * head_dim + x * q_heads * head_dim + i;
                    int V_idx = batch_idx * s * kv_heads * head_dim + y * kv_heads * head_dim + i;
                    atn_scores[atn_idx] += Q[Q_idx] * V[V_idx] / scale;
                }
            }
        }

        // 2: softmax
        // iterate through each column (each query token)
        for(int x = tid_x; x < t; x += blockDim.x) {
            // iterate through each element in column (each kv token)
            // calculate max value in each column
            softmax_mx[tid_x][tid_y] = -1e9;
            __syncthreads();
            for(int y = tid_y; y < s; y += blockDim.y) {
                int atn_idx = batch_idx * t * s * q_heads + x * s * q_heads + y * q_heads + q_head_idx;
                softmax_mx[tid_x][tid_y] = max(softmax_mx[tid_x][tid_y], atn_idx);
            }
            __syncthreads();
            // reduce softmax_mx
            if(tid_y == 0) {
                for(int y = 1; y < 16; y++) {
                    softmax_mx[tid_x][0] = max(softmax_mx[tid_x][0], softmax_mx[tid_x][y])
                }
            }
            __syncthreads();

            // sum column
            softmax_sum[tid_x][tid_y] = 0;
            __syncthreads();
            for(int y = tid_y; y < s; y += blockDim.y) {
                int atn_idx = batch_idx * t * s * q_heads + x * s * q_heads + y * q_heads + q_head_idx;
                softmax_sum[tid_x][tid_y] += exp(atn_idx[idx] - softmax_mx[tid_x][0]);
            } 
            // reduce softmax_sum;
            if(tid_y == 0) {
                for(int y = 1; y < 16; y++) {
                    softmax_sum[tid_x][0] += softmax_sum[y];
                }
            }
            __syncthreads();
            
            // apply softmax
            for(int y = tid_y; y < s; y += blockDim.y) {
                int atn_idx = batch_idx * t * s * q_heads + x * s * q_heads + y * q_heads + q_head_idx;
                atn_scores[atn_idx] = exp(atn_scores[atn_idx] - softmax_mx[tid_x][0]) / softmax_sum[tid_x][0];
            }
            __syncthreads();
        }

        // 3: calculate V
        // iterate through each column (query tokens)
        for(int x = tid_x; x < t; x += blockDim.x) {
            // iterate through each element in column (kv tokens)
            for(int y = tid_y; y < s; y += blockDim.y) {
                int atn_idx = batch_idx * t * s * q_heads + x * s * q_heads + y * q_heads + q_head_idx;
                float atn_score = atn_scores[atn_idx];
                for(int i = 0; i < head_dim; i++) {
                    int out_idx = batch_idx * t * q_heads * head_dim + x * q_heads * head_dim + q_head_idx * head_dim + i;
                    int V_idx = batch_idx * s * kv_heads * head_dim + y * kv_heads * head_dim + kv_head_idx * head_dim + i;
                    output[out_idx] += atn_score * V[V_idx];
                }
            }
        }
    }
}

__host__ torch::Tensor mha_custom(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int b = Q.size(0);
    int t = Q.size(1);
    int s = K.size(1);
    int kv_heads = K.size(2);
    int q_heads = Q.size(2) * Q.size(3);
    int head_dim = K.size(3);

    torch::Tensor atn_scores = torch::empty({b, t, s, q_heads});
    torch::Tensor output = torch::zeros({b, t, q_heads, head_dim});

    dim3 block_dim(16, 16);
    dim3 grid_dim(b, q_heads);
    slow_mha<<<grid_dim, block_dim>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        atn_scores.data_ptr<float>(),
        output.data_ptr<float>(),
        b, t, s, head_dim, q_heads, kv_heads
    );
    cudaDeviceSynchronize();

    return output;
}