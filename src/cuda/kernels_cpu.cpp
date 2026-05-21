#include <cmath>

void layer_norm_cpu(float* x, float* output, float gamma, float beta,
    int batch_size, int embedding_dim, float epsilon) {
    for (int i = 0; i < batch_size; i++) {
        float mean = 0.0f;
        float variance = 0.0f;

        for (int j = 0; j < embedding_dim; j++) {
            mean += x[i * embedding_dim + j];
        }
        mean /= embedding_dim;

        for (int j = 0; j < embedding_dim; j++) {
            float diff = x[i * embedding_dim + j] - mean;
            variance += diff * diff;
        }
        variance /= embedding_dim;

        for (int j = 0; j < embedding_dim; j++) {
            output[i * embedding_dim + j] = gamma * (x[i * embedding_dim + j] - mean) / sqrtf(variance + epsilon) + beta;
        }
    }
}

void softmax_scale_cpu(const float* x, float* output, float scale,
    int batch_size, int n_tokens) {
    for (int i = 0; i < batch_size; i++) {
        float max_val = -1e9f;
        for (int j = 0; j < n_tokens; j++) {
            float val = x[i * n_tokens + j] * scale;
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            sum_exp += expf(x[i * n_tokens + j] * scale - max_val);
        }

        for (int j = 0; j < n_tokens; j++) {
            output[i * n_tokens + j] = expf(x[i * n_tokens + j] * scale - max_val) / sum_exp;
        }
    }
}