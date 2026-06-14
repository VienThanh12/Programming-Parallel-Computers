#include "llm.h"

#include <cstddef>

namespace {

inline float dot16(const float *__restrict a, const float *__restrict b, int n) {
    float acc[16];
    for (int j = 0; j < 16; ++j) {
        acc[j] = 0.0f;
    }
    for (int k = 0; k < n; k += 16) {
        for (int j = 0; j < 16; ++j) {
            acc[j] += a[k + j] * b[k + j];
        }
    }
    float s = 0.0f;
    for (int j = 0; j < 16; ++j) {
        s += acc[j];
    }
    return s;
}

void matmul_batched(float *__restrict out, const float *__restrict x,
                    const float *__restrict w, int n, int d, int T) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        const float *__restrict wi = w + (std::size_t)i * n;
        for (int t = 0; t < T; ++t) {
            out[(std::size_t)t * d + i] = dot16(x + (std::size_t)t * n, wi, n);
        }
    }
}

} // namespace

void llm(LLamaConfig config, LLamaParameters params, const std::vector<token_t> &tokens, std::vector<float> &logits) {
    using namespace utils;

    const int dim = config.dim;
    const int hidden_dim = config.hidden_dim;
    const int head_size = config.head_size();
    const int n_heads = config.n_heads;
    const int vocab_size = config.vocab_size;
    const int T = (int)tokens.size();

    if (T == 0) {
        return;
    }

    std::vector<float> activation((std::size_t)T * dim);
    for (int t = 0; t < T; ++t) {
        int token_id = (int)tokens[t];
        std::copy(params.TokenEmbeddingMatrix.begin() + (std::size_t)token_id * dim,
                  params.TokenEmbeddingMatrix.begin() + (std::size_t)(token_id + 1) * dim,
                  activation.begin() + (std::size_t)t * dim);
    }

    std::vector<float> normed((std::size_t)T * dim);
    std::vector<float> query((std::size_t)T * dim);
    std::vector<float> keys((std::size_t)T * dim);
    std::vector<float> values((std::size_t)T * dim);
    std::vector<float> att_out((std::size_t)T * dim);
    std::vector<float> proj((std::size_t)T * dim);
    std::vector<float> hidden1((std::size_t)T * hidden_dim);
    std::vector<float> hidden3((std::size_t)T * hidden_dim);

    for (int l = 0; l < config.n_layers; ++l) {
        auto &layer = params.LayerWeights[l];

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            rmsnorm(normed.data() + (std::size_t)t * dim,
                    activation.data() + (std::size_t)t * dim,
                    layer.rms_attention.data(), dim);
        }

        matmul_batched(query.data(), normed.data(), layer.query_weight_matrix.data(), dim, dim, T);
        matmul_batched(keys.data(), normed.data(), layer.key_weight_matrix.data(), dim, dim, T);
        matmul_batched(values.data(), normed.data(), layer.value_weight_matrix.data(), dim, dim, T);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            rope(config, query.data() + (std::size_t)t * dim,
                 keys.data() + (std::size_t)t * dim, t);
        }

#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                std::vector<float> attention(t + 1);
                const float *q = query.data() + (std::size_t)t * dim + h * head_size;
                calculate_attention(config, attention.data(), q, t,
                                    keys.data() + h * head_size);
                lookup_with_attention(config, attention.data(),
                                      att_out.data() + (std::size_t)t * dim + h * head_size,
                                      t,
                                      values.data() + h * head_size);
            }
        }

        matmul_batched(proj.data(), att_out.data(), layer.out_weight_matrix.data(), dim, dim, T);

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < (std::size_t)T * dim; ++i) {
            activation[i] += proj[i];
        }

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            rmsnorm(normed.data() + (std::size_t)t * dim,
                    activation.data() + (std::size_t)t * dim,
                    layer.rms_feed_forward.data(), dim);
        }

        matmul_batched(hidden1.data(), normed.data(), layer.feed_forward_w1.data(), dim, hidden_dim, T);
        matmul_batched(hidden3.data(), normed.data(), layer.feed_forward_w3.data(), dim, hidden_dim, T);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            swiglu(hidden1.data() + (std::size_t)t * hidden_dim,
                   hidden1.data() + (std::size_t)t * hidden_dim,
                   hidden3.data() + (std::size_t)t * hidden_dim, hidden_dim);
        }

        matmul_batched(proj.data(), hidden1.data(), layer.feed_forward_w2.data(), hidden_dim, dim, T);

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < (std::size_t)T * dim; ++i) {
            activation[i] += proj[i];
        }
    }

#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        rmsnorm(activation.data() + (std::size_t)t * dim,
                activation.data() + (std::size_t)t * dim,
                params.RmsFinal.data(), dim);
    }

    matmul_batched(logits.data(), activation.data(), params.TokenOutputMatrix.data(),
                   dim, vocab_size, T);
}
