#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

/// This structure contains the shape specifications for the Transformer model
struct LLamaConfig {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length

    [[nodiscard]] int head_size() const {
        return dim / n_heads;
    }
};

/// This structure contains the weights of a single transformer block/layer
struct LLamaLayer {
    std::vector<float> rms_attention;    // (dim,)
    std::vector<float> rms_feed_forward; // (dim,)

    std::vector<float> query_weight_matrix; // (dim, dim)
    std::vector<float> key_weight_matrix;   // (dim, dim)
    std::vector<float> value_weight_matrix; // (dim, dim)
    std::vector<float> out_weight_matrix;   // (dim, dim)

    std::vector<float> feed_forward_w1; // (dim, hidden_dim)
    std::vector<float> feed_forward_w2; // (hidden_dim, dim)
    std::vector<float> feed_forward_w3; // (dim, hidden_dim)
};

/// This structure contains the weights of the entire model
struct LLamaParameters {
    // token embedding table
    std::vector<float> TokenEmbeddingMatrix; // (vocab_size, dim)
    std::vector<float> TokenOutputMatrix;    // (vocab_size, dim)
    std::vector<float> RmsFinal;             // (dim,)
    std::vector<LLamaLayer> LayerWeights;
};

/// Special type to indicate a token id.
enum class token_t : int {
    BOS = 1, // beginning of sequence
    EOS = 2  // end of sequence
};

// Function to be implemented
void llm(LLamaConfig config, LLamaParameters params, const std::vector<token_t> &tokens, std::vector<float> &logits);

// Utility functions:
// You may use these, but are allowed to provide your own alternatives

namespace utils {

/// Normalize the vector in `x`, and scale each coordinate by the corresponding weight.
/// The result is stored in `out`.
/// `x`, `out`, and `weight` should all be arrays of length `size`.
inline void rmsnorm(float *out, const float *x, const float *weights, int size) {
    // accumulate in double to improve numerical stability
    float ss = (float)std::inner_product(x, x + size, x, 0.0) / (float)size;
    for (int i = 0; i < size; ++i) {
        // add a small epsilon to ensure numerical stability
        out[i] = x[i] * weights[i] / std::sqrt(ss + 1e-5f);
    }
}

/// Turn un-normalized scores `x` into a probability distribution. In-place operation.
inline void softmax(float *x, int size) {
    float max = *std::max_element(x, x + size);
    // Subtract the maximum score. This does not affect the result mathematically,
    // but prevents overflows in floating-point representations.
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max);
    }

    // calculate normalization factor and normalize. Accumulate in double to improve
    // numerical stability
    float sum = (float)std::accumulate(x, x + size, 0.0);
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / sum;
    }
}

/// activation function used in LLama
inline float silu(float x) {
    return x / (1.f + std::exp(-x));
}

/// activation function used in LLama
inline void swiglu(float *out, const float *a, const float *b, int size) {
    for (int i = 0; i < size; i++) {
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        // elementwise multiply with w3(x)
        out[i] = silu(a[i]) * b[i];
    }
}

/// Multiply the complex number pointed to by `target` with
/// real part `fcr` and imaginary part `fci`.
inline void rotate(float *target, float fcr, float fci) {
    target[0] = target[0] * fcr - target[1] * fci;
    target[1] = target[0] * fci + target[1] * fcr;
}

/// ROtary Position Encoding. See https://arxiv.org/abs/2104.09864 for motivation.
inline void rope(const LLamaConfig &config, float *query, float *key, int pos) {
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < config.dim; i += 2) {
        int head_dim = i % config.head_size();
        float freq = 1.0f / std::pow(10000.0f, head_dim / (float)config.head_size());
        float val = pos * freq;
        float fcr = std::cos(val);
        float fci = std::sin(val);
        // rotate query vector
        rotate(query + i, fcr, fci);
        // rotate key vector
        rotate(key + i, fcr, fci);
    }
}

/// Given attention scores `attention` and stored values `values_cache`, this calculates
/// the interpolation which is the output of the attention processing, in `result`.
inline void lookup_with_attention(const LLamaConfig &config,
                                  const float *attention, float *result, int pos,
                                  const float *value_cache) {
    // initialize result to zero
    std::fill(result, result + config.head_size(), 0.f);
    // iterate over all previous (and the current) tokens.
    for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        const float *v = value_cache + t * config.dim;
        // get the attention weight for this timestep
        float a = attention[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < config.head_size(); i++) {
            result[i] += a * v[i];
        }
    }
}

inline void calculate_attention(const LLamaConfig &config,
                                float *attention, const float *query, int pos,
                                const float *key_cache) {
    float norm = 1.f / sqrtf(config.head_size());
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        const float *key = key_cache + t * config.dim;
        // calculate the attention score as the dot product of query and k
        float score = 0.0f;
        for (int i = 0; i < config.head_size(); i++) {
            score += query[i] * key[i];
        }
        // save the score to the attention buffer
        attention[t] = score * norm;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(attention, pos + 1);
}

} // namespace utils