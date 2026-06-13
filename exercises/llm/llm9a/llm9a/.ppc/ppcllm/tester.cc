#include "llm.h"
#include "ppc.h"
#include <fstream>
#include <omp.h>
#include <utility>

namespace ppc {
std::vector<float> llama_reference(LLamaConfig config, LLamaParameters params, std::vector<token_t> prompt);

std::pair<LLamaConfig, LLamaParameters> make_test_model(LLamaConfig config,
                                                        int multiplier,
                                                        const std::string &modifier,
                                                        ppc::random rng,
                                                        bool for_test);

bool test_against_ref_implementation(std::ostream *stream, LLamaConfig base_config, ppc::random rng,
                                     int multiplier, const std::string &model_modifier,
                                     std::vector<token_t> input, const std::vector<float> &logits);
} // namespace ppc

int main(int argc, const char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }

    std::unique_ptr<ppc::fdostream> stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    if (argc < 1 || argc > 2) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    bool test = false;
    if (argv[0] == std::string("--test")) {
        test = true;
        argc--;
        argv++;
    }

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return 2;
    }

    std::string time_out;
    CHECK_READ(input_file >> time_out);
    if (time_out == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string model_modifier;
    int model_multiplier;
    int num_tokens, seed;
    auto read_cfg = [&](const std::string &expect, auto &target) {
        std::string line;
        std::getline(input_file, line);
        std::stringstream line_data(line);
        CHECK_READ(line_data >> line);
        if (line != expect) {
            std::cerr << "Invalid test spec: Expected " << expect << ". got " << line << std::endl;
            std::exit(1);
        }
        // no CHECK_READ macro; the check below does the same thing, but gives a better error message
        line_data >> target;
        if (!line_data) {
            std::cerr << "Error while reading test spec " << expect << "." << std::endl;
            std::exit(1);
        }
    };

    LLamaConfig base_config;
    read_cfg("seed", seed);
    read_cfg("dim", base_config.dim);
    read_cfg("n_layers", base_config.n_layers);
    read_cfg("hidden_dim", base_config.hidden_dim);
    read_cfg("n_heads", base_config.n_heads);
    read_cfg("vocab_size", base_config.vocab_size);
    read_cfg("seq_len", base_config.seq_len);
    read_cfg("multiplier", model_multiplier);
    read_cfg("modifier", model_modifier);
    read_cfg("num_tokens", num_tokens);

    // check that we can actually use the requested multiplier
    if (base_config.n_layers % model_multiplier != 0) {
        std::cerr << "Error, invalid multiplier " << model_multiplier << "  for model with " << base_config.n_layers << "layers" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // validate that we keep our promises
    if (base_config.hidden_dim % 16 != 0) {
        std::cerr << "Error, invalid hidden_dim " << base_config.hidden_dim << ": not a multiple of 16." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (base_config.dim % 16 != 0) {
        std::cerr << "Error, invalid dim " << base_config.dim << ": not a multiple of 16." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (base_config.head_size() % 16 != 0) {
        std::cerr << "Error, invalid head size " << base_config.head_size() << " ("
                  << base_config.dim << " / " << base_config.n_heads << "): not a multiple of 16." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<token_t> prompt_tokens(num_tokens);
    ppc::random rng(seed);
    for (int i = 0; i < num_tokens; ++i) {
        prompt_tokens[i] = (token_t)rng.get_int64(0, base_config.vocab_size);
    }

    auto [config, params] = ppc::make_test_model(base_config, model_multiplier, model_modifier, rng, false);

    std::vector<float> logits(prompt_tokens.size() * config.vocab_size);

    ppc::setup_cuda_device();
    ppc::perf timer;
    timer.start();

    // first, init the model
    llm(config, std::move(params), prompt_tokens, logits);

    // then do the actual run
    timer.stop();
    timer.print_to(*stream);
    ppc::reset_cuda_device();

    // give some context rgd. the model
    *stream << "config.vocab_size\t" << config.vocab_size << "\n";
    *stream << "config.dim\t" << config.dim << "\n";
    *stream << "config.hidden_dim\t" << config.hidden_dim << "\n";
    *stream << "config.n_layers\t" << config.n_layers << "\n";
    *stream << "config.seq_len\t" << config.seq_len << "\n";
    *stream << "config.n_heads\t" << config.n_heads << "\n";
    *stream << "length\t" << num_tokens << "\n";

    if (test) {
        bool pass = ppc::test_against_ref_implementation(stream.get(), base_config, rng,
                                                         model_multiplier, model_modifier,
                                                         prompt_tokens, logits);
        if (!pass) {
            *stream << "seed\t";
            *stream << seed << "\n";
            *stream << std::endl;
            exit(0);
        }

        *stream << "result\tpass\n";
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::endl;
}

std::vector<float> rand_vec(ppc::random &rng, std::size_t size) {
    std::vector<float> vec(size);
    std::generate(vec.begin(), vec.end(), [&]() {
        return rng.get_float(-1, 1);
    });
    return vec;
}

std::pair<LLamaConfig, LLamaParameters> ppc::make_test_model(
    LLamaConfig config,
    int multiplier,
    const std::string &modifier,
    ppc::random rng,
    bool for_test) {
    LLamaParameters weights;
    weights.TokenEmbeddingMatrix = rand_vec(rng, config.vocab_size * config.dim);

    // under-provision the LayerWeights; only config.n_layers / multiplier will be non-trivial, to speed up testing
    weights.LayerWeights.resize(config.n_layers / multiplier);

    auto make_data = [&](auto &&ptr_to_member, int num) {
        for (auto &layer : weights.LayerWeights) {
            layer.*ptr_to_member = rand_vec(rng, num);
        }
    };

    int head_size = config.dim / config.n_heads;
    make_data(&LLamaLayer::rms_attention, config.dim);
    make_data(&LLamaLayer::query_weight_matrix, config.dim * (config.n_heads * head_size));
    make_data(&LLamaLayer::key_weight_matrix, config.dim * (config.n_heads * head_size));
    make_data(&LLamaLayer::value_weight_matrix, config.dim * (config.n_heads * head_size));
    make_data(&LLamaLayer::out_weight_matrix, (config.n_heads * head_size) * config.dim);

    make_data(&LLamaLayer::rms_feed_forward, config.dim);
    make_data(&LLamaLayer::feed_forward_w1, config.dim * config.hidden_dim);
    make_data(&LLamaLayer::feed_forward_w2, config.hidden_dim * config.dim);
    make_data(&LLamaLayer::feed_forward_w3, config.dim * config.hidden_dim);

    weights.RmsFinal = rand_vec(rng, config.dim);
    // unused
    rand_vec(rng, config.seq_len * head_size);
    weights.TokenOutputMatrix = rand_vec(rng, config.vocab_size * config.dim);

    if (modifier == "default") {
    } else if (modifier == "disable-attention") {
        // set out_weight_matrix to zero, effectively disabling the attention module
        for (auto &l : weights.LayerWeights) {
            l.out_weight_matrix.assign(l.out_weight_matrix.size(), 0.f);
        }
    } else if (modifier == "disable-feed-forward") {
        // set ffn w2 to zero, effectively disabling the feed-forward module
        for (auto &l : weights.LayerWeights) {
            l.feed_forward_w2.assign(l.feed_forward_w2.size(), 0.f);
        }
    } else {
        std::cerr << "Invalid modifier '" << modifier << "'" << std::endl;
        std::exit(1);
    }

    // if we have a reference model, the effective number of layers is divided by the multiplier
    // if we have the student's model, add in some dummy layers that effectively get remove by
    // having all their outputs multiplied by zero.
    if (!for_test) {
        for (int k = 0; k < multiplier; ++k) {
            for (int i = 0; i < config.n_layers / multiplier; ++i) {
                weights.LayerWeights.push_back(weights.LayerWeights[i]);
                auto &l = weights.LayerWeights.back();
                l.out_weight_matrix.assign(l.out_weight_matrix.size(), 0.f);
                l.feed_forward_w2.assign(l.feed_forward_w2.size(), 0.f);
            }
        }
    } else {
        config.n_layers = config.n_layers / multiplier;
    }

    return std::make_pair(config, std::move(weights));
}

namespace ppc {

inline void matmul(float *out, const float *x, const float *w, int n, int d) {
// when running grading on student's computers, we're working on a system that is in interactive use.
// trying to get _all_ threads here results in terrible performance. Since this loop is very short,
// if just one of the threads gets stolen away by the OS for another task, everything has to wait.
// Therefore, we leave one thread unused.
#pragma omp parallel for num_threads(std::max(1, omp_get_num_threads() - 1))
    for (int i = 0; i < d; i++) {
        // transform-reduce is a slightly faster version of std::inner_product
        // it is not allowed in student code
        out[i] = std::transform_reduce(x, x + n, w + i * n, 0.0);
    }
}

class LLamaRefModel {
  public:
    LLamaRefModel(LLamaConfig config, LLamaParameters params);
    std::vector<float> predict(int position, token_t token);

  private:
    // helpers running part of the model
    void multi_head_attention(const LLamaLayer &layer,
                              float *activation, float *attention,
                              int position, int layer_id,
                              float *query);

    LLamaConfig Config;
    LLamaParameters Parameters;

    std::vector<float> KeyCache;   // (layer, seq_len, dim)
    std::vector<float> ValueCache; // (layer, seq_len, dim)
};

LLamaRefModel::LLamaRefModel(LLamaConfig config, LLamaParameters params) : Config(config), Parameters(std::move(params)) {
    KeyCache.resize(Config.n_layers * Config.seq_len * Config.dim);
    ValueCache.resize(Config.n_layers * Config.seq_len * Config.dim);
}

std::vector<float> LLamaRefModel::predict(int position, token_t token) {
    using namespace utils;
    // a few convenience variables
    int dim = Config.dim;
    int hidden_dim = Config.hidden_dim;

    // initialize the activation by the embedding of the current token
    std::vector<float> activation(Parameters.TokenEmbeddingMatrix.begin() + (int)token * dim,
                                  Parameters.TokenEmbeddingMatrix.begin() + ((int)token + 1) * dim);

    // scratch buffers to use during the computations
    std::vector<float> buffer(dim);
    std::vector<float> buffer2(dim);
    std::vector<float> hidden_buffer(hidden_dim);
    std::vector<float> hidden_buffer2(hidden_dim);
    std::vector<float> query(dim);
    std::vector<float> attention(Config.n_heads * Config.seq_len);

    // forward all the layers
    for (int l = 0; l < Config.n_layers; l++) {
        auto &layer = Parameters.LayerWeights[l];

        // attention rmsnorm
        rmsnorm(buffer.data(),
                activation.data(),
                layer.rms_attention.data(),
                dim);

        multi_head_attention(layer, buffer.data(), attention.data(), position, l, query.data());

        // final matmul to get the output of the attention
        matmul(buffer2.data(), buffer.data(), layer.out_weight_matrix.data(), dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            activation[i] += buffer2[i];
        }

        rmsnorm(buffer.data(), activation.data(), layer.rms_feed_forward.data(), dim);

        matmul(hidden_buffer.data(), buffer.data(), layer.feed_forward_w1.data(), dim, hidden_dim);
        matmul(hidden_buffer2.data(), buffer.data(), layer.feed_forward_w3.data(), dim, hidden_dim);
        swiglu(hidden_buffer.data(), hidden_buffer.data(), hidden_buffer2.data(), hidden_dim);
        // final matmul to get the output of the ffn
        matmul(buffer.data(), hidden_buffer.data(), layer.feed_forward_w2.data(), hidden_dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            activation[i] += buffer[i];
        }
    }

    // final rmsnorm
    rmsnorm(activation.data(), activation.data(), Parameters.RmsFinal.data(), dim);

    // classifier into logits
    std::vector<float> logits(Config.vocab_size);
    matmul(logits.data(), activation.data(), Parameters.TokenOutputMatrix.data(), dim, Config.vocab_size);
    return logits;
}

void LLamaRefModel::multi_head_attention(
    const LLamaLayer &layer,
    float *activation, float *attention,
    int position, int layer_id,
    float *query) {
    using namespace utils;
    int dim = Config.dim;
    int head_size = Config.head_size();

    // key and value point to the kv cache
    int offset = layer_id * Config.seq_len * dim; // kv cache layer offset for convenience

    float *key = KeyCache.data() + offset + position * dim;
    float *value = ValueCache.data() + offset + position * dim;

    // qkv matmuls for this position. key and value are generated directly at the desired position
    // inside the cache
    matmul(query,
           activation,
           layer.query_weight_matrix.data(),
           dim, dim);
    matmul(key,
           activation,
           layer.key_weight_matrix.data(),
           dim, dim);
    matmul(value,
           activation,
           layer.value_weight_matrix.data(),
           dim, dim);

    rope(Config, query, key, position);

    const float *key_base = KeyCache.data() + offset;
    const float *value_base = ValueCache.data() + offset;

    // multihead attention. iterate over all heads
#pragma omp parallel for
    for (int h = 0; h < Config.n_heads; h++) {
        // get the query vector for this head
        float *q = query + h * head_size;
        // attention scores for this head
        float *att = attention + h * Config.seq_len;

        calculate_attention(Config, att, q, position,
                            key_base + h * head_size);
        lookup_with_attention(Config, att,
                              activation + h * head_size,
                              position,
                              value_base + h * head_size);
    }
}

} // namespace ppc

bool ppc::test_against_ref_implementation(std::ostream *stream, LLamaConfig base_config, ppc::random rng, int model_multiplier,
                                          const std::string &model_modifier,
                                          std::vector<token_t> input, const std::vector<float> &logits) {
    // Load and run the reference model
    auto [config, params] = ppc::make_test_model(base_config, model_multiplier, model_modifier, rng, true);
    auto ref_result = ppc::llama_reference(config, std::move(params), std::move(input));

    const float threshold = 1e-2;
    float max_error = threshold;
    unsigned max_error_pos = -1;
    for (unsigned i = 0; i < logits.size(); ++i) {
        if (std::abs(ref_result[i] - logits[i]) > max_error || std::isnan(logits[i])) {
            max_error = std::abs(ref_result[i] - logits[i]);
            max_error_pos = i;
        }
    }

    if (max_error > threshold || std::isnan(max_error)) {
        *stream << "result\tfail\n";
        *stream << "location_pos\t" << max_error_pos / config.vocab_size << "\n";
        *stream << "location_tok\t" << max_error_pos % config.vocab_size << "\n";
        *stream << "threshold\t" << threshold << "\n";
        *stream << "max_error\t" << max_error << "\n";
        return false;
    }

    return true;
}

std::vector<float> ppc::llama_reference(LLamaConfig config, LLamaParameters params, std::vector<token_t> prompt) {
    LLamaRefModel model(config, std::move(params));
    std::vector<float> result;
    result.reserve(config.vocab_size * prompt.size());
    for (unsigned i = 0; i < prompt.size(); ++i) {
        auto pred = model.predict(i, prompt[i]);
        result.insert(result.end(), pred.begin(), pred.end());
    }
    return result;
}
