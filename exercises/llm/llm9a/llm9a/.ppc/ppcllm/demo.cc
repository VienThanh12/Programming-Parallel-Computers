#include "llm.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

std::int32_t read_int32(std::istream &source) {
    std::int32_t value = -1;
    if (source.read(reinterpret_cast<char *>(&value), sizeof(value))) {
        return value;
    }
    throw std::runtime_error("Could not read integer.");
}

std::vector<float> read_vector(std::istream &source, int num) {
    std::vector<float> value(num);
    if (source.read(reinterpret_cast<char *>(value.data()), sizeof(float) * num)) {
        return value;
    }
    throw std::runtime_error("Could not read parameters.");
}

std::pair<LLamaConfig, LLamaParameters> read_checkpoint(std::istream &source) {
    // Start by reading the basic configuration
    LLamaConfig config;
    config.dim = read_int32(source);
    config.hidden_dim = read_int32(source);
    config.n_layers = read_int32(source);
    config.n_heads = read_int32(source);
    int n_kv_heads = read_int32(source);
    if (n_kv_heads != config.n_heads) {
        std::cerr << "Multiquery models are not supported";
        std::exit(1);
    }

    config.vocab_size = read_int32(source);
    config.seq_len = read_int32(source);

    // llama2.c uses negative vocab_size to indicate non-shared encoding/decoding representation
    bool shared_weights = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size);

    // now actually read the weights
    LLamaParameters weights;
    weights.TokenEmbeddingMatrix = read_vector(source, config.vocab_size * config.dim);
    weights.LayerWeights.resize(config.n_layers);

    // reading rms attention weights
    auto read_data = [&](auto &&ptr_to_member, int num) {
        for (auto &layer : weights.LayerWeights) {
            layer.*ptr_to_member = read_vector(source, num);
        }
    };

    int head_size = config.dim / config.n_heads;
    read_data(&LLamaLayer::rms_attention, config.dim);
    read_data(&LLamaLayer::query_weight_matrix, config.dim * (config.n_heads * head_size));
    read_data(&LLamaLayer::key_weight_matrix, config.dim * (config.n_heads * head_size));
    read_data(&LLamaLayer::value_weight_matrix, config.dim * (config.n_heads * head_size));
    read_data(&LLamaLayer::out_weight_matrix, (config.n_heads * head_size) * config.dim);

    read_data(&LLamaLayer::rms_feed_forward, config.dim);
    read_data(&LLamaLayer::feed_forward_w1, config.dim * config.hidden_dim);
    read_data(&LLamaLayer::feed_forward_w2, config.hidden_dim * config.dim);
    read_data(&LLamaLayer::feed_forward_w3, config.dim * config.hidden_dim);

    weights.RmsFinal = read_vector(source, config.dim);
    // unused
    read_vector(source, config.seq_len * head_size);
    if (shared_weights) {
        weights.TokenOutputMatrix = weights.TokenEmbeddingMatrix;
    } else {
        weights.TokenOutputMatrix = read_vector(source, config.vocab_size * config.dim);
    }

    return std::make_pair(config, weights);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

class Tokenizer {
  public:
    explicit Tokenizer(int vocab_size);
    void load(const char *path);

    std::string decode(token_t prev_token, token_t token);
    std::vector<token_t> encode(const std::string &text, bool bos, bool eos);
    int lookup(const std::string &key) const;

    std::string get_token_str(token_t id) { return vocab[(int)id]; }

  private:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::unordered_map<std::string, int> sorted_vocab;
};

Tokenizer::Tokenizer(int vocab_size) : vocab(vocab_size), vocab_scores(vocab_size) {
}

void Tokenizer::load(const char *path) {
    // read in the file
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned int max_token_length;
    if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (unsigned i = 0; i < vocab.size(); i++) {
        if (fread(vocab_scores.data() + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        vocab[i].resize(len + 1);
        if (fread(vocab[i].data(), len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        vocab[i].pop_back();
    }
    fclose(file);

    // sort the vocabulary
    for (unsigned i = 0; i < vocab.size(); i++) {
        sorted_vocab.emplace(vocab[i], i);
    }
}

std::string Tokenizer::decode(token_t prev_token, token_t token) {
    std::string piece = vocab[(int)token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == token_t::BOS && piece[0] == ' ') {
        piece = piece.substr(1);
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        return {(int)1, (char)byte_val};
    }
    return piece;
}

int Tokenizer::lookup(const std::string &key) const {
    auto found = sorted_vocab.find(key);
    return found != sorted_vocab.end() ? found->second : -1;
}

std::vector<token_t> Tokenizer::encode(const std::string &text, bool bos, bool eos) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    std::string str_buffer;
    std::vector<token_t> result;

    // add optional BOS (=1) token, if desired
    if (bos)
        result.push_back(token_t::BOS);

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int start = lookup(" ");
        if (start != -1) {
            result.push_back(token_t{start});
        }
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text.data(); *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_buffer.clear();
        }

        // append the current byte to the buffer
        str_buffer.push_back(*c);

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overrunning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_buffer.size() < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = lookup(str_buffer);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            result.push_back(token_t{id});
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (char i : str_buffer) {
                result.push_back(token_t{(unsigned char)i + 3});
            }
        }

        str_buffer.clear(); // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (unsigned i = 0; i < result.size() - 1; i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            int id = lookup(vocab[(int)result[i]] + vocab[(int)result[i + 1]]);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        result[best_idx] = token_t{best_id};
        result.erase(result.begin() + best_idx + 1);
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        result.push_back(token_t::EOS);

    return result;
}

std::string make_save(const std::string &s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        if (std::isprint(static_cast<unsigned char>(c))) {
            result.push_back(c);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            result.push_back(' ');
        } else {
            char buf[10];
            std::snprintf(buf, sizeof(buf), "<0x%X/>", static_cast<unsigned char>(c));
            result += buf;
        }
    }
    return result;
}

void print_colorized(std::ostream &stream, const std::string &text, const std::string &format_code) {
    if (text.empty())
        return;

    if (text == "\n") {
        stream << "</div>\n<div>";
    } else {
        stream << "<span class='" << format_code << "'>" << text.c_str() << "</span>";
    }
}

const std::string &format_color(float p) {
    static const std::string formats[] = {"certain", "high", "low", "surprise", "extreme"};
    ;

    if (p > 0.8) {
        return formats[0];
    } else if (p > 0.3) {
        return formats[1];
    } else if (p > 0.1) {
        return formats[2];
    } else if (p > 0.01) {
        return formats[3];
    } else {
        return formats[4];
    }
}

const char *header =
    "<head><style>\n"
    ".surprise {\n"
    "  color: red;\n"
    "}\n"
    ".extreme {\n"
    "  color: red;\n"
    "  font-weight: bold;\n"
    "}\n"
    ".low {\n"
    "  color: orange;\n"
    "}\n"
    ".certain {\n"
    "  color: darkgreen;\n"
    "}\n"
    ".high {\n"
    "  color: black;\n"
    "}\n"
    "</style></head>\n";

int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    std::string prompt = argv[3];
    std::ifstream model_file(argv[1]);

    auto [config, params] = read_checkpoint(model_file);

    Tokenizer tok(config.vocab_size);
    tok.load(argv[2]);

    auto tokens = tok.encode(prompt, true, false);
    std::vector<float> probs(tokens.size() * config.vocab_size);
    llm(config, params, tokens, probs);
    // convert logits to probabilities
    for (unsigned i = 0; i < tokens.size(); ++i) {
        utils::softmax(probs.data() + i * config.vocab_size, config.vocab_size);
    }

    // Print the result
    std::ofstream out("out.html");
    out << header << "<bode>\n<pre>\n<div>";

    for (unsigned i = 1; i < tokens.size(); ++i) {
        // current token probabilities are predicted from *previous* tokens,
        // so we need to index into the (i-1)th part.
        float p = probs[(i - 1) * config.vocab_size + (int)tokens[i]];
        std::string tok_text = make_save(tok.decode(tokens[i - 1], tokens[i]));
        print_colorized(out, tok_text, format_color(p));
    }
    out << "\n</div>\n</pre>\n</body>";
    out << std::endl;

    return EXIT_SUCCESS;
}
