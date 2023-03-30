#include "justlm.hpp"

#include <ggml.h>
#include <llama.h>


namespace LM {
struct State {
    llama_context *ctx = nullptr;
    std::string prompt;
    std::vector<int> embd;
    int n_ctx;
    std::string last_result;
} state;



void Inference::init(const std::string& weights_path) {
    // Allocate state
    state = new State;

    // Get llama parameters
    auto lparams = llama_context_default_params();
    lparams.seed = params.seed;
    lparams.n_ctx = params.n_ctx>0?params.n_ctx:2024;

    // Create context
    state->ctx = llama_init_from_file(weights_path.c_str(), lparams);
    if (!state->ctx) {
        throw Exception("Failed to initialize llama from file");
    }

    // Initialize some variables
    state->n_ctx = llama_n_ctx(state->ctx);
}

Inference::~Inference() {
    if (state->ctx) llama_free(state->ctx);
    delete state;
}

void Inference::append(std::string_view prompt, const std::function<bool (float)> &on_tick) {
    // Check if prompt was empty
    const bool was_empty = state->prompt.empty();

    // Append to current prompt
    state->prompt.append(prompt);

    // Resize buffer for tokens
    const auto old_token_count = state->embd.size();
    state->embd.resize(old_token_count+state->prompt.size()+1);

    // Run tokenizer
    const auto token_count = llama_tokenize(state->ctx, prompt.data(), state->embd.data()+old_token_count, state->embd.size()-old_token_count, was_empty);
    state->embd.resize(old_token_count+token_count);

    // Make sure limit is far from being hit
    if (state->embd.size() > state->n_ctx-6) {
        // Yup. *this MUST be decomposed now.
        throw ContextLengthException();
    }

    // Evaluate new tokens
    // TODO: Larger batch size
    std::cout << "Context size: " << old_token_count << '+' << token_count << '=' << state->embd.size() << '/' << state->n_ctx << std::endl;
    for (int it = old_token_count; it != state->embd.size(); it++) {
        std::cout << llama_token_to_str(state->ctx, state->embd.data()[it]) << std::flush;
        llama_eval(state->ctx, state->embd.data()+it, 1, it, params.n_threads);

        // Tick
        if (on_tick) {
            // Calculate progress
            auto progress = float(it-old_token_count) / (state->embd.size()-old_token_count) * 100.f;
            // Run callback
            if (!on_tick(progress)) break;
        }
    }
    std::cout << std::endl;
}

std::string Inference::run(std::string_view end, const std::function<bool (const char *)> &on_tick) {
    std::string fres;

    // Loop until done
    bool abort = false;
    while (!abort && !ends_with(fres, end)) {
        // Sample top p and top k
        const auto id = llama_sample_top_p_top_k(state->ctx, nullptr, 0, params.top_k, params.top_p, params.temp, 1.0f);

        // Add token
        state->embd.push_back(id);

        // Get token as string
        const auto str = llama_token_to_str(state->ctx, id);

        // Debug
        std::cout << str << std::flush;

        // Append string to function result
        fres.append(str);

        // Evaluate token
        //  TODO: Respect batch size
        llama_eval(state->ctx, state->embd.data()+state->embd.size()-1, 1, state->embd.size()-1, params.n_threads);

        // Tick
        if (on_tick && !on_tick(str)) abort = true;
    }

    // Create final string  TODO: Could be optimized
    state->prompt.append(fres);
    fres = std::string(fres.data(), fres.size()-end.size());

    // Return final string
    return fres;
}
}
