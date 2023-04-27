#include "justlm.hpp"

#include <cstring>
#include <ggml.h>
#include <llama.h>


namespace LM {
class LLaMaInference final : public Inference {
    struct State {
        llama_context *ctx = nullptr;
        std::string prompt; // Mostly here for easy "debugging"
        std::vector<int> tokens;
        int n_ctx;
    };

    State*& get_state() {
        return *reinterpret_cast<State**>(&generic_state);
    }
    State* const& get_state() const {
        return *reinterpret_cast<State* const*>(&generic_state);
    }

    void init(const std::string& weights_path) {
        auto& state = get_state();

        // Allocate state
        state = new State;

        // Get llama parameters
        auto lparams = llama_context_default_params();
        lparams.seed = params.seed;
        lparams.n_ctx = params.n_ctx = params.n_ctx>0?params.n_ctx:2024;
        lparams.use_mlock = params.use_mlock;

        // Create context
        state->ctx = llama_init_from_file(weights_path.c_str(), lparams);
        if (!state->ctx) {
            throw Exception("Failed to initialize llama from file");
        }

        // Initialize some variables
        state->n_ctx = llama_n_ctx(state->ctx);
    }

public:
    LLaMaInference(const std::string& weights_path, const Params& p) : Inference(p) {
        init(weights_path);
    }
    ~LLaMaInference() override {
        auto& state = get_state();

        if (state) {
            if (state->ctx) llama_free(state->ctx);
            delete state;
        }
    }

    void append(std::string_view prompt, const std::function<bool (float)> &on_tick = nullptr) override {
        auto& state = get_state();

        // Check if prompt was empty
        const bool was_empty = state->prompt.empty();

        // Append to current prompt
        state->prompt.append(prompt);

        // Resize buffer for tokens
        const auto old_token_count = state->tokens.size();
        state->tokens.resize(old_token_count+state->prompt.size());

        // Run tokenizer                                          .data() <-- this may be a serious issue!!!
        const auto token_count = llama_tokenize(state->ctx, prompt.data(), state->tokens.data()+old_token_count, state->tokens.size()-old_token_count, was_empty);
        state->tokens.resize(old_token_count+token_count);

        // Make sure limit is far from being hit
        if (state->tokens.size() > state->n_ctx-6) {
            // Yup. *this MUST be decomposed now.
            throw ContextLengthException();
        }

        // Evaluate new tokens in batches
        int it;
        for (it = old_token_count; ; it += params.n_batch) {
            if (it >= ssize_t(state->tokens.size()) - params.n_batch) break;

            // Evaluate
            llama_eval(state->ctx, state->tokens.data()+it, params.n_batch, it, params.n_threads);

            // Tick
            if (on_tick) {
                // Calculate progress
                auto progress = float(it-old_token_count) / (state->tokens.size()-old_token_count) * 100.f;
                // Run callback
                if (!on_tick(progress)) break;
            }
        }
        // Evaluate remaining tokens
        if (it < state->tokens.size()) {
            for (; it != state->tokens.size(); it++) {
                llama_eval(state->ctx, state->tokens.data()+it, 1, it, params.n_threads);
            }
        }
    }

    std::string run(std::string_view end, const std::function<bool (const char *)> &on_tick = nullptr) override {
        auto& state = get_state();
        std::string fres;

        // Loop until done
        bool abort = false;
        unsigned eos_count = 0;
        while (!abort && !ends_with(fres, end)) {
            // Sample top p and top k
            auto id = llama_sample_top_p_top_k(state->ctx, params.n_repeat_last?(state->tokens.data()+state->tokens.size()-params.n_repeat_last):nullptr, params.n_repeat_last, params.top_k, params.top_p, params.temp, params.repeat_penalty);

            if (id == llama_token_eos()) {
                if (eos_count++ == params.eos_ignores) {
                    abort = true;
                    continue;
                }
                state->tokens.push_back(0);
                llama_tokenize(state->ctx, "\n", &state->tokens.back(), 1, false);
                id = state->tokens.back();
            } else {
                // Add token
                state->tokens.push_back(id);
            }

            // Get token as string
            const auto str = llama_token_to_str(state->ctx, id);

            // Append string to function result
            fres.append(str);

            // Evaluate token
            //  TODO: Respect batch size
            llama_eval(state->ctx, state->tokens.data()+state->tokens.size()-1, 1, state->tokens.size()-1, params.n_threads);

            // Tick
            if (on_tick && !on_tick(str)) abort = true;
        }

        // Create final string  TODO: Could be optimized
        state->prompt.append(fres);
        if (!abort) {
            fres = std::string(fres.data(), fres.size()-end.size());
        }

        // Return final string
        return fres;
    }

    void create_savestate(Savestate &sv) const override {
        auto& state = get_state();
        sv.kv.resize(llama_get_kv_cache_size(state->ctx));
        std::memcpy(sv.kv.data(), llama_get_kv_cache(state->ctx), sv.kv.size());
        sv.token_count = state->tokens.size();
        sv.prompt = state->prompt;
        sv.ctx = reinterpret_cast<void*>(state->ctx);
    }
    void restore_savestate(const Savestate &sv) override {
        auto& state = get_state();
        if (sv.ctx != reinterpret_cast<void*>(state->ctx))
            throw Exception("Savestate does not match context");
        llama_set_kv_cache(state->ctx, sv.kv.data(), sv.kv.size(), sv.token_count);
        state->tokens.resize(sv.token_count);
        state->prompt = sv.prompt;
    }

    void serialize(std::ostream &o) const override {
        auto& state = get_state();
        // Get state size
        auto state_size = llama_get_state_size(state->ctx);
        // Write sizes
        for (const uint32_t s : {static_cast<size_t>(state->n_ctx), state->tokens.size(), state->prompt.size(), state_size}) {
            if (!o.write(reinterpret_cast<const char*>(&s), sizeof(s))) {
                throw Exception("Failed to serialize data sizes");
            }
        }
        // Write tokens
        if (!o.write(reinterpret_cast<const char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            throw Exception("Failed to serialize tokens");
        }
        // Write prompt
        if (!o.write(state->prompt.data(), state->prompt.size())) {
            throw Exception("Failed to serialize prompt");
        }
        // Write state
        std::vector<uint8_t> state_buf(state_size);
        llama_copy_state_data(state->ctx, state_buf.data());
        if (!o.write(reinterpret_cast<const char*>(state_buf.data()), state_size)) {
            throw Exception("Failed to serialize state");
        }
    }
    void deserialize(std::istream &i) override {
        auto& state = get_state();
        uint32_t n_ctx, embd_size, prompt_size, state_size;
        // Initialization to prevent compiler complaints
        n_ctx = embd_size = prompt_size = state_size = 0;
        // Read sizes
        for (uint32_t *s : {&n_ctx, &embd_size, &prompt_size, &state_size}) {
            if (!i.read(reinterpret_cast<char*>(s), sizeof(*s))) {
                throw Exception("Failed to deserialize data sizes");
            }
        }
        if (state->n_ctx != n_ctx) {
            throw Exception("Context length differs (My "+std::to_string(state->n_ctx)+" vs. files "+std::to_string(n_ctx)+')');
        }
        // Read tokens
        state->tokens.resize(embd_size);
        if (!i.read(reinterpret_cast<char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            throw Exception("Failed to deserialize tokens");
        }
        // Read prompt
        state->prompt.resize(prompt_size);
        if (!i.read(state->prompt.data(), state->prompt.size())) {
            throw Exception("Failed to deserialize prompt");
        }
        // Read state
        std::vector<uint8_t> state_buf(state_size);
        if (!i.read(reinterpret_cast<char*>(state_buf.data()), state_buf.size())) {
            throw Exception("Failed to deserialize state");
        }
        llama_set_state_data(state->ctx, state_buf.data());
    }

    const std::string &get_prompt() const override {
        return get_state()->prompt;
    }
};
}
