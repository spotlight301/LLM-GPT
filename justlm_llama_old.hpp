#include "justlm.hpp"

#include <cstring>
#include <ggml.h>
#include <llama.h>


namespace LM {
class LLaMAInference final : public Inference {
    struct State {
        llama_context *ctx = nullptr;
        std::string prompt; // Mostly here for easy "debugging"
        std::vector<int> tokens;
        unsigned n_ctx;
    };

    State*& get_state() {
        return *reinterpret_cast<State**>(&generic_state);
    }
    State* const& get_state() const {
        return *reinterpret_cast<State* const*>(&generic_state);
    }

    LM_ERRBOOL init(const std::string& weights_path) LM_NOEXCEPTDECL {
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
            LM_THROW("Failed to initialize llama from file", LM_BOOL_ERROR);
        }

        // Initialize some variables
        state->n_ctx = llama_n_ctx(state->ctx);

        return LM_BOOL_SUCCESS;
    }

    // This function reduces the size of our tokens vector according to some parameters
    // All tokens will be evaluated if scrolling was needed and true will be returned
    LM_SCHEDULABLE(bool) window_scroll() LM_NOEXCEPTDECL {
        auto &state = get_state();
        // Check that we actually need to scroll
        if (state->tokens.size() <= state->n_ctx) {
            // Nope
            LM_CORETURN false;
        }
        // Start scrolling
        if (params.scroll_keep > 0.0f) {
            // "Scroll" down the context window...
            unsigned keep_count = float(state->tokens.size() - params.n_ctx_window_top_bar) * 0.4f; // We keep about 40%
            // Get vector of tokens to keep
            std::vector<int> tokens_in_view(state->tokens.end()-keep_count, state->tokens.end());
            // Cut down tokens vector size
            state->tokens.resize(params.n_ctx_window_top_bar+keep_count);
            // Overwrite tokens after top bar with tokens in view
            std::memcpy(state->tokens.data()+params.n_ctx_window_top_bar, tokens_in_view.data(), tokens_in_view.size()*sizeof(int));
        } else {
            // Cut down tokens vector size to top bar
            state->tokens.resize(params.n_ctx_window_top_bar);
        }
        // Evaluate tokens
        LM_ERROR_FORWARD(LM_COAWAIT evaluate_tokens(0, on_scroll));
        LM_CORETURN true;
    }

    LM_SCHEDULABLE(LM_ERRBOOL) evaluate_tokens(size_t starting_offset, const std::function<bool (float)> &on_tick = nullptr) LM_NOEXCEPTDECL {
        auto& state = get_state();

        // Evaluate tokens in batches
        unsigned it;
        for (it = starting_offset; ; it += params.n_batch) {
            if (it + params.n_batch >= ssize_t(state->tokens.size())) break;

            // Evaluate
            if (llama_eval(state->ctx, state->tokens.data()+it, params.n_batch, it, params.n_threads)) {
                LM_COTHROW("Failed to evaluate tokens in batches", LM_BOOL_ERROR);
            }

            // Tick
            if (on_tick) {
                // Calculate progress
                auto progress = float(it-starting_offset) / (state->tokens.size()-starting_offset) * 100.f;
                // Tick and yield
                if (!on_tick(progress)) LM_CORETURN LM_BOOL_SUCCESS;
                else if (!LM_TASKYIELD) LM_CORETURN LM_BOOL_SUCCESS;
            }
        }

        // Evaluate remaining tokens
        if (it < state->tokens.size()) {
            for (; it != state->tokens.size(); it++) {
                if (llama_eval(state->ctx, state->tokens.data()+it, 1, it, params.n_threads)) {
                    LM_COTHROW("Failed to evaluate individual tokens", LM_BOOL_ERROR);
                }
            }
        }

        // Notify about completion
        if (on_tick) on_tick(100.f);

        LM_CORETURN LM_BOOL_SUCCESS;
    }

public:
    LLaMAInference(const std::string& weights_path, const Params& p) : Inference(p) {
        init(weights_path);
    }
    ~LLaMAInference() override {
        auto& state = get_state();

        if (state) {
            if (state->ctx) llama_free(state->ctx);
            delete state;
        }
    }

    LM_SCHEDULABLE(LM_ERRBOOL) append(const std::string& prompt, const std::function<bool (float)> &on_tick = nullptr) LM_NOEXCEPTDECL override {
        auto& state = get_state();

        // Check if prompt was empty
        const bool was_empty = state->prompt.empty();

        // Append to current prompt
        state->prompt.append(prompt);

        // Resize buffer for tokens
        const auto old_token_count = state->tokens.size();
        state->tokens.resize(old_token_count+state->prompt.size());

        // Run tokenizer
        const auto token_count = llama_tokenize(state->ctx, prompt.c_str(), state->tokens.data()+old_token_count, state->tokens.size()-old_token_count, was_empty);
        state->tokens.resize(old_token_count+token_count);

        // Make sure token limit isn't being hit
        if (LM_COAWAIT window_scroll()) {
            // That function already has evaluated our tokens since scrolling was needed
            LM_CORETURN LM_BOOL_SUCCESS;
        }

        // Evaluate new tokens
        LM_CORETURN LM_COAWAIT evaluate_tokens(old_token_count, on_tick);
    }

    LM_SCHEDULABLE(std::string) run(std::string_view end, const std::function<bool (const char *)> &on_tick = nullptr) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        std::string fres;

        // Loop until done
        bool abort = false;
        unsigned eos_count = 0;
        while (!abort && fres.size() >= end.size() && fres.find(end) != fres.npos) {
            // Sample top p and top k
            const auto n_repeat_last = std::min<size_t>(state->tokens.size(), params.n_repeat_last);
            auto id = llama_sample_top_p_top_k(state->ctx, params.n_repeat_last?(state->tokens.data()+state->tokens.size()-n_repeat_last):nullptr, n_repeat_last, params.top_k, params.top_p, params.temp, params.repeat_penalty);

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

            // Make sure token limit isn't hit
            LM_COAWAIT window_scroll();

            // Get token as string
            const std::string_view str = llama_token_to_str(state->ctx, id);

            // Append string to function result
            state->prompt.append(str);
            fres.append(str);

            // Evaluate token
            //  TODO: Respect batch size
            if (llama_eval(state->ctx, state->tokens.data()+state->tokens.size()-1, 1, state->tokens.size()-1, params.n_threads)) {
                LM_COTHROW("Failed to evaluate new tokens", "");
            }

            // Tick and yield
            if (on_tick && !on_tick(str.data())) abort = true;
            else if (!LM_TASKYIELD) abort = true;
        }

        // Create final string  TODO: Could be optimized
        if (!abort) {
            fres = std::string(fres.data(), fres.size()-end.size());
        }

        // Return final string
        LM_CORETURN fres;
    }

    unsigned get_context_size() const noexcept override {
        return get_state()->tokens.size();
    }

    LM_SCHEDULABLE(LM_ERRBOOL) create_savestate(Savestate &sv) const LM_NOEXCEPTDECL override {
        auto& state = get_state();
        sv.buf.resize(llama_get_state_size(state->ctx));
        llama_copy_state_data(state->ctx, sv.buf.data());
        sv.tokens = state->tokens;
        sv.prompt = state->prompt;
        sv.ctx = generic_state;
        LM_CORETURN LM_BOOL_SUCCESS;
    }
    LM_SCHEDULABLE(LM_ERRBOOL) restore_savestate(const Savestate &sv) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        if (sv.ctx != generic_state)
            LM_COTHROW("Savestate does not match context", LM_BOOL_ERROR);
        llama_set_state_data(state->ctx, sv.buf.data());
        state->tokens = sv.tokens;
        state->prompt = sv.prompt;
        LM_CORETURN LM_BOOL_SUCCESS;
    }

    LM_SCHEDULABLE(LM_ERRBOOL) serialize(std::ostream &o) const LM_NOEXCEPTDECL override {
        auto& state = get_state();
        // Get state size
        auto state_size = llama_get_state_size(state->ctx);
        // Write sizes
        for (const uint32_t s : {static_cast<size_t>(state->n_ctx), state->tokens.size(), state->prompt.size(), state_size}) {
            if (!o.write(reinterpret_cast<const char*>(&s), sizeof(s))) {
                LM_COTHROW("Failed to serialize data sizes", LM_BOOL_ERROR);
            }
        }
        // Write tokens
        if (!o.write(reinterpret_cast<const char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            LM_COTHROW("Failed to serialize tokens", LM_BOOL_ERROR);
        }
        // Write prompt
        if (!o.write(state->prompt.data(), state->prompt.size())) {
            LM_COTHROW("Failed to serialize prompt", LM_BOOL_ERROR);
        }
        // Write state
        std::vector<uint8_t> state_buf(state_size);
        llama_copy_state_data(state->ctx, state_buf.data());
        if (!o.write(reinterpret_cast<const char*>(state_buf.data()), state_size)) {
            LM_COTHROW("Failed to serialize state", LM_BOOL_ERROR);
        }
        LM_CORETURN LM_BOOL_SUCCESS;
    }
    LM_SCHEDULABLE(LM_ERRBOOL) deserialize(std::istream &i) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        uint32_t n_ctx, embd_size, prompt_size, state_size;
        // Initialization to prevent compiler complaints
        n_ctx = embd_size = prompt_size = state_size = 0;
        // Read sizes
        for (uint32_t *s : {&n_ctx, &embd_size, &prompt_size, &state_size}) {
            if (!i.read(reinterpret_cast<char*>(s), sizeof(*s))) {
                LM_COTHROW("Failed to deserialize data sizes", LM_BOOL_ERROR);
            }
        }
        if (state->n_ctx != n_ctx) {
            LM_COTHROW("Context length differs (My "+std::to_string(state->n_ctx)+" vs. files "+std::to_string(n_ctx)+')', LM_BOOL_ERROR);
        }
        // Read tokens
        state->tokens.resize(embd_size);
        if (!i.read(reinterpret_cast<char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            LM_COTHROW("Failed to deserialize tokens", LM_BOOL_ERROR);
        }
        // Read prompt
        state->prompt.resize(prompt_size);
        if (!i.read(state->prompt.data(), state->prompt.size())) {
            LM_COTHROW("Failed to deserialize prompt", LM_BOOL_ERROR);
        }
        // Read state
        std::vector<uint8_t> state_buf(state_size);
        if (!i.read(reinterpret_cast<char*>(state_buf.data()), state_buf.size())) {
            LM_COTHROW("Failed to deserialize state", LM_BOOL_ERROR);
        }
        llama_set_state_data(state->ctx, state_buf.data());
        LM_CORETURN LM_BOOL_SUCCESS;
    }

    const std::string &get_prompt() const LM_NOEXCEPTDECL override {
        return get_state()->prompt;
    }
};
}
