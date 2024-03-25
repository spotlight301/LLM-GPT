#include "justlm.hpp"

#include <fstream>
#include <random>
#include <cstring>
#include "mpt/mpt.hpp"
#include "g4a_common.hpp"


namespace LM {
class MPTInference final : public Inference {
    std::string weights_path;

    struct State {
        gpt_vocab vocab;
        mpt_model model;
        std::string prompt; // Mostly here for easy "debugging"
        std::vector<int> tokens;
        std::vector<float> logits;
        size_t mem_per_token = 0;
        std::mt19937 rng;
        int im_end = 0;

        State(int32_t seed) : rng(seed) {}
    };

    State*& get_state() LM_NOEXCEPTDECL {
        return *reinterpret_cast<State**>(&generic_state);
    }
    State* const& get_state() const LM_NOEXCEPTDECL {
        return *reinterpret_cast<State* const*>(&generic_state);
    }

    LM_ERRBOOL init(const std::string& _weights_path, std::ifstream& f) LM_NOEXCEPTDECL {
        auto& state = get_state();
        weights_path = _weights_path;

        // Allocate state
        state = new State(params.seed);

        // Load model
        if (!mpt_model_load(weights_path, f, state->model, state->vocab)) {
            LM_THROW("Failed to initialize mpt_ from file", LM_BOOL_ERROR);
        }

        // Calculate memory required per token
        static std::vector<gpt_vocab::id> p_instruct;
        static std::vector<gpt_vocab::id> r_instruct;
        mpt_eval(state->model, params.n_threads, 0, { 0, 1, 2, 3 }, state->logits, state->mem_per_token);

        // Find im_end token
        {
            auto res = state->vocab.token_to_id.find("<|im_end|>");
            if (res != state->vocab.token_to_id.end()) {
                state->im_end = res->second;
            }
        }

        return LM_BOOL_SUCCESS;
    }
    void deinit() LM_NOEXCEPTDECL {
        auto& state = get_state();

        if (state) {
            delete state;
        }
    }

    // This function reduces the size of our tokens vector according to some parameters
    // All tokens will be evaluated if scrolling was needed and true will be returned
    bool window_scroll() LM_NOEXCEPTDECL {
        auto &state = get_state();
        // Check that we actually need to scroll
        if (state->tokens.size() <= params.n_ctx) {
            // Nope
            return false;
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
        LM_ERROR_FORWARD(evaluate_tokens(0, on_scroll), LM_BOOL_ERROR);
        return true;
    }

    LM_ERRBOOL evaluate_tokens(size_t starting_offset, const AppendCallback &on_tick) LM_NOEXCEPTDECL {
        auto& state = get_state();

        // Evaluate tokens in batches
        unsigned it;
        for (it = starting_offset; ; it += params.n_batch) {
            if (it + params.n_batch >= ssize_t(state->tokens.size())) break;

            // Evaluate
            std::vector<int> batch(state->tokens.begin()+it, state->tokens.begin()+it+params.n_batch);
            if (!mpt_eval(state->model, params.n_threads, it, batch, state->logits, state->mem_per_token)) {
                LM_THROW("Failed to evaluate tokens in batches", LM_BOOL_ERROR);
            }

            // Tick
            if (on_tick) {
                // Calculate progress
                auto progress = float(it-starting_offset) / (state->tokens.size()-starting_offset) * 100.f;
                // Tick and yield
                if (!on_tick(progress)) return LM_BOOL_SUCCESS;
            }
        }

        // Evaluate remaining tokens
        if (it < state->tokens.size()) {
            for (; it != state->tokens.size(); it++) {
                //TODO: This is extremely inefficient! Don't do that...
                std::vector<int> batch(state->tokens.begin()+it, state->tokens.begin()+it+1);
                if (!mpt_eval(state->model, params.n_threads, it, batch, state->logits, state->mem_per_token)) {
                    LM_THROW("Failed to evaluate individual tokens", LM_BOOL_ERROR);
                }
            }
        }

        // Notify about completion
        if (on_tick) on_tick(100.f);

        return LM_BOOL_SUCCESS;
    }

public:
    MPTInference(const std::string& weights_path, std::ifstream& f, const Params& p) : Inference(p) {
        init(weights_path, f);
    }
    ~MPTInference() LM_NOEXCEPTDECL override {
        deinit();
    }

    LM_ERRBOOL append(const std::string& prompt, const AppendCallback &on_tick) LM_NOEXCEPTDECL override {
        auto& state = get_state();

        // Append to current prompt
        state->prompt.append(prompt);

        // Resize buffer for tokens
        const auto old_token_count = state->tokens.size();

        // Run tokenizer
        const auto tokens = gpt_tokenize(state->vocab, prompt);
        state->tokens.insert(
                    state->tokens.end(),
                    std::make_move_iterator(tokens.begin()),
                    std::make_move_iterator(tokens.end())
        );

        // Make sure token limit isn't being hit
        if (window_scroll()) {
            // That function already has evaluated our tokens since scrolling was needed
            return LM_BOOL_SUCCESS;
        }

        // Evaluate new tokens
        return evaluate_tokens(old_token_count, on_tick);
    }

    std::string run(std::string_view end, const GenerateCallback &on_tick, const GenerateCallback& pre_tick) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        std::string fres;

        // Loop until done
        bool abort = false;
        unsigned eos_count = 0;
        size_t last_size = 0;
        while (!abort && (end.empty() || fres.find(end) == fres.npos)) {
            last_size = fres.size();
            // Sample top p and top k
            const auto n_repeat_last = std::min<size_t>(state->tokens.size(), params.n_repeat_last);
            auto id = gpt_sample_top_k_top_p(state->model.hparams.n_vocab, state->tokens.data()+state->tokens.size()-n_repeat_last, n_repeat_last, state->logits, params.top_k, params.top_p, params.temp, params.repeat_penalty, state->rng);

            if (state->im_end && id == state->im_end) {
                if (eos_count++ == params.n_eos_ignores) {
                    abort = true;
                    continue;
                }
                id = gpt_tokenize(state->vocab, "\n")[0];
            } else if (id == 0) {
                if (eos_count++ == params.n_eos_ignores) {
                    abort = true;
                    continue;
                }
                id = gpt_tokenize(state->vocab, "\n")[0];
            }

            // Add token
            state->tokens.push_back(id);

            // Make sure token limit isn't being hit
            window_scroll();

            // Get token as string
            const std::string_view str = state->vocab.id_to_token[id];

            // Append string to function result
            fres.append(str);
            state->prompt.append(str);

            // Tick
            if (pre_tick && !pre_tick(str.data())) abort = true;
            else {
                // Evaluate token
                //  TODO: Respect batch size
                std::vector<int> batch(state->tokens.begin()+state->tokens.size()-1, state->tokens.begin()+state->tokens.size());
                if (!mpt_eval(state->model, params.n_threads, state->tokens.size()-1, batch, state->logits, state->mem_per_token)) {
                    LM_THROW("Failed to evaluate new tokens", "");
                }
            }

            // Tick
            if (on_tick && !on_tick(str.data())) abort = true;
        }

        // Create final string  TODO: Could be optimized
        if (!abort) {
            fres = std::string(fres.data(), last_size);
        }

        // Return final string
        return fres;
    }

    unsigned get_context_size() const noexcept override {
        return get_state()->tokens.size();
    }

    LM_ERRBOOL create_savestate(Savestate &sv) const LM_NOEXCEPTDECL override {
        auto& state = get_state();
        sv.buf.resize(mpt_get_state_size(state->model));
        mpt_copy_state_data(state->model, state->rng, sv.buf.data());
        sv.tokens = state->tokens;
        sv.prompt = state->prompt;
        sv.ctx = generic_state;
        return LM_BOOL_SUCCESS ;
    }
    LM_ERRBOOL restore_savestate(const Savestate &sv) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        if (sv.ctx != generic_state)
            LM_THROW("Savestate does not match context", LM_BOOL_ERROR);
        mpt_set_state_data(&state->model, &state->rng, sv.buf.data());
        state->tokens = sv.tokens;
        state->prompt = sv.prompt;
        return LM_BOOL_SUCCESS;
    }

    LM_ERRBOOL serialize(std::ostream &o) const LM_NOEXCEPTDECL override {
        auto& state = get_state();
        // Get state size
        auto state_size = mpt_get_state_size(state->model);
        // Write sizes
        for (const uint32_t s : {state->tokens.size(), state->prompt.size(), state_size}) {
            if (!o.write(reinterpret_cast<const char*>(&s), sizeof(s))) {
                LM_THROW("Failed to serialize data sizes", LM_BOOL_ERROR);
            }
        }
        // Write tokens
        if (!o.write(reinterpret_cast<const char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            LM_THROW("Failed to serialize tokens", LM_BOOL_ERROR);
        }
        // Write prompt
        if (!o.write(state->prompt.data(), state->prompt.size())) {
            LM_THROW("Failed to serialize prompt", LM_BOOL_ERROR);
        }
        // Write state
        std::vector<uint8_t> state_buf(state_size);
        mpt_copy_state_data(state->model, state->rng, state_buf.data());
        if (!o.write(reinterpret_cast<const char*>(state_buf.data()), state_size)) {
            LM_THROW("Failed to serialize state", LM_BOOL_ERROR);
        }
        return LM_BOOL_SUCCESS;
    }
    LM_ERRBOOL deserialize(std::istream &i) LM_NOEXCEPTDECL override {
        auto& state = get_state();
        uint32_t embd_size, promptsize, state_size;
        // Initialization to prevent compiler complaints
        embd_size = promptsize = state_size = 0;
        // Read sizes
        for (uint32_t *s : {&embd_size, &promptsize, &state_size}) {
            if (!i.read(reinterpret_cast<char*>(s), sizeof(*s))) {
                LM_THROW("Failed to deserialize data sizes", LM_BOOL_ERROR);
            }
        }
        // Read tokens
        state->tokens.resize(embd_size);
        if (!i.read(reinterpret_cast<char*>(state->tokens.data()), state->tokens.size()*sizeof(int))) {
            LM_THROW("Failed to deserialize tokens", LM_BOOL_ERROR);
        }
        // Read prompt
        state->prompt.resize(promptsize);
        if (!i.read(state->prompt.data(), state->prompt.size())) {
            LM_THROW("Failed to deserialize prompt", LM_BOOL_ERROR);
        }
        // Read state
        std::vector<uint8_t> state_buf(state_size);
        if (!i.read(reinterpret_cast<char*>(state_buf.data()), state_buf.size())) {
            LM_THROW("Failed to deserialize state", LM_BOOL_ERROR);
        }
        mpt_set_state_data(&state->model, &state->rng, state_buf.data());
        return LM_BOOL_SUCCESS;
    }
    const std::string &get_prompt() const LM_NOEXCEPTDECL override {
        return get_state()->prompt;
    }
};
}
