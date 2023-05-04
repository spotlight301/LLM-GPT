#ifndef JUSTLM_GPTJ_HPP
#define JUSTLM_GPTJ_HPP
#include "justlm.hpp"

#include <fstream>
#include <random>
#include <cstring>
#include "gptj/gptj.hpp"
#include "gptj/utils.hpp"


namespace LM {
class GPTJInference final : public Inference {
    std::string weights_path;

    struct State {
        gpt_vocab vocab;
        gptj_model model;
        std::string prompt; // Mostly here for easy "debugging"
        std::vector<int> tokens;
        std::vector<float> logits;
        size_t mem_per_token = 0;
        std::mt19937 rng;

        State(int32_t seed) : rng(seed) {}
    };

    State*& get_state() {
        return *reinterpret_cast<State**>(&generic_state);
    }
    State* const& get_state() const {
        return *reinterpret_cast<State* const*>(&generic_state);
    }

    void init(const std::string& _weights_path, std::ifstream& f) {
        auto& state = get_state();
        weights_path = _weights_path;

        // Allocate state
        state = new State(params.seed);

        // Load model
        if (!gptj_model_load(weights_path, f, state->model, state->vocab)) {
            throw Exception("Failed to initialize gptj from file");
        }

        // Calculate memory required per token
        static std::vector<gpt_vocab::id> p_instruct;
        static std::vector<gpt_vocab::id> r_instruct;
        gptj_eval(state->model, params.n_threads, 0, { 0, 1, 2, 3 }, state->logits, state->mem_per_token);
    }
    void deinit() {
        auto& state = get_state();

        if (state) {
            if (state->model.ctx) ggml_free(state->model.ctx); //TODO: Is that enough?
            delete state;
        }
    }
    void reinit() {
        if (!get_state()->prompt.empty()) {
            deinit();
            std::ifstream f(weights_path, std::ios::binary);
            init(weights_path, f);
        }
    }

    // This function reduces the size of our tokens vector according to some parameters
    // All tokens will be evaluated if scrolling was needed and true will be returned
    LM_SCHEDULABLE(bool) window_scroll() {
        auto &state = get_state();
        // Check that we actually need to scroll
        if (state->tokens.size() <= params.n_ctx) {
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
        LM_COAWAIT evaluate_tokens(0, on_scroll);
        // We've scrolled!
        LM_CORETURN true;
    }

    LM_SCHEDULABLE(void) evaluate_tokens(size_t starting_offset, const std::function<bool (float)> &on_tick = nullptr) {
        auto& state = get_state();

        // Evaluate tokens in batches
        unsigned it;
        for (it = starting_offset; ; it += params.n_batch) {
            if (it + params.n_batch >= ssize_t(state->tokens.size())) break;

            // Evaluate
            std::vector<int> batch(state->tokens.begin()+it, state->tokens.begin()+it+params.n_batch);
            gptj_eval(state->model, params.n_threads, it, batch, state->logits, state->mem_per_token);

            // Tick
            if (on_tick) {
                // Calculate progress
                auto progress = float(it-starting_offset) / (state->tokens.size()-starting_offset) * 100.f;
                // Run callback
                if (!on_tick(progress)) break;
            }
        }

        // Evaluate remaining tokens
        if (it < state->tokens.size()) {
            for (; it != state->tokens.size(); it++) {
                //TODO: This is extremely inefficient! Don't do that...
                std::vector<int> batch(state->tokens.begin()+it, state->tokens.begin()+it+1);
                gptj_eval(state->model, params.n_threads, it, batch, state->logits, state->mem_per_token);
            }
        }

        // Notify about completion
        if (on_tick) on_tick(100.f);

        LM_CORETURN;
    }

public:
    GPTJInference(const std::string& weights_path, std::ifstream& f, const Params& p) : Inference(p) {
        init(weights_path, f);
    }
    ~GPTJInference() override {
        deinit();
    }

    LM_SCHEDULABLE(void) append(const std::string& prompt, const std::function<bool (float)> &on_tick = nullptr) override {
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
        if (LM_COAWAIT window_scroll()) {
            // That function already has evaluated our tokens since scrolling was needed
            LM_CORETURN;
        }

        // Evaluate new tokens
        LM_COAWAIT evaluate_tokens(old_token_count, on_tick);
    }

    LM_SCHEDULABLE(std::string) run(std::string_view end, const std::function<bool (const char *)> &on_tick = nullptr) override {
        auto& state = get_state();
        std::string fres;

        // Loop until done
        bool abort = false;
        unsigned eos_count = 0;
        while (!abort && !ends_with(fres, end)) {
            // Sample top p and top k
            auto id = gpt_sample_top_k_top_p(state->vocab, params.n_repeat_last?(state->tokens.data()+state->tokens.size()-params.n_repeat_last):nullptr, params.n_repeat_last, state->logits, params.top_k, params.top_p, params.temp, params.repeat_penalty, state->rng);

            if (id == 50256) {
                if (eos_count++ == params.eos_ignores) {
                    abort = true;
                    continue;
                }
                id = gpt_tokenize(state->vocab, "\n")[0];
                state->tokens.push_back(id);
            } else {
                // Add token
                state->tokens.push_back(id);
            }

            // Make sure token limit isn't being hit
            LM_COAWAIT window_scroll();

            // Get token as string
            const auto str = state->vocab.id_to_token[id];

            // Append string to function result
            fres.append(str);

            // Evaluate token
            //  TODO: Respect batch size
            std::vector<int> batch(state->tokens.begin()+state->tokens.size()-1, state->tokens.begin()+state->tokens.size());
            gptj_eval(state->model, params.n_threads, state->tokens.size()-1, batch, state->logits, state->mem_per_token);

            // Tick
            if (on_tick && !on_tick(str.c_str())) abort = true;
        }

        // Create final string  TODO: Could be optimized
        state->prompt.append(fres);
        if (!abort) {
            fres = std::string(fres.data(), fres.size()-end.size());
        }

        // Return final string
        LM_CORETURN fres;
    }

    unsigned get_context_size() const override {
        return get_state()->tokens.size();
    }

    //TODO: The following functions are just a stub implementations and should be implemented properly asap
    LM_SCHEDULABLE(void) create_savestate(Savestate &sv) const override {
        auto& state = get_state();
        sv.prompt = state->prompt;
        sv.ctx = generic_state;
        LM_CORETURN;
    }
    LM_SCHEDULABLE(void) restore_savestate(const Savestate &sv) override {
        auto& state = get_state();
        if (sv.ctx != generic_state)
            throw Exception("Savestate does not match context");
        reinit();
        LM_COAWAIT append(sv.prompt);
        LM_CORETURN;
    }

    LM_SCHEDULABLE(void) serialize(std::ostream &o) const override {
        auto& state = get_state();
        size_t size = state->prompt.size();
        o.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (!o.write(state->prompt.data(), size)) {
            throw Exception("Failed to serialize prompt");
        }
        LM_CORETURN;
    }
    LM_SCHEDULABLE(void) deserialize(std::istream &i) override {
        auto& state = get_state();
        std::string prompt;
        size_t size;
        if (!i.read(reinterpret_cast<char*>(&size), sizeof(size))) {
            throw Exception("Failed to deserialize prompt size");
        }
        prompt.resize(size);
        if (!i.read(prompt.data(), size)) {
            throw Exception("Failed to deserialize prompt");
        }
        reinit();
        LM_COAWAIT append(prompt);
        LM_CORETURN;
    }

    const std::string &get_prompt() const override {
        return get_state()->prompt;
    }
};
}

#endif // JUSTLM_GPTJ_HPP
