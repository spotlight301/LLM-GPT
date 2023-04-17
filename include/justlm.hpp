#ifndef LLM_H
#define LLM_H
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <thread>


namespace LM {
class Inference {
    struct State *state;

    void init(const std::string& weights_path);

    static inline
    bool ends_with(std::string_view str, std::string_view suffix) {
        return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
    }

public:
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct ContextLengthException : public Exception {
        ContextLengthException() : Exception("Max. context length exceeded") {}
    };

    struct Params {
        int32_t seed = 0; // RNG seed
        int32_t n_threads = 0;
        union {
            int32_t n_ctx; // Context size, llama.cpp specific
            int32_t n_prompt = -1; // Prompt size, gpt2 specific
        };
        int32_t n_batch = 8; // Batch size
        int32_t n_repeat_last = 0; // llama.cpp specific

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.72f;
        float repeat_penalty = 1.0f; // llama.cpp specific
        unsigned eos_ignores = 0; // llama.cpp specific

        bool use_mlock = true; // llama.cpp specific
    } params;

    struct Savestate {
        std::vector<uint8_t> kv;
        unsigned token_count;
        void *ctx = nullptr;

        bool is_valid() const {
            return ctx != nullptr;
        }
    };

    Inference(const std::string& weights_path, const Params& p) : params(p) {
        // Set random seed
        params.seed = params.seed?params.seed:time(NULL);
        params.n_threads = params.n_threads?params.n_threads:(static_cast<int32_t>(std::thread::hardware_concurrency()) / 2);

        // Initialize llm
        init(weights_path);
    }
    ~Inference();
    Inference(const Inference&) = delete;
    Inference(Inference&) = delete;
    Inference(Inference&&);

    void append(std::string_view prompt, const std::function<bool (float progress)>& on_tick = nullptr);

    std::string run(std::string_view end, const std::function<bool (const char *generated)>& on_tick = nullptr);

    void create_savestate(Savestate&);
    void restore_savestate(const Savestate&);
};
}
#endif // LLM_H
