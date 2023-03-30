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
    struct {
        int32_t seed; // RNG seed
        int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency()) / 2;
        union {
            int32_t n_ctx; // Context size, llama.cpp specific
            int32_t n_prompt = -1; // Prompt size, gpt2 specific
        };
        int32_t n_batch = 8; // Batch size, unused

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.72f;
    } params;

    struct State *state;

    void init(const std::string& weights_path);

    static
    bool ends_with(std::string_view str, std::string_view suffix);

public:
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct ContextLengthException : public Exception {
        ContextLengthException() : Exception("Max. context length exceeded") {}
    };

    Inference(const std::string& weights_path, int32_t seed = 0) {
        // Set random seed
        params.seed = seed?seed:time(NULL);

        // Initialize llm
        init(weights_path);
    }
    ~Inference();

    void append(std::string_view prompt, const std::function<bool (float progress)>& on_tick = nullptr);

    std::string run(std::string_view end, const std::function<bool (const char *generated)>& on_tick = nullptr);
};
}
#endif // LLM_H
