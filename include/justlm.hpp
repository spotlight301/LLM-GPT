#ifndef JUSTLM_HPP
#define JUSTLM_HPP
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <thread>


namespace LM {
class Inference {
protected:
    void *generic_state = nullptr;

    static inline
    bool ends_with(std::string_view str, std::string_view suffix) {
        if (suffix.empty()) return false;
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
        int seed = 0; // RNG seed
        unsigned n_threads = 0;
        unsigned n_ctx; // Context size
        unsigned n_ctx_window_top_bar = 0; // Top bar of visible context. Must be smaller than context size
        unsigned n_batch = 8; // Batch size
        unsigned n_repeat_last = 0; // llama.cpp specific

        unsigned top_k = 40;
        float   top_p = 0.9f;
        float   temp  = 0.72f;
        float repeat_penalty = 1.0f; // llama.cpp specific
        unsigned eos_ignores = 0; // llama.cpp specific

        bool use_mlock = true; // llama.cpp specific
    } params;

    struct Savestate {
        std::vector<uint8_t> buf;
        std::vector<int> tokens;
        std::string prompt;
        void *ctx = nullptr;

        bool is_valid() const {
            return ctx != nullptr;
        }
    };

    Inference(const Params& p) : params(p) {
        // Set random seed
        params.seed = params.seed?params.seed:time(NULL);
        params.n_threads = params.n_threads?params.n_threads:(static_cast<unsigned>(std::thread::hardware_concurrency()) / 2);
    }
    virtual ~Inference() {}
    Inference(const Inference&) = delete;
    Inference(Inference&) = delete;
    Inference(Inference&& o)
        : params(o.params)
        , generic_state(o.generic_state) {
        o.generic_state = nullptr;
    }

    static
    Inference *construct(const std::string& weights_path, const Params& p);

    virtual void append(const std::string& prompt, const std::function<bool (float progress)>& on_tick = nullptr) = 0;

    virtual std::string run(std::string_view end = "", const std::function<bool (const char *generated)>& on_tick = nullptr) = 0;

    virtual unsigned get_token_count() const = 0;

    virtual void create_savestate(Savestate&) const = 0;
    virtual void restore_savestate(const Savestate&) = 0;

    virtual void serialize(std::ostream&) const = 0;
    virtual void deserialize(std::istream&) = 0;

    virtual const std::string& get_prompt() const = 0;
};
}
#endif // JUSTLM_HPP
