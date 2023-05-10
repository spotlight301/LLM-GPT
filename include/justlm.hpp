#ifndef JUSTLM_HPP
#define JUSTLM_HPP
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <thread>

#ifdef LM_COSCHED
#   include <scheduler.hpp>
#   define LM_SCHEDULABLE(type) ::CoSched::AwaitableTask<type>
#   define LM_CORETURN co_return
#   define LM_COAWAIT co_await
#   define LM_TASKYIELD (co_await ::CoSched::Task::get_current().yield())
#else
#   define LM_SCHEDULABLE(type) type
#   define LM_CORETURN return
#   define LM_COAWAIT
#   define LM_TASKYIELD (true)
#endif

#ifdef LM_NOEXCEPT
#   define LM_NOEXCEPTDECL noexcept
#   define LM_THROW(t, r) this->last_error = (t); LM_CORETURN r;
#   define LM_LAST_ERROR_STORAGE mutable std::string last_error;
#   define LM_LAST_ERROR_GETTER const std::string& get_last_error() const {return last_error;}
#   define LM_ERRBOOL bool
#   define LM_BOOL_ERROR false
#   define LM_BOOL_SUCCESS true
#   define LM_IF_ERROR(x) if (!x)
#   define LM_ERROR_FORWARD(x) {auto v = x; if (!v) LM_CORETURN x;} 0
#else
#   define LM_NOEXCEPTDECL
#   define LM_THROW(t, r) throw Exception(t)
#   define LM_LAST_ERROR_STORAGE
#   define LM_LAST_ERROR_GETTER
#   define LM_ERRBOOL void
#   define LM_BOOL_ERROR
#   define LM_BOOL_SUCCESS
#   define LM_IF_ERROR(x) if (false)
#   define LM_ERROR_FORWARD(x) {x;}
#endif

#if _MSC_VER
#include <BaseTsd.h>
#endif


namespace LM {
using ssize_t = unsigned;
#if _MSC_VER
using ssize_t = SSIZE_T;
#endif

class Inference {
protected:
    std::function<bool (float)> on_scroll = nullptr;

    void *generic_state = nullptr;

    static inline
    bool ends_with(std::string_view str, std::string_view suffix) noexcept {
        if (suffix.empty()) return false;
        return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
    }

    LM_LAST_ERROR_STORAGE

public:
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    struct Params {
        int seed = 0; // RNG seed
        unsigned n_threads = 0;
        unsigned n_ctx = 2012; // Context size
        unsigned n_ctx_window_top_bar = 0; // Top bar of context window. Must be smaller than context size
        unsigned n_batch = 8; // Batch size
        unsigned n_repeat_last = 0; // llama.cpp specific

        float scroll_keep = 0.0f; // 0.4f to keep 40% of context below top bar when scrolling; 0.0f to remove everything after top bar

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
            : generic_state(o.generic_state)
            , params(o.params) {
        o.generic_state = nullptr;
    }

    static
    Inference *construct(const std::string& weights_path, const Params& p);

    void set_scroll_callback(const std::function<bool (float)>& scroll_cb) noexcept {
        on_scroll = scroll_cb;
    }

    // This must be called with a non-empty prompt!
    virtual LM_SCHEDULABLE(LM_ERRBOOL) append(const std::string& prompt, const std::function<bool (float progress)>& on_tick = nullptr) LM_NOEXCEPTDECL = 0;

    // append() must have been called at least once before calling this!
    virtual LM_SCHEDULABLE(std::string) run(std::string_view end = "", const std::function<bool (const char *generated)>& on_tick = nullptr) LM_NOEXCEPTDECL = 0;

    virtual unsigned get_context_size() const noexcept = 0;

    virtual LM_SCHEDULABLE(LM_ERRBOOL) create_savestate(Savestate&) const LM_NOEXCEPTDECL = 0;
    virtual LM_SCHEDULABLE(LM_ERRBOOL) restore_savestate(const Savestate&) LM_NOEXCEPTDECL = 0;

    virtual LM_SCHEDULABLE(LM_ERRBOOL) serialize(std::ostream&) const LM_NOEXCEPTDECL = 0;
    virtual LM_SCHEDULABLE(LM_ERRBOOL) deserialize(std::istream&) LM_NOEXCEPTDECL = 0;

    virtual const std::string& get_prompt() const LM_NOEXCEPTDECL = 0;

    LM_LAST_ERROR_GETTER
};
}
#endif // JUSTLM_HPP
