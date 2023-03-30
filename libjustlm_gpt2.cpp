#include "justlm.hpp"
#include "gpt2/gpt2tc.h"

#include <filesystem>
#include <cstring>


namespace LM {
struct State {
    std::string prompt;
    std::string model_path;
    GPT2ModelEnum model;
} state;



void Inference::init(const std::string& weights_path) {
    state->model_path = weights_path;
    // Get weight file size
    auto weights_size = std::filesystem::file_size(weights_path);
    // Determine weight size
    switch (weights_size) {
    case 250700242: state->model = GPT2_MODEL_117M; break;
    case 3120522738: state->model = GPT2_MODEL_1558M; break;
    case 712396722: state->model = GPT2_MODEL_345M; break;
    case 1551900050: state->model = GPT2_MODEL_774M; break;
    default: throw Exception("Unknown model size");
    }
}

Inference::~Inference() {
    delete state;
}

void Inference::append(std::string_view prompt, const std::function<bool (float)> &on_tick) {
    state->prompt.append(prompt);
}

std::string Inference::run(std::string_view end, const std::function<bool (const char *)> &on_tick) {
    std::string fres;
    TextCompleteGlobalState *tcs;
    TextGenContext *ts;
    int count;
    struct timeval tv;
    struct list_head ts_list;

    // Initialize completion
    tcs = text_complete_global_init(state->model, state->model_path.c_str());

    // Run completion
    ts = text_complete_start(tcs, state->prompt.c_str(), params.top_k, params.top_p, params.temp,
                             params.seed, params.n_prompt>0?params.n_prompt:0xfffffff - state->prompt.size());
    bool abort = false;
    while (!abort && !ends_with(fres, end)) {
        // Run completion
        init_list_head(&ts_list);
        list_add_tail(&ts->link, &ts_list);
        text_complete_next(tcs, &ts_list);
        if (ts->out_text_len == 0)
            break;
        auto str = std::string_view{ts->out_text, static_cast<std::string_view::size_type>(ts->out_text_len)};

        // Append result to fres
        fres.append(str);

        // Tick
        if (on_tick && !on_tick(std::string(str).c_str()) /*Huge overhead in favor of llama.cpp*/) abort = true;
    }
    // End completion
    text_complete_end(ts);

    text_complete_global_end(tcs);

    // Create final string  TODO: Could be optimized
    state->prompt.append(fres);
    fres = std::string(fres.data(), fres.size()-end.size());

    // Return final string
    return fres;
}
}
