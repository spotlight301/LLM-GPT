#ifndef GPTJ_HPP
#define GPTJ_HPP
#include <string>
#include <vector>
#include <map>
#include <ggml.h>

#include "utils.hpp"


// default hparams (GPT-J 6B)
struct gptj_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct gptj_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // attention
    struct ggml_tensor * c_attn_q_proj_w;
    struct ggml_tensor * c_attn_k_proj_w;
    struct ggml_tensor * c_attn_v_proj_w;

    struct ggml_tensor * c_attn_proj_w;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gptj_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] addr;
        addr = new uint8_t[size];
        this->size = size;
    }

    ~gptj_buffer() {
        delete[] addr;
    }
};

struct gptj_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = NULL;

    gptj_buffer buf;

    int n; // number of tokens currently in the cache

    ~gptj_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gptj_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head
    struct ggml_tensor * lmh_b; // language model bias

    std::vector<gptj_layer> layers;

    // key + value memory
    struct gptj_kv_cache kv_self;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;

    gptj_buffer buf;

    ~gptj_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};


bool gptj_model_load(const std::string &fname, std::istream &fin, gptj_model & model, gpt_vocab & vocab);
bool gptj_model_load(const std::string & fname, gptj_model & model, gpt_vocab & vocab);
bool gptj_eval(gptj_model& model, const int n_threads, const int n_past, const std::vector<gpt_vocab::id>& embd_inp, std::vector<float>& embd_w, size_t& mem_per_token);
size_t gptj_get_state_size(const gptj_model &model);
size_t gptj_copy_state_data(const gptj_model &model, const std::mt19937 &rng, uint8_t *dest);
size_t gptj_set_state_data(gptj_model *model, std::mt19937 *rng, const uint8_t *src);
#endif // GPTJ_HPP
