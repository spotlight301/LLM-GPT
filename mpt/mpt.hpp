#ifndef MPT_H
#define MPT_H
#include "../g4a_common.hpp"

#include <string>
#include <vector>
#include <map>
#include <random>
#include <ggml.h>


// default hparams (MPT 7B)
struct mpt_hparams {
    int32_t n_vocab      = 50432;
    int32_t n_ctx        = 2048;
    int32_t n_embd       = 4096;
    int32_t n_head       = 32;
    int32_t n_layer      = 32;
    float alibi_bias_max = 8;
    float clip_qkv       = 0;
    int32_t expand       = 4;
    int32_t f16          = 1;
};

struct mpt_layer {
    // normalization
    struct ggml_tensor * norm_1_w;
    struct ggml_tensor * norm_2_w;

    // attention
    struct ggml_tensor * attn_Wqkv_w;
    struct ggml_tensor * attn_out_proj_w;

    // ff
    struct ggml_tensor * ffn_up_proj_w;
    struct ggml_tensor * ffn_down_proj_w;
};

struct mpt_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] addr;
        addr = new uint8_t[size];
        this->size = size;
    }

    ~mpt_buffer() {
        fflush(stdout);
        delete[] addr;
    }
};

struct mpt_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = NULL;

    mpt_buffer buf;

    int n; // number of tokens currently in the cache

    ~mpt_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct mpt_model {
    mpt_hparams hparams;

    // normalization
    struct ggml_tensor * norm_f_w;

    struct ggml_tensor * wte; // position embedding

    // mpt does weight tying

    std::vector<mpt_layer> layers;

    struct mpt_kv_cache kv_self;
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;

    mpt_buffer buf;

    ~mpt_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};


bool mpt_model_load(const std::string &fname, std::istream &fin, mpt_model & model, gpt_vocab& vocab);
bool mpt_eval(mpt_model& model, const int n_threads, const int n_past, const std::vector<int>& embd_inp, std::vector<float>& embd_w, size_t& mem_per_token);
size_t mpt_get_state_size(const mpt_model &model);
size_t mpt_copy_state_data(const mpt_model &model, const std::mt19937& rng, uint8_t *dest);
size_t mpt_set_state_data(mpt_model *model, std::mt19937 *rng, const uint8_t *src);
#endif // MPT_H
