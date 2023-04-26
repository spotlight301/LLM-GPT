#include "additions.hpp"
#include "gptj.hpp"

#define MAX_RNG_STATE 0x10000



int llama_get_kv_cache_token_count(gptj_model& model) {
    return model.ctx->
}

// Returns the size of the state
size_t gptj_get_state_size(gptj_model& model, size_t logits_capacity, size_t embedding_size, size_t kv_buf_size) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    constexpr size_t s_rng_size        = sizeof(size_t);
    constexpr size_t s_rng             = 0x10000;
    constexpr size_t s_logits_capacity = sizeof(size_t);
    constexpr size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits              = logits_capacity * sizeof(float);
    constexpr size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding           = embedding_size * sizeof(float);
    constexpr size_t s_kv_size         = sizeof(size_t);
    constexpr size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv                  = kv_buf_size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// Copies the state to the specified destination address
size_t gptj_copy_state_data(gptj_model& model, uint8_t * dest) {
    uint8_t * out = dest;

    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[MAX_RNG_STATE];

        memset(&rng_buf[0], 0, MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        memcpy(out, &rng_size,   sizeof(rng_size));    out += sizeof(rng_size);
        memcpy(out, &rng_buf[0], MAX_RNG_STATE); out += MAX_RNG_STATE;
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        memcpy(out, &logits_cap,  sizeof(logits_cap));  out += sizeof(logits_cap);
        memcpy(out, &logits_size, sizeof(logits_size)); out += sizeof(logits_size);

        if (logits_size) {
            memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
        }

        out += logits_cap * sizeof(float);
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        memcpy(out, &embedding_size, sizeof(embedding_size)); out += sizeof(embedding_size);

        if (embedding_size) {
            memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
            out += embedding_size * sizeof(float);
        }
    }

    // copy kv cache
    {
        const size_t kv_size = ctx->model.kv_self.buf.size;
        const int    kv_ntok = llama_get_kv_cache_token_count(ctx);

        memcpy(out, &kv_size, sizeof(kv_size)); out += sizeof(kv_size);
        memcpy(out, &kv_ntok, sizeof(kv_ntok)); out += sizeof(kv_ntok);

        if (kv_size) {
            memcpy(out, ctx->model.kv_self.buf.addr, kv_size); out += kv_size;
        }
    }

    const size_t written  = out - dest;
    const size_t expected = gptj_get_state_size(model);

    LLAMA_ASSERT(written == expected);

    return written;
}

// Sets the state reading from the specified source address
size_t gptj_set_state_data(gptj_model& model, const uint8_t * src) {
    const uint8_t * in = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[MAX_RNG_STATE];

        memcpy(&rng_size,   in, sizeof(rng_size));    in += sizeof(rng_size);
        memcpy(&rng_buf[0], in, MAX_RNG_STATE); in += MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        LLAMA_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  in, sizeof(logits_cap));  in += sizeof(logits_cap);
        memcpy(&logits_size, in, sizeof(logits_size)); in += sizeof(logits_size);

        LLAMA_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), in, logits_size * sizeof(float));
        }

        in += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, in, sizeof(embedding_size)); in += sizeof(embedding_size);

        LLAMA_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), in, embedding_size * sizeof(float));
            in += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, in, sizeof(kv_size)); in += sizeof(kv_size);
        memcpy(&kv_ntok, in, sizeof(kv_ntok)); in += sizeof(kv_ntok);

        if (kv_size) {
            LLAMA_ASSERT(ctx->model.kv_self.buf.size == kv_size);

            void * k_data = ctx->model.kv_self.k->data; // remember data pointers
            void * v_data = ctx->model.kv_self.v->data; // because their value is stored in buf and overwritten by memcpy

            memcpy(ctx->model.kv_self.buf.addr, in, kv_size); in += kv_size;

            ctx->model.kv_self.k->data = k_data; // restore correct data pointers
            ctx->model.kv_self.v->data = v_data;

        }

        ctx->model.kv_self.n = kv_ntok;
    }

    const size_t nread    = in - src;
    const size_t expected = gptj_get_state_size(model);

    LLAMA_ASSERT(nread == expected);

    return nread;
}
