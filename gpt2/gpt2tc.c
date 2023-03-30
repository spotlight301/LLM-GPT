/*
 * Text Completion with GPT-2 Transformer
 * 
 * Copyright (c) 2019-2021 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>
#include <stdarg.h>
#include <sys/time.h>
#include <ctype.h>
#include <pthread.h>

#include "cutils.h"
#include "arith.h"
#include "libnc.h"
#include "cp_utils.h"
#include "list.h"
#include "gpt2tc.h"


/************************************************/
/* Transformer model */

static int nb_threads = 1;

/* [seg_len, d_model] -> 
   [n_head, seg_len, d_model/n_head] */
static NCTensor *split_head(NCTensor *x, int n_head)
{
    const size_t *dims;
    int n_dims, axis[3];
    
    dims = nc_tensor_get_dims(x, &n_dims);
    assert(n_dims == 2);
    assert((dims[0] % n_head) == 0);
    x = nc_reshape_3d(x, dims[0] / n_head, n_head, dims[1]);
    /* [seg_len, n_head, d_model/n_head] */
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 1;
    return nc_permute(x, 3, axis);
}

/* [n_head, seg_len, d_value]
   -> [seg_len, d_value * n_head] */
static NCTensor *concat_head(NCTensor *x)
{
    const size_t *dims;
    int n_dims, axis[3];
    
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 1;
    x = nc_permute(x, 3, axis);
    dims = nc_tensor_get_dims(x, &n_dims);
    assert(n_dims == 3);
    /* [seg_len, n_head, d_value] */
    return nc_reshape_2d(x, dims[0] * dims[1], dims[2]);
}

#define MAT_STRIDE 64

/* convert the matrix to strided representation */
static void convert_mat(NCTensor **pw)
{
    NCTensor *w;
    int m, n, n_dims, r;
    const size_t *dims;
    int axis[3];
    
    w = *pw;
    dims = nc_tensor_get_dims(w, &n_dims);
    assert(n_dims == 2);
    m = dims[0];
    n = dims[1];
    r = (-m) % MAT_STRIDE;
    if (r < 0)
        r += MAT_STRIDE;
    w = nc_pad(w, 0, NC_PAD_ZERO, r, NC_PAD_ZERO);
    w = nc_reshape_3d(w, MAT_STRIDE, (m + MAT_STRIDE - 1) / MAT_STRIDE, n);
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 1;
    w = nc_permute(w, 3, axis);
    *pw = w;
}

static TransformerModel *trf_init(const TransformerModelParams *p,
                                  const char *coefs_filename)
{
    TransformerModel *s;
    NCContext *m;
    NCDevice *d;
    int layer_idx;
    TransformerLayer *layers, *tl;
    
    s = nc_mallocz(sizeof(*s));
    rnd_init(&s->rnd_state, p->seed);
    s->n_layer = p->n_layer;
    s->d_model = p->d_model;
    s->n_head = p->n_head;
    s->d_key = p->d_key;
    s->d_value = p->d_value;
    s->d_inner = p->d_inner;
    s->n_ctx = p->n_ctx;
    s->n_symbols = p->n_symbols;
    
    m = nc_context_init(nb_threads);
    s->model = m;
    d = nc_new_cpu_device(m);
    s->device = d;
    
    nc_param_list_init(&s->param_list);
    /* disable graph for the parameters */
    nc_param_list_set_graph(&s->param_list, FALSE);
    
    layers = nc_mallocz(sizeof(layers[0]) * s->n_layer);
    s->layers = layers;
    for(layer_idx = 0; layer_idx < s->n_layer; layer_idx++) {
        tl = &layers[layer_idx];
        tl->ln_1_g = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->ln_1_g, "h%d/ln_1/g", layer_idx);

        tl->ln_1_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->ln_1_b, "h%d/ln_1/b", layer_idx);
        
        tl->attn_w = nc_new_tensor_2d(d, NC_TYPE_F16, s->n_head * s->d_key * 3,
                                      s->d_model);
        nc_new_param(&s->param_list, &tl->attn_w,
                     "h%d/attn/c_attn/w", layer_idx);

        tl->attn_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->n_head * s->d_key * 3);
        nc_new_param(&s->param_list, &tl->attn_b,
                     "h%d/attn/c_attn/b", layer_idx);
        
        tl->attn_proj_w = nc_new_tensor_2d(d, NC_TYPE_F16, s->d_model,
                                           s->n_head * s->d_value);
        nc_new_param(&s->param_list, &tl->attn_proj_w,
                     "h%d/attn/c_proj/w", layer_idx);

        tl->attn_proj_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->attn_proj_b,
                     "h%d/attn/c_proj/b", layer_idx);
        
        tl->ln_2_g = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->ln_2_g, "h%d/ln_2/g", layer_idx);
        
        tl->ln_2_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->ln_2_b, "h%d/ln_2/b", layer_idx);
        
        tl->mlp_fc_w = nc_new_tensor_2d(d, NC_TYPE_F16, s->d_inner,
                                        s->d_model);
        nc_new_param(&s->param_list, &tl->mlp_fc_w,
                     "h%d/mlp/c_fc/w", layer_idx);

        tl->mlp_fc_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_inner);
        nc_new_param(&s->param_list, &tl->mlp_fc_b,
                     "h%d/mlp/c_fc/b", layer_idx);
        
        tl->mlp_proj_w = nc_new_tensor_2d(d, NC_TYPE_F16, s->d_model,
                                          s->d_inner);
        nc_new_param(&s->param_list, &tl->mlp_proj_w,
                     "h%d/mlp/c_proj/w", layer_idx);

        tl->mlp_proj_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
        nc_new_param(&s->param_list, &tl->mlp_proj_b,
                     "h%d/mlp/c_proj/b", layer_idx);
    }
    
    s->ln_f_g = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
    nc_new_param(&s->param_list, &s->ln_f_g, "ln_f/g");
    
    s->ln_f_b = nc_new_tensor_1d(d, NC_TYPE_F32, s->d_model);
    nc_new_param(&s->param_list, &s->ln_f_b, "ln_f/b");
    
    s->wte = nc_new_tensor_2d(d, NC_TYPE_F16, s->d_model,
                              s->n_symbols);
    nc_new_param(&s->param_list, &s->wte, "wte");
    
    s->wpe = nc_new_tensor_2d(d, NC_TYPE_F32, s->d_model,
                              s->n_ctx);
    nc_new_param(&s->param_list, &s->wpe, "wpe");
    
    nc_load_coefs(&s->param_list, coefs_filename);

    /* optimize the variable storage */
    s->wte_trans = nc_transpose(nc_dup_tensor(s->wte));

    convert_mat(&s->wte_trans);
    
    for(layer_idx = 0; layer_idx < s->n_layer; layer_idx++) {
        tl = &layers[layer_idx];
        convert_mat(&tl->attn_w);
        convert_mat(&tl->attn_proj_w);
        convert_mat(&tl->mlp_fc_w);
        convert_mat(&tl->mlp_proj_w);
    }
    return s;
}

typedef struct {
    int mem_len;
    NCTensor **mem_k;
    NCTensor **mem_v;
} BatchEntry;

/* dimensions: output[train_len * n_streams][n_symbols],
   input[train_len * n_streams], tab_mem[n_streams], mem_k[n_layer]
   mem_v[n_layer]. */
static NCTensor *trf_eval(TransformerModel *s, int train_len,
                          int n_streams, BatchEntry *tab_mem,
                          NCTensor *input)
{
    NCTensor *layer_input, **tab_tmp, *output, *position;
    TransformerLayer *tl;
    int layer_idx, i, j, *ptr;
    BatchEntry *be;
    
    tab_tmp = nc_mallocz(sizeof(tab_tmp[0]) *
                          max_int(max_int(3, train_len),
                                  max_int(s->n_head, s->n_layer)));

    position = nc_new_tensor_1d(s->device, NC_TYPE_I32,
                                train_len * n_streams);
    ptr = nc_tensor_get_ptr(position, NULL);
    for(i = 0; i < train_len; i++) {
        for(j = 0; j < n_streams; j++) {
            ptr[i * n_streams + j] = tab_mem[j].mem_len + i;
        }
    }
    
    layer_input = nc_get_col(nc_dup_tensor(s->wte), input);
    layer_input = nc_convert(layer_input, NC_TYPE_F32);
    layer_input = nc_add(layer_input, nc_get_col(nc_dup_tensor(s->wpe),
                                                 position));
                         
    for(layer_idx = 0; layer_idx < s->n_layer; layer_idx++) {
        NCTensor *query, *key, *value, *ff_input, *t0, **tab_tmp2;

        tl = &s->layers[layer_idx];

        t0 = nc_add(nc_mul(nc_layer_norm(nc_dup_tensor(layer_input), 1e-5),
                           nc_dup_tensor(tl->ln_1_g)),
                    nc_dup_tensor(tl->ln_1_b));

        t0 = nc_add(nc_matmul_stride(nc_dup_tensor(tl->attn_w), t0),
                    nc_dup_tensor(tl->attn_b));
        tab_tmp2 = nc_mallocz(sizeof(tab_tmp2[0]) * n_streams);

        /* [ train_len * n_streams d_model * 3] ->
           n_streams * [ train_len d_model * 3] */
        nc_hsplit(tab_tmp2, t0, n_streams, NULL);
        for(i = 0; i < n_streams; i++) {
            be = &tab_mem[i];
            
            t0 = tab_tmp2[i];
            nc_vsplit(tab_tmp, t0, 3, NULL);
            query = tab_tmp[0];
            key = tab_tmp[1];
            value = tab_tmp[2];

            /* split query, key and value for each head */
            key = split_head(key, s->n_head);
            query = split_head(query, s->n_head);
            value = split_head(value, s->n_head);

            /* save the key and value to the memory */
            t0 = nc_slice_alias(be->mem_k[layer_idx],
                                1, be->mem_len, be->mem_len + train_len);
            nc_tensor_copy(t0, key);
            nc_free_tensor(t0);
            nc_free_tensor(key);

            t0 = nc_slice_alias(be->mem_v[layer_idx],
                                1, be->mem_len, be->mem_len + train_len);
            nc_tensor_copy(t0, value);
            nc_free_tensor(t0);
            nc_free_tensor(value);
            
            key = nc_slice_alias(be->mem_k[layer_idx], 
                                 1, 0, be->mem_len + train_len);
            value = nc_slice_alias(be->mem_v[layer_idx],
                                   1, 0, be->mem_len + train_len);
            
            /* cross product term */
            t0 = nc_matmul_add(key, query, NULL,
                               TRUE, FALSE);
            t0 = nc_mul(t0, nc_new_f32(s->device, 1.0f / sqrtf(s->d_key)));

            /* set the future cross products to -infinity so that they
               don't change the softmax result */
            t0 = nc_slt_mat_set(t0, be->mem_len + 1, -INFINITY);
        
            t0 = nc_soft_max(t0);
            t0 = nc_matmul(value, t0);

            /* merge all the heads */
            tab_tmp2[i] = concat_head(t0);
        }

        t0 = nc_hconcat(tab_tmp2, n_streams);
        nc_free(tab_tmp2);
        
        /* projection */
        t0 = nc_add(nc_matmul_stride(nc_dup_tensor(tl->attn_proj_w), t0),
                    nc_dup_tensor(tl->attn_proj_b));
        
        t0 = nc_add(t0, layer_input);

        ff_input = nc_dup_tensor(t0);

        t0 = nc_add(nc_mul(nc_layer_norm(t0, 1e-5),
                           nc_dup_tensor(tl->ln_2_g)),
                    nc_dup_tensor(tl->ln_2_b));
        
        t0 = nc_add(nc_matmul_stride(nc_dup_tensor(tl->mlp_fc_w), t0),
                    nc_dup_tensor(tl->mlp_fc_b));
        t0 = nc_gelu(t0);
            
        t0 = nc_add(nc_matmul_stride(nc_dup_tensor(tl->mlp_proj_w), t0),
                    nc_dup_tensor(tl->mlp_proj_b));
        
        layer_input = nc_add(t0, ff_input);
    }
    
    {
        NCTensor *t0;
        t0 = nc_add(nc_mul(nc_layer_norm(layer_input, 1e-5),
                           nc_dup_tensor(s->ln_f_g)),
                    nc_dup_tensor(s->ln_f_b));

        t0 = nc_matmul_stride(nc_dup_tensor(s->wte_trans), t0);
        /* need to resize the output to the exact size because the
           strided matrix is larger */
        output = nc_resize(t0, s->n_symbols);
    }
    nc_free(tab_tmp);
    return output;
}

static void trf_end(TransformerModel *s)
{
    nc_free_tensor(s->wte_trans);

    nc_param_list_end(&s->param_list);
    nc_free(s->layers);
    nc_context_end(s->model);
    nc_free(s);
}

static const char *gpt2_model_name[] = { "117M", "345M", "774M", "1558M" };

GPT2ModelEnum parse_model(const char *str)
{
    int i;
    for(i = 0; i < countof(gpt2_model_name); i++) {
        if (!strcmp(gpt2_model_name[i], str))
            return i;
    }
    return (GPT2ModelEnum)-1;
}

void trf_set_params(TransformerModelParams *p, GPT2ModelEnum model)
{
    memset(p, 0, sizeof(*p));
    p->seed = 123;
    switch(model) {
    case GPT2_MODEL_117M:
        p->n_layer = 12;
        p->d_model = 768;
        break;
    case GPT2_MODEL_345M:
        p->n_layer = 24;
        p->d_model = 1024;
        break;
    case GPT2_MODEL_774M:
        p->n_layer = 36;
        p->d_model = 1280;
        break;
    case GPT2_MODEL_1558M:
        p->n_layer = 48;
        p->d_model = 1600;
        break;
    default:
        abort();
    }
    p->d_key = 64;
    p->n_head = p->d_model / p->d_key;
    p->d_value = p->d_key;
    p->d_inner = p->d_model * 4;
    p->n_ctx = 1024;
    p->n_symbols = 50257;
}

typedef uint16_t DataSymbol;

/****************************************************************/
/* preprocessor */

static uint32_t hash_calc(const uint8_t *buf, int len, int n_bits)
{
    uint32_t h;
    int i;

    h = 1;
    for(i = 0; i < len; i++) {
        h = h * 263 + buf[i];
    }
    return h & ((1 << n_bits) - 1);
}

static void hash_resize(WordList *s, int hash_bits)
{
    int i, h;
    Word *p;
    
    s->hash_bits = hash_bits;
    s->hash_size = 1 << hash_bits;
    free(s->hash_table);
    s->hash_table = malloc(sizeof(s->hash_table[0]) * s->hash_size);
    for(i = 0; i < s->hash_size; i++)
        s->hash_table[i] = -1;
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        h = hash_calc(p->buf, p->len, s->hash_bits);
        p->next = s->hash_table[h];
        s->hash_table[h] = i;
    }
}

static WordList *word_list_init(void)
{
    WordList *s;
    
    s = malloc(sizeof(WordList));
    memset(s, 0, sizeof(*s));
    s->word_count = 0;
    s->word_size = 0;
    hash_resize(s, 12);
    return s;
}

static void word_list_end(WordList *s)
{
    int i;
    Word *p;
    
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        free(p->buf);
    }
    free(s->words);
    free(s->hash_table);
    free(s);
}

static int64_t hash_lookup_count;
static int64_t hash_it_count;

/* the hash size contains HASH_SIZE_FACTOR times more entries */
#define HASH_SIZE_FACTOR 2

static Word *word_find_add(WordList *s, const uint8_t *buf, int len, int add)
{
    uint32_t h, idx;
    Word *p;

    h = hash_calc(buf, len, s->hash_bits);
    idx = s->hash_table[h];
    hash_lookup_count++;
    while (idx != -1) {
        hash_it_count++;
        p = &s->words[idx];
        if (p->len == len && !memcmp(p->buf, buf, len))
            return p;
        idx = p->next;
    }

    if (!add)
        return NULL;

    if (s->word_count >= s->word_size) {
        size_t new_size = s->word_size + s->word_size / 2;
        if (new_size < 32)
            new_size = 32;
        if (s->word_count + 1 > new_size)
            new_size = s->word_count + 1;
        s->words = realloc(s->words, new_size * sizeof(s->words[0]));
        s->word_size = new_size;

    }
    /* resize the hash table when needed */
    if ((s->word_count * HASH_SIZE_FACTOR) > s->hash_size) {
        int hash_bits = s->hash_bits;
        while ((s->word_count * HASH_SIZE_FACTOR) > (1 << hash_bits))
            hash_bits++;
        hash_resize(s, hash_bits);
        
        /* recompute the hash with the new hash table size */
        h = hash_calc(buf, len, s->hash_bits);
    }

    idx = s->word_count++;
    p = &s->words[idx];
    p->len = len;
    p->buf = malloc(len + 1);
    memcpy(p->buf, buf, len);
    p->buf[len] = 0;
    p->next = s->hash_table[h];
    s->hash_table[h] = idx;
    return p;
}

static void word_load(WordList *s, const char *filename)
{
    FILE *f;
    uint8_t buf[1024];
    int len, c;
    
    f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(1);
    }
    len = 0;
    for(;;) {
        c = fgetc(f);
        if (c < 0)
            break;
        if (c == '\n') {
            if (len > 0) {
                word_find_add(s, buf, len, TRUE);
            }
            len = 0;
        } else {
            if (c == '\\') {
                c = fgetc(f);
                if (c < 0)
                    break;
                if (c == 'n') {
                    c = '\n';
                } else if (c != '\\') {
                    fprintf(stderr, "Invalid escape\n");
                    exit(1);
                }
            }
            if (len >= sizeof(buf)) {
                fprintf(stderr, "Word too long\n");
                exit(1);
            }
            buf[len++] = c;
        }
    }
    fclose(f);
}

typedef enum {
    CAT_SPACE,
    CAT_LETTER,
    CAT_NUMBER,
    CAT_OTHER,
} CharCatEnum;

static int get_char_cat(int c)
{
    if (c == ' ') {
        return CAT_SPACE;
    } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
               c >= 128) {
        return CAT_LETTER;
    } else if (c >= '0' && c <= '9') {
        return CAT_NUMBER;
    } else {
        return CAT_OTHER;
    }
}

static BOOL match(size_t *pmatch_len,
                  const uint8_t *buf, size_t buf_len, const char *str)
{
    size_t len;
    len = strlen(str);
    if (len <= buf_len && !memcmp(buf, str, len)) {
        *pmatch_len = len;
        return TRUE;
    } else {
        *pmatch_len = 0;
        return FALSE;
    }
}

static size_t gpt2_get_word(const uint8_t *buf, size_t buf_len)
{
    size_t len, p;
    int cat;
    
    if (buf_len == 0)
        return 0;
    if (buf[0] == '\'' &&
        (match(&len, buf, buf_len, "'s") ||
         match(&len, buf, buf_len, "'t") ||
         match(&len, buf, buf_len, "'re") ||
         match(&len, buf, buf_len, "'ve") ||
         match(&len, buf, buf_len, "'m") ||
         match(&len, buf, buf_len, "'ll") ||
         match(&len, buf, buf_len, "'d"))) {
        return len;
    }
    p = 0;
    if (buf[0] == ' ' && buf_len >= 2)
        p++;
    if (buf[p] != ' ') {
        cat = get_char_cat(buf[p]);
        len = 1 + p;
        while (len < buf_len && get_char_cat(buf[len]) == cat)
            len++;
        return len;
    } else {
        return 1;
    }
}

static __unused void print_word(const uint8_t *buf, size_t len)
{
    size_t i;
    int c;
    for(i = 0; i < len; i++) {
        c = buf[i];
        if (c >= ' ' && c <= '~')
            putchar(c);
        else
            printf("\\x%02x", c);
    }
}

void gpt2_pp_encode(const char *word_filename,
                    const char *in_filename, const char *out_filename)
{
    FILE *f, *fo;
    size_t buf_size, buf_pos, word_len, len, i;
    uint8_t *buf;
    WordList *s;
    Word *p;
    
    f = fopen(in_filename, "rb");
    if (!f) {
        perror(in_filename);
        exit(1);
    }
    
    fseek(f, 0, SEEK_END);
    buf_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc(buf_size * sizeof(buf[0]));
    fread(buf, 1, buf_size, f);
    fclose(f);

    s = word_list_init();
    word_load(s, word_filename);
    
    fo = fopen(out_filename, "wb");
    if (!fo) {
        perror(out_filename);
        exit(1);
    }

    for(buf_pos = 0; buf_pos < buf_size; buf_pos += word_len) {
        word_len = gpt2_get_word(buf + buf_pos, buf_size - buf_pos);
#if 0
        print_word(buf + buf_pos, word_len);
        printf("\n");
#endif
        /* find the longest word(s) */
        for(i = 0; i < word_len; i += len) {
            for(len = word_len - i; len >= 1; len--) {
                p = word_find_add(s, buf + buf_pos + i, len, FALSE);
                if (p)
                    break;
            }
            assert(len >= 1);
            fput_be16(fo, p - s->words);
        }
    }

    free(buf);
    
    fclose(fo);

    word_list_end(s);
}

#define SYMB_EOT 50256

static void add_char(DataSymbol **pbuf,
                     size_t *psize, size_t *plen, DataSymbol c)
{
    size_t len = *plen, size = *psize;
    if ((len + 1) > size) {
        size = max_size_t(max_size_t(len + 1, 4),
                          size * 3 / 2);
        *pbuf = realloc(*pbuf, sizeof(**pbuf) * size);
        *psize = size;
    }
    (*pbuf)[len++] = c;
    *plen = len;
}

static void gpt2_pp_encode_buf1(WordList *s, 
                                DataSymbol **pout_buf,
                                size_t *pout_buf_size, size_t *pout_buf_len,
                                const uint8_t *buf, size_t buf_size)
{
    size_t buf_pos, word_len, len, i;
    Word *p;
    
    for(buf_pos = 0; buf_pos < buf_size; buf_pos += word_len) {
        word_len = gpt2_get_word(buf + buf_pos, buf_size - buf_pos);
#if 0
        print_word(buf + buf_pos, word_len);
        printf("\n");
#endif
        /* find the longest word(s) */
        for(i = 0; i < word_len; i += len) {
            for(len = word_len - i; len >= 1; len--) {
                p = word_find_add(s, buf + buf_pos + i, len, FALSE);
                if (p)
                    break;
            }
            assert(len >= 1);
            add_char(pout_buf, pout_buf_size, pout_buf_len, p - s->words);
        }
    }
}

size_t gpt2_pp_encode_buf(WordList *s, DataSymbol **pout_buf,
                          const uint8_t *buf, size_t buf_size)
{
    size_t out_buf_len, out_buf_size;
    DataSymbol *out_buf;

    out_buf_len = 0;
    out_buf_size = 0;
    out_buf = NULL;
    gpt2_pp_encode_buf1(s, &out_buf, &out_buf_size, &out_buf_len,
                        buf, buf_size);
    *pout_buf = out_buf;
    return out_buf_len;
}

void gpt2_pp_decode(const char *word_filename,
                    const char *in_filename, const char *out_filename)
{
    WordList *s;
    FILE *f, *fo;
    uint16_t c;
    Word *p;
    
    s = word_list_init();
    word_load(s, word_filename);

    f = fopen(in_filename, "rb");
    if (!f) {
        perror(in_filename);
        exit(1);
    }
    
    fo = fopen(out_filename, "wb");
    if (!fo) {
        perror(out_filename);
        exit(1);
    }

    for(;;) {
        if (fget_be16(f, &c))
            break;
        if (c >= s->word_count) {
            fprintf(stderr, "Invalid symbol: %d\n", c);
            exit(1);
        }
        p = &s->words[c];
        fwrite(p->buf, 1, p->len, fo);
    }

    fclose(fo);

    fclose(f);
    
    word_list_end(s);
}

static struct option options[] = {
    { NULL },
};

/****************************************************************/
/* text completion */

static int get_random_symb_topk(float *prob, size_t n_symb, int topk,
                                float topp, RNDState *rnd_state)
{
    NCTopKEntry *tab;
    int i, c, k;
    float p;
    double sum;
    
    assert(n_symb >= 1);

    prof_start(PROF_WRITE_SYM);
    k = nc_topk(&tab, &sum, prob, n_symb, topk, topp);
    prof_end(PROF_WRITE_SYM);
    
    p = rnd_unif(rnd_state) * sum;
    
    sum = 0;
    for(i = 0; i < k - 1; i++) {
        sum += prob[tab[i].idx];
        if (p < sum)
            break;
    }
    c = tab[i].idx;
    nc_free(tab);
    return c;
}

static void dump_pred_symb(float *prob, size_t n_symb, int k,
                           WordList *wl)
{
#if 0
    int *tab, i, c;
    Word *wp;

    assert(n_symb >= 1);
    tab = malloc(sizeof(tab[0]) * n_symb);
    for(i = 0; i < n_symb; i++)
        tab[i] = i;
    topk_sort(tab, n_symb, prob);

    k = min_int(n_symb, k);
    for(i = 0; i < k; i++) {
        c = tab[i];
        printf("%d: %10.3g '", i, prob[c]);
        wp = &wl->words[c];
        fwrite(wp->buf, 1, wp->len, stdout);
        printf("'\n");
    }
    free(tab);
#endif
}

char *trim_text(const char *str)
{
    size_t len;
    char *new_str;
    while (*str == ' ')
        str++;
    len = strlen(str);
    while (len > 0 && str[len - 1] == ' ')
        len--;
    new_str = malloc(len + 1);
    memcpy(new_str, str, len + 1);
    return new_str;
}

TextCompleteGlobalState *text_complete_global_init(GPT2ModelEnum model,
                                                   const char *filename)
{
    WordList *wl;
    TransformerModelParams p_s, *p = &p_s;
    TransformerModel *s;
    TextCompleteGlobalState *tcs;
    char coefs_filename[128];
    
    tcs = nc_mallocz(sizeof(*tcs));
    
    trf_set_params(p, model);
    if (!filename) {
        snprintf(coefs_filename, sizeof(coefs_filename),
                 "gpt2_%s.bin", gpt2_model_name[model]);
        filename = coefs_filename;
    }
    s = trf_init(p, filename);
    
    wl = word_list_init();
    word_load(wl, "gpt2vocab.txt");
    tcs->wl = wl;
    tcs->trf_state = s;
    return tcs;
}

void text_complete_global_end(TextCompleteGlobalState *tcs)
{
    trf_end(tcs->trf_state);
    word_list_end(tcs->wl);
    nc_free(tcs);
}

TextGenContext *text_complete_start(TextCompleteGlobalState *tcs,
                                    const char *input_text,
                                    int top_k, float top_p, float temperature,
                                    int seed, int max_output_len)
{
    TransformerModel *s = tcs->trf_state;
    WordList *wl = tcs->wl;
    TextGenContext *ts;
    int i, mem_len;
    
    ts = nc_mallocz(sizeof(*ts));
    ts->global_state = tcs;
    ts->top_k = top_k;
    ts->top_p = top_p;
    ts->temperature = temperature;
    rnd_init(&ts->rnd_state, seed);
    ts->max_output_len = max_output_len;
    ts->input_buf_len = gpt2_pp_encode_buf(wl, &ts->input_buf,
                                           (const uint8_t *)input_text,
                                           strlen(input_text));
    if (ts->input_buf_len > MAX_INITIAL_TEXT_LEN) {
        memmove(ts->input_buf, ts->input_buf + ts->input_buf_len - MAX_INITIAL_TEXT_LEN, MAX_INITIAL_TEXT_LEN * sizeof(ts->input_buf[0]));
        ts->input_buf_len = MAX_INITIAL_TEXT_LEN;
        ts->input_buf = realloc(ts->input_buf,
                                ts->input_buf_len * sizeof(ts->input_buf[0]));
    }

#if 0
    for(i = 0; i < ts->input_buf_len; i++) {
        printf(" %04x", ts->input_buf[i]);
    }
    printf("\n");
#endif

    ts->mem_k = nc_mallocz(sizeof(ts->mem_k[0]) * s->n_layer);
    ts->mem_v = nc_mallocz(sizeof(ts->mem_v[0]) * s->n_layer);
    mem_len = ts->input_buf_len + max_output_len;
    for(i = 0; i < s->n_layer; i++) {
        ts->mem_k[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                        s->d_key, mem_len, s->n_head);
        nc_tensor_set_name(ts->mem_k[i], "mem_k_%d", i);
        ts->mem_v[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                        s->d_value, mem_len, s->n_head);
        nc_tensor_set_name(ts->mem_v[i], "mem_v_%d", i);
    }
    ts->text_len = ts->input_buf_len;
    ts->is_first = TRUE;
    return ts;
}

static void text_complete_symb(TextCompleteGlobalState *tcs,
                               TextGenContext *ts, NCTensor *logits)
{
    TransformerModel *s = tcs->trf_state;
    WordList *wl = tcs->wl;
    Word *wp;
    NCTensorData xbuf, *x;
    int c, out_len;
    NCTensor *t0;

    t0 = logits;
    if (ts->temperature != 1.0)
        t0 = nc_mul(t0, nc_new_f32(s->device, 1.0f / ts->temperature));
    t0 = nc_soft_max(t0);
    x = nc_tensor_get_data(&xbuf, t0);
    
    if (0) {
        printf("\n");
        dump_pred_symb((float *)x->data, s->n_symbols, 10, wl);
    }
    c = get_random_symb_topk((float *)x->data,
                             s->n_symbols, ts->top_k, ts->top_p,
                             &ts->rnd_state);
    if (c == SYMB_EOT) {
        ts->out_text_len = 0;
        ts->out_text[0] = '\0';
    } else {
        wp = &wl->words[c];
        out_len = min_int(sizeof(ts->out_text) - 1, wp->len);
        memcpy(ts->out_text, wp->buf, out_len);
        ts->out_text[out_len] = '\0';
        ts->out_text_len = out_len;
    }
    ts->last_c = c;

    nc_free_tensor(t0);
}

/* Note: ts_list is emptied */
void text_complete_next(TextCompleteGlobalState *tcs,
                        struct list_head *ts_list)
{
    TransformerModel *s = tcs->trf_state;
    int i, k;
    NCTensor *output, *input;
    int32_t *ptr;
    struct list_head *el, *el1;
    TextGenContext *ts, **ts_tab;
    int batch_size;
    BatchEntry tab_mem[BATCH_SIZE_MAX];

    list_for_each_safe(el, el1, ts_list) {
        ts = list_entry(el, TextGenContext, link);
        if (ts->text_len >= s->n_ctx ||
            (ts->text_len - ts->input_buf_len) >= ts->max_output_len) {
            ts->out_text_len = 0;
            ts->out_text[0] = '\0';
            list_del(&ts->link);
        } else if (ts->is_first) {
            input = nc_new_tensor_1d(s->device, NC_TYPE_I32, ts->text_len);
            ptr = nc_tensor_get_ptr(input, NULL);
            for(i = 0; i < ts->text_len; i++) {
                ptr[i] = ts->input_buf[i];
            }
            
            prof_start(PROF_EVAL);
            tab_mem[0].mem_len = 0;
            tab_mem[0].mem_k = ts->mem_k;
            tab_mem[0].mem_v = ts->mem_v;
            output = trf_eval(s, ts->text_len, 1, tab_mem, input);
            prof_end(PROF_EVAL);
            
            text_complete_symb(tcs, ts, nc_slice_alias(output, 1, ts->text_len - 1, ts->text_len));
            nc_free_tensor(output);
            
            ts->text_len++;
            ts->is_first = FALSE;
            list_del(&ts->link);
        }
    }

    ts_tab = nc_mallocz(sizeof(ts_tab[0]) * BATCH_SIZE_MAX);
    for(;;) {
        k = 0;
        list_for_each_safe(el, el1, ts_list) {
            ts = list_entry(el, TextGenContext, link);
            ts_tab[k++] = ts;
            list_del(&ts->link);
            if (k >= BATCH_SIZE_MAX)
                break;
        }
        if (k == 0)
            break;
        batch_size = k;
        //        printf("batch_size=%d\n", k);
        
        for(k = 0; k < batch_size; k++) {
            ts = ts_tab[k];
            tab_mem[k].mem_len = ts->text_len - 1;
            tab_mem[k].mem_k = ts->mem_k;
            tab_mem[k].mem_v = ts->mem_v;
        }
        
        /* compute the next probabilities */
        input = nc_new_tensor_1d(s->device, NC_TYPE_I32, batch_size);
        ptr = nc_tensor_get_ptr(input, NULL);
        for(k = 0; k < batch_size; k++) {
            ts = ts_tab[k];
            ptr[k] = ts->last_c;
        }
        
        prof_start(PROF_EVAL);
        output = trf_eval(s, 1, batch_size, tab_mem, input);
        prof_end(PROF_EVAL);
        
        for(k = 0; k < batch_size; k++) {
            ts = ts_tab[k];
            text_complete_symb(tcs, ts,
                               nc_slice_alias(output, 1, k, k + 1));
            
            ts->text_len++;
            ts->is_first = FALSE;
        }
        nc_free_tensor(output);
    }
    nc_free(ts_tab);
}

void text_complete_end(TextGenContext *ts)
{
    TransformerModel *s = ts->global_state->trf_state;
    int i;
    
    for(i = 0; i < s->n_layer; i++) {
        nc_free_tensor(ts->mem_k[i]);
        nc_free_tensor(ts->mem_v[i]);
    }
    nc_free(ts->mem_k);
    nc_free(ts->mem_v);

    free(ts->input_buf);
    nc_free(ts);
}

void text_complete(GPT2ModelEnum model, const char *model_filename,
                   const char *input_text,
                   int top_k, float top_p, float temperature,
                   int max_output_len, int batch_size, int seed,
                   BOOL verbose)
{
    TextCompleteGlobalState *tcs;
    TextGenContext *ts;
    int count;
    struct timeval tv;
    const char *input_text1;
    struct list_head ts_list;
    int64_t ti;
    
    tcs = text_complete_global_init(model, model_filename);
    
    if (seed == 0) {
        gettimeofday(&tv, NULL);
        seed = tv.tv_sec + tv.tv_usec;
    }

    input_text1 = trim_text(input_text);
    if (input_text1[0] == '\0')
        input_text1 = strdup(" ");
    printf("%s", input_text1);
    fflush(stdout);
    prof_start(PROF_TOTAL);
    if (batch_size == 0) {
        ts = text_complete_start(tcs, input_text1, top_k, top_p, temperature,
                                 seed, max_output_len);
        
        ti = get_time_ms();
        count = 0;
        for(;;) {
            init_list_head(&ts_list);
            list_add_tail(&ts->link, &ts_list);
            text_complete_next(tcs, &ts_list);
            if (ts->out_text_len == 0)
                break;
            fwrite(ts->out_text, 1, ts->out_text_len, stdout);
            fflush(stdout);
            count++;
        }
        printf("\n");
        text_complete_end(ts);
    } else {
        TextGenContext **ts_tab;
        int i;

        /* test for batch processing (the same text is generated by
           each job) */
        
        ts_tab = nc_mallocz(sizeof(ts_tab[0]) * batch_size);
        
        for(i = 0; i < batch_size; i++) {
            ts = text_complete_start(tcs, input_text1, top_k, top_p,
                                     temperature, seed, max_output_len);
            ts_tab[i] = ts;
        }
    
        ti = get_time_ms();
        count = 0;
        for(;;) {
            init_list_head(&ts_list);
            for(i = 0; i < batch_size; i++) {
                ts = ts_tab[i];
                if (ts->is_first || ts->out_text_len > 0) {
                    list_add_tail(&ts->link, &ts_list);
                }
            }
            if (list_empty(&ts_list))
                break;
            text_complete_next(tcs, &ts_list);

            for(i = 0; i < batch_size; i++) {
                ts = ts_tab[i];
                if (ts->out_text_len > 0 && i == 0) {
                    fwrite(ts->out_text, 1, ts->out_text_len, stdout);
                    fflush(stdout);
                }
            }
            count++;
        }
        printf("\n");
        
        for(i = 0; i < batch_size; i++) {
            ts = ts_tab[i];
            text_complete_end(ts);
        }
        nc_free(ts_tab);
    }
    ti = get_time_ms() - ti;
    if (verbose) {
        printf("time=%0.1f word/s\n",
               (double)count / ti * 1000);
    }
    prof_end(PROF_TOTAL);
    text_complete_global_end(tcs);

    nc_prof_dump();
}

/******************************************************************/
/* short text compression */

/* Note: at most 31 bits are encoded. At most UTF8_CHAR_LEN_MAX bytes
   are output. */
int unicode_to_utf8(uint8_t *buf, unsigned int c)
{
    uint8_t *q = buf;

    if (c < 0x80) {
        *q++ = c;
    } else {
        if (c < 0x800) {
            *q++ = (c >> 6) | 0xc0;
        } else {
            if (c < 0x10000) {
                *q++ = (c >> 12) | 0xe0;
            } else {
                if (c < 0x00200000) {
                    *q++ = (c >> 18) | 0xf0;
                } else {
                    if (c < 0x04000000) {
                        *q++ = (c >> 24) | 0xf8;
                    } else if (c < 0x80000000) {
                        *q++ = (c >> 30) | 0xfc;
                        *q++ = ((c >> 24) & 0x3f) | 0x80;
                    } else {
                        return 0;
                    }
                    *q++ = ((c >> 18) & 0x3f) | 0x80;
                }
                *q++ = ((c >> 12) & 0x3f) | 0x80;
            }
            *q++ = ((c >> 6) & 0x3f) | 0x80;
        }
        *q++ = (c & 0x3f) | 0x80;
    }
    return q - buf;
}

static const unsigned int utf8_min_code[5] = {
    0x80, 0x800, 0x10000, 0x00200000, 0x04000000,
};

static const unsigned char utf8_first_code_mask[5] = {
    0x1f, 0xf, 0x7, 0x3, 0x1,
};

/* return -1 if error. *pp is not updated in this case. max_len must
   be >= 1. The maximum length for a UTF8 byte sequence is 6 bytes. */
int unicode_from_utf8(const uint8_t *p, int max_len, const uint8_t **pp)
{
    int l, c, b, i;

    c = *p++;
    if (c < 0x80) {
        *pp = p;
        return c;
    }
    switch(c) {
    case 0xc0 ... 0xdf:
        l = 1;
        break;
    case 0xe0 ... 0xef:
        l = 2;
        break;
    case 0xf0 ... 0xf7:
        l = 3;
        break;
    case 0xf8 ... 0xfb:
        l = 4;
        break;
    case 0xfc ... 0xfd:
        l = 5;
        break;
    default:
        return -1;
    }
    /* check that we have enough characters */
    if (l > (max_len - 1))
        return -1;
    c &= utf8_first_code_mask[l - 1];
    for(i = 0; i < l; i++) {
        b = *p++;
        if (b < 0x80 || b >= 0xc0)
            return -1;
        c = (c << 6) | (b & 0x3f);
    }
    if (c < utf8_min_code[l - 1])
        return -1;
    *pp = p;
    return c;
}

static inline int simple_get_bit(const uint8_t *data, size_t index)
{
    return (data[index >> 3] >> (7 - (index & 7))) & 1;
}

static inline void simple_put_bit(uint8_t *data, size_t index, int bit)
{
    data[index >> 3] |= bit << (7 - (index & 7));
}

static uint16_t ranges[3][2] = {
    { 0x3400, 0x4DB5 },
    { 0x4e00, 0x9fcf },
    { 0xAC00, 0xD7A3 },
};

static int c15_to_unicode(int c)
{
    int i, n, count;
    for(i = 0; i < countof(ranges); i++) {
        count = ranges[i][1] - ranges[i][0] + 1;
        n = count;
        if (c < n) {
            return ranges[i][0] + c;
        }
        c -= count;
    }
    return -1;
}

static int unicode_to_c15(int c)
{
    int i, b;
    b = 0;
    for(i = 0; i < countof(ranges); i++) {
        if (c >= ranges[i][0] && c <= ranges[i][1])
            return b + c - ranges[i][0];
        b += ranges[i][1] - ranges[i][0] + 1;
    }
    return -1;
}

size_t convert_to_chars(char **pout_buf, uint8_t *buf, size_t n_bits)
{
    size_t idx, out_buf_len;
    int c, i, l, len;
    char buf1[8], *out_buf;
    
    out_buf = malloc(4 * ((n_bits + 14) / 15) + 1);
    out_buf_len = 0;
    for(idx = 0; idx < n_bits; idx += 15) {
        l = min_size_t(15, n_bits - idx);
        c = 0;
        for(i = 0; i < l; i++) {
            c |= simple_get_bit(buf, idx + i) << (14 - i);
        }
        c = c15_to_unicode(c);
        len = unicode_to_utf8((uint8_t *)buf1, c);
        memcpy(out_buf + out_buf_len, buf1, len);
        out_buf_len += len;
    }
    out_buf[out_buf_len] = '\0';
    *pout_buf = out_buf;
    return out_buf_len;
}

/* return -1 if error */
ssize_t convert_from_chars(uint8_t **pout_buf, const char *str)
{
    const char *str_end;
    int c, i;
    uint8_t *out_buf;
    size_t str_len, len;
    
    str_len = strlen(str);
    str_end = str + str_len;
    /* Note: the exact length of out_buf is smaller */
    out_buf = malloc(str_len);
    memset(out_buf, 0, str_len);
    
    len = 0;
    while (*str != '\0') {
        c = unicode_from_utf8((uint8_t *)str, str_end - str, (const uint8_t **)&str);
        if (c < 0)
            goto fail;
        c = unicode_to_c15(c);
        if (c < 0 || c >= 32768)
            goto fail;
        for(i = 0; i < 15; i++) {
            simple_put_bit(out_buf, len * 15 + i, (c >> (14 - i)) & 1);
        }
        len++;
    }
    *pout_buf = out_buf;
    return (len * 15 + 7) / 8;
 fail:
    free(out_buf);
    return -1;
}

#define LENGTH_K 2

int encode_length(PutBitState *pb, uint32_t val)
{
    uint32_t n, a, b, i;
    a = val;
    n = 1;
    for(;;) {
        b = 1 << (LENGTH_K * n);
        if (a < b)
            break;
        n++;
        a -= b;
    }
    for(i = 0; i < n - 1; i++)
        put_bit_raw(pb, 0);
    put_bit_raw(pb, 1);
    for(i = 0; i < (LENGTH_K * n); i++) {
        put_bit_raw(pb, (a >> (LENGTH_K * n - 1 - i)) & 1);
    }
    return n + LENGTH_K * n;
}

int decode_length(GetBitState *gb)
{
    int n, val, a, i;
    n = 1;
    a = 0;
    for(;;) {
        if (get_bit_raw(gb))
            break;
        if (n >= 10) /* arbitrary limit */
            return -1;
        a += 1 << (LENGTH_K * n);
        n++;
    }
    val = 0;
    for(i = 0; i < (LENGTH_K * n); i++) {
        val |= get_bit_raw(gb) << (LENGTH_K * n - 1 - i);
    }
    return val + a;
}

static void realloc_buf(char **pbuf,
                        size_t *psize, size_t len)
{
    size_t size = *psize;
    if (len > size) {
        size = max_size_t(len, size * 3 / 2);
        *pbuf = realloc(*pbuf, sizeof(**pbuf) * size);
        *psize = size;
    }
}


#define CTEXT_LEN_MAX 256

int text_decompress(TextCompleteGlobalState *tcs,
                    char **poutput_text, const char *input_text)
{
    TransformerModel *s = tcs->trf_state;
    WordList *wl = tcs->wl;
    uint8_t *data_buf;
    ssize_t data_buf_len, text_len, mem_len;
    GetBitState gb_s, *gb = &gb_s;
    BatchEntry tab_mem[1];
    NCTensor **mem_k, **mem_v;
    DataSymbol *text_buf;
    NCTensorData xbuf, *x;
    int c, i;
    char *out_str;
    size_t out_str_len, out_str_size;

    *poutput_text = NULL;

    /* XXX: handle zero length ? */
    data_buf_len = convert_from_chars(&data_buf, input_text);
    if (data_buf_len < 0)
        return -1;
    if (data_buf_len == 0) {
        *poutput_text = strdup("");
        free(data_buf);
        return 0;
    }
#if 0
    {
        int i;
        printf("data_buf=");
        for(i = 0; i < data_buf_len; i++)
            printf(" %02x", data_buf[i]);
        printf("\n");
    }
#endif
    get_bit_init(gb, data_buf, data_buf_len, NULL, NULL);

    text_len = decode_length(gb);
    if (text_len < 0 || text_len > CTEXT_LEN_MAX) {
        free(data_buf);
        return -1;
    }
    text_len++;

    text_buf = nc_malloc(sizeof(text_buf[0]) * text_len);

    mem_k = nc_mallocz(sizeof(mem_k[0]) * s->n_layer);
    mem_v = nc_mallocz(sizeof(mem_v[0]) * s->n_layer);
    mem_len = text_len;
    for(i = 0; i < s->n_layer; i++) {
        mem_k[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_key, mem_len, s->n_head);
        nc_tensor_set_name(mem_k[i], "mem_k_%d", i);
        mem_v[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_value, mem_len, s->n_head);
        nc_tensor_set_name(mem_v[i], "mem_v_%d", i);
    }
    tab_mem[0].mem_k = mem_k;
    tab_mem[0].mem_v = mem_v;

    text_buf[0] = SYMB_EOT;

    for(i = 0; i < text_len - 1; i++) {
        NCTensor *t0, *input;
        int32_t *ptr;

        input = nc_new_tensor_1d(s->device, NC_TYPE_I32, 1);
        ptr = nc_tensor_get_ptr(input, NULL);
        ptr[0] = text_buf[i];
        tab_mem[0].mem_len = i;
        t0 = trf_eval(s, 1, 1, tab_mem, input);

        t0 = nc_soft_max(t0);
        x = nc_tensor_get_data(&xbuf, t0);
        c = read_sym(gb, (float *)x->data, x->dims[0]);
        text_buf[i + 1] = c;
        nc_free_tensor(t0);
    }

    /* convert back to a string */
    out_str = NULL;
    out_str_len = 0;
    out_str_size = 0;
    for(i = 1; i < text_len; i++) {
        Word *wp;
        wp = &wl->words[text_buf[i]];
        realloc_buf(&out_str, &out_str_size, out_str_len + wp->len);
        memcpy(out_str + out_str_len, wp->buf, wp->len);
        out_str_len += wp->len;
    }
    realloc_buf(&out_str, &out_str_size, out_str_len + 1);
    out_str[out_str_len] = '\0';

    for(i = 0; i < s->n_layer; i++) {
        nc_free_tensor(mem_k[i]);
        nc_free_tensor(mem_v[i]);
    }
    nc_free(mem_k);
    nc_free(mem_v);
    nc_free(text_buf);
    free(data_buf);

    *poutput_text = out_str;

    return 0;
}

#define TEXT_OUTPUT_BUF_LEN 4096

static void text_arith_write_buf(void *opaque, const uint8_t *buf, size_t buf_size)
{
    /* we assume the output is small enough to fit the buffer */
}

int text_compress(TextCompleteGlobalState *tcs,
                  char **poutput_text,
                  const char *input_text, BOOL dump_stats)
{
    TransformerModel *s = tcs->trf_state;
    DataSymbol *input_buf;
    int i, mem_len;
    NCTensorData xbuf, *x;
    double n_bits;
    BatchEntry tab_mem[1];
    NCTensor **mem_k, **mem_v, *output, *input;
    PutBitState pb_s, *pb = &pb_s;
    size_t input_buf_len, input_buf_size, out_buf_len;
    uint8_t *out_buf;
    char *out_str;
    int32_t *ptr;

    *poutput_text = NULL;

    input_buf = NULL;
    input_buf_size = 0;
    input_buf_len = 0;

    add_char(&input_buf, &input_buf_size, &input_buf_len, SYMB_EOT);
    gpt2_pp_encode_buf1(tcs->wl, &input_buf, &input_buf_size, &input_buf_len,
                        (const uint8_t *)input_text,
                        strlen(input_text));
    if (input_buf_len > CTEXT_LEN_MAX) {
        free(input_buf);
        return -1;
    }
    if (input_buf_len == 1) {
        free(input_buf);
        *poutput_text = strdup("");
        return 0;
    }

#if 0
    for(i = 0; i < input_buf_len; i++) {
        printf(" %04x", input_buf[i]);
    }
    printf("\n");
#endif
    prof_start(PROF_EVAL);
    input = nc_new_tensor_1d(s->device, NC_TYPE_I32, input_buf_len);
    ptr = nc_tensor_get_ptr(input, NULL);
    for(i = 0; i < input_buf_len; i++) {
        ptr[i] = input_buf[i];
    }

    mem_k = nc_mallocz(sizeof(mem_k[0]) * s->n_layer);
    mem_v = nc_mallocz(sizeof(mem_v[0]) * s->n_layer);
    mem_len = input_buf_len;
    for(i = 0; i < s->n_layer; i++) {
        mem_k[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_key, mem_len, s->n_head);
        nc_tensor_set_name(mem_k[i], "mem_k_%d", i);
        mem_v[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_value, mem_len, s->n_head);
        nc_tensor_set_name(mem_v[i], "mem_v_%d", i);
    }
    tab_mem[0].mem_len = 0;
    tab_mem[0].mem_k = mem_k;
    tab_mem[0].mem_v = mem_v;

    output = trf_eval(s, input_buf_len, 1, tab_mem, input);
    prof_end(PROF_EVAL);

    out_buf = malloc(TEXT_OUTPUT_BUF_LEN);
    put_bit_init(pb, out_buf, TEXT_OUTPUT_BUF_LEN, text_arith_write_buf, NULL);

    n_bits = encode_length(pb, input_buf_len - 1);

    for(i = 0; i < input_buf_len - 1; i++) {
        double v;
        NCTensor *t0;
        t0 = nc_soft_max(nc_slice_alias(output, 1, i, i + 1));
        x = nc_tensor_get_data(&xbuf, t0);
        write_sym(pb, (float *)x->data, x->dims[0], input_buf[i + 1]);
        v = -log2(((float *)x->data)[input_buf[i + 1]]);
        //        printf("%d: %0.1f\n", i, v);
        nc_free_tensor(t0);
        n_bits += v;
    }
    nc_free_tensor(output);
    out_buf_len = put_bit_flush(pb);
#if 0
    {
        printf("out_buf=");
        for(i = 0; i < (out_buf_len + 7) / 8; i++)
            printf(" %02x", out_buf[i]);
        printf("\n");
    }
#endif
    convert_to_chars(&out_str, out_buf, out_buf_len);
    if (dump_stats) {
        printf("%d chars, %" PRId64 " symbols, %" PRId64 " bits (ref=%0.1f bits) (%d compressed chars)\n",
               (int)strlen(input_text),
               (int64_t)input_buf_len,
               (int64_t)out_buf_len,
               n_bits,
               (int)((out_buf_len + 14) / 15));
    }

    free(out_buf);
    free(input_buf);
    for(i = 0; i < s->n_layer; i++) {
        nc_free_tensor(mem_k[i]);
        nc_free_tensor(mem_v[i]);
    }
    nc_free(mem_k);
    nc_free(mem_v);
    *poutput_text = out_str;
    return 0;
}

void text_compress_test(GPT2ModelEnum model, const char *model_filename,
                        const char *input_text,
                        BOOL is_decode, BOOL verbose)
{
    TextCompleteGlobalState *tcs;
    char *out_str;

    tcs = text_complete_global_init(model, model_filename);

    if (is_decode) {
        if (text_decompress(tcs, &out_str, input_text) < 0) {
            printf("Error\n");
        } else {
            printf("%s\n", out_str);
        }
        free(out_str);
    } else {
        if (text_compress(tcs, &out_str, input_text, verbose) < 0) {
            printf("Error\n");
        } else {
            printf("%s\n", out_str);
        }
        free(out_str);
    }
    text_complete_global_end(tcs);
}

/*************************************************/
/* file compression */

static uint8_t *load_file(size_t *psize, const char *filename)
{
    FILE *f;
    size_t size;
    uint8_t *buf;

    f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc(size + 1);
    if (fread(buf, 1, size, f) != size) {
        fprintf(stderr, "%s: I/O error\n", filename);
        exit(1);
    }
    buf[size] = '\0';
    fclose(f);
    *psize = size;
    return buf;
}

/* check if CRLF can be converted to LF losslessly */
static BOOL check_lossless_crlf(const uint8_t *buf, size_t len)
{
    size_t i;
    BOOL has_crlf;
    has_crlf = FALSE;
    for(i = 0; i < len - 1;) {
        if (buf[i] == '\r' && buf[i + 1] == '\n') {
            has_crlf = TRUE;
            i += 2;
        } else if (buf[i] == '\n') {
            return FALSE;
        } else {
            i++;
        }
    }
    return has_crlf;
}

static size_t convert_crlf_to_lf(uint8_t *buf, size_t len)
{
    size_t i, j;
    j = 0;
    for(i = 0; i < len - 1;) {
        if (buf[i] == '\r' && buf[i + 1] == '\n')
            i++;
        buf[j++] = buf[i++];
    }
    if (i < len)
        buf[j++] = buf[i++];
    return j;
}

#define ARITH_BUF_LEN 65536

static void arith_write_buf(void *opaque, const uint8_t *buf, size_t buf_size)
{
    FILE *f = opaque;
    fwrite(buf, 1, buf_size, f);
}

/* XXX: should use a large batch size */
int file_compress(TextCompleteGlobalState *tcs,
                  const char *infilename, const char *outfilename)
{
    TransformerModel *s = tcs->trf_state;
    DataSymbol *input_buf;
    int i, mem_len, len;
    NCTensorData xbuf, *x;
    BatchEntry tab_mem[1];
    NCTensor **mem_k, **mem_v, *output, *input;
    PutBitState pb_s, *pb = &pb_s;
    size_t input_buf_len, input_buf_size, input_text_len;
    int64_t n_output_bits;
    size_t input_buf_pos;
    uint8_t *input_text, *arith_buf;
    FILE *f;
    BOOL convert_crlf;
    int32_t *ptr;

    input_text = load_file(&input_text_len, infilename);

    convert_crlf = check_lossless_crlf(input_text, input_text_len);
    //    printf("convert_crlf=%d\n", convert_crlf);

    if (convert_crlf) {
        input_text_len = convert_crlf_to_lf(input_text, input_text_len);
    }

    input_buf = NULL;
    input_buf_size = 0;
    input_buf_len = 0;

    add_char(&input_buf, &input_buf_size, &input_buf_len, SYMB_EOT);
    gpt2_pp_encode_buf1(tcs->wl, &input_buf, &input_buf_size, &input_buf_len,
                        input_text, input_text_len);
    add_char(&input_buf, &input_buf_size, &input_buf_len, SYMB_EOT);

#if 0
    for(i = 0; i < input_buf_len; i++) {
        printf(" %04x", input_buf[i]);
    }
    printf("\n");
#endif
    prof_start(PROF_EVAL);
    mem_k = nc_mallocz(sizeof(mem_k[0]) * s->n_layer);
    mem_v = nc_mallocz(sizeof(mem_v[0]) * s->n_layer);
    for(i = 0; i < s->n_layer; i++) {
        mem_k[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_key, s->n_ctx, s->n_head);
        nc_tensor_set_name(mem_k[i], "mem_k_%d", i);
        mem_v[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_value, s->n_ctx, s->n_head);
        nc_tensor_set_name(mem_v[i], "mem_v_%d", i);
    }

    f = fopen(outfilename, "wb");
    if (!f) {
        perror(outfilename);
        exit(1);
    }

    arith_buf = nc_malloc(ARITH_BUF_LEN);
    put_bit_init(pb, arith_buf, ARITH_BUF_LEN, arith_write_buf, f);

    put_bit_raw(pb, convert_crlf);

    mem_len = 0;
    input_buf_pos = 0;
    while (input_buf_pos < (input_buf_len - 1)) {
        len = min_size_t(input_buf_len - 1 - input_buf_pos, s->n_ctx - mem_len);
        printf("%5.1f%%   \r", (double)input_buf_pos / (double)input_buf_len * 100);
        fflush(stdout);
        //        printf("pos=%d mem_len=%d len=%d\n", (int)input_buf_pos, mem_len, len);

        input = nc_new_tensor_1d(s->device, NC_TYPE_I32, mem_len + len);
        ptr = nc_tensor_get_ptr(input, NULL);
        for(i = 0; i < mem_len + len; i++) {
            ptr[i] = input_buf[input_buf_pos - mem_len + i];
        }
        tab_mem[0].mem_len = 0;
        tab_mem[0].mem_k = mem_k;
        tab_mem[0].mem_v = mem_v;

        output = trf_eval(s, mem_len + len, 1, tab_mem, input);

        for(i = 0; i < len; i++) {
            NCTensor *t0;
            t0 = nc_soft_max(nc_slice_alias(output, 1, mem_len + i,
                                            mem_len + i + 1));
            x = nc_tensor_get_data(&xbuf, t0);
            write_sym(pb, (float *)x->data,
                      x->dims[0], input_buf[input_buf_pos + i + 1]);
            nc_free_tensor(t0);
        }
        nc_free_tensor(output);

        input_buf_pos += len;
        mem_len = min_int(mem_len + len, s->n_ctx / 2);
    }

    prof_end(PROF_EVAL);

    n_output_bits = put_bit_flush(pb);

    printf("-> %" PRId64 " bytes\n", (n_output_bits + 7) / 8);
    fclose(f);
    nc_free(arith_buf);

    free(input_buf);
    for(i = 0; i < s->n_layer; i++) {
        nc_free_tensor(mem_k[i]);
        nc_free_tensor(mem_v[i]);
    }
    nc_free(mem_k);
    nc_free(mem_v);
    return 0;
}

int file_decompress(TextCompleteGlobalState *tcs,
                    const char *infilename, const char *outfilename)
{
    TransformerModel *s = tcs->trf_state;
    WordList *wl = tcs->wl;
    uint8_t *data_buf;
    ssize_t data_buf_len;
    GetBitState gb_s, *gb = &gb_s;
    BatchEntry tab_mem[1];
    NCTensor **mem_k, **mem_v, *input, *t0;
    DataSymbol *text_buf;
    NCTensorData xbuf, *x;
    Word *wp;
    int c, i, pos;
    FILE *f;
    BOOL convert_crlf;
    int32_t *ptr;

    data_buf = load_file((size_t *)&data_buf_len, infilename);
#if 0
    {
        int i;
        printf("data_buf=");
        for(i = 0; i < data_buf_len; i++)
            printf(" %02x", data_buf[i]);
        printf("\n");
    }
#endif
    get_bit_init(gb, data_buf, data_buf_len, NULL, NULL);

    convert_crlf = get_bit_raw(gb);

    text_buf = nc_malloc(sizeof(text_buf[0]) * s->n_ctx);

    mem_k = nc_mallocz(sizeof(mem_k[0]) * s->n_layer);
    mem_v = nc_mallocz(sizeof(mem_v[0]) * s->n_layer);
    for(i = 0; i < s->n_layer; i++) {
        mem_k[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_key, s->n_ctx, s->n_head);
        nc_tensor_set_name(mem_k[i], "mem_k_%d", i);
        mem_v[i] = nc_new_tensor_3d(s->device, NC_TYPE_F32,
                                    s->d_value, s->n_ctx, s->n_head);
        nc_tensor_set_name(mem_v[i], "mem_v_%d", i);
    }
    tab_mem[0].mem_k = mem_k;
    tab_mem[0].mem_v = mem_v;

    text_buf[0] = SYMB_EOT;

    f = fopen(outfilename, "wb");
    if (!f)
        perror(outfilename);

    pos = 0;
    for(;;) {
        input = nc_new_tensor_1d(s->device, NC_TYPE_I32, 1);
        ptr = nc_tensor_get_ptr(input, NULL);
        ptr[0] = text_buf[pos];
        tab_mem[0].mem_len = pos;
        t0 = trf_eval(s, 1, 1, tab_mem, input);
        t0 = nc_soft_max(t0);
        x = nc_tensor_get_data(&xbuf, t0);
        c = read_sym(gb, (float *)x->data, x->dims[0]);
        nc_free_tensor(t0);
        if (c == SYMB_EOT)
            break;
        wp = &wl->words[c];
        if (convert_crlf) {
            for(i = 0; i < wp->len; i++) {
                if (wp->buf[i] == '\n')
                    fputc('\r', f);
                fputc(wp->buf[i], f);
            }
        } else {
            fwrite(wp->buf, 1, wp->len, f);
        }
        fflush(f);
        pos++;
        if (pos >= s->n_ctx) {
            int n;
            /* buffer full: restart with the last n_ctx / 2 symbols */
            n = s->n_ctx / 2;
            for(i = 0; i < n; i++)
                text_buf[i] = text_buf[pos - n + i];

            input = nc_new_tensor_1d(s->device, NC_TYPE_I32, n);
            ptr = nc_tensor_get_ptr(input, NULL);
            for(i = 0; i < n; i++)
                ptr[i] = text_buf[i];
            tab_mem[0].mem_len = 0;
            t0 = trf_eval(s, n, 1, tab_mem, input);
            nc_free_tensor(t0);
            pos = n;
        }
        text_buf[pos] = c;
    }

    fclose(f);

    for(i = 0; i < s->n_layer; i++) {
        nc_free_tensor(mem_k[i]);
        nc_free_tensor(mem_v[i]);
    }
    nc_free(mem_k);
    nc_free(mem_v);
    nc_free(text_buf);
    free(data_buf);

    return 0;
}
