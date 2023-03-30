#ifndef _GPT2TC_H
#define _GPT2TC_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "cutils.h"
#include "arith.h"
#include "cp_utils.h"
#include "list.h"
#include "libnc.h"

#define MAX_INITIAL_TEXT_LEN 256 /* in symbols */
#define MAX_OUTPUT_LEN 100
#define DEFAULT_TOP_K 40
#define DEFAULT_TOP_P 0.9
#define BATCH_SIZE_MAX 16
//#define BATCH_SIZE_MAX 1


typedef uint16_t DataSymbol;

typedef enum {
    GPT2_MODEL_117M,
    GPT2_MODEL_345M,
    GPT2_MODEL_774M,
    GPT2_MODEL_1558M,
} GPT2ModelEnum;

typedef struct {
    BOOL is_decoder;
    int n_layer;
    int d_model;
    int n_head;
    int d_key;
    int d_value;
    int d_inner;
    int n_ctx;
    int n_symbols;
    uint32_t seed;
} TransformerModelParams;

typedef struct {
    NCTensor *ln_1_g, *ln_1_b;
    NCTensor *attn_w, *attn_b;
    NCTensor *attn_proj_w, *attn_proj_b;

    NCTensor *ln_2_g, *ln_2_b;
    NCTensor *mlp_fc_w, *mlp_fc_b;
    NCTensor *mlp_proj_w, *mlp_proj_b;
} TransformerLayer;

typedef struct {
    RNDState rnd_state;
    NCContext *model;
    NCDevice *device;
    int n_layer;
    int d_model;
    int n_head;
    int d_key;
    int d_value;
    int d_inner;
    int n_symbols;
    int n_ctx;

    /* parameters */
    NCParamList param_list;
    TransformerLayer *layers;
    NCTensor *wte, *wpe, *wte_trans;
    NCTensor *ln_f_g, *ln_f_b;
} TransformerModel;

typedef struct Word {
    uint32_t next; /* -1 = end */
    uint32_t len;
    uint8_t *buf;
} Word;

typedef struct {
    Word *words;
    size_t word_count;
    size_t word_size;
    uint32_t *hash_table;
    int hash_size;
    int hash_bits;
} WordList;

typedef struct {
    TransformerModel *trf_state;
    WordList *wl;
} TextCompleteGlobalState;

typedef struct {
    struct list_head link;
    TextCompleteGlobalState *global_state;
    int top_k;
    float top_p;
    float temperature;
    RNDState rnd_state;
    NCTensor **mem_k, **mem_v;
    DataSymbol *input_buf;
    int input_buf_len;
    int text_len; /* current input text len */
    BOOL is_first;
    int last_c;
    int max_output_len;

    /* output */
    char out_text[1024];
    int out_text_len; /* 0 means end of output */
} TextGenContext;

GPT2ModelEnum parse_model(const char *str);
void trf_set_params(TransformerModelParams *p, GPT2ModelEnum model);
void gpt2_pp_encode(const char *word_filename, const char *in_filename, const char *out_filename);
size_t gpt2_pp_encode_buf(WordList *s, DataSymbol **pout_buf, const uint8_t *buf, size_t buf_size);
void gpt2_pp_decode(const char *word_filename, const char *in_filename, const char *out_filename);
char *trim_text(const char *str);
TextCompleteGlobalState *text_complete_global_init(GPT2ModelEnum model, const char *filename);
void text_complete_global_end(TextCompleteGlobalState *tcs);
TextGenContext *text_complete_start(TextCompleteGlobalState *tcs, const char *input_text, int top_k, float top_p, float temperature, int seed, int max_output_len);
void text_complete_next(TextCompleteGlobalState *tcs, struct list_head *ts_list);
void text_complete_end(TextGenContext *ts);
void text_complete(GPT2ModelEnum model, const char *model_filename, const char *input_text, int top_k, float top_p, float temperature, int max_output_len, int batch_size, int seed, BOOL verbose);
int unicode_to_utf8(uint8_t *buf, unsigned int c);
int unicode_from_utf8(const uint8_t *p, int max_len, const uint8_t **pp);
size_t convert_to_chars(char **pout_buf, uint8_t *buf, size_t n_bits);
ssize_t convert_from_chars(uint8_t **pout_buf, const char *str);
int encode_length(PutBitState *pb, uint32_t val);
int decode_length(GetBitState *gb);
int text_decompress(TextCompleteGlobalState *tcs, char **poutput_text, const char *input_text);
int text_compress(TextCompleteGlobalState *tcs, char **poutput_text, const char *input_text, BOOL dump_stats);
void text_compress_test(GPT2ModelEnum model, const char *model_filename, const char *input_text, BOOL is_decode, BOOL verbose);
int file_compress(TextCompleteGlobalState *tcs, const char *infilename, const char *outfilename);
int file_decompress(TextCompleteGlobalState *tcs, const char *infilename, const char *outfilename);
#ifdef __cplusplus
}
#endif
#endif
