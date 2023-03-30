/*
 * Compression utilities
 * 
 * Copyright (c) 2018-2019 Fabrice Bellard
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
#include "arith.h"
#include "libnc.h"

void __attribute__((noreturn, format(printf, 1, 2))) fatal_error(const char *fmt, ...);

int64_t get_time_ms(void);
void fput_u8(FILE *f, uint8_t v);
int fget_u8(FILE *f, uint8_t *pv);
void fput_be16(FILE *f, uint16_t v);
int fget_be16(FILE *f, uint16_t *pv);
void fput_be32(FILE *f, uint32_t v);
int fget_be32(FILE *f, uint32_t *pv);
void fput_f32(FILE *f, float v);
int fget_f32(FILE *f, float *pv);
void fput_sgd_opt(FILE *f, const SGDOptParams *p);
int fget_sgd_opt(FILE *f, SGDOptParams *p);
void dump_sgd_opt_params(FILE *f, const SGDOptParams *p);

void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym);
int read_sym(GetBitState *gb, const float *prob_table, int n_symb);

void create_debug_dir(char *debug_dir, size_t debug_dir_size,
                      const char *debug_path, const char *prefix);
char *get_si_prefix(char *buf, int buf_size, uint64_t val);

