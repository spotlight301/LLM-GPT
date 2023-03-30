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
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

#include "cutils.h"
#include "libnc.h"
#include "cp_utils.h"

void fatal_error(const char *fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    fprintf(stderr, "Fatal error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    exit(1);
}

int64_t get_time_ms(void)
{
#ifdef _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (tv.tv_usec / 1000U);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (ts.tv_nsec / 1000000U);
#endif
}

void fput_u8(FILE *f, uint8_t v)
{
    fputc(v, f);
}

int fget_u8(FILE *f, uint8_t *pv)
{
    int c;
    c = fgetc(f);
    if (c < 0)
        return -1;
    *pv = c;
    return 0;
}

void fput_be16(FILE *f, uint16_t v)
{
    fputc(v >> 8, f);
    fputc(v >> 0, f);
}

int fget_be16(FILE *f, uint16_t *pv)
{
    uint8_t buf[2];
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
        return -1;
    *pv = (buf[0] << 8) |
        (buf[1] << 0);
    return 0;
}

void fput_be32(FILE *f, uint32_t v)
{
    fputc(v >> 24, f);
    fputc(v >> 16, f);
    fputc(v >> 8, f);
    fputc(v >> 0, f);
}

int fget_be32(FILE *f, uint32_t *pv)
{
    uint8_t buf[4];
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
        return -1;
    *pv = (buf[0] << 24) |
        (buf[1] << 16) |
        (buf[2] << 8) |
        (buf[3] << 0);
    return 0;
}

void fput_sgd_opt(FILE *f, const SGDOptParams *p)
{
    fput_u8(f, p->algo);
    switch(p->algo) {
    case SGD_OPT_BASIC:
        break;
    case SGD_OPT_ADAM:
        fput_f32(f, p->u.adam.beta1);
        fput_f32(f, p->u.adam.beta2);
        fput_f32(f, p->u.adam.eps);
        fput_f32(f, p->u.adam.gradient_clip);
        break;
    default:
        abort();
    }
}

int fget_sgd_opt(FILE *f, SGDOptParams *p)
{
    uint8_t v8;
    
    if (fget_u8(f, &v8))
        return -1;
    p->algo = v8;
    switch(p->algo) {
    case SGD_OPT_BASIC:
        break;
    case SGD_OPT_ADAM:
        if (fget_f32(f, &p->u.adam.beta1))
            return -1;
        if (fget_f32(f, &p->u.adam.beta2))
            return -1;
        if (fget_f32(f, &p->u.adam.eps))
            return -1;
        if (fget_f32(f, &p->u.adam.gradient_clip))
            return -1;
        break;
    default:
        return -1;
    }
    return 0;
}

void dump_sgd_opt_params(FILE *f, const SGDOptParams *p)
{
    switch(p->algo) {
    case SGD_OPT_BASIC:
        fprintf(f, " sgd_opt=%s", 
               "none");
        break;
    case SGD_OPT_ADAM:
        fprintf(f, " sgd_opt=%s beta1=%g beta2=%g eps=%g gclip=%g",
               "adam",
                p->u.adam.beta1,
                p->u.adam.beta2,
                p->u.adam.eps,
                p->u.adam.gradient_clip);
        break;
    default:
        abort();
    }
}

typedef union {
    float f;
    uint32_t u32;
} f32;

void fput_f32(FILE *f, float v)
{
    f32 u;
    u.f = v;
    fput_be32(f, u.u32);
}

int fget_f32(FILE *f, float *pv)
{
    f32 u;
    if (fget_be32(f, &u.u32))
        return -1;
    *pv = u.f;
    return 0;
}

void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym)
{
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0; /* invariant: p=sum(prob_table[start...start + range]) */
    while (range > 1) {
        range0 = range >> 1;
        p0 = vec_sum_f32(prob_table + start, range0);
        prob0 = lrintf(p0 * PROB_UNIT / p);
        prob0 = clamp_int(prob0, 1, PROB_UNIT - 1);
        bit = sym >= (start + range0);
        put_bit(pb, prob0, bit);
        if (bit) {
            start += range0;
            range = range - range0;
            p = p - p0;
        } else {
            p = p0;
            range = range0;
        }
    }
}

int read_sym(GetBitState *gb, const float *prob_table, int n_symb)
{
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0; /* invariant: p=sum(prob_table[start...start + range]) */
    while (range > 1) {
        range0 = range >> 1;
        p0 = vec_sum_f32(prob_table + start, range0);
        prob0 = lrintf(p0 * PROB_UNIT / p);
        prob0 = clamp_int(prob0, 1, PROB_UNIT - 1);
        bit = get_bit(gb, prob0);
        if (bit) {
            start += range0;
            range = range - range0;
            p = p - p0;
        } else {
            p = p0;
            range = range0;
        }
    }
    return start;
}

void create_debug_dir(char *debug_dir, size_t debug_dir_size,
                      const char *debug_path, const char *prefix)
{
    char name1[1024];
    struct tm *tm;
    time_t ti;
    
    snprintf(name1, sizeof(name1), "%s/%s", debug_path, prefix);
#ifdef _WIN32
    _mkdir(name1);
#else
    mkdir(name1, 0777);
#endif
    
    ti = time(NULL);
    tm = localtime(&ti);
    snprintf(debug_dir, debug_dir_size, "%s/%04u%02u%02u-%02u%02u%02u",
             name1,
             tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec);
#ifdef _WIN32
    _mkdir(debug_dir);
#else
    mkdir(debug_dir, 0777);
#endif
}

/* we print at least 3 significant digits with at most 5 chars, except
   if larger than 9999T. The value is rounded to zero. */
char *get_si_prefix(char *buf, int buf_size, uint64_t val)
{
    static const char suffixes[4] = "kMGT";
    uint64_t base;
    int i;

    if (val <= 999) {
        snprintf(buf, buf_size, "%" PRId64, val);
    } else {
        base = 1000;
        for(i=0;i<4;i++) {
            /* Note: we round to 0 */
            if (val < base * 10) {
                snprintf(buf, buf_size, "%0.2f%c", 
                         floor((val * 100.0) / base) / 100.0,
                         suffixes[i]);
                break;
            } else if (val < base * 100) {
                snprintf(buf, buf_size, "%0.1f%c", 
                         floor((val * 10.0) / base) / 10.0,
                         suffixes[i]);
                break;
            } else if (val < base * 1000 || (i == 3)) {
                snprintf(buf, buf_size,
                         "%" PRId64 "%c", 
                         val / base,
                         suffixes[i]);
                break;
            }
            base = base * 1000;
        }
    }
    return buf;
}
