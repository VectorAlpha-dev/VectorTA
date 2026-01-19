


#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef SMA_NAN
#define SMA_NAN (__int_as_float(0x7fffffff))
#endif





#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif




















extern "C" __global__ void sma_exclusive_prefix_f64(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    double* __restrict__ prefix
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    prefix[0] = 0.0;
    double acc = 0.0;
    for (int t = 0; t < series_len; ++t) {
        const double x = (t < first_valid) ? 0.0 : static_cast<double>(prices[t]);
        acc += x;
        prefix[t + 1] = acc;
    }
}

extern "C" __global__ void sma_batch_from_prefix_f64(
    const double* __restrict__ prefix,
    const int* __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int row_off = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    if (UNLIKELY(series_len <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        while (t < series_len) {
            out[row_off + t] = SMA_NAN;
            t += stride;
        }
        return;
    }

    if (UNLIKELY(period <= 0 || period > series_len || (series_len - first_valid) < period)) {
        while (t < series_len) {
            out[row_off + t] = SMA_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    const double inv_p = 1.0 / static_cast<double>(period);

    while (t < series_len) {
        if (t < warm) {
            out[row_off + t] = SMA_NAN;
        } else {
            const int t1 = t + 1;
            const int start = t1 - period;
            const double sum = prefix[t1] - prefix[start];
            out[row_off + t] = static_cast<float>(sum * inv_p);
        }
        t += stride;
    }
}

extern "C" __global__ void sma_batch_from_prefix_f64_tm(
    const double* __restrict__ prefix,
    const int* __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_tm
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    if (UNLIKELY(series_len <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        while (t < series_len) {
            out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = SMA_NAN;
            t += stride;
        }
        return;
    }

    if (UNLIKELY(period <= 0 || period > series_len || (series_len - first_valid) < period)) {
        while (t < series_len) {
            out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = SMA_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    const double inv_p = 1.0 / static_cast<double>(period);

    while (t < series_len) {
        float outv = SMA_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            const int start = t1 - period;
            const double sum = prefix[t1] - prefix[start];
            outv = static_cast<float>(sum * inv_p);
        }
        out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = outv;
        t += stride;
    }
}

extern "C" __global__ void sma_batch_f32(const float* __restrict__ prices,
                                         const int*   __restrict__ periods,
                                         int series_len,
                                         int n_combos,
                                         int first_valid,
                                         float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int  period  = periods[combo];
    const int  base    = combo * series_len;
    float*     out_ptr = out + base;


    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = SMA_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = SMA_NAN;
        return;
    }

    const int   warm = first_valid + period - 1;
    const float inv  = 1.0f / static_cast<float>(period);


    for (int i = 0; i < warm; ++i) out_ptr[i] = SMA_NAN;

    if (period == 1) {

        const float* src = prices + first_valid;
        float*       dst = out_ptr + first_valid;
        for (int i = first_valid; i < series_len; ++i) *dst++ = *src++;
        return;
    }


    float sum = 0.0f;
#ifdef SMA_USE_KAHAN
    float c = 0.0f;
#endif
    const float* p0 = prices + first_valid;
    for (int k = 0; k < period; ++k) {
#ifdef SMA_USE_KAHAN
        float y = p0[k] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
#else
        sum += p0[k];
#endif
    }
    out_ptr[warm] = sum * inv;


    const float* cur = prices + (warm + 1);
    const float* old = prices + first_valid;
    float*       dst = out_ptr + (warm + 1);
    for (int i = warm + 1; i < series_len; ++i) {
#ifdef SMA_USE_KAHAN
        float delta = (*cur++) - (*old++);
        float y = delta - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        *dst++ = sum * inv;
#else
        sum += *cur++;
        sum -= *old++;
        *dst++ = sum * inv;
#endif
    }
}

extern "C" __global__ void sma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;


    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;


    if (UNLIKELY(period <= 0 || period > series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int   warm = first_valid + period - 1;
    const float inv  = 1.0f / static_cast<float>(period);


    {
        float* o = col_out;
        for (int row = 0; row < warm; ++row, o += num_series) *o = SMA_NAN;
    }

    if (period == 1) {

        const float* src = col_in  + static_cast<size_t>(first_valid) * num_series;
        float*       dst = col_out + static_cast<size_t>(first_valid) * num_series;
        for (int row = first_valid; row < series_len; ++row, src += num_series, dst += num_series)
            *dst = *src;
        return;
    }


    float sum = 0.0f;
#ifdef SMA_USE_KAHAN
    float c = 0.0f;
#endif
    const float* p = col_in + static_cast<size_t>(first_valid) * num_series;
    for (int k = 0; k < period; ++k, p += num_series) {
#ifdef SMA_USE_KAHAN
        float y = *p - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
#else
        sum += *p;
#endif
    }


    *(col_out + static_cast<size_t>(warm) * num_series) = sum * inv;


    const float* cur = col_in  + static_cast<size_t>(warm + 1)    * num_series;
    const float* old = col_in  + static_cast<size_t>(first_valid) * num_series;
    float*       dst = col_out + static_cast<size_t>(warm + 1)    * num_series;

    for (int row = warm + 1; row < series_len; ++row) {
#ifdef SMA_USE_KAHAN
        float delta = (*cur) - (*old);
        float y = delta - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        *dst = sum * inv;
#else
        sum += *cur;
        sum -= *old;
        *dst = sum * inv;
#endif
        cur += num_series;
        old += num_series;
        dst += num_series;
    }
}
