// CUDA kernels for Donchian Channels (upper/middle/lower) in FP32.
//
// Semantics:
// - Warmup: indices < (first_valid + period - 1) are NaN
// - Any NaN in the window gates the entire output at that index to NaN
// - Upper = max(high), Lower = min(low), Middle = (Upper + Lower)/2
//
// Batch variant: one series × many params (periods)
// Many-series variant: time-major layout × one param

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef DCH_NAN
#define DCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void donchian_batch_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ out_upper,
                                               float* __restrict__ out_middle,
                                               float* __restrict__ out_lower) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* uo = out_upper + base;
    float* mo = out_middle + base;
    float* lo = out_lower + base;

    // Guard invalid inputs by writing NaNs
    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) {
            uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN;
        }
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < series_len; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }

    if (period == 1) {
        for (int i = first_valid; i < series_len; ++i) {
            const float h = high[i];
            const float l = low[i];
            if (isnan(h) || isnan(l)) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
            else { uo[i] = h; lo[i] = l; mo[i] = 0.5f * (h + l); }
        }
        return;
    }

    // Naive window scan per output index (kept simple for correctness parity)
    for (int i = warm; i < series_len; ++i) {
        const int start = i + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;
        for (int k = 0; k < period; ++k) {
            const float h = high[start + k];
            const float l = low[start + k];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (any_nan) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        else { uo[i] = maxv; lo[i] = minv; mo[i] = 0.5f * (maxv + minv); }
    }
}

// Many series × one param (time-major I/O)
extern "C" __global__ void donchian_many_series_one_param_f32(
    const float* __restrict__ high_tm,   // time-major: [row * num_series + series]
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids, // per series (column)
    int num_series,
    int series_len,
    int period,
    float* __restrict__ upper_tm,
    float* __restrict__ middle_tm,
    float* __restrict__ lower_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len || period <= 0 || period > series_len || (series_len - first_valid) < period) {
        // All NaN outputs
        for (int row = 0; row < series_len; ++row) {
            const int idx = row * num_series + series;
            upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int row = 0; row < warm; ++row) {
        const int idx = row * num_series + series;
        upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
    }

    if (period == 1) {
        for (int row = first_valid; row < series_len; ++row) {
            const int idx = row * num_series + series;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            if (isnan(h) || isnan(l)) { upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN; }
            else { upper_tm[idx] = h; lower_tm[idx] = l; middle_tm[idx] = 0.5f * (h + l); }
        }
        return;
    }

    for (int row = warm; row < series_len; ++row) {
        const int start = row + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;
        // strided access in time-major layout
        for (int k = 0; k < period; ++k) {
            const int idx = (start + k) * num_series + series;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        const int idx_out = row * num_series + series;
        if (any_nan) { upper_tm[idx_out] = DCH_NAN; middle_tm[idx_out] = DCH_NAN; lower_tm[idx_out] = DCH_NAN; }
        else { upper_tm[idx_out] = maxv; lower_tm[idx_out] = minv; middle_tm[idx_out] = 0.5f * (maxv + minv); }
    }
}
