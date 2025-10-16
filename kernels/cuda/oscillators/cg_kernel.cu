// CUDA kernels for Center of Gravity (CG)
//
// Math pattern: dot-product style with fixed linear weights 1..(period-1).
// Warmup semantics: first valid index is `first_valid + period`; prefix is NaN.
// Division-by-zero or NaN denominator yields 0.0f, matching scalar policy.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CG_NAN
#define CG_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void cg_batch_f32(const float* __restrict__ prices,
                                         const int*   __restrict__ periods,
                                         int series_len,
                                         int n_combos,
                                         int first_valid,
                                         float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* out_ptr   = out + base;

    // Basic input validation mirrors CPU path behavior.
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        // Not enough valid points for CG warmup policy (needs period + 1)
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period; // first computed index for CG
    const int window = period - 1;           // number of terms in dot-product

    // Prefill NaN prefix up to (warm-1)
    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;

    if (window <= 0) {
        // Degenerate case: write zeros from warm to end
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }

    for (int i = warm; i < series_len; ++i) {
        const float* base_ptr = prices + i;
        float num = 0.0f;
        float den = 0.0f;
        // Accumulate newest (weight=1) to oldest (weight=window)
        for (int k = 0; k < window; ++k) {
            const float p = base_ptr[-k];
            num += (float)(k + 1) * p;
            den += p;
        }
        // Match scalar: tiny/NaN denominator -> 0.0
        if (!isfinite(den) || fabsf(den) <= 1.1920929e-7f /* FLT_EPSILON */) {
            out_ptr[i] = 0.0f;
        } else {
            out_ptr[i] = -num / den;
        }
    }
}

// prices_tm: time-major layout [row * num_series + series]
extern "C" __global__ void cg_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        // Write NaN column
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;

    // Prefill NaN up to warm-1
    {
        float* o = col_out;
        for (int row = 0; row < warm; ++row, o += num_series) *o = CG_NAN;
    }

    if (window <= 0) {
        // Degenerate: zeros from warm
        float* o = col_out + (size_t)warm * num_series;
        for (int row = warm; row < series_len; ++row, o += num_series) *o = 0.0f;
        return;
    }

    // For each row >= warm, compute dot over (window) elements, strided by num_series
    for (int row = warm; row < series_len; ++row) {
        const float* base = col_in + (size_t)row * num_series;
        float num = 0.0f;
        float den = 0.0f;
        for (int k = 0; k < window; ++k) {
            const float p = *(base - (size_t)k * num_series);
            num += (float)(k + 1) * p;
            den += p;
        }
        float* dst = col_out + (size_t)row * num_series;
        if (!isfinite(den) || fabsf(den) <= 1.1920929e-7f) {
            *dst = 0.0f;
        } else {
            *dst = -num / den;
        }
    }
}

