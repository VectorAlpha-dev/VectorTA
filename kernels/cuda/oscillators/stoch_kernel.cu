// CUDA kernels for Stochastic Oscillator (Stoch)
//
// We provide two core kernels:
// - stoch_k_raw_from_hhll_f32: compute raw %K given precomputed rolling
//   highest-high (hh) and lowest-low (ll) arrays for a single series.
// - stoch_many_series_one_param_f32: compute raw %K for many series laid out
//   in time-major format (columns = series), using a shared fastk_period.
//
// Notes:
// - All math is FP32 to match the broader CUDA integration in this crate.
// - Warmup semantics: indices before (first_valid + fastk - 1) are set to NaN.
// - NaN propagation: any NaN in the inputs (close, hh, ll) yields NaN.
// - Zero (or near-zero) denominator: write 50.0 (matches scalar Stoch policy).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCH_NAN
#define STOCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__
void stoch_k_raw_from_hhll_f32(const float* __restrict__ close,
                               const float* __restrict__ hh,
                               const float* __restrict__ ll,
                               int series_len,
                               int first_valid,
                               int fastk_period,
                               float* __restrict__ out) {
    // Parallel prefill of NaNs for warmup region
    const int warm = first_valid + fastk_period - 1;
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[i] = (i < warm) ? STOCH_NAN : out[i];
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane sequential scan for simplicity

    if (UNLIKELY(series_len <= 0 || fastk_period <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(warm >= series_len)) return;

    for (int t = warm; t < series_len; ++t) {
        const float c = close[t];
        const float h = hh[t];
        const float l = ll[t];
        if (!(isfinite(c) && isfinite(h) && isfinite(l))) {
            out[t] = STOCH_NAN;
            continue;
        }
        const float denom = h - l;
        if (fabsf(denom) < 1e-12f) {
            out[t] = 50.0f;
        } else {
            out[t] = (c - l) * (100.0f / denom);
        }
    }
}

// Time-major many-series kernel (shared fastk, naive O(period) window scan per step)
// prices are laid out time-major: idx = row * num_series + series
extern "C" __global__
void stoch_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const float* __restrict__ close_tm,
                                     const int*   __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int fastk_period,
                                     float* __restrict__ k_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id (column)
    if (s >= num_series) return;

    // Invalid inputs -> full NaN column
    if (UNLIKELY(fastk_period <= 0 || fastk_period > series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }

    const int first_valid = first_valids[s];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }
    const int warm = first_valid + fastk_period - 1;

    // Prefill NaN up to warm-1
    {
        float* out_col = k_tm + s;
        for (int row = 0; row < warm; ++row, out_col += num_series) *out_col = STOCH_NAN;
    }

    // Row pointers (time-major strides by num_series)
    const float* h0 = high_tm  + ((size_t)first_valid) * num_series + s;
    const float* l0 = low_tm   + ((size_t)first_valid) * num_series + s;
    const float* c0 = close_tm + ((size_t)first_valid) * num_series + s;
    float*       o0 = k_tm     + ((size_t)first_valid) * num_series + s;

    if (fastk_period == 1) {
        // Just map close -> %K (undefined range degenerates to 50), but by policy
        // we still need hh/ll; with period 1, hh=high, ll=low at the same row.
        for (int row = first_valid; row < series_len; ++row) {
            const float h = *h0; const float l = *l0; const float c = *c0;
            *o0 = (isfinite(h) && isfinite(l) && isfinite(c)) ? ((h == l) ? 50.0f : (c - l) * (100.0f / (h - l))) : STOCH_NAN;
            h0 += num_series; l0 += num_series; c0 += num_series; o0 += num_series;
        }
        return;
    }

    // Naive O(period) window scan per row; fastk ~14 typical -> acceptable.
    // For larger windows this could be replaced by monotone deques.
    for (int row = warm; row < series_len; ++row) {
        // Evaluate window [row - fastk + 1, row]
        float hmax = -INFINITY;
        float lmin =  INFINITY;
        bool any_nan = false;
        int start = row - fastk_period + 1;
        const float* hptr = high_tm  + ((size_t)start) * num_series + s;
        const float* lptr = low_tm   + ((size_t)start) * num_series + s;
        for (int k = 0; k < fastk_period; ++k) {
            float hv = *hptr; float lv = *lptr;
            if (!(isfinite(hv) && isfinite(lv))) { any_nan = true; break; }
            hmax = fmaxf(hmax, hv);
            lmin = fminf(lmin, lv);
            hptr += num_series; lptr += num_series;
        }

        float* outp = k_tm + ((size_t)row) * num_series + s;
        const float c = close_tm[((size_t)row) * num_series + s];
        if (any_nan || !isfinite(c) || !isfinite(hmax) || !isfinite(lmin)) {
            *outp = STOCH_NAN;
            continue;
        }
        const float denom = hmax - lmin;
        *outp = (fabsf(denom) < 1e-12f) ? 50.0f : (c - lmin) * (100.0f / denom);
    }
}
