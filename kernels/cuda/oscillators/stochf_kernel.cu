// CUDA kernels for StochF (Fast Stochastic Oscillator: %K and %D).
//
// Batch (one series × many params):
//  - Uses WILLR sparse tables (st_max/st_min + log2/offsets + nan_psum) built on host
//    to answer HH/LL queries in O(1) per index.
//  - %K is raw stochastic: 100 * (C - LL) / (HH - LL), with degenerate window
//    handling matching scalar: if HH==LL, K = (C==HH ? 100 : 0).
//  - %D supports SMA only (matype==0). For other matypes, outputs remain NaN
//    to match scalar.
//
// Many-series × one-param (time-major):
//  - Naive O(fastk) scan per time step to compute HH/LL per series.
//  - Same warmup/NaN semantics as scalar: K warmup = fv + fastk - 1; D warmup =
//    fv + fastk + fastd - 2. Write NaN before warmup.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCHF_QNAN
#define STOCHF_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// %K from WILLR sparse tables
__device__ __forceinline__ float stoch_from_tables(
    int t,
    int fast_k,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
) {
    const int start = t - fast_k + 1;
    // Any NaN in window → NaN (conservative; typical datasets have no NaNs post-warm)
    if (nan_psum[t + 1] - nan_psum[start] != 0) return STOCHF_QNAN;

    const int window = fast_k;
    const int k = log2_tbl[window];
    const int offset = 1 << k;
    const int level_base = level_offsets[k];
    const int idx_a = level_base + start;
    const int idx_b = level_base + (t + 1 - offset);
    const double h = (double)fmaxf(st_max[idx_a], st_max[idx_b]);
    const double l = (double)fminf(st_min[idx_a], st_min[idx_b]);
    const double c = (double)close[t];
    if (!isfinite(h) || !isfinite(l) || !isfinite(c)) return STOCHF_QNAN;
    const double den = h - l;
    if (den == 0.0) {
        // Match scalar: if c==hh (==l), return 100; else 0
        return (c == h) ? 100.0f : 0.0f;
    }
    return (float)(100.0 * ((c - l) / den));
}

extern "C" __global__ void stochf_batch_f32(
    // Inputs
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    const int*   __restrict__ fastk_arr,
    const int*   __restrict__ fastd_arr,
    const int*   __restrict__ matype_arr, // 0=SMA; others → D stays NaN
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    // Outputs (row-major)
    float* __restrict__ out_k,
    float* __restrict__ out_d
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Initialize entire row to NaN
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_k[base + i] = STOCHF_QNAN;
        out_d[base + i] = STOCHF_QNAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane sequential scan per combo

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(level_count <= 0)) return;

    const int fk = fastk_arr[combo];
    const int fd = fastd_arr[combo];
    const int mt = matype_arr[combo];
    if (UNLIKELY(fk <= 0 || fd <= 0)) return;

    const int k_warm = first_valid + fk - 1;
    const int d_warm = k_warm + fd - 1;
    if (UNLIKELY(k_warm >= series_len)) return;

    // %K raw stochastic per index
    for (int t = k_warm; t < series_len; ++t) {
        float kv = stoch_from_tables(t, fk, close, log2_tbl, level_offsets, st_max, st_min, nan_psum);
        out_k[base + t] = kv;
    }

    // %D only SMA (matype==0). Others remain NaN (match scalar contract)
    if (mt == 0) {
        double d_sum = 0.0; int cnt = 0;
        for (int t = k_warm; t < series_len; ++t) {
            const float kv = out_k[base + t];
            if (kv == kv) {
                if (cnt < fd) { d_sum += (double)kv; cnt += 1; }
                if (t < d_warm) {
                    out_d[base + t] = STOCHF_QNAN;
                } else if (t == d_warm) {
                    // First complete window
                    out_d[base + t] = (cnt == fd) ? (float)(d_sum / (double)fd) : STOCHF_QNAN;
                } else {
                    // Rolling window: subtract k[t-fd], add kv
                    const float oldk = out_k[base + (t - fd)];
                    d_sum += (double)kv - (double)oldk;
                    out_d[base + t] = (float)(d_sum / (double)fd);
                }
            } else {
                // kv is NaN → D is NaN; keep d_sum/cnt unchanged until next valid run
                out_d[base + t] = STOCHF_QNAN;
            }
        }
    }
}

// ---------------- Many-series × one-param (time-major) ----------------
// Inputs are time-major: x[t * num_series + s]
extern "C" __global__ void stochf_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int fast_k,
    int fast_d,
    int matype, // 0=SMA; others → D stays NaN
    float* __restrict__ k_out_tm,
    float* __restrict__ d_out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    // Initialize outputs to NaN for entire series
    for (int t = 0; t < series_len; ++t) {
        *(k_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
        *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
    }
    if (UNLIKELY(fv < 0 || fv >= series_len || fast_k <= 0 || fast_d <= 0)) return;

    const int k_warm = fv + fast_k - 1;
    const int d_warm = k_warm + fast_d - 1;
    if (UNLIKELY(k_warm >= series_len)) return;

    auto load_tm = [num_series, series](const float* base, int t)->float {
        return *(base + (size_t)t * num_series + series);
    };

    auto stoch_naive = [&](int t)->float {
        const int start = t - fast_k + 1;
        double h = -INFINITY, l = INFINITY;
        for (int i = start; i <= t; ++i) {
            const double hi = (double)load_tm(high_tm, i);
            const double lo = (double)load_tm(low_tm,  i);
            if (!(hi == hi) || !(lo == lo)) return STOCHF_QNAN; // NaN in window
            h = fmax(h, hi);
            l = fmin(l, lo);
        }
        const double c = (double)load_tm(close_tm, t);
        if (!(c == c)) return STOCHF_QNAN;
        const double den = h - l;
        if (den == 0.0) return (c == h) ? 100.0f : 0.0f;
        return (float)(100.0 * ((c - l) / den));
    };

    // Fill K
    for (int t = k_warm; t < series_len; ++t) {
        float kv = stoch_naive(t);
        *(k_out_tm + (size_t)t * num_series + series) = kv;
    }

    // Fill D (SMA only)
    if (matype == 0) {
        double d_sum = 0.0; int cnt = 0;
        for (int t = k_warm; t < series_len; ++t) {
            const float kv = *(k_out_tm + (size_t)t * num_series + series);
            if (kv == kv) {
                if (cnt < fast_d) { d_sum += (double)kv; cnt += 1; }
                if (t < d_warm) {
                    *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
                } else if (t == d_warm) {
                    *(d_out_tm + (size_t)t * num_series + series) = (cnt == fast_d) ? (float)(d_sum / (double)fast_d) : STOCHF_QNAN;
                } else {
                    const float oldk = *(k_out_tm + (size_t)(t - fast_d) * num_series + series);
                    d_sum += (double)kv - (double)oldk;
                    *(d_out_tm + (size_t)t * num_series + series) = (float)(d_sum / (double)fast_d);
                }
            } else {
                *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
            }
        }
    }
}

