// CUDA kernels for TTM Trend (close > SMA(source, period) ? 1 : 0)
//
// Batch path: one series × many params. Uses host-precomputed inclusive
// prefix sums of `source` (in FP64) to compute window averages in O(1).
// Many-series path: time-major layout, one thread per series scans time
// sequentially with a rolling sum.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ------------------ Batch: prefix-sum based ------------------
// prefix_src: inclusive prefix where prefix[first] = source[first]
// close:      input close series (f32)
// periods:    per-row period (int)
// warm_idx:   per-row warm index = first_valid + period - 1
// series_len: length
// first_valid: index of first non-NaN pair (source, close) on host
// n_combos:   number of parameter rows
// out:        row-major [n_combos * series_len], 0.0 for warmup, 1.0/0.0 after
extern "C" __global__ void ttm_trend_batch_prefix_f64(
    const double* __restrict__ prefix_src,
    const float*  __restrict__ close,
    const int*    __restrict__ periods,
    const int*    __restrict__ warm_idx,
    int series_len,
    int first_valid,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int p    = periods[combo];
    const int warm = warm_idx[combo];
    if (p <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;

    // Clear row to 0.0 (warmup semantics for boolean output)
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = 0.0f;
    }
    __syncthreads();

    // Compute outputs starting at warm
    const double invp = 1.0 / (double)p;
    // Ensure first element is handled by a single thread to avoid races
    if (threadIdx.x == 0) {
        const double avg0 = prefix_src[warm] * invp; // window [first_valid .. warm]
        out[base + warm] = ((double)close[warm] > avg0) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Remaining indices: each thread strides over time
    for (int i = warm + 1 + threadIdx.x; i < series_len; i += blockDim.x) {
        const double sum = prefix_src[i] - prefix_src[i - p];
        const double avg = sum * invp;
        out[base + i] = ((double)close[i] > avg) ? 1.0f : 0.0f;
    }
}

// ------------------ Many-series × one param (time-major) ------------------
// Layout: prices_tm[row * num_series + series]
extern "C" __global__ void ttm_trend_many_series_one_param_time_major_f32(
    const float* __restrict__ source_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (period <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int fv     = first_valids[series];
    if (fv < 0 || fv >= series_len) return;
    const int warm = fv + period - 1;
    const int col  = series;

    // Warmup prefix: zeros
    for (int r = 0; r < warm && r < series_len; ++r) {
        out_tm[(size_t)r * stride + col] = 0.0f;
    }
    if (warm >= series_len) return;

    // Initial window sum over [fv .. warm]
    double sum = 0.0;
    for (int k = fv; k <= warm; ++k) {
        sum += (double)source_tm[(size_t)k * stride + col];
    }
    const double invp = 1.0 / (double)period;
    double avg = sum * invp;
    out_tm[(size_t)warm * stride + col] = ((double)close_tm[(size_t)warm * stride + col] > avg) ? 1.0f : 0.0f;

    // Slide window
    for (int t = warm + 1; t < series_len; ++t) {
        const double add = (double)source_tm[(size_t)t * stride + col];
        const double sub = (double)source_tm[(size_t)(t - period) * stride + col];
        sum += add - sub;
        avg = sum * invp;
        out_tm[(size_t)t * stride + col] = ((double)close_tm[(size_t)t * stride + col] > avg) ? 1.0f : 0.0f;
    }
}

