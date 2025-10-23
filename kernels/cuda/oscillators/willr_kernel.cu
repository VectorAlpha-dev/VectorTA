// CUDA kernel for Williams' %R (WILLR) batch evaluation — optimized.
//
// Each block = one combo (period). Threads in the block parallelize across time.
// Warmup semantics preserved, FP32 throughout.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void willr_batch_f32(const float* __restrict__ close,
                     const int* __restrict__ periods,
                     const int* __restrict__ log2_tbl,
                     const int* __restrict__ level_offsets,
                     const float* __restrict__ st_max,
                     const float* __restrict__ st_min,
                     const int* __restrict__ nan_psum,
                     int series_len,
                     int first_valid,
                     int level_count,
                     int n_combos,
                     float* __restrict__ out) {

    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;
    float* __restrict__ out_row = out + base;

    // Guard invalid input by writing full NaN row.
    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x)
            out_row[i] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;

    // Prefill only the warmup prefix with NaNs (no barrier needed).
    const int warm_clamped = (warm < series_len) ? warm : series_len;
    for (int i = threadIdx.x; i < warm_clamped; i += blockDim.x)
        out_row[i] = NAN;

    if (warm >= series_len) return;

    // Sparse-table constants for this combo (O(1) RMQ per query).
    const int k = log2_tbl[period];
    if (k < 0 || k >= level_count) {
        for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x)
            out_row[t] = NAN;
        return;
    }
    const int offset     = 1 << k;
    const int level_base = level_offsets[k];

    // Parallelize over time: each thread computes a strided subset of t.
    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float c = close[t];
        if (isnan(c)) { out_row[t] = NAN; continue; }

        const int start = t - period + 1;

        // Any NaN in [start..t] -> NaN
        if (nan_psum[t + 1] - nan_psum[start] != 0) {
            out_row[t] = NAN;
            continue;
        }

        // O(1) range max/min via sparse table (two overlapping blocks).
        const int idx_a  = level_base + start;
        const int idx_b  = level_base + (t + 1 - offset);
        const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
        const float lmin = fminf(st_min[idx_a], st_min[idx_b]);

        const float denom = hmax - lmin;
        out_row[t] = (denom == 0.0f) ? 0.0f : ((hmax - c) / denom) * -100.0f;
    }
}

// Many-series × one-param (time-major) kernel — streamlined.
// One thread = one series. Warmup NaNs for [0..warm), then compute in place.
extern "C" __global__
void willr_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    if (period <= 0) {
        // Entire column becomes NaN
        for (int t = 0; t < rows; ++t) out_tm[t * cols + series] = NAN;
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + period - 1;

    // Warmup NaNs for prefix only
    const int wclamp = (warm < rows) ? warm : rows;
    for (int t = 0; t < wclamp; ++t)
        out_tm[t * cols + series] = NAN;

    if (warm >= rows) return;

    for (int t = warm; t < rows; ++t) {
        const int idx = t * cols + series;
        const float c = close_tm[idx];
        if (isnan(c)) { out_tm[idx] = NAN; continue; }

        const int start = t - period + 1;
        float h = -INFINITY, l = INFINITY;
        bool any_nan = false;

        // Naive scan (simple & branch-friendly). Early-out on NaN.
        for (int j = start; j <= t; ++j) {
            const int jidx = j * cols + series;
            const float hj = high_tm[jidx];
            const float lj = low_tm[jidx];
            if (isnan(hj) || isnan(lj)) { any_nan = true; break; }
            if (hj > h) h = hj;
            if (lj < l) l = lj;
        }

        if (any_nan) { out_tm[idx] = NAN; continue; }

        const float denom = h - l;
        out_tm[idx] = (denom == 0.0f) ? 0.0f : ((h - c) / denom) * -100.0f;
    }
}
