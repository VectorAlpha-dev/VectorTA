// CUDA kernel for Williams' %R (WILLR) batch evaluation.
//
// Each CUDA block processes a single period combination across the entire
// series. The implementation mirrors the scalar Rust semantics: outputs are
// warmup-prefixed with NaNs, any NaN in the high/low window yields NaN, and a
// zero denominator returns 0.0. Computation is performed in FP32 to match the
// existing GPU infrastructure (ALMA/DEMA).

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
    if (combo >= n_combos) {
        return;
    }

    const int base = combo * series_len;

    // Initialize output row with NaNs in parallel so warmup semantics match CPU.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // Single lane performs sequential window scans.
    if (threadIdx.x != 0) {
        return;
    }

    if (first_valid >= series_len) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0) {
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        return;
    }

    for (int t = warm; t < series_len; ++t) {
        const float c = close[t];
        if (isnan(c)) {
            out[base + t] = NAN;
            continue;
        }

        const int start = t - period + 1;
        if (nan_psum[t + 1] - nan_psum[start] != 0) {
            out[base + t] = NAN;
            continue;
        }

        const int window = period;
        int k = log2_tbl[window];
        if (k < 0 || k >= level_count) {
            out[base + t] = NAN;
            continue;
        }

        const int offset = 1 << k;
        const int level_base = level_offsets[k];
        const int idx_a = level_base + start;
        const int idx_b = level_base + (t + 1 - offset);
        const float h_max = fmaxf(st_max[idx_a], st_max[idx_b]);
        const float l_min = fminf(st_min[idx_a], st_min[idx_b]);

        if (!isfinite(h_max) || !isfinite(l_min)) {
            out[base + t] = NAN;
            continue;
        }

        const float denom = h_max - l_min;
        out[base + t] = (denom == 0.0f) ? 0.0f : (h_max - c) / denom * -100.0f;
    }
}

// Many-series × one-param (time-major) kernel.
//
// Layout:
// - Inputs are time-major matrices of shape [rows, cols] flattened, i.e.,
//   index = t * cols + series.
// - One thread handles one series (column) sequentially across time.
// - Warmup and NaN semantics mirror the scalar implementation:
//   * Output is NaN before first_valid[series] + period - 1.
//   * If any NaN appears in the [t-period+1..t] window of (high|low), result is NaN.
//   * If close[t] is NaN, result is NaN.
//   * If (max(high)-min(low)) == 0, write 0.0.
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
    if (series >= cols) {
        return;
    }

    // Prefill column with NaNs
    const float nan = nanf("");
    for (int t = 0; t < rows; ++t) {
        out_tm[t * cols + series] = nan;
    }

    if (period <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + period - 1;
    if (warm >= rows) {
        return;
    }

    for (int t = warm; t < rows; ++t) {
        const float c = close_tm[t * cols + series];
        if (isnan(c)) {
            // propagate NaN in close
            out_tm[t * cols + series] = nan;
            continue;
        }

        const int start = t - period + 1;
        // Scan window for max(high) and min(low); abort on NaN in either
        float h = -INFINITY;
        float l = INFINITY;
        bool any_nan = false;
        for (int j = start; j <= t; ++j) {
            const float hj = high_tm[j * cols + series];
            const float lj = low_tm[j * cols + series];
            if (isnan(hj) || isnan(lj)) {
                any_nan = true;
                break;
            }
            if (hj > h) h = hj;
            if (lj < l) l = lj;
        }

        if (any_nan || !isfinite(h) || !isfinite(l)) {
            out_tm[t * cols + series] = nan;
            continue;
        }

        const float denom = h - l;
        out_tm[t * cols + series] = (denom == 0.0f) ? 0.0f : (h - c) / denom * -100.0f;
    }
}
