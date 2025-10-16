// CUDA kernels for rolling Standard Deviation (population) indicator.
//
// Semantics mirror src/indicators/stddev.rs scalar path:
// - Warmup: output is NaN before warm = first_valid + period - 1
// - NaN handling: if any NaN exists in a window, output is NaN for that index
// - Variance <= 0 -> output 0.0 (scaled by nbdev still yields 0.0)
// - Accumulations in float64; outputs in float32

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ----------------- One-series × many-params (prefix-sum based) -----------------
// Uses prefix sums for x and x^2, and a prefix of NaN counts. Each block-y is
// a parameter row. Threads in x sweep time indices.
extern "C" __global__ void stddev_batch_f32(
    const double* __restrict__ ps_x,    // [len+1]
    const double* __restrict__ ps_x2,   // [len+1]
    const int*    __restrict__ ps_nan,  // [len+1]
    int len,
    int first_valid,
    const int* __restrict__ periods,    // [n_combos]
    const float* __restrict__ nbdevs,   // [n_combos]
    int n_combos,
    float* __restrict__ out             // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float nb = nbdevs[combo];

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_n = 1.0 / (double)period;

    while (t < len) {
        float outv = nan_f;
        if (t >= warm) {
            if (nb != 0.0f) {
                const int end = t + 1;
                int start = end - period;
                if (start < 0) start = 0;
                const int nan_count = ps_nan[end] - ps_nan[start];
                if (nan_count == 0) {
                    const double s1 = ps_x[end]  - ps_x[start];
                    const double s2 = ps_x2[end] - ps_x2[start];
                    const double mean = s1 * inv_n;
                    const double var  = (s2 * inv_n) - (mean * mean);
                    if (var > 0.0) {
                        outv = (float)(sqrt(var) * (double)nb);
                    } else {
                        outv = 0.0f; // parity with scalar: non-positive variance -> 0.0
                    }
                }
            } else {
                outv = 0.0f; // nbdev=0 => 0 regardless of var when valid
            }
        }
        out[row_off + t] = outv;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------
// Time-major layout [t][series]. Each block handles one series (column).
// Thread 0 performs the sequential sliding-window scan; other threads help
// initialize the output with NaNs.
extern "C" __global__ void stddev_many_series_one_param_f32(
    const float* __restrict__ data_tm,   // [rows * cols], time-major
    const int* __restrict__ first_valids,// [cols]
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm           // [rows * cols], time-major
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;

    // fill column with NaN cooperatively
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm = first_valid + period - 1;
    const double inv_n = 1.0 / (double)period;

    // Bootstrap raw sums over initial window [first_valid .. warm]
    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else {
            const double d = (double)v;
            s1 += d;
            s2 += d * d;
        }
    }

    if (warm < rows && nan_in_win == 0) {
        const double mean = s1 * inv_n;
        const double var  = (s2 * inv_n) - (mean * mean);
        out_tm[warm * stride + series] = (var > 0.0) ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
    }

    // Slide window forward one step at a time
    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {
            // Rebuild over the current window
            s1 = 0.0; s2 = 0.0; nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else { const double d = (double)vv; s1 += d; s2 += d * d; }
            }
        } else {
            // O(1) update of raw sums
            const double od = (double)old_v;
            const double nd = (double)new_v;
            s1 += nd - od;
            s2 += (nd * nd) - (od * od);
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            const double mean = s1 * inv_n;
            const double var  = (s2 * inv_n) - (mean * mean);
            out_tm[t * stride + series] = (var > 0.0) ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        }
    }
}

