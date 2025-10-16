// ROCP (Rate of Change Percentage without 100x) CUDA kernels
//
// Semantics follow src/indicators/rocp.rs exactly:
// - Output length equals input length
// - Warmup prefix: NaN up to index (first_valid + period - 1); first valid at (first_valid + period)
// - No special-case for zero denominator in ROCP (unlike ROC); IEEE-754 division semantics apply
// - NaNs in inputs naturally propagate

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// One-series × many-params (batch). Each block handles one period combo.
// Inputs:
//  - data: price series (len)
//  - inv: reciprocals of price series (len), i.e., inv[i] = 1.0f / data[i]
//  - periods: per-row period values (n_combos)
//  - len: length of series
//  - first_valid: index of first non-NaN in data
//  - n_combos: number of parameter rows
// Output row-major: out[row * len + t]
extern "C" __global__
void rocp_batch_f32(const float* __restrict__ data,
                    const float* __restrict__ inv,
                    const int* __restrict__ periods,
                    int len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= n_combos) return;
    const int period = periods[row];
    if (period <= 0) return;

    const int base = row * len;

    // Initialize row to NaN in parallel to match warmup semantics
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    const int start = first_valid + period;
    if (start >= len) return;

    // Parallel compute across time with striding threads
    for (int t = start + threadIdx.x; t < len; t += blockDim.x) {
        const float c = data[t];
        const float ip = inv[t - period]; // 1/prev
        out[base + t] = fmaf(c, ip, -1.0f); // (c/prev) - 1
    }
}

// Many-series × one-param, time-major layout.
// Inputs:
//  - data_tm: time-major [rows x cols] (index = t*cols + s)
//  - firsts: per-series first_valid indices (len = cols)
//  - cols: number of series
//  - rows: number of timesteps
//  - period: lookback window
// Output time-major in-place: out[t*cols + s]
extern "C" __global__
void rocp_many_series_one_param_f32(const float* __restrict__ data_tm,
                                    const int* __restrict__ firsts,
                                    int cols,
                                    int rows,
                                    int period,
                                    float* __restrict__ out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols || period <= 0) return;

    const int first = firsts[s];
    if (first >= rows) {
        // Initialize column to NaN (rare case); keep behavior consistent
        for (int t = 0; t < rows; ++t) {
            out[t * cols + s] = NAN;
        }
        return;
    }

    const int warm = first + period; // first output index
    // Prefix NaNs
    for (int t = 0; t < warm && t < rows; ++t) {
        out[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    // Compute remainder sequentially for this series
    for (int t = warm; t < rows; ++t) {
        const float c = data_tm[t * cols + s];
        const float p = data_tm[(t - period) * cols + s];
        out[t * cols + s] = (c - p) / p; // match scalar ROCP exactly
    }
}

