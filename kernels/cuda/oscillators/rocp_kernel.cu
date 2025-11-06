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

// ------------------------------
// Helpers
// ------------------------------
#ifndef ROCP_QNAN
// Canonical quiet NaN payload (IEEE-754): 0x7fc00000
__device__ __forceinline__ float rocp_qnan() {
    return __int_as_float(0x7fc00000);
}
#define ROCP_QNAN rocp_qnan()
#endif

// One-series × many-params (batch), FP32 only.
// Each block handles one period combo (row).
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

    const int start = first_valid + period; // first valid output index

    // Write only the warm-up prefix to NaN. Avoid full-row initialization.
    const int warm = (start < len) ? start : len;
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = ROCP_QNAN;
    }

    if (start >= len) return; // nothing to compute

    // Grid-stride loop with 4x unroll for ILP.
    int t = start + threadIdx.x;
    const int stride = blockDim.x;

    // Main unrolled body: process 4 elements per iteration.
    for (; t + 3*stride < len; t += 4*stride) {
        const float c0  = data[t];
        const float ip0 = inv[t - period];
        out[base + t] = fmaf(c0, ip0, -1.0f); // (c/prev) - 1

        const int t1 = t + stride;
        const float c1  = data[t1];
        const float ip1 = inv[t1 - period];
        out[base + t1] = fmaf(c1, ip1, -1.0f);

        const int t2 = t + 2*stride;
        const float c2  = data[t2];
        const float ip2 = inv[t2 - period];
        out[base + t2] = fmaf(c2, ip2, -1.0f);

        const int t3 = t + 3*stride;
        const float c3  = data[t3];
        const float ip3 = inv[t3 - period];
        out[base + t3] = fmaf(c3, ip3, -1.0f);
    }

    // Tail
    for (; t < len; t += stride) {
        const float c  = data[t];
        const float ip = inv[t - period];
        out[base + t] = fmaf(c, ip, -1.0f);
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
            out[t * cols + s] = ROCP_QNAN;
        }
        return;
    }

    const int warm = first + period; // first output index
    // Prefix NaNs
    const int limit = (warm < rows) ? warm : rows;
    for (int t = 0; t < limit; ++t) {
        out[t * cols + s] = ROCP_QNAN;
    }
    if (warm >= rows) return;

    // Compute remainder sequentially for this series
    for (int t = warm; t < rows; ++t) {
        const float c = data_tm[t * cols + s];
        const float p = data_tm[(t - period) * cols + s];
        out[t * cols + s] = (c - p) / p; // match scalar ROCP exactly
    }
}

