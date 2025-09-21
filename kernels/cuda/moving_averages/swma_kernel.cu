// CUDA kernels for Symmetric Weighted Moving Average (SWMA).
//
// Mirrors the TRIMA layout: each batch kernel block precomputes normalized
// triangular weights in shared memory before striding across the timeline. A
// separate many-series × one-parameter kernel consumes time-major input.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// One-series × many-parameter kernel (batch path).
extern "C" __global__
void swma_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const int* __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    extern __shared__ float weights[];

    float norm;
    if (period <= 2) {
        // period == 1 -> norm = 1, period == 2 -> norm = 2
        norm = (period == 1) ? 1.0f : 2.0f;
    } else if (period % 2 == 0) {
        const float half = float(period / 2);
        norm = half * (half + 1.0f);
    } else {
        const float half_plus = float((period + 1) / 2);
        norm = half_plus * half_plus;
    }
    const float inv_norm = 1.0f / norm;

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        const int left = idx + 1;
        const int right = period - idx;
        const int w = left < right ? left : right;
        weights[idx] = float(w) * inv_norm;
    }
    __syncthreads();

    const int warm = warm_indices[combo];
    const int first = warm - period + 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm || t - period + 1 < first) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}

// Many-series × one-parameter kernel (time-major input).
extern "C" __global__
void swma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];

    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_weights[i] = weights[i];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                acc = fmaf(prices_tm[in_idx], shared_weights[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}
