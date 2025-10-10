// CUDA kernels for the Weighted Moving Average (WMA).
//
// These kernels evaluate a single price series across multiple period choices
// or many series sharing a single period. They operate entirely in FP32 and
// precompute linear weights in shared memory per block, mirroring the on-device
// weighting strategy used by the ALMA CUDA implementation.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void wma_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 1) {
        return;
    }

    extern __shared__ float weights[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        weights[i] = float(i + 1);
    }
    __syncthreads();

    const float inv_norm = 2.0f / (float(period) * float(period + 1));
    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], weights[k], acc);
            }
            out[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}

// Prefix-sum variant for batch: uses A and B prefixes to compute windowed WMA in O(1).
extern "C" __global__
void wma_batch_prefix_f32(const float* __restrict__ pref_a,
                          const float* __restrict__ pref_b,
                          const int* __restrict__ periods,
                          int series_len,
                          int n_combos,
                          int first_valid,
                          float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const float inv_norm = 2.0f / (float(period) * float(period + 1));
    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            // Window [t - period + 1 .. t]
            const int start = t + 1 - period; // 1-based start index for prefixes
            const float s_a = pref_a[t + 1] - pref_a[start];
            const float s_b = pref_b[t + 1] - pref_b[start];
            const float wsum = fmaf(-float(t - period), s_a, s_b);
            out[out_idx] = wsum * inv_norm;
        }
        t += stride;
    }
}

extern "C" __global__
void wma_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    if (period <= 1) {
        return;
    }

    extern __shared__ float weights[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        weights[i] = float(i + 1);
    }
    __syncthreads();

    const float inv_norm = 2.0f / (float(period) * float(period + 1));

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
                acc = fmaf(prices_tm[in_idx], weights[k], acc);
            }
            out_tm[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}
