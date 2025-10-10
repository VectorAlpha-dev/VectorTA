// CUDA kernels for the Sine Weighted Moving Average (SINWMA).
//
// The kernels mirror the ALMA/WMA GPU implementations: weights are generated
// on-device in shared memory so no host-side transfer is required, and all
// arithmetic stays in FP32. Two entry points are provided:
//   * sinwma_batch_f32:     single price series × many period choices
//   * sinwma_many_series_one_param_time_major_f32: many series (time-major)
//     sharing a single period.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void sinwma_batch_f32(const float* __restrict__ prices,
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
    if (period <= 0) {
        return;
    }

    extern __shared__ float weights[];
    __shared__ float norm;

    if (threadIdx.x == 0) {
        norm = 0.0f;
    }
    __syncthreads();

    const float denom = float(period + 1);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float angle = (float(i + 1) * CUDART_PI_F) / denom;
        const float w = sinf(angle);
        weights[i] = w;
        atomicAdd(&norm, w);
    }
    __syncthreads();

    if (norm <= 0.0f) {
        // Degenerate: fill warmup with NaN and post-warm with 0.0 to match
        // the "zero denominator -> 0.0" guidance (batch path).
        const int warm = first_valid + period - 1;
        const int base_out = combo * series_len;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            const int out_idx = base_out + t;
            out[out_idx] = (t < warm) ? NAN : 0.0f;
            t += stride;
        }
        return;
    }

    const float inv_norm = 1.0f / norm;
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

extern "C" __global__
void sinwma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    if (period <= 0) {
        return;
    }

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    extern __shared__ float weights[];
    __shared__ float norm;

    if (threadIdx.x == 0) {
        norm = 0.0f;
    }
    __syncthreads();

    const float denom = float(period + 1);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float angle = (float(i + 1) * CUDART_PI_F) / denom;
        const float w = sinf(angle);
        weights[i] = w;
        atomicAdd(&norm, w);
    }
    __syncthreads();

    if (norm <= 0.0f) {
        // Degenerate normalization: write warmup NaNs then 0.0s post-warm.
        const int warm = first_valids[series_idx] + period - 1;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            const int out_idx = t * num_series + series_idx;
            out_tm[out_idx] = (t < warm) ? NAN : 0.0f;
            t += stride;
        }
        return;
    }

    const float inv_norm = 1.0f / norm;

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
