// CUDA kernels for the Square Weighted Moving Average (SQWMA).
//
// Each parameter combination (period) maps to a block in the Y dimension.
// We precompute the squared weights in shared memory so that every output
// sample reuses the same coefficients instead of rebuilding them per index.
// The kernels operate in FP32 and mirror the warm-up semantics of the scalar
// Rust implementation: the first valid output appears at index
// `first_valid + period + 1`.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float sqwma_weight_sum(int period) {
    // Sum of squares from 2^2 up to period^2.
    double p = static_cast<double>(period);
    double sum = (p * (p + 1.0) * (2.0 * p + 1.0)) / 6.0 - 1.0;
    return static_cast<float>(sum);
}

extern "C" __global__
void sqwma_batch_f32(const float* __restrict__ prices,
                     const int* __restrict__ periods,
                     int series_len,
                     int n_combos,
                     int first_valid,
                     float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }
    if (series_len <= 0) {
        return;
    }

    const int period = periods[combo];
    if (period <= 1) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int base = combo * series_len;
        while (t < series_len) {
            out[base + t] = NAN;
            t += stride;
        }
        return;
    }

    const int weights_len = period - 1;
    const float inv_weight_sum = 1.0f / sqwma_weight_sum(period);
    const int warm = first_valid + period + 1;
    const int base_out = combo * series_len;

    extern __shared__ float weights[];
    for (int i = threadIdx.x; i < weights_len; i += blockDim.x) {
        const int w = period - i;
        weights[i] = static_cast<float>(w * w);
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float value = NAN;
        if (t >= warm && t < series_len) {
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < weights_len; ++k) {
                acc += prices[t - k] * weights[k];
            }
            value = acc * inv_weight_sum;
        }
        out[base_out + t] = value;
        t += stride;
    }
}

extern "C" __global__
void sqwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    if (period <= 1) {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            const int out_idx = t * num_series + series_idx;
            out_tm[out_idx] = NAN;
            t += stride;
        }
        return;
    }

    const int weights_len = period - 1;
    const float inv_weight_sum = 1.0f / sqwma_weight_sum(period);
    const int warm = first_valids[series_idx] + period + 1;

    extern __shared__ float weights[];
    for (int i = threadIdx.x; i < weights_len; i += blockDim.x) {
        const int w = period - i;
        weights[i] = static_cast<float>(w * w);
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        float value = NAN;
        if (t >= warm && t < series_len) {
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < weights_len; ++k) {
                const int idx = (t - k) * num_series + series_idx;
                acc += prices_tm[idx] * weights[k];
            }
            value = acc * inv_weight_sum;
        }
        out_tm[out_idx] = value;
        t += stride;
    }
}
