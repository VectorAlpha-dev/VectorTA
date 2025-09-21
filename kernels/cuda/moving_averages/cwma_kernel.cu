// CUDA kernels for the Cubic Weighted Moving Average (CWMA).
//
// These kernels operate purely in single precision and mirror the scalar
// implementation used by the Rust CWMA indicator. The batch kernel handles a
// single price series evaluated against multiple period choices, while the
// time-major kernel supports many series (columns) sharing one parameter set.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void cwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int* __restrict__ periods,
                    const float* __restrict__ inv_norms,
                    int max_period,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int weight_len = (period > 0) ? period - 1 : 0;
    const float inv_norm = inv_norms[combo];

    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < weight_len; i += blockDim.x) {
        shared_weights[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                sum = fmaf(prices[t - k], shared_weights[k], sum);
            }
            out[out_idx] = sum * inv_norm;
        }
        t += stride;
    }
}

extern "C" __global__
void cwma_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ weights,
    int period,
    float inv_norm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int weight_len = (period > 0) ? period - 1 : 0;

    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < weight_len; i += blockDim.x) {
        shared_weights[i] = weights[i];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                const int in_idx = (t - k) * num_series + series_idx;
                sum = fmaf(prices_tm[in_idx], shared_weights[k], sum);
            }
            out_tm[out_idx] = sum * inv_norm;
        }
        t += stride;
    }
}
