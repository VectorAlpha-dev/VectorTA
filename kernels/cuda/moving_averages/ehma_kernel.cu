// CUDA kernels for Ehlers Hann Moving Average (EHMA).
//
// The batch kernel mirrors the ALMA/SWMA layout: each parameter combination
// resides in blockIdx.y while blockIdx.x/threadIdx.x stride across the time
// dimension. Hann weights are generated in shared memory per kernel launch so
// host code only transfers the price series and parameter metadata. A second
// kernel handles the many-series × one-parameter time-major path with
// pre-normalized weights supplied by the Rust wrapper.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float ehma_hann_weight(int period, int idx) {
    // idx is [0, period) where 0 indexes the oldest sample.
    const float pi = 3.14159265358979323846f;
    const float i = static_cast<float>(period - idx);
    const float angle = (2.0f * pi * i) / (static_cast<float>(period) + 1.0f);
    return 1.0f - cosf(angle);
}

// One-series × many-parameter kernel (batch mode).
extern "C" __global__
void ehma_batch_f32(const float* __restrict__ prices,
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

    __shared__ float norm;
    if (threadIdx.x == 0) {
        norm = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        const float wt = ehma_hann_weight(period, idx);
        weights[idx] = wt;
        atomicAdd(&norm, wt);
    }
    __syncthreads();

    if (norm <= 0.0f) {
        return;
    }
    const float inv_norm = 1.0f / norm;

    const int warm = warm_indices[combo];
    const int first = warm - period + 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm || (t - period + 1) < first) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc * inv_norm;
        }
        t += stride;
    }
}

// Many-series × one-parameter kernel (time-major input).
extern "C" __global__
void ehma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int first = first_valids[series_idx];
    const int warm = first + period - 1;

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
