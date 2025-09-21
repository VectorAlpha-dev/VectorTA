// CUDA kernels for the Square Root Weighted Moving Average (SRWMA).
//
// The batch variant processes a single price series against multiple period
// combinations, reusing precomputed square-root weights staged in shared
// memory. A companion kernel handles the time-major many-series × one-period
// flow used by the zero-copy Rust wrapper and Python bindings.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void srwma_batch_f32(const float* __restrict__ prices,
                     const float* __restrict__ weights_flat,
                     const int* __restrict__ periods,
                     const int* __restrict__ warm_indices,
                     const float* __restrict__ inv_norms,
                     int max_wlen,
                     int series_len,
                     int n_combos,
                     float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int warm = warm_indices[combo];
    if (period <= 1 || series_len <= 0) {
        return;
    }

    const int wlen = period - 1;
    const int row_offset = combo * series_len;

    extern __shared__ float shared_weights[];
    const int weight_base = combo * max_wlen;
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        shared_weights[k] = weights_flat[weight_base + k];
    }
    __syncthreads();

    const float inv_norm = inv_norms[combo];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = row_offset + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < wlen; ++k) {
                const int src_idx = t - k;
                if (src_idx < 0) {
                    break;
                }
                sum += prices[src_idx] * shared_weights[k];
            }
            out[out_idx] = sum * inv_norm;
        }
        t += stride;
    }
}

extern "C" __global__
void srwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int* __restrict__ first_valids,
                                     const float* __restrict__ weights,
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }
    if (period <= 1 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int wlen = period - 1;
    extern __shared__ float shared_weights[];
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        shared_weights[k] = weights[k];
    }
    __syncthreads();

    const int first_valid = first_valids[series_idx];
    const int warm = first_valid + period + 1;
    const int stride = num_series;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int offset = t * stride + series_idx;
        if (t < warm) {
            out_tm[offset] = NAN;
        } else {
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < wlen; ++k) {
                const int src_t = t - k;
                if (src_t < 0) {
                    break;
                }
                const float price = prices_tm[src_t * stride + series_idx];
                sum += price * shared_weights[k];
            }
            out_tm[offset] = sum * inv_norm;
        }
        t += step;
    }
}
