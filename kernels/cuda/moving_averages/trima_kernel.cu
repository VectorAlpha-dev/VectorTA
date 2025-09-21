// CUDA kernel for Triangular Moving Average (TRIMA) batch computation.
//
// Each block processes one parameter combination (period) across the entire
// time series. We preload the triangular weights for the target period into
// shared memory, then have threads stride across the timeline applying the
// weighted sum.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void trima_batch_f32(const float* __restrict__ prices,
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

    int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    const int warm = warm_indices[combo];
    const int first = warm - period + 1;

    extern __shared__ float weights[];

    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const float inv_norm = 1.0f / float(m1 * m2);

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        int w;
        if (idx < m1) {
            w = idx + 1;
        } else if (idx < m2) {
            w = m1;
        } else {
            w = (m1 + m2 - 1) - idx;
        }
        if (w < 0) {
            w = 0;
        }
        weights[idx] = float(w) * inv_norm;
    }
    __syncthreads();

    const int base_out = combo * series_len;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

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
void trima_multi_series_one_param_f32(const float* __restrict__ prices_tm,
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
    if (series_idx >= num_series) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int warm = first_valids[series_idx] + period - 1;

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
