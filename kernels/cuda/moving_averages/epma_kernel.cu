// CUDA kernels for the Endpoint Moving Average (EPMA).
//
// Follows the VRAM-first approach used across the moving averages suite:
// kernels operate on FP32 device buffers while using FP64 accumulators to
// reduce drift versus the CPU reference. The batch variant evaluates a single
// price series across multiple (period, offset) combinations, whereas the
// many-series variant processes a time-major matrix where all columns share a
// common parameter pair.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {
__device__ __forceinline__ double epma_weight_at(int period_minus_one,
                                                 int offset,
                                                 int idx) {
    // Matches build_weights_rev in the scalar Rust implementation.
    const int rev = period_minus_one - 1 - idx;
    return static_cast<double>(period_minus_one + 1 - rev - offset);
}
}

extern "C" __global__
void epma_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const int* __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1 = period - 1;
    if (p1 <= 0) {
        return;
    }

    extern __shared__ float weights[];
    __shared__ double weight_sum_shared;

    for (int k = threadIdx.x; k < p1; k += blockDim.x) {
        const double w = epma_weight_at(p1, offset, k);
        weights[k] = static_cast<float>(w);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int k = 0; k < p1; ++k) {
            sum += static_cast<double>(weights[k]);
        }
        weight_sum_shared = sum;
    }
    __syncthreads();

    const double weight_sum = weight_sum_shared;
    const double inv_weight_sum = (weight_sum == 0.0) ? 0.0 : (1.0 / weight_sum);
    const int warm = first_valid + period + offset + 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            const int start = t + 1 - p1;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < p1; ++k) {
                const double price = static_cast<double>(prices[start + k]);
                const double weight = static_cast<double>(weights[k]);
                acc = fma(price, weight, acc);
            }
            out[out_idx] = static_cast<float>(acc * inv_weight_sum);
        }
        t += stride;
    }
}

extern "C" __global__
void epma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int p1 = period - 1;
    if (p1 <= 0) {
        return;
    }

    extern __shared__ float weights[];
    __shared__ double weight_sum_shared;

    for (int k = threadIdx.x; k < p1; k += blockDim.x) {
        const double w = epma_weight_at(p1, offset, k);
        weights[k] = static_cast<float>(w);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int k = 0; k < p1; ++k) {
            sum += static_cast<double>(weights[k]);
        }
        weight_sum_shared = sum;
    }
    __syncthreads();

    const double weight_sum = weight_sum_shared;
    const double inv_weight_sum = (weight_sum == 0.0) ? 0.0 : (1.0 / weight_sum);

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period + offset + 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t + 1 - p1;
            double acc = 0.0;
#pragma unroll 4
            for (int k = 0; k < p1; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                const double price = static_cast<double>(prices_tm[in_idx]);
                const double weight = static_cast<double>(weights[k]);
                acc = fma(price, weight, acc);
            }
            out_tm[out_idx] = static_cast<float>(acc * inv_weight_sum);
        }
        t += stride;
    }
}
