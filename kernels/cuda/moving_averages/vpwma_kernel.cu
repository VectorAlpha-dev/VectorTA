// CUDA kernel for VPWMA (Variable Power Weighted Moving Average) batch evaluation.
//
// Each thread is responsible for a single parameter combination (row). We rely on
// host-side preprocessing to provide flattened weight buffers and inverse
// normalization factors so the kernel only performs the rolling dot products.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef VPWMA_NAN
#define VPWMA_NAN (__int_as_float(0x7fffffff))
#endif

extern "C" __global__
void vpwma_batch_f32(const float* __restrict__ prices,
                     const int* __restrict__ periods,
                     const int* __restrict__ win_lengths,
                     const float* __restrict__ weights,
                     const float* __restrict__ inv_norms,
                     int series_len,
                     int stride,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int win_len = win_lengths[combo];
    if (win_len <= 0 || period <= 1) {
        return;
    }

    const float inv_norm = inv_norms[combo];
    const int warm = first_valid + win_len;
    const int row_offset = combo * series_len;
    const int weight_offset = combo * stride;

    // Warmup prefix: everything before the first fully-formed window is NaN.
    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int i = 0; i < warm_clamped; ++i) {
        out[row_offset + i] = VPWMA_NAN;
    }

    if (warm >= series_len) {
        return;
    }

    const float* __restrict__ w_row = weights + weight_offset;
    const int start = first_valid + win_len;

    for (int idx = start; idx < series_len; ++idx) {
        float acc = 0.0f;
        for (int k = 0; k < win_len; ++k) {
            acc = fmaf(prices[idx - k], w_row[k], acc);
        }
        out[row_offset + idx] = acc * inv_norm;
    }
}

extern "C" __global__
void vpwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int* __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int period,
                                     const float* __restrict__ weights,
                                     float inv_norm,
                                     float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    const int first_valid = first_valids[series];
    const int win_len = period - 1;
    if (win_len <= 0) {
        return;
    }

    const int warm = first_valid + win_len;
    const int stride = num_series;

    for (int t = 0; t < series_len; ++t) {
        out_tm[t * stride + series] = VPWMA_NAN;
    }

    if (warm >= series_len) {
        return;
    }

    for (int t = warm; t < series_len; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < win_len; ++k) {
            const int idx = (t - k) * stride + series;
            acc = fmaf(prices_tm[idx], weights[k], acc);
        }
        out_tm[t * stride + series] = acc * inv_norm;
    }
}
