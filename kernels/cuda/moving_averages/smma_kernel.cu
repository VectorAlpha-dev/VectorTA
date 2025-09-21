// CUDA kernels for Smoothed Moving Average (SMMA).
//
// The batch kernel assigns one thread per parameter combination and walks the
// series sequentially (SMMA values depend on the previous output). A second
// kernel handles the many-series / one-parameter case using time-major input.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

extern "C" __global__
void smma_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const int* __restrict__ warm_indices,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0) {
        return;
    }

    const int warm = warm_indices[combo];
    const int base = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int idx = 0; idx < warm_clamped; ++idx) {
        out[base + idx] = nan_f;
    }

    if (warm >= series_len) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < period; ++k) {
        sum += prices[first_valid + k];
    }
    float prev = sum / static_cast<float>(period);
    out[base + warm] = prev;

    for (int t = warm + 1; t < series_len; ++t) {
        const float price = prices[t];
        prev = (prev * static_cast<float>(period - 1) + price) /
               static_cast<float>(period);
        out[base + t] = prev;
    }
}

extern "C" __global__
void smma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) {
        return;
    }

    const int first = first_valids[series_idx];
    if (period <= 0 || first < 0 || first >= series_len) {
        return;
    }

    const int warm = first + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[t * num_series + series_idx] = nan_f;
    }

    if (warm >= series_len) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < period; ++k) {
        const int idx = (first + k) * num_series + series_idx;
        sum += prices_tm[idx];
    }
    float prev = sum / static_cast<float>(period);
    out_tm[warm * num_series + series_idx] = prev;

    for (int t = warm + 1; t < series_len; ++t) {
        const int idx = t * num_series + series_idx;
        prev = (prev * static_cast<float>(period - 1) + prices_tm[idx]) /
               static_cast<float>(period);
        out_tm[idx] = prev;
    }
}
