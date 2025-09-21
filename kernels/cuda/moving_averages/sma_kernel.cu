// CUDA kernels for the Simple Moving Average (SMA).
//
// Each batch kernel thread evaluates one parameter combination sequentially.
// The many-series variant operates on time-major input (rows = time, columns =
// series) and reuses precomputed first-valid indices supplied by the host.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef SMA_NAN
#define SMA_NAN (__int_as_float(0x7fffffff))
#endif

extern "C" __global__ void sma_batch_f32(const float* __restrict__ prices,
                                          const int* __restrict__ periods,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int base = combo * series_len;

    for (int idx = 0; idx < series_len; ++idx) {
        out[base + idx] = SMA_NAN;
    }

    if (period <= 0 || period > series_len) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const int warm = first_valid + period - 1;
    const float inv = 1.0f / static_cast<float>(period);

    if (period == 1) {
        for (int idx = first_valid; idx < series_len; ++idx) {
            out[base + idx] = prices[idx];
        }
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < period; ++k) {
        sum += prices[first_valid + k];
    }
    out[base + warm] = sum * inv;

    for (int idx = warm + 1; idx < series_len; ++idx) {
        sum += prices[idx];
        sum -= prices[idx - period];
        out[base + idx] = sum * inv;
    }
}

extern "C" __global__ void sma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    auto idx = [num_series, series](int row) { return row * num_series + series; };

    for (int row = 0; row < series_len; ++row) {
        out_tm[idx(row)] = SMA_NAN;
    }

    if (period <= 0 || period > series_len) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const int warm = first_valid + period - 1;
    const float inv = 1.0f / static_cast<float>(period);

    if (period == 1) {
        for (int row = first_valid; row < series_len; ++row) {
            out_tm[idx(row)] = prices_tm[idx(row)];
        }
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < period; ++k) {
        sum += prices_tm[idx(first_valid + k)];
    }
    out_tm[idx(warm)] = sum * inv;

    for (int row = warm + 1; row < series_len; ++row) {
        sum += prices_tm[idx(row)];
        sum -= prices_tm[idx(row - period)];
        out_tm[idx(row)] = sum * inv;
    }
}
