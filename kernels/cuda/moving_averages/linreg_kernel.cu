// CUDA kernels for the Linear Regression (LINREG) indicator.
//
// Each batch-thread processes one parameter combination sequentially to
// preserve the rolling window dependencies. A companion kernel supports the
// many-series × one-parameter scenario using time-major layout, matching the
// ergonomics provided by the Rust/Python zero-copy wrappers.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef LINREG_NAN
#define LINREG_NAN (__int_as_float(0x7fffffff))
#endif

extern "C" __global__ void linreg_batch_f32(const float* __restrict__ prices,
                                             const int* __restrict__ periods,
                                             const float* __restrict__ x_sums,
                                             const float* __restrict__ denom_invs,
                                             const float* __restrict__ inv_periods,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int base = combo * series_len;
    for (int idx = 0; idx < series_len; ++idx) {
        out[base + idx] = LINREG_NAN;
    }

    const int period = periods[combo];
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
    const float period_f = static_cast<float>(period);
    const float x_sum = x_sums[combo];
    const float denom_inv = denom_invs[combo];
    const float inv_period = inv_periods[combo];

    float y_sum = 0.0f;
    float xy_sum = 0.0f;
    for (int k = 0; k < period - 1; ++k) {
        const float val = prices[first_valid + k];
        const float x = static_cast<float>(k + 1);
        y_sum += val;
        xy_sum += val * x;
    }

    for (int idx = warm; idx < series_len; ++idx) {
        const float latest = prices[idx];
        y_sum += latest;
        xy_sum += latest * period_f;

        const float b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        const float a = (y_sum - b * x_sum) * inv_period;
        out[base + idx] = a + b * period_f;

        xy_sum -= y_sum;
        const int oldest = idx - period + 1;
        y_sum -= prices[oldest];
    }
}

extern "C" __global__ void linreg_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float x_sum,
    float denom_inv,
    float inv_period,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    auto idx = [num_series, series](int row) { return row * num_series + series; };

    for (int row = 0; row < series_len; ++row) {
        out_tm[idx(row)] = LINREG_NAN;
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
    const float period_f = static_cast<float>(period);

    float y_sum = 0.0f;
    float xy_sum = 0.0f;
    for (int k = 0; k < period - 1; ++k) {
        const int row = first_valid + k;
        const float val = prices_tm[idx(row)];
        const float x = static_cast<float>(k + 1);
        y_sum += val;
        xy_sum += val * x;
    }

    for (int row = warm; row < series_len; ++row) {
        const float latest = prices_tm[idx(row)];
        y_sum += latest;
        xy_sum += latest * period_f;

        const float b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        const float a = (y_sum - b * x_sum) * inv_period;
        out_tm[idx(row)] = a + b * period_f;

        xy_sum -= y_sum;
        const int oldest_row = row - period + 1;
        y_sum -= prices_tm[idx(oldest_row)];
    }
}
