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
    const double period_f = static_cast<double>(period);
    const double x_sum = static_cast<double>(x_sums[combo]);
    const double denom_inv = static_cast<double>(denom_invs[combo]);
    const double inv_period = static_cast<double>(inv_periods[combo]);

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int k = 0; k < period - 1; ++k) {
        const double val = static_cast<double>(prices[first_valid + k]);
        const double x = static_cast<double>(k + 1);
        y_sum += val;
        xy_sum += val * x;
    }

    for (int idx = warm; idx < series_len; ++idx) {
        const double latest = static_cast<double>(prices[idx]);
        y_sum += latest;
        xy_sum += latest * period_f;

        const double b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        const double a = (y_sum - b * x_sum) * inv_period;
        out[base + idx] = static_cast<float>(a + b * period_f);

        xy_sum -= y_sum;
        const int oldest = idx - period + 1;
        y_sum -= static_cast<double>(prices[oldest]);
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
    const double period_f = static_cast<double>(period);

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int k = 0; k < period - 1; ++k) {
        const int row = first_valid + k;
        const double val = static_cast<double>(prices_tm[idx(row)]);
        const double x = static_cast<double>(k + 1);
        y_sum += val;
        xy_sum += val * x;
    }

    for (int row = warm; row < series_len; ++row) {
        const double latest = static_cast<double>(prices_tm[idx(row)]);
        y_sum += latest;
        xy_sum += latest * period_f;

        const double x_sum_d = static_cast<double>(x_sum);
        const double denom_inv_d = static_cast<double>(denom_inv);
        const double inv_period_d = static_cast<double>(inv_period);
        const double b = (period_f * xy_sum - x_sum_d * y_sum) * denom_inv_d;
        const double a = (y_sum - b * x_sum_d) * inv_period_d;
        out_tm[idx(row)] = static_cast<float>(a + b * period_f);

        xy_sum -= y_sum;
        const int oldest_row = row - period + 1;
        y_sum -= static_cast<double>(prices_tm[idx(oldest_row)]);
    }
}
