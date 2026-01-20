








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef LINREG_INTERCEPT_NAN
#define LINREG_INTERCEPT_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LRI_LAUNCH_BOUNDS
#define LRI_LAUNCH_BOUNDS 256, 2
#endif


extern "C" __global__ void linearreg_intercept_exclusive_prefix_y_yi_f64(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    double* __restrict__ prefix_y,
    double* __restrict__ prefix_yi
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    if (first_valid < 0) first_valid = 0;
    if (first_valid > series_len) first_valid = series_len;

    prefix_y[0]  = 0.0;
    prefix_yi[0] = 0.0;

    double acc_y  = 0.0;
    double acc_yi = 0.0;
    for (int t = 0; t < series_len; ++t) {
        const double v = (t < first_valid) ? 0.0 : static_cast<double>(prices[t]);
        acc_y  += v;
        acc_yi  = fma(v, static_cast<double>(t), acc_yi);
        prefix_y[t + 1]  = acc_y;
        prefix_yi[t + 1] = acc_yi;
    }
}

extern "C" __global__
__launch_bounds__(LRI_LAUNCH_BOUNDS)
void linearreg_intercept_batch_from_prefix_f64(
    const double* __restrict__ prefix_y,
    const double* __restrict__ prefix_yi,
    const int*   __restrict__ periods,
    const float* __restrict__ x_sums,
    const float* __restrict__ denom_invs,
    const float* __restrict__ inv_periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.y);
    if (combo >= n_combos) return;

    const int period  = periods[combo];
    const int row_off = combo * series_len;

    int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);

    if (series_len <= 0) return;

    if (first_valid < 0 || first_valid >= series_len) {
        while (t < series_len) {
            out[row_off + t] = LINREG_INTERCEPT_NAN;
            t += stride;
        }
        return;
    }

    if (period <= 0 || period > series_len || (series_len - first_valid) < period) {
        while (t < series_len) {
            out[row_off + t] = LINREG_INTERCEPT_NAN;
            t += stride;
        }
        return;
    }

    const int warm = first_valid + period - 1;


    if (period == 1) {
        while (t < series_len) {
            if (t < warm) {
                out[row_off + t] = LINREG_INTERCEPT_NAN;
            } else {
                out[row_off + t] = static_cast<float>(prefix_y[t + 1] - prefix_y[t]);
            }
            t += stride;
        }
        return;
    }

    const double period_f   = static_cast<double>(period);
    const double x_sum      = static_cast<double>(x_sums[combo]);
    const double denom_inv  = static_cast<double>(denom_invs[combo]);
    const double inv_period = static_cast<double>(inv_periods[combo]);
    const double k          = 1.0 - x_sum * inv_period;

    while (t < series_len) {
        if (t < warm) {
            out[row_off + t] = LINREG_INTERCEPT_NAN;
        } else {
            const int t1    = t + 1;
            const int start = t1 - period;
            const double sum_y  = prefix_y[t1]  - prefix_y[start];
            const double sum_yi = prefix_yi[t1] - prefix_yi[start];
            const double xy_sum = fma((period_f - static_cast<double>(t)), sum_y, sum_yi);

            const double b_num = fma(period_f, xy_sum, -x_sum * sum_y);
            const double b     = b_num * denom_inv;
            out[row_off + t]   = static_cast<float>(sum_y * inv_period + b * k);
        }
        t += stride;
    }
}



extern "C" __global__
__launch_bounds__(LRI_LAUNCH_BOUNDS)
void linearreg_intercept_batch_f32(const float* __restrict__ prices,
                                   const int*   __restrict__ periods,
                                   const float* __restrict__ x_sums,
                                   const float* __restrict__ denom_invs,
                                   const float* __restrict__ inv_periods,
                                   int series_len,
                                   int n_combos,
                                   int first_valid,
                                   float* __restrict__ out)
{
    const int stride = blockDim.x * gridDim.x;

    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += stride)
    {
        const int base   = combo * series_len;
        const int period = periods[combo];


        if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LINREG_INTERCEPT_NAN;
            continue;
        }
        const int tail = series_len - first_valid;
        if (tail < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LINREG_INTERCEPT_NAN;
            continue;
        }

        const int    warm       = first_valid + period - 1;
        const double period_f   = static_cast<double>(period);
        const double x_sum      = static_cast<double>(x_sums[combo]);
        const double denom_inv  = static_cast<double>(denom_invs[combo]);
        const double inv_period = static_cast<double>(inv_periods[combo]);
        const double k          = 1.0 - x_sum * inv_period;


        for (int i = 0; i < warm; ++i) out[base + i] = LINREG_INTERCEPT_NAN;


        if (period == 1) {
            for (int idx = warm; idx < series_len; ++idx) out[base + idx] = prices[idx];
            continue;
        }


        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int kx = 0; kx < period - 1; ++kx) {
            const double v = static_cast<double>(prices[first_valid + kx]);
            const double x = static_cast<double>(kx + 1);
            y_sum += v;
            xy_sum = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices[warm]);

        for (int idx = warm; idx < series_len; ++idx) {
            y_sum += latest;
            xy_sum = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double y     = y_sum * inv_period + b * k;
            out[base + idx] = static_cast<float>(y);

            xy_sum -= y_sum;
            const int oldest = idx - period + 1;
            y_sum -= static_cast<double>(prices[oldest]);
            if (idx + 1 < series_len)
                latest = static_cast<double>(prices[idx + 1]);
        }
    }
}



static __device__ __forceinline__ int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(LRI_LAUNCH_BOUNDS)
void linearreg_intercept_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                                   const int*   __restrict__ first_valids,
                                                   int num_series,
                                                   int series_len,
                                                   int period,
                                                   float x_sum_f,
                                                   float denom_inv_f,
                                                   float inv_period_f,
                                                   float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const double period_f   = static_cast<double>(period);
    const double x_sum      = static_cast<double>(x_sum_f);
    const double denom_inv  = static_cast<double>(denom_inv_f);
    const double inv_period = static_cast<double>(inv_period_f);
    const double k          = 1.0 - x_sum * inv_period;

    for (int series = blockIdx.x * blockDim.x + threadIdx.x;
         series < num_series;
         series += stride)
    {
        if (period <= 0 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_INTERCEPT_NAN;
            continue;
        }

        const int first_valid = first_valids[series];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_INTERCEPT_NAN;
            continue;
        }
        const int tail = series_len - first_valid;
        if (tail < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = LINREG_INTERCEPT_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx(row, num_series, series)] = LINREG_INTERCEPT_NAN;

        if (period == 1) {
            for (int row = warm; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = prices_tm[tm_idx(row, num_series, series)];
            continue;
        }

        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int kx = 0; kx < period - 1; ++kx) {
            const int row = first_valid + kx;
            const double v = static_cast<double>(prices_tm[tm_idx(row, num_series, series)]);
            const double x = static_cast<double>(kx + 1);
            y_sum += v;
            xy_sum = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices_tm[tm_idx(warm, num_series, series)]);
        for (int row = warm; row < series_len; ++row) {
            y_sum += latest;
            xy_sum = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double y     = y_sum * inv_period + b * k;
            out_tm[tm_idx(row, num_series, series)] = static_cast<float>(y);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum -= static_cast<double>(prices_tm[tm_idx(oldest_row, num_series, series)]);
            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx(row + 1, num_series, series)]);
        }
    }
}

