// CUDA kernels for the Time Series Forecast (TSF) indicator.
// TSF is equivalent to forecasting the next value from a rolling linear
// regression over a window of size `period`. The math matches the LINREG
// kernels; symbol names are TSF-specific.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef TSF_NAN
#define TSF_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef TSF_LAUNCH_BOUNDS
#define TSF_LAUNCH_BOUNDS 256, 2
#endif

// -------------------------- Batch kernel (one series × many params) --------------------------

extern "C" __global__
__launch_bounds__(TSF_LAUNCH_BOUNDS)
void tsf_batch_f32(const float* __restrict__ prices,
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

        // Guard invalid params as NaN row (wrapper should validate first).
        if (period <= 1 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = TSF_NAN;
            continue;
        }

        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = TSF_NAN;
            continue;
        }

        const int    warm       = first_valid + period - 1;
        const double period_f   = static_cast<double>(period);
        const double x_sum      = static_cast<double>(x_sums[combo]);
        const double denom_inv  = static_cast<double>(denom_invs[combo]);
        const double inv_period = static_cast<double>(inv_periods[combo]);

        // Prefix warmup = NaN
        for (int i = 0; i < warm; ++i) out[base + i] = TSF_NAN;

        // Initialize rolling sums over the first period-1 values (x = 1..period-1)
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const double val = static_cast<double>(prices[first_valid + k]);
            const double x   = static_cast<double>(k + 1);
            y_sum  += val;
            xy_sum  = fma(val, x, xy_sum);
        }

        // Prefetch the first fully-included value at index `warm`
        double latest = static_cast<double>(prices[warm]);

        // Main loop
        const double period_next = period_f + 1.0;
        for (int idx = warm; idx < series_len; ++idx) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;
            out[base + idx] = static_cast<float>(a + b * period_next);

            xy_sum -= y_sum;
            const int oldest = idx - period + 1;
            y_sum  -= static_cast<double>(prices[oldest]);

            if (idx + 1 < series_len) {
                latest = static_cast<double>(prices[idx + 1]);
            }
        }
    }
}

// -------------------------- Many-series × one param (time-major) --------------------------

static __device__ __forceinline__
int tm_idx(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(TSF_LAUNCH_BOUNDS)
void tsf_many_series_one_param_f32(const float* __restrict__ prices_tm,
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

    for (int series = blockIdx.x * blockDim.x + threadIdx.x;
         series < num_series;
         series += stride)
    {
        if (period <= 1 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        const int first_valid = first_valids[series];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        if (series_len - first_valid < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx(row, num_series, series)] = TSF_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx(row, num_series, series)] = TSF_NAN;

        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const int row   = first_valid + k;
            const double v  = static_cast<double>(prices_tm[tm_idx(row, num_series, series)]);
            const double x  = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices_tm[tm_idx(warm, num_series, series)]);

        const double period_next = period_f + 1.0;
        for (int row = warm; row < series_len; ++row) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            const double a     = (y_sum - b * x_sum) * inv_period;
            out_tm[tm_idx(row, num_series, series)] = static_cast<float>(a + b * period_next);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum  -= static_cast<double>(prices_tm[tm_idx(oldest_row, num_series, series)]);

            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx(row + 1, num_series, series)]);
        }
    }
}
