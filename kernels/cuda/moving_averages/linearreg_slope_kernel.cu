// CUDA kernels for Linear Regression Slope (outputs the slope b of the LS line).
// Pattern mirrors linreg_kernel.cu but writes b instead of the fitted value.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef LRS_NAN
#define LRS_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LRS_LAUNCH_BOUNDS
#define LRS_LAUNCH_BOUNDS 256, 2
#endif

extern "C" __global__
__launch_bounds__(LRS_LAUNCH_BOUNDS)
void linearreg_slope_batch_f32(const float* __restrict__ prices,
                               const int*   __restrict__ periods,
                               const float* __restrict__ x_sums,
                               const float* __restrict__ denom_invs,
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

        if (period < 2 || period > series_len || first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LRS_NAN;
            continue;
        }
        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int i = 0; i < series_len; ++i) out[base + i] = LRS_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        const double period_f  = static_cast<double>(period);
        const double x_sum     = static_cast<double>(x_sums[combo]);
        const double denom_inv = static_cast<double>(denom_invs[combo]);

        for (int i = 0; i < warm; ++i) out[base + i] = LRS_NAN;

        // Warm-up rolling sums for first period-1 elements (x = 1..period-1)
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const double v = static_cast<double>(prices[first_valid + k]);
            const double x = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }

        double latest = static_cast<double>(prices[warm]);
        for (int idx = warm; idx < series_len; ++idx) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum); // add latest * period

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            out[base + idx] = static_cast<float>(b);

            xy_sum -= y_sum; // shift weights by -1 => subtract y_sum
            const int oldest = idx - period + 1;
            y_sum  -= static_cast<double>(prices[oldest]);
            if (idx + 1 < series_len)
                latest = static_cast<double>(prices[idx + 1]);
        }
    }
}

static __device__ __forceinline__
int tm_idx_lrs(int row, int num_series, int series) {
    return row * num_series + series;
}

extern "C" __global__
__launch_bounds__(LRS_LAUNCH_BOUNDS)
void linearreg_slope_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                               const int*   __restrict__ first_valids,
                                               int num_series,
                                               int series_len,
                                               int period,
                                               float x_sum_f,
                                               float denom_inv_f,
                                               float* __restrict__ out_tm)
{
    const int stride = blockDim.x * gridDim.x;
    const double period_f  = static_cast<double>(period);
    const double x_sum     = static_cast<double>(x_sum_f);
    const double denom_inv = static_cast<double>(denom_inv_f);

    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_series; s += stride) {
        if (period < 2 || period > series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }
        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }
        const int tail_len = series_len - first_valid;
        if (tail_len < period) {
            for (int row = 0; row < series_len; ++row)
                out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;
            continue;
        }

        const int warm = first_valid + period - 1;
        for (int row = 0; row < warm; ++row)
            out_tm[tm_idx_lrs(row, num_series, s)] = LRS_NAN;

        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int k = 0; k < period - 1; ++k) {
            const int r = first_valid + k;
            const double v = static_cast<double>(prices_tm[tm_idx_lrs(r, num_series, s)]);
            const double x = static_cast<double>(k + 1);
            y_sum  += v;
            xy_sum  = fma(v, x, xy_sum);
        }
        double latest = static_cast<double>(prices_tm[tm_idx_lrs(warm, num_series, s)]);
        for (int row = warm; row < series_len; ++row) {
            y_sum  += latest;
            xy_sum  = fma(latest, period_f, xy_sum);

            const double b_num = fma(period_f, xy_sum, -x_sum * y_sum);
            const double b     = b_num * denom_inv;
            out_tm[tm_idx_lrs(row, num_series, s)] = static_cast<float>(b);

            xy_sum -= y_sum;
            const int oldest_row = row - period + 1;
            y_sum  -= static_cast<double>(prices_tm[tm_idx_lrs(oldest_row, num_series, s)]);
            if (row + 1 < series_len)
                latest = static_cast<double>(prices_tm[tm_idx_lrs(row + 1, num_series, s)]);
        }
    }
}

