// CUDA kernels for TrendFlex filter.
//
// Each parameter combination is processed by one thread in the batch kernel.
// We precompute the super smoother sequence into a scratch buffer and then
// apply the rolling volatility logic to fill the output row. A companion kernel
// handles the many-series × one-parameter scenario using the same two-pass
// strategy.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef TRENDFLEX_NAN
#define TRENDFLEX_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ float trendflex_round_half(float v) {
    return roundf(v);
}

extern "C" __global__ void trendflex_batch_f32(const float* __restrict__ prices,
                                               const int* __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ ssf_buf,
                                               float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int base = combo * series_len;
    float* __restrict__ row_out = out + base;
    float* __restrict__ row_ssf = ssf_buf + base;

    for (int i = 0; i < series_len; ++i) {
        row_out[i] = TRENDFLEX_NAN;
        row_ssf[i] = 0.0f;
    }

    if (period <= 0 || period >= series_len) {
        return;
    }
    if (first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const float PI = 3.14159265358979323846f;
    const float ROOT2 = 1.41421356237f;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) {
        ss_period = 1;
    }
    if (tail_len < ss_period) {
        return;
    }

    const float inv_ss = 1.0f / (float)ss_period;
    const float a = expf(-ROOT2 * PI * inv_ss);
    const float a_sq = a * a;
    const float b = 2.0f * a * cosf(ROOT2 * PI * inv_ss);
    const float c = (1.0f + a_sq - b) * 0.5f;

    // Build super smoother sequence in scratch buffer (aligned with output row)
    const int first_idx = first_valid;
    float prev2 = prices[first_idx];
    row_ssf[first_idx] = prev2;
    float prev1 = prev2;
    if (tail_len > 1) {
        prev1 = prices[first_idx + 1];
        row_ssf[first_idx + 1] = prev1;
    }

    for (int t = 2; t < tail_len; ++t) {
        const int idx = first_idx + t;
        const float cur_price = prices[idx];
        const float prev_price = prices[idx - 1];
        const float ss = c * (cur_price + prev_price) + b * prev1 - a_sq * prev2;
        row_ssf[idx] = ss;
        prev2 = prev1;
        prev1 = ss;
    }

    const int warm = first_valid + period;
    if (warm >= series_len) {
        return;
    }

    float rolling_sum = 0.0f;
    for (int t = 0; t < period; ++t) {
        rolling_sum += row_ssf[first_idx + t];
    }

    const float tp_f = (float)period;
    const float inv_tp = 1.0f / tp_f;
    float ms_prev = 0.0f;

    for (int idx = warm; idx < series_len; ++idx) {
        const float ss = row_ssf[idx];
        const float my_sum = (tp_f * ss - rolling_sum) * inv_tp;
        const float ms_current = 0.04f * my_sum * my_sum + 0.96f * ms_prev;
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0f) {
            out_val = my_sum / sqrtf(ms_current);
        }
        row_out[idx] = out_val;

        const float ss_old = row_ssf[idx - period];
        rolling_sum += ss - ss_old;
    }
}

extern "C" __global__ void trendflex_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ ssf_tm,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series];
    float* __restrict__ ssf_col = ssf_tm + series;
    float* __restrict__ out_col = out_tm + series;

    for (int row = 0; row < series_len; ++row) {
        out_col[row * stride] = TRENDFLEX_NAN;
        ssf_col[row * stride] = 0.0f;
    }

    if (period <= 0 || period >= series_len) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        return;
    }

    const float PI = 3.14159265358979323846f;
    const float ROOT2 = 1.41421356237f;

    int ss_period = (int)trendflex_round_half(0.5f * (float)period);
    if (ss_period < 1) {
        ss_period = 1;
    }
    if (tail_len < ss_period) {
        return;
    }

    const float inv_ss = 1.0f / (float)ss_period;
    const float a = expf(-ROOT2 * PI * inv_ss);
    const float a_sq = a * a;
    const float b = 2.0f * a * cosf(ROOT2 * PI * inv_ss);
    const float c = (1.0f + a_sq - b) * 0.5f;

    const int first_idx = first_valid;
    auto idx_tm = [stride, series](int row) { return row * stride + series; };

    float prev2 = prices_tm[idx_tm(first_idx)];
    ssf_col[idx_tm(first_idx)] = prev2;
    float prev1 = prev2;
    if (tail_len > 1) {
        prev1 = prices_tm[idx_tm(first_idx + 1)];
        ssf_col[idx_tm(first_idx + 1)] = prev1;
    }

    for (int t = 2; t < tail_len; ++t) {
        const int row = first_idx + t;
        const float cur_price = prices_tm[idx_tm(row)];
        const float prev_price = prices_tm[idx_tm(row - 1)];
        const float ss = c * (cur_price + prev_price) + b * prev1 - a_sq * prev2;
        ssf_col[idx_tm(row)] = ss;
        prev2 = prev1;
        prev1 = ss;
    }

    const int warm = first_valid + period;
    if (warm >= series_len) {
        return;
    }

    float rolling_sum = 0.0f;
    for (int t = 0; t < period; ++t) {
        rolling_sum += ssf_col[idx_tm(first_idx + t)];
    }

    const float tp_f = (float)period;
    const float inv_tp = 1.0f / tp_f;
    float ms_prev = 0.0f;

    for (int row = warm; row < series_len; ++row) {
        const float ss = ssf_col[idx_tm(row)];
        const float my_sum = (tp_f * ss - rolling_sum) * inv_tp;
        const float ms_current = 0.04f * my_sum * my_sum + 0.96f * ms_prev;
        ms_prev = ms_current;

        float out_val = 0.0f;
        if (ms_current > 0.0f) {
            out_val = my_sum / sqrtf(ms_current);
        }
        out_col[idx_tm(row)] = out_val;

        const float ss_old = ssf_col[idx_tm(row - period)];
        rolling_sum += ss - ss_old;
    }
}
