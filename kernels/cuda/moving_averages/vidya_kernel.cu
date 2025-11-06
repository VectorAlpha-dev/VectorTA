// CUDA kernels for Variable Index Dynamic Average (VIDYA)
//
// Pattern: recurrence/IIR. Each block handles one parameter combo (batch)
// or one series (many-series, time-major). Thread 0 performs the sequential
// scan; other threads parallelize prefix NaN initialization.
//
// Semantics are aligned with the scalar Rust implementation in
// src/indicators/vidya.rs:
// - warmup index = first_valid + long_period - 2
// - out[warmup-2] = price[warmup-2]
// - out[warmup-1] = EMA-style update using k = alpha * (short_std / long_std)
// - Thereafter, rolling updates for sums/sumsq and EMA-style recurrence.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void vidya_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ short_periods,
                     const int*   __restrict__ long_periods,
                     const float* __restrict__ alphas,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    const float alpha = alphas[combo];
    const int base = combo * series_len;

    // Basic validation mirroring CPU-side guards; on invalid, fill NaNs.
    bool invalid = (sp < 2) || (lp < sp) || (lp < 2) || (alpha < 0.0f) || (alpha > 1.0f) ||
                   (first_valid < 0) || (first_valid >= series_len) ||
                   (lp > (series_len - first_valid));

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = CUDART_NAN_F;
        }
        return;
    }

    const int warm_end = first_valid + lp; // first index after initial window
    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;
    const int warmup_prefix = idx_m2; // exclusive end for NaN prefix

    // Prefix NaN initialization [0, warmup_prefix)
    for (int i = threadIdx.x; i < warmup_prefix; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }

    if (threadIdx.x != 0) return;

    // Rolling accumulators over initial window [first_valid .. warm_end)
    double long_sum = 0.0;
    double long_sum2 = 0.0;
    double short_sum = 0.0;
    double short_sum2 = 0.0;

    const int short_head = warm_end - sp;
    for (int i = first_valid; i < short_head; ++i) {
        const double x = static_cast<double>(prices[i]);
        long_sum += x;
        long_sum2 += x * x;
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = static_cast<double>(prices[i]);
        long_sum += x;
        long_sum2 += x * x;
        short_sum += x;
        short_sum2 += x * x;
    }

    // First two defined outputs
    float val = prices[idx_m2];
    out[base + idx_m2] = val;

    if (idx_m1 < series_len) {
        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);

        const float x = prices[idx_m1];
        val = fmaf(x - val, static_cast<float>(k), val);
        out[base + idx_m1] = val;
    }

    // Main rolling loop
    for (int t = warm_end; t < series_len; ++t) {
        const double x_new = static_cast<double>(prices[t]);
        const double x_new2 = x_new * x_new;

        // push new
        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;

        // pop old
        const double x_long_out = static_cast<double>(prices[t - lp]);
        const double x_short_out = static_cast<double>(prices[t - sp]);
        long_sum -= x_long_out;
        long_sum2 -= x_long_out * x_long_out;
        short_sum -= x_short_out;
        short_sum2 -= x_short_out * x_short_out;

        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);

        const float x = prices[t];
        val = fmaf(x - val, static_cast<float>(k), val);
        out[base + t] = val;
    }
}

extern "C" __global__
void vidya_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
                                     int short_period,
                                     int long_period,
                                     float alpha,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x; // one block per series (compat mode)
    if (series_idx >= num_series || series_len <= 0) return;

    const int sp = short_period;
    const int lp = long_period;
    int first_valid = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const bool invalid = (sp < 2) || (lp < sp) || (lp < 2) || (alpha < 0.0f) || (alpha > 1.0f) ||
                         (lp > (series_len - first_valid));
    const int stride = num_series; // time-major

    if (invalid) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = CUDART_NAN_F;
        }
        return;
    }

    const int warm_end = first_valid + lp;
    const int idx_m2 = warm_end - 2;
    const int idx_m1 = warm_end - 1;

    // Prefix NaN per series
    for (int t = threadIdx.x; t < idx_m2; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }

    if (threadIdx.x != 0) return;

    // Rolling accumulators over initial window
    double long_sum = 0.0;
    double long_sum2 = 0.0;
    double short_sum = 0.0;
    double short_sum2 = 0.0;
    const int short_head = warm_end - sp;
    for (int i = first_valid; i < short_head; ++i) {
        const double x = static_cast<double>(prices_tm[i * stride + series_idx]);
        long_sum += x;
        long_sum2 += x * x;
    }
    for (int i = short_head; i < warm_end; ++i) {
        const double x = static_cast<double>(prices_tm[i * stride + series_idx]);
        long_sum += x;
        long_sum2 += x * x;
        short_sum += x;
        short_sum2 += x * x;
    }

    float val = prices_tm[idx_m2 * stride + series_idx];
    out_tm[idx_m2 * stride + series_idx] = val;

    if (idx_m1 < series_len) {
        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - (short_mean * short_mean);
        const double long_var  = long_sum2 * long_inv - (long_mean * long_mean);
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);
        const float x = prices_tm[idx_m1 * stride + series_idx];
        val = fmaf(x - val, static_cast<float>(k), val);
        out_tm[idx_m1 * stride + series_idx] = val;
    }

    for (int t = warm_end; t < series_len; ++t) {
        const double x_new = static_cast<double>(prices_tm[t * stride + series_idx]);
        const double x_new2 = x_new * x_new;
        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;
        const double x_long_out = static_cast<double>(prices_tm[(t - lp) * stride + series_idx]);
        const double x_short_out = static_cast<double>(prices_tm[(t - sp) * stride + series_idx]);
        long_sum -= x_long_out;
        long_sum2 -= x_long_out * x_long_out;
        short_sum -= x_short_out;
        short_sum2 -= x_short_out * x_short_out;

        const double short_inv = 1.0 / static_cast<double>(sp);
        const double long_inv  = 1.0 / static_cast<double>(lp);
        const double short_mean = short_sum * short_inv;
        const double long_mean  = long_sum * long_inv;
        const double short_var = short_sum2 * short_inv - short_mean * short_mean;
        const double long_var  = long_sum2 * long_inv - long_mean * long_mean;
        const double short_std = sqrt(fmax(0.0, short_var));
        const double long_std  = sqrt(fmax(0.0, long_var));
        double k = (long_std == 0.0) ? 0.0 : (short_std / long_std);
        k *= static_cast<double>(alpha);
        const float x = prices_tm[t * stride + series_idx];
        val = fmaf(x - val, static_cast<float>(k), val);
        out_tm[t * stride + series_idx] = val;
    }
}

