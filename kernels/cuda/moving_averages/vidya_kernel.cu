// CUDA kernels for Variable Index Dynamic Average (VIDYA)
//
// Pattern: Recurrence/IIR per-parameter (or per-series) time scan.
// Warmup semantics mirror the scalar implementation in src/indicators/vidya.rs:
// - first_valid = index of first non-NaN input
// - warm_end = first_valid + long_period
// - The first defined output is at warm_end - 2
// - Indices [0, warm_end - 2) are set to NaN
// - Output at warm_end - 1 uses k computed from initial windows
// - Thereafter, rolling windows update by push/pop

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void vidya_batch_f32(const float* __restrict__ prices,
                     const int*   __restrict__ shorts,
                     const int*   __restrict__ longs,
                     const float* __restrict__ alphas,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int sp = shorts[combo];
    const int lp = longs[combo];
    if (sp <= 0 || lp <= 0 || first_valid >= series_len) return;
    const float alpha = alphas[combo];

    const int base = combo * series_len;

    // Warmup bounds
    const int warm_end = min(series_len, first_valid + lp);
    const int warmup_prefix = max(0, warm_end - 2); // first defined output index

    // Initialize prefix [0, warmup_prefix) to NaN
    for (int idx = threadIdx.x; idx < warmup_prefix; idx += blockDim.x) {
        out[base + idx] = NAN;
    }

    // Only a single thread performs the sequential scan
    if (threadIdx.x != 0) return;

    if (first_valid >= series_len) return;

    // Rolling accumulators for sums and sums of squares
    double long_sum  = 0.0;
    double long_sum2 = 0.0;
    double short_sum  = 0.0;
    double short_sum2 = 0.0;

    // Phase 1: accumulate long only until short window starts contributing
    const int short_head = max(first_valid, warm_end - sp);
    for (int i = first_valid; i < short_head; ++i) {
        const float x = prices[i];
        long_sum  += (double)x;
        long_sum2 += (double)x * (double)x;
    }
    // Phase 2: accumulate both long and short up to warm_end (exclusive)
    for (int i = short_head; i < warm_end; ++i) {
        const float x = prices[i];
        long_sum  += (double)x;
        long_sum2 += (double)x * (double)x;
        short_sum  += (double)x;
        short_sum2 += (double)x * (double)x;
    }

    if (warm_end - 2 >= 0 && warm_end - 2 < series_len) {
        float val = prices[warm_end - 2];
        out[base + (warm_end - 2)] = val;

        if (warm_end - 1 < series_len) {
            const double sp_inv = 1.0 / (double)sp;
            const double lp_inv = 1.0 / (double)lp;
            const double short_mean = short_sum * sp_inv;
            const double long_mean  = long_sum * lp_inv;
            const double short_var  = short_sum2 * sp_inv - short_mean * short_mean;
            const double long_var   = long_sum2 * lp_inv - long_mean * long_mean;
            double k = 0.0;
            if (long_var > 0.0 && short_var > 0.0) {
                k = sqrt(short_var / long_var) * (double)alpha;
            }
            const float x1 = prices[warm_end - 1];
            val = fmaf(x1 - val, (float)k, val);
            out[base + (warm_end - 1)] = val;

            // Main loop
            for (int t = warm_end; t < series_len; ++t) {
                const float x_new = prices[t];
                const double x2   = (double)x_new * (double)x_new;

                // push new
                long_sum  += (double)x_new;
                long_sum2 += x2;
                short_sum  += (double)x_new;
                short_sum2 += x2;

                // pop old
                const float x_long_out  = prices[t - lp];
                const float x_short_out = prices[t - sp];
                long_sum  -= (double)x_long_out;
                long_sum2 -= (double)x_long_out * (double)x_long_out;
                short_sum  -= (double)x_short_out;
                short_sum2 -= (double)x_short_out * (double)x_short_out;

                const double short_mean2 = short_sum * sp_inv;
                const double long_mean2  = long_sum * lp_inv;
                const double short_var2  = short_sum2 * sp_inv - short_mean2 * short_mean2;
                const double long_var2   = long_sum2 * lp_inv - long_mean2 * long_mean2;
                double k2 = 0.0;
                if (long_var2 > 0.0 && short_var2 > 0.0) {
                    k2 = sqrt(short_var2 / long_var2) * (double)alpha;
                }
                val = fmaf(x_new - val, (float)k2, val);
                out[base + t] = val;
            }
        }
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
    const int series_idx = blockIdx.x; // one block per series (compat mapping)
    if (series_idx >= num_series || series_len <= 0) return;
    if (short_period <= 0 || long_period <= 0) return;

    const int stride = num_series; // time-major layout: [t][series]
    int first_valid = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_end = min(series_len, first_valid + long_period);
    const int warmup_prefix = max(0, warm_end - 2);

    // Fill NaN prefix for this series
    for (int t = threadIdx.x; t < warmup_prefix; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }

    if (threadIdx.x != 0) return;

    // Accumulators
    double long_sum  = 0.0, long_sum2  = 0.0;
    double short_sum = 0.0, short_sum2 = 0.0;

    const int short_head = max(first_valid, warm_end - short_period);
    for (int t = first_valid; t < short_head; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        long_sum  += (double)x;
        long_sum2 += (double)x * (double)x;
    }
    for (int t = short_head; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        long_sum  += (double)x;
        long_sum2 += (double)x * (double)x;
        short_sum  += (double)x;
        short_sum2 += (double)x * (double)x;
    }

    if (warm_end - 2 >= 0 && warm_end - 2 < series_len) {
        float val = prices_tm[(warm_end - 2) * stride + series_idx];
        out_tm[(warm_end - 2) * stride + series_idx] = val;

        if (warm_end - 1 < series_len) {
            const double sp_inv = 1.0 / (double)short_period;
            const double lp_inv = 1.0 / (double)long_period;
            const double short_mean = short_sum * sp_inv;
            const double long_mean  = long_sum * lp_inv;
            const double short_var  = short_sum2 * sp_inv - short_mean * short_mean;
            const double long_var   = long_sum2 * lp_inv - long_mean * long_mean;
            double k = 0.0;
            if (long_var > 0.0 && short_var > 0.0) {
                k = sqrt(short_var / long_var) * (double)alpha;
            }
            const float x1 = prices_tm[(warm_end - 1) * stride + series_idx];
            val = fmaf(x1 - val, (float)k, val);
            out_tm[(warm_end - 1) * stride + series_idx] = val;

            for (int t = warm_end; t < series_len; ++t) {
                const float x_new = prices_tm[t * stride + series_idx];
                const double x2   = (double)x_new * (double)x_new;

                long_sum  += (double)x_new;
                long_sum2 += x2;
                short_sum  += (double)x_new;
                short_sum2 += x2;

                const float x_long_out  = prices_tm[(t - long_period) * stride + series_idx];
                const float x_short_out = prices_tm[(t - short_period) * stride + series_idx];
                long_sum  -= (double)x_long_out;
                long_sum2 -= (double)x_long_out * (double)x_long_out;
                short_sum  -= (double)x_short_out;
                short_sum2 -= (double)x_short_out * (double)x_short_out;

                const double short_mean2 = short_sum * sp_inv;
                const double long_mean2  = long_sum * lp_inv;
                const double short_var2  = short_sum2 * sp_inv - short_mean2 * short_mean2;
                const double long_var2   = long_sum2 * lp_inv - long_mean2 * long_mean2;
                double k2 = 0.0;
                if (long_var2 > 0.0 && short_var2 > 0.0) {
                    k2 = sqrt(short_var2 / long_var2) * (double)alpha;
                }
                val = fmaf(x_new - val, (float)k2, val);
                out_tm[t * stride + series_idx] = val;
            }
        }
    }
}

