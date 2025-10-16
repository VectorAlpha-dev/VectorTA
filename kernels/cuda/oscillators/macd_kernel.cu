// CUDA kernels for MACD (EMA-only fast path)
//
// Implements two entry points:
//  - macd_batch_f32: one series × many params (rows = combos, cols = len)
//  - macd_many_series_one_param_f32: many series × one param (time-major)
//
// Behavior mirrors the scalar EMA path:
//  - Seed fast/slow EMAs by SMA windows starting at first_valid
//  - Advance fast EMA to align with slow window at macd_warmup = first + slow - 1
//  - First MACD at macd_warmup, then signal is SMA-seeded over `signal` MACD values
//  - Write NaN before warmups

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void macd_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ fasts,
                    const int*   __restrict__ slows,
                    const int*   __restrict__ signals,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ macd_out,
                    float* __restrict__ signal_out,
                    float* __restrict__ hist_out) {
    const int combo = blockIdx.x; // one block per parameter row
    if (combo >= n_combos || series_len <= 0) return;

    const int fast   = fasts[combo];
    const int slow   = slows[combo];
    const int signal = signals[combo];
    if (fast <= 0 || slow <= 0 || signal <= 0) return;
    if (first_valid >= series_len) return;

    const int row_base = combo * series_len;
    const int macd_warmup   = first_valid + slow - 1;
    const int signal_warmup = first_valid + slow + signal - 2;

    // Prefix NaNs
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        if (i < macd_warmup) macd_out[row_base + i] = NAN;
        if (i < signal_warmup) {
            signal_out[row_base + i] = NAN;
            hist_out[row_base + i]   = NAN;
        }
    }

    // Sequential section in thread 0
    if (threadIdx.x != 0) return;

    // Clamp end guards
    int mwu = macd_warmup < series_len ? macd_warmup : series_len - 1;
    int swu = signal_warmup < series_len ? signal_warmup : series_len - 1;

    // Seed fast EMA via SMA over [first_valid .. first_valid+fast)
    float fsum = 0.0f; for (int i = 0; i < fast && first_valid + i < series_len; ++i) fsum += prices[first_valid + i];
    float fast_ema = fsum / (float)fast;

    // Seed slow EMA via SMA over [first_valid .. first_valid+slow)
    float ssum = 0.0f; for (int i = 0; i < slow && first_valid + i < series_len; ++i) ssum += prices[first_valid + i];
    float slow_ema = ssum / (float)slow;

    // Advance fast EMA up to macd_warmup to align with slow window
    const float af = 2.0f / (fast + 1.0f);
    const float omf = 1.0f - af;
    const float aslow = 2.0f / (slow + 1.0f);
    const float oms   = 1.0f - aslow;
    const float asig  = 2.0f / (signal + 1.0f);
    const float omsi  = 1.0f - asig;

    for (int t = first_valid + fast; t <= mwu; ++t) {
        const float x = prices[t];
        if (isfinite(x)) fast_ema = fmaf(x - fast_ema, af, fast_ema);
    }

    // First MACD value at macd_warmup
    if (macd_warmup < series_len) {
        macd_out[row_base + macd_warmup] = fast_ema - slow_ema;
    }

    // Signal seeding
    bool have_seed = false;
    float se = 0.0f;
    if (signal == 1 && signal_warmup < series_len) {
        se = macd_out[row_base + signal_warmup];
        have_seed = true;
        signal_out[row_base + signal_warmup] = se;
        hist_out[row_base + signal_warmup] = macd_out[row_base + signal_warmup] - se;
    }
    float sig_accum = (signal > 1 && macd_warmup < series_len) ? macd_out[row_base + macd_warmup] : 0.0f;

    // Main forward pass
    for (int k = macd_warmup + 1; k < series_len; ++k) {
        const float x = prices[k];
        if (isfinite(x)) {
            fast_ema = fmaf(x - fast_ema, af, fast_ema);
            slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
        }
        const float m = fast_ema - slow_ema;
        macd_out[row_base + k] = m;

        if (!have_seed) {
            if (signal > 1 && k <= signal_warmup) {
                sig_accum += m;
                if (k == signal_warmup) {
                    se = sig_accum / (float)signal;
                    have_seed = true;
                    signal_out[row_base + k] = se;
                    hist_out[row_base + k] = m - se;
                }
            }
        } else {
            se = fmaf(m - se, asig, se); // se += asig*(m-se)
            if (k >= signal_warmup) {
                signal_out[row_base + k] = se;
                hist_out[row_base + k] = m - se;
            }
        }
    }
}

extern "C" __global__
void macd_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int cols,
                                    int rows,
                                    int fast,
                                    int slow,
                                    int signal,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ macd_tm,
                                    float* __restrict__ signal_tm,
                                    float* __restrict__ hist_tm) {
    const int series_idx = blockIdx.x; // one block per series
    if (series_idx >= cols || rows <= 0 || fast <= 0 || slow <= 0 || signal <= 0) return;

    const int stride = cols; // time-major layout
    int first_valid = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= rows) return;

    const int macd_warmup   = first_valid + slow - 1;
    const int signal_warmup = first_valid + slow + signal - 2;

    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        if (t < macd_warmup)   macd_tm[t * stride + series_idx]   = NAN;
        if (t < signal_warmup) {
            signal_tm[t * stride + series_idx] = NAN;
            hist_tm[t * stride + series_idx]   = NAN;
        }
    }
    if (threadIdx.x != 0) return;

    // Seed EMAs
    float fsum = 0.0f; for (int i = 0; i < fast && first_valid + i < rows; ++i) fsum += prices_tm[(first_valid + i) * stride + series_idx];
    float ssum = 0.0f; for (int i = 0; i < slow && first_valid + i < rows; ++i) ssum += prices_tm[(first_valid + i) * stride + series_idx];
    float fast_ema = fsum / (float)fast;
    float slow_ema = ssum / (float)slow;

    const float af = 2.0f / (fast + 1.0f);
    const float omf = 1.0f - af;
    const float aslow = 2.0f / (slow + 1.0f);
    const float oms   = 1.0f - aslow;
    const float asig  = 2.0f / (signal + 1.0f);

    for (int t = first_valid + fast; t <= macd_warmup && t < rows; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) fast_ema = fmaf(x - fast_ema, af, fast_ema);
    }
    if (macd_warmup < rows) macd_tm[macd_warmup * stride + series_idx] = fast_ema - slow_ema;

    bool have_seed = false; float se = 0.0f; float sig_accum = (signal > 1 && macd_warmup < rows) ? macd_tm[macd_warmup * stride + series_idx] : 0.0f;
    if (signal == 1 && signal_warmup < rows) {
        se = macd_tm[signal_warmup * stride + series_idx];
        have_seed = true;
        signal_tm[signal_warmup * stride + series_idx] = se;
        hist_tm[signal_warmup * stride + series_idx] = macd_tm[signal_warmup * stride + series_idx] - se;
    }

    for (int t = macd_warmup + 1; t < rows; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            fast_ema = fmaf(x - fast_ema, af, fast_ema);
            slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
        }
        const float m = fast_ema - slow_ema;
        macd_tm[t * stride + series_idx] = m;
        if (!have_seed) {
            if (signal > 1 && t <= signal_warmup) {
                sig_accum += m;
                if (t == signal_warmup) {
                    se = sig_accum / (float)signal; have_seed = true;
                    signal_tm[t * stride + series_idx] = se;
                    hist_tm[t * stride + series_idx] = m - se;
                }
            }
        } else {
            se = fmaf(m - se, asig, se);
            if (t >= signal_warmup) {
                signal_tm[t * stride + series_idx] = se;
                hist_tm[t * stride + series_idx] = m - se;
            }
        }
    }
}

