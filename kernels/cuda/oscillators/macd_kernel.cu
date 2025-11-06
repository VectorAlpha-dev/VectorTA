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

// --- small helpers (header-only, no extra compilation switches) ---
__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    // Kahan compensated add in FP32
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

// ===================================================================
// 1) One price series × many params (rows = combos, cols = len)
//    One thread per combo (grid-stride over combos).
// ===================================================================
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
    // Grid-stride over parameter rows (combos)
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x) {

        if (series_len <= 0) continue;

        const int fast   = fasts[combo];
        const int slow   = slows[combo];
        const int signal = signals[combo];

        if (fast <= 0 || slow <= 0 || signal <= 0) continue;

        int fv = first_valid;
        if (fv >= series_len) {
            // Nothing to compute; still need to leave outputs untouched
            continue;
        }
        fv = imax(fv, 0);

        const int row_base = combo * series_len;
        const int macd_warmup   = fv + slow - 1;
        const int signal_warmup = fv + slow + signal - 2;

        // Prefix NaNs (done by the same thread to avoid an extra pass / sync)
        const int macd_nan_end   = imin(macd_warmup, series_len);
        const int signal_nan_end = imin(signal_warmup, series_len);
        for (int i = 0; i < macd_nan_end; ++i) {
            macd_out[row_base + i] = NAN;
        }
        for (int i = 0; i < signal_nan_end; ++i) {
            signal_out[row_base + i] = NAN;
            hist_out[row_base + i]   = NAN;
        }

        // If the first MACD index is already beyond the series, we are done
        if (macd_warmup >= series_len) continue;

        // -------------------------
        // Seed EMAs via SMA windows
        // -------------------------
        // Use compensated sums for the SMA seeds; divide by window length (mirrors scalar path).
        float fsum = 0.f, fc = 0.f;
        const int fcap = imin(fast, series_len - fv);
        for (int i = 0; i < fcap; ++i) {
            kahan_add(prices[fv + i], fsum, fc);
        }
        float fast_ema = fsum / (float)fast;

        float ssum = 0.f, sc = 0.f;
        const int scap = imin(slow, series_len - fv);
        for (int i = 0; i < scap; ++i) {
            kahan_add(prices[fv + i], ssum, sc);
        }
        float slow_ema = ssum / (float)slow;

        // Precompute alphas (float only)
        const float af    = 2.0f / (fast   + 1.0f);
        const float aslow = 2.0f / (slow   + 1.0f);
        const float asig  = 2.0f / (signal + 1.0f);

        // Advance fast EMA up to macd_warmup to align with the slow window
        const int mwu = imin(macd_warmup, series_len - 1);
        for (int t = fv + fast; t <= mwu; ++t) {
            const float x = prices[t];
            if (isfinite(x)) {
                // fast_ema += af * (x - fast_ema)  in a single rounding step
                fast_ema = fmaf(x - fast_ema, af, fast_ema);
            }
        }

        // First MACD value at macd_warmup
        macd_out[row_base + macd_warmup] = fast_ema - slow_ema;

        // -------------------------
        // Seed signal
        // -------------------------
        bool  have_seed = false;
        float se        = 0.0f;

        // Accumulate for SMA seed (if signal > 1)
        float sig_acc = (signal > 1) ? macd_out[row_base + macd_warmup] : 0.0f;
        float sig_c   = 0.0f;  // Kahan compensation for signal SMA warmup

        if (signal == 1 && signal_warmup < series_len) {
            se = macd_out[row_base + signal_warmup];
            have_seed = true;
            signal_out[row_base + signal_warmup] = se;
            hist_out  [row_base + signal_warmup] = macd_out[row_base + signal_warmup] - se;
        }

        // -------------------------
        // Main forward pass (sequential per combo/thread)
        // -------------------------
        for (int k = macd_warmup + 1; k < series_len; ++k) {
            const float x = prices[k];
            if (isfinite(x)) {
                fast_ema = fmaf(x - fast_ema, af,    fast_ema);
                slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
            }
            const float m = fast_ema - slow_ema;
            macd_out[row_base + k] = m;

            if (!have_seed) {
                if (signal > 1 && k <= signal_warmup) {
                    kahan_add(m, sig_acc, sig_c);
                    if (k == signal_warmup) {
                        se = sig_acc / (float)signal;
                        have_seed = true;
                        signal_out[row_base + k] = se;
                        hist_out  [row_base + k] = m - se;
                    }
                }
            } else {
                // se += asig * (m - se)
                se = fmaf(m - se, asig, se);
                if (k >= signal_warmup) {
                    signal_out[row_base + k] = se;
                    hist_out  [row_base + k] = m - se;
                }
            }
        }
    }
}

// ===================================================================
// 2) Many series × one param (time-major). One thread per series.
//    Grid-stride over columns (series).
// ===================================================================
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
    if (rows <= 0 || cols <= 0 || fast <= 0 || slow <= 0 || signal <= 0) return;

    const int stride = cols; // time-major layout

    for (int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
         series_idx < cols;
         series_idx += blockDim.x * gridDim.x) {

        int fv = first_valids[series_idx];
        if (fv < 0) fv = 0;
        if (fv >= rows) continue;

        const int macd_warmup   = fv + slow - 1;
        const int signal_warmup = fv + slow + signal - 2;

        // Prefix NaNs for this series
        const int macd_nan_end   = imin(macd_warmup, rows);
        const int signal_nan_end = imin(signal_warmup, rows);
        for (int t = 0; t < macd_nan_end; ++t) {
            macd_tm[t * stride + series_idx] = NAN;
        }
        for (int t = 0; t < signal_nan_end; ++t) {
            signal_tm[t * stride + series_idx] = NAN;
            hist_tm  [t * stride + series_idx] = NAN;
        }

        if (macd_warmup >= rows) continue;

        // Seed EMAs via SMA over [fv .. fv+fast/slow)
        float fsum = 0.f, fc = 0.f;
        const int fcap = imin(fast, rows - fv);
        for (int i = 0; i < fcap; ++i) {
            kahan_add(prices_tm[(fv + i) * stride + series_idx], fsum, fc);
        }
        float fast_ema = fsum / (float)fast;

        float ssum = 0.f, sc = 0.f;
        const int scap = imin(slow, rows - fv);
        for (int i = 0; i < scap; ++i) {
            kahan_add(prices_tm[(fv + i) * stride + series_idx], ssum, sc);
        }
        float slow_ema = ssum / (float)slow;

        const float af    = 2.0f / (fast   + 1.0f);
        const float aslow = 2.0f / (slow   + 1.0f);
        const float asig  = 2.0f / (signal + 1.0f);

        // Advance fast EMA to macd_warmup
        const int mwu = imin(macd_warmup, rows - 1);
        for (int t = fv + fast; t <= mwu; ++t) {
            const float x = prices_tm[t * stride + series_idx];
            if (isfinite(x)) {
                fast_ema = fmaf(x - fast_ema, af, fast_ema);
            }
        }

        // First MACD
        macd_tm[macd_warmup * stride + series_idx] = fast_ema - slow_ema;

        // Signal seeding
        bool  have_seed = false;
        float se        = 0.0f;

        float sig_acc = (signal > 1) ? macd_tm[macd_warmup * stride + series_idx] : 0.0f;
        float sig_c   = 0.0f;

        if (signal == 1 && signal_warmup < rows) {
            se = macd_tm[signal_warmup * stride + series_idx];
            have_seed = true;
            signal_tm[signal_warmup * stride + series_idx] = se;
            hist_tm  [signal_warmup * stride + series_idx] = macd_tm[signal_warmup * stride + series_idx] - se;
        }

        // Main pass
        for (int t = macd_warmup + 1; t < rows; ++t) {
            const float x = prices_tm[t * stride + series_idx];
            if (isfinite(x)) {
                fast_ema = fmaf(x - fast_ema, af,    fast_ema);
                slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
            }
            const float m = fast_ema - slow_ema;
            macd_tm[t * stride + series_idx] = m;

            if (!have_seed) {
                if (signal > 1 && t <= signal_warmup) {
                    kahan_add(m, sig_acc, sig_c);
                    if (t == signal_warmup) {
                        se = sig_acc / (float)signal;
                        have_seed = true;
                        signal_tm[t * stride + series_idx] = se;
                        hist_tm  [t * stride + series_idx] = m - se;
                    }
                }
            } else {
                se = fmaf(m - se, asig, se);
                if (t >= signal_warmup) {
                    signal_tm[t * stride + series_idx] = se;
                    hist_tm  [t * stride + series_idx] = m - se;
                }
            }
        }
    }
}

