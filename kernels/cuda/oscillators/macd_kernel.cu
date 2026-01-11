// CUDA kernels for MACD (EMA-only fast path)
//
// Implements two entry points:
//  - macd_batch_f32: one series x many params (rows = combos, cols = len)
//  - macd_many_series_one_param_f32: many series x one param (time-major)
//
// Behavior mirrors the scalar EMA path:
//  - Seed fast/slow EMAs by SMA windows starting at first_valid
//  - Advance fast EMA to align with slow window at macd_warmup = first + slow - 1
//  - First MACD at macd_warmup, then signal is SMA-seeded over `signal` MACD values
//  - Write NaN before warmups
//
// Batch kernel is optimized for the primary use-case (one series x many params):
// - One warp per parameter row (combo)
// - Warp-level affine scans compute EMA recurrences 32 timesteps at a time
// - Row-major outputs remain coalesced within each warp

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

// Warp-wide inclusive scan over affine transforms (A,B) where:
//   y = A * y_prev + B
// Composition is associative:
//   (A1,B1) ∘ (A2,B2) = (A1*A2, A1*B2 + B1)
__device__ __forceinline__ void warp_inclusive_scan_affine(float &A, float &B, unsigned lane, unsigned mask) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float A_prev = __shfl_up_sync(mask, A, offset);
        const float B_prev = __shfl_up_sync(mask, B, offset);
        if (lane >= static_cast<unsigned>(offset)) {
            const float A_cur = A;
            const float B_cur = B;
            A = A_cur * A_prev;
            B = __fmaf_rn(A_cur, B_prev, B_cur);
        }
    }
}

// ===================================================================
// 1) One price series x many params (rows = combos, cols = len)
//    Warp-per-combo with warp-scan for EMA recurrences.
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
    if (series_len <= 0) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int combo = blockIdx.x * warps_per_block + warp_id;
    if (combo >= n_combos) return;
    const unsigned mask = 0xffffffffu;

    const int fast   = fasts[combo];
    const int slow   = slows[combo];
    const int signal = signals[combo];
    if (fast <= 0 || slow <= 0 || signal <= 0) return;

    int fv = first_valid;
    if (fv >= series_len) return;
    fv = imax(fv, 0);

    const size_t row_base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);
    const int macd_warmup   = fv + slow - 1;
    const int signal_warmup = fv + slow + signal - 2;

    // Prefix NaNs (coalesced within warp).
    const float nanv = NAN;
    const int macd_nan_end   = imin(macd_warmup, series_len);
    const int signal_nan_end = imin(signal_warmup, series_len);
    for (int i = lane; i < macd_nan_end; i += 32) {
        macd_out[row_base + static_cast<size_t>(i)] = nanv;
    }
    for (int i = lane; i < signal_nan_end; i += 32) {
        const size_t idx = row_base + static_cast<size_t>(i);
        signal_out[idx] = nanv;
        hist_out[idx]   = nanv;
    }

    if (macd_warmup >= series_len) return;

    // Lane 0 handles the short scalar warmup work; warp-scan handles the long tail.
    float fast_prev = 0.0f;
    float slow_prev = 0.0f;
    float se_prev   = 0.0f;
    int   have_seed = 0;

    const float af    = 2.0f / (static_cast<float>(fast)   + 1.0f);
    const float aslow = 2.0f / (static_cast<float>(slow)   + 1.0f);
    const float asig  = 2.0f / (static_cast<float>(signal) + 1.0f);
    const float bf    = 1.0f - af;
    const float bslow = 1.0f - aslow;
    const float bsig  = 1.0f - asig;

    if (lane == 0) {
        // Seed EMAs via SMA windows (match scalar seeding)
        float fsum = 0.0f, fc = 0.0f;
        const int fcap = imin(fast, series_len - fv);
        for (int i = 0; i < fcap; ++i) {
            kahan_add(prices[fv + i], fsum, fc);
        }
        float fast_ema = fsum / static_cast<float>(fast);

        float ssum = 0.0f, sc = 0.0f;
        const int scap = imin(slow, series_len - fv);
        for (int i = 0; i < scap; ++i) {
            kahan_add(prices[fv + i], ssum, sc);
        }
        float slow_ema = ssum / static_cast<float>(slow);

        // Advance fast EMA to macd_warmup (align with slow window at slow-1)
        const int mwu = imin(macd_warmup, series_len - 1);
        for (int t = fv + fast; t <= mwu; ++t) {
            const float x = prices[t];
            if (isfinite(x)) {
                // fast_ema += af * (x - fast_ema)  (single fma)
                fast_ema = fmaf(x - fast_ema, af, fast_ema);
            }
        }

        // First MACD value at macd_warmup
        float m0 = fast_ema - slow_ema;
        macd_out[row_base + static_cast<size_t>(macd_warmup)] = m0;

        // Seed signal (SMA over first `signal` MACD values) if it exists in-bounds
        if (signal_warmup < series_len) {
            if (signal == 1) {
                se_prev = m0;
                signal_out[row_base + static_cast<size_t>(signal_warmup)] = se_prev;
                hist_out[row_base + static_cast<size_t>(signal_warmup)]   = 0.0f;
                have_seed = 1;
            } else {
                float sig_acc = m0;
                float sig_c = 0.0f;

                // Emit MACD values up to signal_warmup so the SMA seed matches scalar.
                for (int k = macd_warmup + 1; k <= signal_warmup; ++k) {
                    const float x = prices[k];
                    if (isfinite(x)) {
                        fast_ema = fmaf(x - fast_ema, af,    fast_ema);
                        slow_ema = fmaf(x - slow_ema, aslow, slow_ema);
                    }
                    const float m = fast_ema - slow_ema;
                    macd_out[row_base + static_cast<size_t>(k)] = m;
                    kahan_add(m, sig_acc, sig_c);
                }

                se_prev = sig_acc / static_cast<float>(signal);
                signal_out[row_base + static_cast<size_t>(signal_warmup)] = se_prev;
                const float m_seed = macd_out[row_base + static_cast<size_t>(signal_warmup)];
                hist_out[row_base + static_cast<size_t>(signal_warmup)] = m_seed - se_prev;
                have_seed = 1;
            }
        } else {
            have_seed = 0;
        }

        fast_prev = fast_ema;
        slow_prev = slow_ema;
    }

    // Broadcast warmup state.
    fast_prev = __shfl_sync(mask, fast_prev, 0);
    slow_prev = __shfl_sync(mask, slow_prev, 0);
    se_prev   = __shfl_sync(mask, se_prev,   0);
    have_seed = __shfl_sync(mask, have_seed, 0);

    int t0 = have_seed ? (signal_warmup + 1) : (macd_warmup + 1);

    // Main steady-state tiles: 32 timesteps per loop using warp scans.
    for (; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;

        // FAST EMA transform at t
        float Af = 1.0f;
        float Bf = 0.0f;
        // SLOW EMA transform at t
        float As = 1.0f;
        float Bs = 0.0f;

        float x = 0.0f;
        int x_finite = 0;
        if (t < series_len) {
            x = prices[t];
            x_finite = isfinite(x) ? 1 : 0;
        }
        if (x_finite) {
            Af = bf;    Bf = af * x;
            As = bslow; Bs = aslow * x;
        }

        warp_inclusive_scan_affine(Af, Bf, lane, mask);
        warp_inclusive_scan_affine(As, Bs, lane, mask);

        const float fast_y = __fmaf_rn(Af, fast_prev, Bf);
        const float slow_y = __fmaf_rn(As, slow_prev, Bs);

        float m = 0.0f;
        if (t < series_len) {
            m = fast_y - slow_y;
            macd_out[row_base + static_cast<size_t>(t)] = m;
        }

        float sig_y = nanv;
        if (have_seed) {
            float Ase = 1.0f;
            float Bse = 0.0f;
            if (t < series_len) {
                Ase = bsig;
                Bse = asig * m;
            }
            warp_inclusive_scan_affine(Ase, Bse, lane, mask);
            sig_y = __fmaf_rn(Ase, se_prev, Bse);

            if (t < series_len) {
                const size_t idx = row_base + static_cast<size_t>(t);
                signal_out[idx] = sig_y;
                hist_out[idx]   = m - sig_y;
            }
        }

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        fast_prev = __shfl_sync(mask, fast_y, last_lane);
        slow_prev = __shfl_sync(mask, slow_y, last_lane);
        if (have_seed) {
            se_prev = __shfl_sync(mask, sig_y, last_lane);
        }
    }
}

// ===================================================================
// 2) Many series x one param (time-major). One thread per series.
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
