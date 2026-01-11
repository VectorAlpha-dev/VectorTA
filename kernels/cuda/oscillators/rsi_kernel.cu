// CUDA kernels for RSI (Relative Strength Index)
//
// Drop-in rewrite focused on one-series×many-params batch case with shared-memory
// tiling and per-row register-resident Wilder updates. FP32 only; outputs clamped
// post-division to [0,100]. Matches existing warmup/NaN semantics.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef RSI_NAN
#define RSI_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ float clamp_rsi(float x) {
    x = fminf(100.0f, x);
    x = fmaxf(0.0f, x);
    return x;
}

// ===============================
// One series × many params (batch)
// prices: length = series_len
// periods: length = n_combos
// out: rows=n_combos, cols=series_len (row-major)
// ===============================
extern "C" __global__
void rsi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    // One warp per combo. Each lane advances 1 timestep; warp scan emits 32 outputs per iteration.
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_row = out + (size_t)combo * (size_t)series_len;

    // Basic validation mirroring wrapper guards
    if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }
    const int fv = first_valid;
    const int tail = series_len - fv;
    if (tail <= period) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }

    const int warm = fv + period; // first index with a defined output

    // Prefill NaN up to (warm-1) only (avoid full-row writes)
    for (int i = (int)lane; i < warm; i += 32) out_row[i] = RSI_NAN;

    // Precompute alpha & beta for Wilder smoothing (FP32)
    const float inv_p = 1.0f / (float)period;
    const float beta  = inv_p;
    const float alpha = 1.0f - inv_p; // (period - 1) / period

    // ----- Initial averages over (fv+1 ..= warm) computed by lane 0 -----
    float avg_g = 0.0f;
    float avg_l = 0.0f;
    int dead_i = 0;
    if (lane == 0) {
        float prev = prices[fv];
        float sum_g = 0.0f;
        float sum_l = 0.0f;
        for (int i = fv + 1; i <= warm; ++i) {
            const float curr = prices[i];
            const float d = curr - prev;
            prev = curr;
            if (!isfinite(d)) {
                dead_i = 1;
                break;
            }
            if (d > 0.0f) sum_g += d;
            else if (d < 0.0f) sum_l -= d;
        }
        if (!dead_i) {
            avg_g = sum_g * beta;
            avg_l = sum_l * beta;
            const float denom = avg_g + avg_l;
            float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
            out_row[warm] = clamp_rsi(rsi);
        }
    }

    const unsigned mask = 0xffffffffu;
    avg_g = __shfl_sync(mask, avg_g, 0);
    avg_l = __shfl_sync(mask, avg_l, 0);
    dead_i = __shfl_sync(mask, dead_i, 0);

    if (dead_i) {
        for (int i = (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
        return;
    }

    // Rolling update: process 32 timesteps per iteration, starting at warm+1
    for (int t0 = warm + 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;

        float A  = 1.0f;
        float Bg = 0.0f;
        float Bl = 0.0f;
        bool ok = true;
        if (t < series_len) {
            const float p1 = prices[t];
            const float p0 = prices[t - 1];
            const float d = p1 - p0;
            ok = isfinite(d);
            if (ok) {
                const float g = fmaxf(d, 0.0f);
                const float l = fmaxf(-d, 0.0f);
                A  = alpha;
                Bg = beta * g;
                Bl = beta * l;
            }
        }

        const unsigned invalid_mask = __ballot_sync(mask, (t < series_len) && (!ok));

        // Inclusive warp scan composing (A,B) left-to-right.
        // Composition: (A1,B1) ∘ (A2,B2) = (A1*A2, A1*B2 + B1).
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev  = __shfl_up_sync(mask, A, offset);
            const float Bg_prev = __shfl_up_sync(mask, Bg, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float A_cur  = A;
                const float Bg_cur = Bg;
                const float Bl_cur = Bl;
                A  = A_cur * A_prev;
                Bg = __fmaf_rn(A_cur, Bg_prev, Bg_cur);
                Bl = __fmaf_rn(A_cur, Bl_prev, Bl_cur);
            }
        }

        const float yg = __fmaf_rn(A, avg_g, Bg);
        const float yl = __fmaf_rn(A, avg_l, Bl);

        if (t < series_len) {
            if (invalid_mask) {
                const int first_bad = __ffs(invalid_mask) - 1;
                if ((int)lane >= first_bad) {
                    out_row[t] = RSI_NAN;
                } else {
                    const float denom = yg + yl;
                    float rsi = (denom == 0.0f) ? 50.0f : (100.0f * yg / denom);
                    out_row[t] = clamp_rsi(rsi);
                }
            } else {
                const float denom = yg + yl;
                float rsi = (denom == 0.0f) ? 50.0f : (100.0f * yg / denom);
                out_row[t] = clamp_rsi(rsi);
            }
        }

        // If a non-finite delta occurred in this window, the row becomes "dead" (all future NaN).
        if (invalid_mask) {
            const int remaining = series_len - t0;
            const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
            const int first_bad = __ffs(invalid_mask) - 1;
            if (first_bad <= last_lane) {
                for (int i = t0 + 32 + (int)lane; i < series_len; i += 32) out_row[i] = RSI_NAN;
                return;
            }
        }

        // Advance to next window using the last valid lane.
        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        avg_g = __shfl_sync(mask, yg, last_lane);
        avg_l = __shfl_sync(mask, yl, last_lane);
    }
}

// ===============================
// Many series × one param (time-major)
// prices_tm/out_tm index: t * cols + s
// ===============================
extern "C" __global__
void rsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;
    if (period <= 0) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }

    const int fv   = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }

    const int warm = fv + period;
    for (int t = 0; t <= warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    const float inv_p = 1.0f / (float)period;
    const float beta  = 1.0f - inv_p;

    // Warmup sums across first `period` deltas for this series
    float avg_g = 0.0f, avg_l = 0.0f;
    float sum_g = 0.0f, sum_l = 0.0f;
    bool  has_nan = false;

    for (int t = fv + 1; t <= warm; ++t) {
        const float d = prices_tm[t * cols + s] - prices_tm[(t - 1) * cols + s];
        if (!isfinite(d)) { has_nan = true; break; }
        if (d > 0.0f) sum_g += d;
        else if (d < 0.0f) sum_l -= d;
    }

    if (has_nan) {
        out_tm[warm * cols + s] = NAN;
        avg_g = avg_l = NAN;
    } else {
        avg_g = sum_g * inv_p;
        avg_l = sum_l * inv_p;
        const float denom = avg_g + avg_l;
        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
        out_tm[warm * cols + s] = clamp_rsi(rsi);
    }

    // Recursive updates; semantics: treat NaN deltas after warmup as zero change
    for (int t = warm + 1; t < rows; ++t) {
        const float d = prices_tm[t * cols + s] - prices_tm[(t - 1) * cols + s];
        const float g = (d > 0.0f) ? d : 0.0f; // comparisons false for NaN -> 0
        const float l = (d < 0.0f) ? -d : 0.0f;
        avg_g = fmaf(beta, avg_g, inv_p * g);
        avg_l = fmaf(beta, avg_l, inv_p * l);
        const float denom = avg_g + avg_l;
        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
        out_tm[t * cols + s] = clamp_rsi(rsi);
    }
}
