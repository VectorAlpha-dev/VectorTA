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

// Tunables (safe defaults; launch may override via block/grid sizing)
#ifndef RSI_BLOCK_THREADS
#define RSI_BLOCK_THREADS 256
#endif
#ifndef RSI_TILE
#define RSI_TILE 1024
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
    // Early exit for blocks entirely out of range (compat with existing launcher)
    if (blockIdx.x * blockDim.x >= (unsigned)n_combos) {
        return;
    }

    // Map one thread -> one parameter row
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos);

    // Shared tiles: prices (T+1), gains, losses, finiteness mask
    __shared__ float   s_pr[RSI_TILE + 1];
    __shared__ float   s_g [RSI_TILE];
    __shared__ float   s_l [RSI_TILE];
    __shared__ uint8_t s_ok[RSI_TILE];

    // Per-row registers
    int   period = 0, warm = 0, fv = first_valid;
    float inv_p = 0.0f, beta = 0.0f;
    float sum_g = 0.0f, sum_l = 0.0f;   // warmup sums
    float avg_g = 0.0f, avg_l = 0.0f;   // Wilder state after warm
    bool  dead  = false;                // non-finite encountered → row remains NaN

    if (active) {
        period = periods[combo];
        if (period <= 0) {
            const int base = combo * series_len;
            for (int t = 0; t < series_len; ++t) out[base + t] = NAN;
        }
        if (period > 0) {
            warm  = fv + period;
            inv_p = 1.0f / (float)period;
            beta  = 1.0f - inv_p;
        }
    }

    // Tile over timeline t = 0..series_len-1
    for (int t0 = 0; t0 < series_len; t0 += RSI_TILE) {
        const int tile_len = min(RSI_TILE, series_len - t0);

        // ---- Stage prices into shared memory (cooperative, coalesced) ----
        // Need prices[t0-1 .. t0+tile_len]; for t0==0 reuse prices[0] at s_pr[0]
        for (int i = threadIdx.x; i < tile_len + 1; i += blockDim.x) {
            int gi = t0 + i - 1;
            gi = (gi < 0) ? 0 : gi;
            s_pr[i] = prices[gi];
        }
        __syncthreads();

        // Compute per-step delta → gain/loss + finiteness mask once per block
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            const int t = t0 + i;
            if (t == 0) {
                s_g[i]  = 0.0f;
                s_l[i]  = 0.0f;
                s_ok[i] = 1;
            } else {
                const float d = s_pr[i + 1] - s_pr[i];
                const uint8_t ok = isfinite(d) ? 1u : 0u;
                s_ok[i] = ok;
                const float g = ok ? ((d > 0.0f) ? d : 0.0f) : 0.0f;
                const float l = ok ? ((d < 0.0f) ? -d : 0.0f) : 0.0f;
                s_g[i] = g;
                s_l[i] = l;
            }
        }
        __syncthreads();

        // ---- Per-row sequential use of the staged tile ----
        if (active && period > 0) {
            const int base = combo * series_len;

            for (int i = 0; i < tile_len; ++i) {
                const int t = t0 + i;

                // Pre-warm region always NaN
                if (t < series_len && t < warm) {
                    out[base + t] = NAN;
                }

                // Accumulate warmup sums over deltas in [fv+1 .. warm]
                if (!dead && (t >= (fv + 1)) && (t < warm) && (t > 0)) {
                    if (!s_ok[i]) { dead = true; }
                    else { sum_g += s_g[i]; sum_l += s_l[i]; }
                }

                // Emit first RSI at t == warm (includes delta at warm)
                if (t == warm && t < series_len) {
                    if (!dead) {
                        if (!s_ok[i]) { // non-finite at warm
                            dead = true;
                            avg_g = avg_l = NAN;
                            out[base + t] = NAN;
                        } else {
                            sum_g += s_g[i]; sum_l += s_l[i];
                            avg_g = sum_g * inv_p;
                            avg_l = sum_l * inv_p;
                            const float denom = avg_g + avg_l;
                            float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
                            out[base + t] = clamp_rsi(rsi);
                        }
                    } else {
                        out[base + t] = NAN;
                    }
                }

                // Recursive updates after warm
                if (t > warm && t < series_len) {
                    if (dead) {
                        out[base + t] = NAN;
                    } else if (!s_ok[i]) {
                        dead = true;
                        out[base + t] = NAN;
                    } else {
                        avg_g = fmaf(beta, avg_g, inv_p * s_g[i]);
                        avg_l = fmaf(beta, avg_l, inv_p * s_l[i]);
                        const float denom = avg_g + avg_l;
                        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
                        out[base + t] = clamp_rsi(rsi);
                    }
                }

                // If warm is beyond end, we still must paint all outputs NaN
                if (warm >= series_len && t < series_len) {
                    out[base + t] = NAN;
                }
            }
        }

        __syncthreads();
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
        const float g = (d > 0.0f) ? d : 0.0f; // comparisons false for NaN → 0
        const float l = (d < 0.0f) ? -d : 0.0f;
        avg_g = fmaf(beta, avg_g, inv_p * g);
        avg_l = fmaf(beta, avg_l, inv_p * l);
        const float denom = avg_g + avg_l;
        float rsi = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
        out_tm[t * cols + s] = clamp_rsi(rsi);
    }
}
