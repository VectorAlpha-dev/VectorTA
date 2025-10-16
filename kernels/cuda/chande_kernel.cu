// CUDA kernels for Chande (Chandelier Exit) indicator.
//
// For each parameter combo (period, mult, direction), compute:
//  - Long:  HighestHigh(period) - mult * ATR(period)
//  - Short: LowestLow(period)   + mult * ATR(period)
//
// ATR uses Wilder's RMA recurrence with True Range defined as in scalar:
//  TR(t) = max(high-low, |high-prev_close|, |low-prev_close|),
//  with t==first_valid seeded as (high-low).
//
// Warmup/NaN semantics:
//  warm = first_valid + period - 1; out[0..warm) = NaN; out[warm..] valid.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

static __forceinline__ __device__ float tr_at(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int t,
    int first_valid)
{
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        return hi - lo;
    }
    const float pc = close[t - 1];
    float tr = hi - lo;
    float hc = fabsf(hi - pc);
    if (hc > tr) tr = hc;
    float lc = fabsf(lo - pc);
    if (lc > tr) tr = lc;
    return tr;
}

extern "C" __global__
void chande_batch_f32(const float* __restrict__ high,
                      const float* __restrict__ low,
                      const float* __restrict__ close,
                      const int* __restrict__ periods,
                      const float* __restrict__ mults,
                      const int* __restrict__ dirs,   // 1=long, 0=short
                      const float* __restrict__ alphas,
                      const int* __restrict__ warm_indices,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;
    // Initialize entire row to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane sequential kernel per combo

    // Seed ATR over first window [first_valid, first_valid+period)
    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) {
        sum_tr += (double)tr_at(high, low, close, t, first_valid);
    }
    double atr = sum_tr / (double)period;
    // Write first output at warm index
    // Compute rolling max/min via naive scan over current window for simplicity and correctness
    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }

    // Steady state
    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr_at(high, low, close, t, first_valid);
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}

// Optimized batch variant that reuses host-precomputed TR (True Range) across parameter rows.
extern "C" __global__
void chande_batch_from_tr_f32(const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ tr,
                              const int* __restrict__ periods,
                              const float* __restrict__ mults,
                              const int* __restrict__ dirs,
                              const float* __restrict__ alphas,
                              const int* __restrict__ warm_indices,
                              int series_len,
                              int first_valid,
                              int n_combos,
                              float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float mult   = mults[combo];
    const int   dir    = dirs[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) { out[base + idx] = NAN; }
    __syncthreads();
    if (threadIdx.x != 0) return;

    // Seed ATR from TR
    double sum_tr = 0.0;
    for (int t = first_valid; t < first_valid + period; ++t) { sum_tr += (double)tr[t]; }
    double atr = sum_tr / (double)period;

    {
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        const int wstart = warm + 1 - period;
        for (int t = wstart; t <= warm; ++t) {
            const float v = (dir != 0) ? high[t] : low[t];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + warm] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr[t];
        atr = fma((double)tri - atr, (double)alpha, atr);
        const int wstart = t + 1 - period;
        float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
        for (int k = wstart; k <= t; ++k) {
            const float v = (dir != 0) ? high[k] : low[k];
            if (dir != 0) { if (v > extrema) extrema = v; }
            else          { if (v < extrema) extrema = v; }
        }
        out[base + t] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
    }
}

// Many-series × one-parameter (time-major). Each warp handles a series (lane 0 does sequential work).
extern "C" __global__
void chande_many_series_one_param_f32(const float* __restrict__ high_tm,
                                      const float* __restrict__ low_tm,
                                      const float* __restrict__ close_tm,
                                      const int* __restrict__ first_valids, // per series (column)
                                      int period,
                                      float mult,
                                      int dir, // 1=long, 0=short
                                      float alpha,
                                      int num_series,
                                      int series_len,
                                      float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        // Initialize column to NaN cooperatively
        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + s] = NAN;
        }
        if (first_valid < 0 || first_valid >= series_len) continue;
        const int warm = first_valid + period - 1;
        if (warm >= series_len) continue;

        if (lane == 0) {
            // Seed ATR
            double sum_tr = 0.0;
            for (int t = first_valid; t < first_valid + period; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                float tri;
                if (t == first_valid) {
                    tri = hi - lo;
                } else {
                    const float pc = close_tm[(t - 1) * stride + s];
                    float tr = hi - lo;
                    float hc = fabsf(hi - pc);
                    if (hc > tr) tr = hc;
                    float lc = fabsf(lo - pc);
                    if (lc > tr) tr = lc;
                    tri = tr;
                }
                sum_tr += (double)tri;
            }
            double atr = sum_tr / (double)period;
            // First output
            {
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                const int wstart = warm + 1 - period;
                for (int t = wstart; t <= warm; ++t) {
                    const float v = (dir != 0) ? high_tm[t * stride + s] : low_tm[t * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[warm * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }
            // Steady state
            for (int t = warm + 1; t < series_len; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                float tr = hi - lo;
                float hc = fabsf(hi - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(lo - pc);
                if (lc > tr) tr = lc;
                atr = fma((double)tr - atr, (double)alpha, atr);
                const int wstart = t + 1 - period;
                float extrema = (dir != 0) ? -FLT_MAX : FLT_MAX;
                for (int k = wstart; k <= t; ++k) {
                    const float v = (dir != 0) ? high_tm[k * stride + s] : low_tm[k * stride + s];
                    if (dir != 0) { if (v > extrema) extrema = v; }
                    else          { if (v < extrema) extrema = v; }
                }
                out_tm[t * stride + s] = (dir != 0) ? (extrema - mult * (float)atr) : (extrema + mult * (float)atr);
            }
        }
    }
}
