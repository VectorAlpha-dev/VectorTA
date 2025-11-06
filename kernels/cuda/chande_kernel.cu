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

// === Optimized one-series × many-params path ===============================
// Drop-in kernels that prioritize the common "one price series – many params"
// use case. They implement rolling extrema via a monotone deque (amortized
// O(1) per step) and seed/update ATR in FP32 using Kahan compensation and FMA.
//
// These kernels are additive and do not change existing entry points. Host
// wrappers may opt-in to these names for better performance:
//   - chande_one_series_many_params_f32(...)
//   - chande_one_series_many_params_from_tr_f32(...)

// Compute True Range from high/low/prev_close (broadcast per warp).
static __forceinline__ __device__ float tr_from_hlpc(
    float hi, float lo, float pc, int t, int first_valid)
{
    if (t == first_valid) return hi - lo;
    float tr  = hi - lo;
    float hc  = fabsf(hi - pc);
    float lc  = fabsf(lo - pc);
    if (hc > tr) tr = hc;
    if (lc > tr) tr = lc;
    return tr;
}

// Power-of-two ring buffer ops (mask = cap-1). Head points to valid front.
// We store both index and value so we don't reload historical v from global.
static __forceinline__ __device__ void dq_push_monotone(
    int* __restrict__ idx_buf,
    float* __restrict__ val_buf,
    unsigned int mask,
    int& head, int& tail,
    int idx_new, float val_new, bool keep_max)
{
    // Pop from back while monotonicity is violated.
    while (head != tail) {
        unsigned int last = (static_cast<unsigned int>(tail - 1)) & mask;
        float back_val = val_buf[last];
        if (keep_max ? (back_val >= val_new) : (back_val <= val_new)) break;
        tail = static_cast<int>(last);
    }
    val_buf[tail] = val_new;
    idx_buf[tail] = idx_new;
    tail = static_cast<int>((static_cast<unsigned int>(tail) + 1u) & mask);
}

// Drop expired entries (index < window_start).
static __forceinline__ __device__ void dq_pop_expired(
    const int* __restrict__ idx_buf,
    unsigned int mask,
    int& head, int tail, int window_start)
{
    while (head != tail) {
        if (idx_buf[head] >= window_start) break;
        head = static_cast<int>((static_cast<unsigned int>(head) + 1u) & mask);
    }
}

// Return current extremum value at deque front. Assumes not empty.
static __forceinline__ __device__ float dq_front_value(
    const float* __restrict__ val_buf, unsigned int mask, int head)
{
    return val_buf[head & mask];
}

// Inputs:
//  - high, low, close: single series (length = series_len)
//  - periods, mults, dirs, alphas: size = n_combos
//  - first_valid: common across combos for this series
//  - queue_cap: power-of-two >= (max_period + 1) across combos
//  - dq_idx, dq_val: workspace sized n_combos * queue_cap (int / float)
// Output layout: row-major [combo][t], same as existing batch kernels.
extern "C" __global__
void chande_one_series_many_params_f32(const float* __restrict__ high,
                                       const float* __restrict__ low,
                                       const float* __restrict__ close,
                                       const int*   __restrict__ periods,
                                       const float* __restrict__ mults,
                                       const int*   __restrict__ dirs,   // 1=long, 0=short
                                       const float* __restrict__ alphas,
                                       int first_valid,
                                       int series_len,
                                       int n_combos,
                                       int queue_cap,          // power-of-two >= max_period+1
                                       int*   __restrict__ dq_idx,
                                       float* __restrict__ dq_val,
                                       float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        // Initialize entire row to NaN (one lane per combo does full row).
        for (int t0 = 0; t0 < series_len; ++t0) {
            out[base + t0] = NAN;
        }

        if (period <= 0 || warm >= series_len || first_valid >= series_len)
            continue;

        // Per-thread deque state mapped into the flat workspaces
        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;

        // ATR state: Kahan-compensated sum for seeding, then FMA recurrence.
        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;

        // Main time loop with warp-level broadcast of inputs.
        float prev_close_b = 0.0f; // broadcast pc at t-1
        for (int t = 0; t < series_len; ++t) {
            // Lane0 loads scalars once per warp; broadcast to others.
            float hi = 0.0f, lo = 0.0f, pc = 0.0f;
            if (lane == 0) {
                hi = high[t];
                lo = low[t];
                if (t > 0) pc = close[t - 1];
            }
            hi = __shfl_sync(full_mask, hi, 0);
            lo = __shfl_sync(full_mask, lo, 0);
            if (t > 0) prev_close_b = __shfl_sync(full_mask, pc, 0);

            // Push into deque only once we enter the valid region.
            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo; // per-combo choice
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, /*keep_max=*/(dir != 0));
                // Expire elements that fell out of this combo's window
                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }

            // ATR seeding for this combo over its own period
            if (t >= first_valid && !atr_seeded) {
                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);
                // Kahan compensated sum
                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;

                    // First valid output at warm
                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {
                // Wilder's RMA (FMA keeps one rounding)
                const float tri = tr_from_hlpc(hi, lo, prev_close_b, t, first_valid);
                atr = __fmaf_rn(alpha, (tri - atr), atr); // atr += alpha*(tri - atr)

                // Output
                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}

// Variant that reuses precomputed TR[t] for the series, broadcast across warp.
extern "C" __global__
void chande_one_series_many_params_from_tr_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const float* __restrict__ tr,
                                               const int*   __restrict__ periods,
                                               const float* __restrict__ mults,
                                               const int*   __restrict__ dirs,
                                               const float* __restrict__ alphas,
                                               int first_valid,
                                               int series_len,
                                               int n_combos,
                                               int queue_cap,          // power-of-two >= max_period+1
                                               int*   __restrict__ dq_idx,
                                               float* __restrict__ dq_val,
                                               float* __restrict__ out)
{
    const int lane            = threadIdx.x & 31;
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;
    const unsigned full_mask = 0xFFFFFFFFu;
    const unsigned int qmask = static_cast<unsigned int>(queue_cap - 1);

    for (int w = warp_idx; w < (n_combos + 31) / 32; w += total_warps) {
        const int combo = (w << 5) + lane;
        if (combo >= n_combos) continue;

        const int   period = periods[combo];
        const float mult   = mults[combo];
        const int   dir    = dirs[combo];
        const float alpha  = alphas[combo];

        const int warm = first_valid + period - 1;
        const int base = combo * series_len;

        for (int t0 = 0; t0 < series_len; ++t0) out[base + t0] = NAN;
        if (period <= 0 || warm >= series_len || first_valid >= series_len) continue;

        int*   ring_idx = dq_idx + combo * queue_cap;
        float* ring_val = dq_val + combo * queue_cap;
        int head = 0, tail = 0;

        float seed_sum = 0.0f, c = 0.0f;
        float atr = 0.0f;
        bool  atr_seeded = false;

        for (int t = 0; t < series_len; ++t) {
            float hi = 0.0f, lo = 0.0f, tri = 0.0f;
            if (lane == 0) {
                hi  = high[t];
                lo  = low[t];
                tri = tr[t];
            }
            hi  = __shfl_sync(full_mask, hi, 0);
            lo  = __shfl_sync(full_mask, lo, 0);
            tri = __shfl_sync(full_mask, tri, 0);

            if (t >= first_valid) {
                const float v = (dir != 0) ? hi : lo;
                dq_push_monotone(ring_idx, ring_val, qmask, head, tail, t, v, /*keep_max=*/(dir != 0));
                const int wstart = t + 1 - period;
                dq_pop_expired(ring_idx, qmask, head, tail, wstart);
            }

            if (t >= first_valid && !atr_seeded) {
                const float y = tri - c;
                const float tmp = seed_sum + y;
                c = (tmp - seed_sum) - y;
                seed_sum = tmp;

                if (t == warm) {
                    atr = seed_sum / static_cast<float>(period);
                    atr_seeded = true;
                    const float ext = dq_front_value(ring_val, qmask, head);
                    out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
                }
            } else if (atr_seeded && t > warm) {
                atr = __fmaf_rn(alpha, (tri - atr), atr);
                const float ext = dq_front_value(ring_val, qmask, head);
                out[base + t] = (dir != 0) ? (ext - mult * atr) : (ext + mult * atr);
            }
        }
    }
}
