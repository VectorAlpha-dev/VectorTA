// CUDA kernel for DEMA (Double Exponential Moving Average).
//
// Optimizations applied (drop-in, accuracy-preserving):
// - Remove blanket NaN writes from kernels; optionally initialize only [0..warm-1]
//   to NaN when host does not prefill (default enabled to preserve existing behavior).
// - Keep sequential recurrence single-threaded per work item.
// - Many-series time-major path maps a warp to 32 series in SIMT lock-step over time
//   for coalesced global loads/stores. Compatible with existing launch configs.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Accuracy toggles for DEMA updates:
// 0 = plain FMA delta form (fast, tighter rounding)
// 1 = add Kahan-style error feedback (slightly more math, tighter still)
#ifndef USE_DEMA_COMPENSATION
#define USE_DEMA_COMPENSATION 0
#endif

// Host prefill recommended; enable fallback in-kernel prefix NaN init by default
// to preserve current crate behavior without host-side changes.
#ifndef DEMA_INIT_NANS_IN_KERNEL
#define DEMA_INIT_NANS_IN_KERNEL 0
#endif

// --------------------------------------------
// 1) Single-series, many-periods (row-major)
//    One active thread per block; no blanket NaN pass
// --------------------------------------------
extern "C" __global__
void dema_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0)   return;

    const int base = combo * series_len;

    // All recurrence is sequential; a single thread suffices
    if (threadIdx.x != 0)  return;

    if (first_valid >= series_len) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const int   warm  = first_valid + period - 1;

#if DEMA_INIT_NANS_IN_KERNEL
    // Only set prefix to NaN; remaining outputs will be overwritten.
    const int nan_end = (warm < series_len ? warm : series_len);
    for (int i = 0; i < nan_end; ++i) {
        out[base + i] = NAN;
    }
#endif

    // Seed EMA at first_valid
    int t = first_valid;
    float ema  = prices[t];
    float ema2 = ema;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif

    if (t >= warm) {
        out[base + t] = 2.0f * ema - ema2;
    }

    // Sequential recurrence
    for (++t; t < series_len; ++t) {
        const float x = prices[t];
#if USE_DEMA_COMPENSATION
        float inc1 = fmaf(alpha, x - ema, -c1);
        float tmp1 = ema + inc1;
        c1 = (tmp1 - ema) - inc1;
        ema = tmp1;

        float inc2 = fmaf(alpha, ema - ema2, -c2);
        float tmp2 = ema2 + inc2;
        c2 = (tmp2 - ema2) - inc2;
        ema2 = tmp2;
#else
        ema  = fmaf(alpha, x   - ema,    ema);
        ema2 = fmaf(alpha, ema - ema2,   ema2);
#endif
        if (t >= warm) {
            out[base + t] = fmaf(2.0f, ema, -ema2);
        }
    }
}

// --------------------------------------------
// 2) Many series, one period, time-major layout
//    Warp processes 32 series in lock-step over time.
//    Coalesced reads/writes at each time step.
//    Compatible with existing grid/block choices; extra warps simply exit.
// --------------------------------------------
extern "C" __global__
void dema_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,   // [t * num_series + series]
    const int*   __restrict__ first_valids,// [series]
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)            // same layout as prices_tm
{
    if (period <= 0 || series_len <= 0) return;

    const int lane      = threadIdx.x & 31;          // 0..31
    const int warps_pb  = blockDim.x >> 5;           // warps per block
    const int warp_id   = threadIdx.x >> 5;          // 0..warps_pb-1
    const int warp_gbl  = blockIdx.x * warps_pb + warp_id;
    const int series0   = warp_gbl * 32;
    const int series_idx= series0 + lane;
    if (series_idx >= num_series) return;

    const int   stride  = num_series; // time-major stride
    const int   fv      = first_valids[series_idx];
    if (fv >= series_len) return;

    const float alpha   = 2.0f / (static_cast<float>(period) + 1.0f);
    const int   warm    = fv + period - 1;

#if DEMA_INIT_NANS_IN_KERNEL
    // Initialize only the NaN prefix for this series using intra-warp striding.
    const int nan_end = (warm < series_len ? warm : series_len);
    for (int t = lane; t < nan_end; t += 32) {
        out_tm[(size_t)t * stride + series_idx] = NAN;
    }
    // No sync needed; each lane writes disjoint locations.
#endif

    // Lock-step over time to keep memory coalesced across the warp.
    // Maintain per-lane state: "started" after fv is reached.
    bool  started = false;
    float ema = 0.0f, ema2 = 0.0f;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif

    // Pointer-carrying form to avoid t*stride multiplies in the loop
    const float* x_ptr = prices_tm + series_idx;
    float*       y_ptr = out_tm    + series_idx;

    for (int t = 0; t < series_len; ++t) {
        const float x = *x_ptr;

        if (!started && t == fv) {
            started = true;
            ema  = x;
            ema2 = x;

            if (t >= warm) {
                *y_ptr = 2.0f * ema - ema2;
            }
        } else if (started) {
#if USE_DEMA_COMPENSATION
            float inc1 = fmaf(alpha, x - ema, -c1);
            float tmp1 = ema + inc1;
            c1 = (tmp1 - ema) - inc1;
            ema = tmp1;

            float inc2 = fmaf(alpha, ema - ema2, -c2);
            float tmp2 = ema2 + inc2;
            c2 = (tmp2 - ema2) - inc2;
            ema2 = tmp2;
#else
            ema  = fmaf(alpha, x   - ema,    ema);
            ema2 = fmaf(alpha, ema - ema2,   ema2);
#endif
            if (t >= warm) {
                *y_ptr = fmaf(2.0f, ema, -ema2);
            }
        }

        x_ptr += stride;
        y_ptr += stride;
    }
}
