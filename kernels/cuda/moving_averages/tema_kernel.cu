// CUDA kernels for TEMA (Triple Exponential Moving Average), optimized.
// - One warp processes up to 32 independent sequences in parallel.
// - For the param-sweep kernel, a warp covers 32 combos and broadcasts the price.
// - For the multi-series kernel (time-major layout), a warp covers 32 series.
// - Uses EMA difference-form with FMA for accuracy/perf.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef TEMA_WARPS_PER_BLOCK
// Tunable: warps per block (1..8). 4 is a good default on Ada/Lovelace.
#define TEMA_WARPS_PER_BLOCK 4
#endif

// Quiet NaN bit-pattern for IEEE-754 single (canonical form commonly used).
// Using a constant avoids device-side math headers creating doubles.
#ifndef TEMA_QNAN_U32
#define TEMA_QNAN_U32 0x7fc00000u
#endif

static __device__ __forceinline__ float tema_qnan() {
    return __int_as_float((int)TEMA_QNAN_U32);
}

// FMA-based EMA update: prev += alpha * (x - prev)
static __device__ __forceinline__ float ema_step(float prev, float x, float alpha) {
    return fmaf(alpha, x - prev, prev);
}

// Broadcast from lane 0 to the warp
static __device__ __forceinline__ float warp_broadcast0(float v) {
    unsigned m = __activemask();            // mask of participating lanes
    return __shfl_sync(m, v, 0);            // broadcast lane 0 to all
}

// -------------------------------------------
// 1) Param-sweep over periods (one price series)
//    One warp handles up to 32 combos (periods).
// -------------------------------------------
extern "C" __global__
__launch_bounds__(TEMA_WARPS_PER_BLOCK * 32, 2)
void tema_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;

    // Warp decomposition
    const int lane     = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5; // / 32
    const int warp_global   = blockIdx.x * TEMA_WARPS_PER_BLOCK + warp_in_block;
    const int combo         = warp_global * 32 + lane;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int base_out = combo * series_len;

    // Lookback / warm indices (period-specific)
    const int lookback   = (period - 1) * 3;
    const int ema2_start = first_valid + (period - 1);
    const int ema3_start = first_valid + 2 * (period - 1);
    const int warm       = first_valid + lookback;

    // Prefill NaN only up to warm (earlier code wrote whole series).
    // If first_valid is out of range, mark all NaN like before.
    if (first_valid >= series_len) {
        float qn = tema_qnan();
        for (int i = 0; i < series_len; ++i) out[base_out + i] = qn;
        return;
    } else {
        const int nan_to = warm < series_len ? warm : series_len;
        float qn = tema_qnan();
        for (int i = 0; i < nan_to; ++i) out[base_out + i] = qn;
    }

    // Per-combo constants
    const float alpha = 2.0f / (float(period) + 1.0f);

    // Initialize EMA1 from prices[first_valid], broadcast so only one global load per warp
    float p0 = (lane == 0) ? prices[first_valid] : 0.0f;
    p0 = warp_broadcast0(p0);
    float ema1 = p0;
    float ema2 = 0.0f;
    float ema3 = 0.0f;

    // Sequential time loop; broadcast price each step so every combo in the warp reuses it
    for (int t = first_valid; t < series_len; ++t) {
        float px = (lane == 0) ? prices[t] : 0.0f;
        px = warp_broadcast0(px);

        // EMA1
        ema1 = ema_step(ema1, px, alpha);

        // EMA2 (seed on first step it becomes active, then update)
        if (t >= ema2_start) {
            if (t == ema2_start) ema2 = ema1;
            ema2 = ema_step(ema2, ema1, alpha);
        }

        // EMA3
        if (t >= ema3_start) {
            if (t == ema3_start) ema3 = ema2;
            ema3 = ema_step(ema3, ema2, alpha);
        }

        if (t >= warm) {
            // TEMA = ema3 + 3*(ema1 - ema2)
            out[base_out + t] = fmaf(3.0f, (ema1 - ema2), ema3);
        }
    }
}

// --------------------------------------------------------------
// 2) Multi-series, one period, time-major layout (prices_tm[t*N + s])
//    One warp handles up to 32 series. Loads are naturally coalesced.
// --------------------------------------------------------------
extern "C" __global__
__launch_bounds__(TEMA_WARPS_PER_BLOCK * 32, 2)
void tema_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    if (series_len <= 0 || period <= 0 || num_series <= 0) return;

    const int lane     = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_global   = blockIdx.x * TEMA_WARPS_PER_BLOCK + warp_in_block;
    const int sidx          = warp_global * 32 + lane;
    if (sidx >= num_series) return;

    int fv = first_valids[sidx];
    if (fv < 0) fv = 0;

    const int lookback   = (period - 1) * 3;
    const int ema2_start = fv + (period - 1);
    const int ema3_start = fv + 2 * (period - 1);
    const int warm       = fv + lookback;

    // Prefill NaN up to warm for this series (time-major write)
    if (fv >= series_len) {
        float qn = tema_qnan();
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + sidx] = qn;
        }
        return;
    } else {
        float qn = tema_qnan();
        const int nan_to = warm < series_len ? warm : series_len;
        for (int t = 0; t < nan_to; ++t) {
            out_tm[t * num_series + sidx] = qn;
        }
    }

    const float alpha = 2.0f / (float(period) + 1.0f);

    // Initialize EMA1 from first valid price for this series
    float ema1 = prices_tm[fv * num_series + sidx];
    float ema2 = 0.0f;
    float ema3 = 0.0f;

    // Time loop; loads and stores are coalesced across lanes because of time-major layout.
    for (int t = fv; t < series_len; ++t) {
        const float px = prices_tm[t * num_series + sidx];

        ema1 = ema_step(ema1, px, alpha);

        if (t >= ema2_start) {
            if (t == ema2_start) ema2 = ema1;
            ema2 = ema_step(ema2, ema1, alpha);
        }
        if (t >= ema3_start) {
            if (t == ema3_start) ema3 = ema2;
            ema3 = ema_step(ema3, ema2, alpha);
        }
        if (t >= warm) {
            out_tm[t * num_series + sidx] = fmaf(3.0f, (ema1 - ema2), ema3);
        }
    }
}
