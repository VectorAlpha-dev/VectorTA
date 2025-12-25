// CUDA kernels for TTM Trend (close > SMA(source, period) ? 1 : 0)
//
// Optimizations applied:
// - Remove FP64 from hot path.
// - Batch (one series × many params): 2-D tiled kernel with shared-memory reuse
//   over time to cut global traffic; uses float-float (double-single) prefix
//   with error-free transforms to compute window sums via prefix differences.
// - Many-series path: FP32-only Kahan summation for numerical stability.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// Utilities: float-float (double-single) helpers using error-free transforms
// ============================================================================

// TwoSum: exact decomposition of a+b = s + e  (Dekker/Shewchuk)
__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// ff2_sub: (a_hi+a_lo) - (b_hi+b_lo) -> float2{hi, lo}
__device__ __forceinline__ float2 ff2_sub(const float2 a, const float2 b) {
    float s, e;
    two_sum(a.x, -b.x, s, e);      // subtract hi parts
    e += (a.y - b.y);              // accumulate low parts to error
    float hi, lo;
    two_sum(s, e, hi, lo);         // renormalize
    return make_float2(hi, lo);
}

// ff2_scale: (a_hi+a_lo) * s -> float2{hi, lo}, using one FMA to capture error
__device__ __forceinline__ float2 ff2_scale(const float2 a, const float s) {
    float hi = a.x * s;
    // error in hi plus scaled low part; fmaf captures the rounding error of a.x*s
    float err = fmaf(a.x, s, -hi) + a.y * s;
    float rhi, rlo;
    two_sum(hi, err, rhi, rlo);
    return make_float2(rhi, rlo);
}

// ============================================================================
// 1) Batch path: ONE series × MANY params, prefix-sum based (FP32+compensation)
//
// - prefix_ff2: inclusive prefix of `source` as float2 (hi, lo) per index
// - close:      input close series (f32)
// - periods:    per-row period (int)
// - warm_idx:   per-row warm index = first_valid + period - 1
// - series_len: length
// - n_combos:   number of parameter rows
// - out:        row-major [n_combos * series_len], assumed pre-zeroed
//
// Tiling: 2-D grid over (time, params). Each block caches a tile of
// close[t] and prefix[t] in shared memory and reuses them across the
// small parameter tile.
// ============================================================================

#ifndef TTM_TILE_TIME
#define TTM_TILE_TIME 256     // threads along time (x)
#endif
#ifndef TTM_TILE_PARAMS
#define TTM_TILE_PARAMS 4     // threads along params (y)
#endif

extern "C" __global__
void ttm_trend_batch_prefix_ff2_tiled(
    const float2* __restrict__ prefix_ff2, // inclusive prefix: prefix[first_valid] = source[first_valid]
    const float*  __restrict__ close,      // close[t]
    const int*    __restrict__ periods,    // per param
    const int*    __restrict__ warm_idx,   // per param
    int series_len,
    int n_combos,
    float* __restrict__ out)               // row-major [row*series_len + t]; pre-zeroed
{
    const int tx = threadIdx.x;                       // [0, TTM_TILE_TIME)
    const int ty = threadIdx.y;                       // [0, TTM_TILE_PARAMS)
    const int t0 = blockIdx.x * TTM_TILE_TIME;        // time tile start
    const int p0 = blockIdx.y * TTM_TILE_PARAMS;      // param tile start
    const int t  = t0 + tx;
    const int row = p0 + ty;

    // Shared tiles (time-major caches reused across up to TTM_TILE_PARAMS rows)
    __shared__ float  sh_close[TTM_TILE_TIME];
    __shared__ float2 sh_pref [TTM_TILE_TIME];
    __shared__ int    sh_period[TTM_TILE_PARAMS];
    __shared__ int    sh_warm  [TTM_TILE_PARAMS];

    // Load time tile once (by ty==0)
    if (ty == 0 && t < series_len) {
        sh_close[tx] = close[t];
        sh_pref[tx]  = prefix_ff2[t];
    }

    // Load per-row metadata once (by tx==0)
    if (tx == 0) {
        if (row < n_combos) {
            sh_period[ty] = periods[row];
            sh_warm  [ty] = warm_idx[row];
        } else {
            sh_period[ty] = 0;
            sh_warm  [ty] = INT_MAX;
        }
    }
    __syncthreads();

    if (row >= n_combos || t >= series_len) return;

    const int p    = sh_period[ty];
    const int warm = sh_warm  [ty];
    if (p <= 0) return;

    // We assume 'out' is already memset to 0.0f. Only write for t >= warm.
    if (t < warm) return;

    const float invp = 1.0f / (float)p;
    float avg;

    if (t == warm) {
        // avg = prefix[warm] * (1/p)
        float2 scaled = ff2_scale(sh_pref[tx], invp);
        avg = scaled.x + scaled.y;
    } else {
        // avg = (prefix[t] - prefix[t-p]) * (1/p)
        const int j = t - p;
        float2 pref_t  = sh_pref[tx];
        float2 pref_j  = prefix_ff2[j];         // outside time tile; still coalesced across warp
        float2 diff    = ff2_sub(pref_t, pref_j);
        float2 scaled  = ff2_scale(diff, invp);
        avg = scaled.x + scaled.y;
    }

    const float cv = sh_close[tx];
    out[(size_t)row * series_len + t] = (cv > avg) ? 1.0f : 0.0f;
}

// ============================================================================
// 2) Many-series × one param (time-major), FP32-only with Kahan summation
//
// Layout: time-major (row = time), column = series
//   source_tm[t * num_series + s], close_tm[t * num_series + s]
// Assumes 'out_tm' pre-zeroed.
// ============================================================================

extern "C" __global__
void ttm_trend_many_series_one_param_time_major_f32(
    const float* __restrict__ source_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series || period <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int fv     = first_valids[series];
    if (fv < 0 || fv >= series_len) return;

    const int warm = fv + period - 1;
    if (warm >= series_len) return;

    // Initial window [fv..warm] using Kahan compensated summation
    float s = 0.0f, c = 0.0f;
    for (int k = fv; k <= warm; ++k) {
        const float x = source_tm[(size_t)k * stride + series];
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }

    const float invp = 1.0f / (float)period;
    float avg = s * invp;
    out_tm[(size_t)warm * stride + series] =
        (close_tm[(size_t)warm * stride + series] > avg) ? 1.0f : 0.0f;

    // Slide window: s += add; s -= sub; both with compensation
    for (int t = warm + 1; t < series_len; ++t) {
        const float add = source_tm[(size_t)t * stride + series];
        const float sub = source_tm[(size_t)(t - period) * stride + series];

        // Kahan add
        float y1 = add - c;
        float u1 = s + y1;
        c = (u1 - s) - y1;
        s = u1;

        // Kahan subtract
        float y2 = -sub - c;
        float u2 = s + y2;
        c = (u2 - s) - y2;
        s = u2;

        avg = s * invp;
        out_tm[(size_t)t * stride + series] =
            (close_tm[(size_t)t * stride + series] > avg) ? 1.0f : 0.0f;
    }
}

