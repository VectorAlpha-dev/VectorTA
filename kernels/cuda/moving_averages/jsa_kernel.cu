// CUDA kernels for the Jump Step Average (JSA).
//
// Guide-driven optimizations applied:
// - Remove full-row NaN fill and block barrier; only write the prefix [0, start)
//   where start = max(warm, period[, first_valid]), then compute from start.
// - Guard past reads by starting at 'start' so (t - period) is always valid.
// - Provide an additional coalesced time‑major kernel variant (not used by the
//   existing host launcher, but available for future heuristic dispatch).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <stdint.h>

// Portable IEEE-754 quiet NaN without <math.h> / libm linkage
static __device__ __forceinline__ float jsaf_qnan() {
    return __int_as_float(0x7fc00000);
}

extern "C" __global__
void jsa_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   const int*   __restrict__ warm_indices,
                   int first_valid,
                   int series_len,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0 || n_combos <= 0) return;

    const int period = periods[combo];
    const int warm   = warm_indices[combo];

    // Basic validation.
    if (period <= 0 || warm < first_valid || warm > series_len) return;

    // Start index at which it is legal to read (t - period).
    const int start = max(warm, period);
    const int row_off = combo * series_len;
    float* __restrict__ out_row = out + row_off;

    // 1) Prefix NaN fill: [0, start)
    for (int t = threadIdx.x; t < min(start, series_len); t += blockDim.x) {
        out_row[t] = jsaf_qnan();
    }

    if (start >= series_len) return;

    // 2) Compute averages for t in [start, series_len)
    for (int t = start + threadIdx.x; t < series_len; t += blockDim.x) {
        const float c = prices[t];
        const float p = prices[t - period];
        out_row[t] = 0.5f * (c + p);
    }
}

extern "C" __global__
void jsa_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   const int*   __restrict__ warm_indices,
                                   int period,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || num_series <= 0 || series_len <= 0) return;
    if (period <= 0) return;

    const int stride      = num_series;
    const int first_valid = first_valids[series_idx];
    const int warm        = warm_indices[series_idx];
    if (first_valid < 0 || first_valid >= series_len) return;

    // Earliest legal compute index for this series
    const int start = max(max(warm, period), first_valid);

    // 1) NaN prefix [0, start)
    for (int t = threadIdx.x; t < min(start, series_len); t += blockDim.x) {
        out_tm[t * stride + series_idx] = jsaf_qnan();
    }

    if (start >= series_len) return;

    // 2) Compute [start, end)
    for (int t = start + threadIdx.x; t < series_len; t += blockDim.x) {
        const int curr = t * stride + series_idx;
        const int prev = (t - period) * stride + series_idx;
        const float c = prices_tm[curr];
        const float p = prices_tm[prev];
        out_tm[curr] = 0.5f * (c + p);
    }
}

// ---------------- Optional coalesced time‑major variant (host must use 2D grid) ----------------

#ifndef JSA_BLOCK_X
#define JSA_BLOCK_X 256
#endif
#ifndef JSA_TIME_TILE
#define JSA_TIME_TILE 64
#endif

extern "C" __global__
void jsa_many_series_one_param_f32_coalesced(const float* __restrict__ prices_tm,
                                             const int*   __restrict__ first_valids,
                                             const int*   __restrict__ warm_indices,
                                             int period,
                                             int num_series,
                                             int series_len,
                                             float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    // Series lane handled by this thread:
    const int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    // Time tile this block operates on:
    const int t0   = blockIdx.x * JSA_TIME_TILE;
    const int tEnd = min(t0 + JSA_TIME_TILE, series_len);

    // Per-series gating
    int fv   = first_valids[s];
    int warm = warm_indices[s];
    if (fv < 0 || fv >= series_len) return;

    // Earliest legal compute index for this series
    const int start = max(max(warm, period), fv);

    // Walk the tile, writing either NaN (t < start) or the average (t >= start).
    for (int t = t0; t < tEnd; ++t) {
        const int offset = t * stride + s;
        if (t < start) {
            out_tm[offset] = jsaf_qnan();
        } else {
            const int prev = (t - period) * stride + s;
            const float c = prices_tm[offset];
            const float p = prices_tm[prev];
            out_tm[offset] = 0.5f * (c + p);
        }
    }
}
