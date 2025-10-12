// CUDA kernels for Smoothed Moving Average (SMMA) — optimized.
//
// Key changes:
//  - Warp-broadcast of input prices to eliminate redundant loads across combos
//  - FMA form of the recurrence for speed and numerical stability
//  - Fewer integer ops inside hot loops; pointer/index math hoisted
//  - Same external signatures for drop-in compatibility

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// --- small device helpers
static __device__ __forceinline__ unsigned lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

static __device__ __forceinline__ int warp_reduce_max(int v) {
    // Full-warp reduction for widest compatibility (pre-Volta safe)
    const unsigned full = 0xFFFFFFFFu;
    for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1) {
        int other = __shfl_down_sync(full, v, ofs);
        v = max(v, other);
    }
    return v;
}

extern "C" __global__
void smma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ warm_indices,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned full = 0xFFFFFFFFu;

    // Per-thread parameters (guard against out-of-range lanes)
    int   period = 0;
    int   warm   = 0;
    int   base   = 0;
    bool  valid  = false;
    float alpha  = 0.0f;  // 1/period
    float beta   = 0.0f;  // 1 - alpha

    if (combo < n_combos) {
        period = periods[combo];
        warm   = warm_indices[combo];
        base   = combo * series_len;
        valid  = (period > 0);
        if (valid) {
            alpha = 1.0f / static_cast<float>(period);
            beta  = 1.0f - alpha;
        }
    }

    const float nan_f = __int_as_float(0x7fffffff);

    // ---- Compute the first value: average of prices[first_valid ... first_valid+period-1]
    // We broadcast each price once per warp and let each lane accumulate up to its own period.
    float prev = 0.0f;
    const bool needs_first = (combo < n_combos) && valid && (warm < series_len);
    // Reduce the maximum period across the full warp (lanes without work contribute 0)
    const int myP = needs_first ? period : 0;
    const int maxP = warp_reduce_max(myP);
    const int leader = 0; // use lane 0 for widest compatibility

    float sum = 0.0f;
    for (int k = 0; k < maxP; ++k) {
        float v = 0.0f;
        if (lane_id() == (unsigned)leader) {
            // all combos share the same prices window for initialization
            v = prices[first_valid + k];
        }
        const float p_k = __shfl_sync(full, v, leader);
        if (needs_first && k < period) sum += p_k;
    }
    if (needs_first) prev = sum * alpha;

    // ---- Single pass over time: broadcast price[t] once per warp,
    // and each lane writes according to its own warm/period.
    const int leader_all = 0; // lane 0 broadcasts prices[t] for the warp

    for (int t = 0; t < series_len; ++t) {
        // Broadcast prices[t] to the whole warp (only one global load per warp)
        float v = 0.0f;
        if (lane_id() == (unsigned)leader_all) {
            v = prices[t];
        }
        const float price_t = __shfl_sync(full, v, leader_all);

        if ((combo < n_combos) && valid) {
            const int warm_clamped = (warm < series_len ? warm : series_len);
            if (t < warm_clamped) {
                out[base + t] = nan_f;
            } else if (t == warm) {
                out[base + t] = prev;
            } else if (t > warm) {
                // SMMA recurrence in FMA form: prev = beta*prev + alpha*price
                prev = fmaf(prev, beta, price_t * alpha);
                out[base + t] = prev;
            }
        }
        // Threads with invalid period participate in shuffles but do not write — matches original semantics.
    }
}

extern "C" __global__
void smma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int first = first_valids[series_idx];
    if (period <= 0 || first < 0 || first >= series_len) return;

    const int warm  = first + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    // Precompute constants once
    const float alpha = 1.0f / static_cast<float>(period);
    const float beta  = 1.0f - alpha;

    // Stride between consecutive timesteps in time-major layout
    const size_t stride = static_cast<size_t>(num_series);

    // Fill [0, warm) with NaN (guard for warm beyond end)
    const int warm_clamped = (warm < series_len ? warm : series_len);
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[static_cast<size_t>(t) * stride + series_idx] = nan_f;
    }
    if (warm >= series_len) return;

    // Initial average over [first, first+period)
    const size_t col = static_cast<size_t>(series_idx);
    size_t p = static_cast<size_t>(first) * stride + col;
    float sum = 0.0f;
#pragma unroll 4
    for (int k = 0; k < period; ++k) {
        sum += prices_tm[p];
        p   += stride;
    }
    float prev = sum * alpha;
    out_tm[static_cast<size_t>(warm) * stride + col] = prev;

    // Recurrence: sequential in time for this series, but memory is coalesced across the warp
    size_t idx = static_cast<size_t>(warm + 1) * stride + col;
    for (int t = warm + 1; t < series_len; ++t, idx += stride) {
        prev = fmaf(prev, beta, prices_tm[idx] * alpha); // prev = beta*prev + alpha*x
        out_tm[idx] = prev;
    }
}
