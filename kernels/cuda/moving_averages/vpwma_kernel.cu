// ============================================================
// VPWMA optimized kernels (CUDA 13) – Ada/Ampere friendly
//
// Drop-in refactor with two execution paths:
//  - Optimized CTA-per-combo tiled path (when launched with grid.x == n_combos
//    and dynamic shared memory provided by the host).
//  - Backward-compatible thread-per-combo path (matches previous behavior)
//    to preserve existing launch configs used by wrappers/benchmarks.
// ============================================================

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Optional async copy path (SM >= 80). Auto-disables if not available.
#ifndef VPWMA_USE_ASYNC
#define VPWMA_USE_ASYNC 0
#endif

#if VPWMA_USE_ASYNC && (__CUDA_ARCH__ >= 800) && (__cplusplus >= 201703L)
  #include <cooperative_groups.h>
  #include <cuda/pipeline>
  namespace cg = cooperative_groups;
#else
  #undef VPWMA_USE_ASYNC
  #define VPWMA_USE_ASYNC 0
#endif

#ifndef VPWMA_NAN
#define VPWMA_NAN (__int_as_float(0x7fffffff))
#endif

// Tuneable tile size for the optimized path
#ifndef VPWMA_TILE_T
#define VPWMA_TILE_T 128
#endif

// ---------------------------------------------------------------------------
// 1) Many parameter combinations, one price series
//    - Optimized: one CTA per combo, shared-memory tiling
//    - Fallback: one thread per combo (original path, preserves API/launch)
// ---------------------------------------------------------------------------
extern "C" __global__
void vpwma_batch_f32(const float* __restrict__ prices,   // [series_len]
                     const int*   __restrict__ periods,  // [n_combos]
                     const int*   __restrict__ win_lengths, // [n_combos]
                     const float* __restrict__ weights,  // [n_combos][stride]
                     const float* __restrict__ inv_norms,// [n_combos]
                     int series_len,
                     int stride,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out) {          // [n_combos][series_len]

    // Heuristic: if launched with one CTA per combo, use the tiled path.
    // Otherwise, fall back to the legacy thread-per-combo behavior.
    const bool cta_per_combo = (gridDim.x == (unsigned)n_combos);

    if (cta_per_combo) {
        const int combo = blockIdx.x;
        if (combo >= n_combos) return;

        const int period  = periods[combo];
        const int win_len = win_lengths[combo];
        if (win_len <= 0 || period <= 1) return;

        const int row_offset    = combo * series_len;
        const int weight_offset = combo * stride;
        const float inv_norm    = inv_norms[combo];

        const int warm = first_valid + win_len;
        const int warm_clamped = warm < series_len ? warm : series_len;

        // Shared memory layout: [weights | price_tile]
        extern __shared__ float smem[];
        float* __restrict__ s_w = smem;                     // [win_len]
        float* __restrict__ s_x = smem + win_len;           // [VPWMA_TILE_T + win_len - 1]

        // Cache weights once per block
        for (int k = threadIdx.x; k < win_len; k += blockDim.x) {
            s_w[k] = weights[weight_offset + k];
        }
        __syncthreads();

        // Parallel warmup prefix
        for (int i = threadIdx.x; i < warm_clamped; i += blockDim.x) {
            out[row_offset + i] = VPWMA_NAN;
        }
        __syncthreads();
        if (warm >= series_len) return;

        // Process time in tiles
        for (int t0 = warm; t0 < series_len; t0 += VPWMA_TILE_T) {
            const int tile_w   = min(VPWMA_TILE_T, series_len - t0);
            const int g_start  = t0 - (win_len - 1);
            const int load_len = tile_w + win_len - 1;

            // Load contiguous price slice into shared memory
            #if VPWMA_USE_ASYNC
                // Async copy path disabled by default for portability; enabling
                // requires pipeline or barrier management. Fallback below.
            #endif
                for (int o = threadIdx.x; o < load_len; o += blockDim.x) {
                    s_x[o] = prices[g_start + o];
                }
                __syncthreads();
            

            // Each thread computes one output in the tile
            const int tid = threadIdx.x;
            if (tid < tile_w) {
                float acc = 0.0f;
                // w[0] multiplies prices[t], i.e., s_x[tid + win_len - 1]
                #pragma unroll 4
                for (int k = 0; k < win_len; ++k) {
                    acc = fmaf(s_w[k], s_x[tid + (win_len - 1) - k], acc);
                }
                out[row_offset + (t0 + tid)] = acc * inv_norm;
            }
            __syncthreads();
        }
        return;
    }

    // -------- Fallback path: one thread per combo (original implementation) --------
    {
        const int combo = blockIdx.x * blockDim.x + threadIdx.x;
        if (combo >= n_combos) return;

        const int period  = periods[combo];
        const int win_len = win_lengths[combo];
        if (win_len <= 0 || period <= 1) return;

        const float inv_norm    = inv_norms[combo];
        const int row_offset    = combo * series_len;
        const int weight_offset = combo * stride;
        const int warm          = first_valid + win_len;

        const int warm_clamped = warm < series_len ? warm : series_len;
        for (int i = 0; i < warm_clamped; ++i) {
            out[row_offset + i] = VPWMA_NAN;
        }
        if (warm >= series_len) return;

        const float* __restrict__ w_row = weights + weight_offset;
        for (int t = warm; t < series_len; ++t) {
            float acc = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < win_len; ++k) {
                acc = fmaf(prices[t - k], w_row[k], acc);
            }
            out[row_offset + t] = acc * inv_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// 2) Many series, one parameter (time-major layout)
//    We keep the one-thread-per-series mapping for compatibility and only
//    optimize the warmup prefix write. (Shared-weight path is optional and can
//    be enabled via launch-time dynamic shared memory in future wrappers.)
// ---------------------------------------------------------------------------
extern "C" __global__
void vpwma_many_series_one_param_f32(const float* __restrict__ prices_tm,  // [series_len][num_series]
                                     const int*   __restrict__ first_valids, // [num_series]
                                     int num_series,
                                     int series_len,
                                     int period,
                                     const float* __restrict__ weights,    // [win_len]
                                     float inv_norm,
                                     float* __restrict__ out_tm) {         // [series_len][num_series]
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (series < num_series);

    const int win_len = period - 1;
    if (win_len <= 0) return;

    const int stride      = num_series;
    const int first_valid = active ? first_valids[series] : 0;
    const int warm        = active ? (first_valid + win_len) : 0;

    // Shared weights once per block
    extern __shared__ float s_w[];
    for (int k = threadIdx.x; k < win_len; k += blockDim.x) {
        s_w[k] = weights[k];
    }
    __syncthreads();

    // Write only the required NaN prefix
    if (active) {
        const int until = warm < series_len ? warm : series_len;
        for (int t = 0; t < until; ++t) {
            out_tm[t * stride + series] = VPWMA_NAN;
        }
    }
    if (!active || warm >= series_len) {
        // Inactive threads still participate in following __syncthreads() due to block-level control flow
    }

    // Rolling dot products (coalesced across series)
    if (active) {
        for (int t = warm; t < series_len; ++t) {
            float acc = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < win_len; ++k) {
                acc = fmaf(s_w[k], prices_tm[(t - k) * stride + series], acc);
            }
            out_tm[t * stride + series] = acc * inv_norm;
        }
    }
}
