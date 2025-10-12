// CUDA kernels for Symmetric Weighted Moving Average (SWMA).
//
// Optimized variants:
// - Batch (one-series × many-parameter): tiles prices into shared memory and
//   builds normalized triangular weights once per block.
// - Many-series × one-parameter (time-major): processes multiple series per
//   block to improve coalescing and (optionally) uses constant-memory weights.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ------------------- Tunables (safe defaults) -------------------
#ifndef SWMA_BLOCK_X
#define SWMA_BLOCK_X 128           // threads per block in X (both kernels)
#endif
#ifndef SWMA_OUTS_PER_THREAD
#define SWMA_OUTS_PER_THREAD 2     // batch kernel: outputs/thread per tile
#endif
#ifndef SWMA_SERIES_PER_BLOCK
#define SWMA_SERIES_PER_BLOCK 8    // many-series kernel: series lanes per block (blockDim.y)
#endif
#ifndef SWMA_MAX_PERIOD
#define SWMA_MAX_PERIOD 4096       // bounds __constant__ array when enabled
#endif
#ifndef SWMA_USE_CONST_WEIGHTS
#define SWMA_USE_CONST_WEIGHTS 1   // 1=use __constant__ weights in many-series kernel
#endif

#if SWMA_USE_CONST_WEIGHTS
__constant__ float c_swma_weights[SWMA_MAX_PERIOD];
#endif

// ------------------- helpers -------------------
static __device__ __forceinline__ float swma_norm_inv(int period) {
    if (period <= 2) return (period == 1) ? 1.0f : 0.5f;
    if ((period & 1) == 0) {
        float m = float(period >> 1);
        return 1.0f / (m * (m + 1.0f));
    } else {
        float m = float((period + 1) >> 1);
        return 1.0f / (m * m);
    }
}

static __device__ __forceinline__ void fill_tri_weights(float* __restrict__ w_sh, int period, float inv_norm) {
    int half = period >> 1;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float v = float(i + 1) * inv_norm;
        w_sh[i] = v;
        w_sh[period - 1 - i] = v;
    }
    if ((period & 1) && threadIdx.x == 0) {
        w_sh[half] = float(half + 1) * inv_norm;
    }
}

// ======================================================
//  Kernel 1: one-series × many-parameter (batched) path
//  - Tiles the input series into shared memory for reuse.
//  - Computes weights once per block in shared memory.
// ======================================================
extern "C" __global__
void swma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    // Shared memory layout: [weights (max_period)] [tile prices (tileT + period - 1)]
    extern __shared__ float smem[];
    float* const w_sh = smem;
    float* const tile = smem + max_period;

    // Precompute weights (triangular) once per block:
    const float inv_norm = swma_norm_inv(period);
    __syncthreads();
    fill_tri_weights(w_sh, period, inv_norm);
    __syncthreads();

    const int warm  = warm_indices[combo];
    const int first = warm - period + 1;      // retained for parity with original guard
    const int out_base = combo * series_len;

    // Tiling along time
    constexpr int OUTS_PER_THREAD = SWMA_OUTS_PER_THREAD;
    const int TILE_OUT = blockDim.x * OUTS_PER_THREAD;       // outputs covered by one tile per block
    const int GRID_TILE_STRIDE = gridDim.x * TILE_OUT;

    // Each block iterates over its strided tiles
    int tile_start_out = blockIdx.x * TILE_OUT;
    while (tile_start_out < series_len) {
        // Determine how many outputs we actually cover in this tile
        const int n_outs = min(TILE_OUT, series_len - tile_start_out);

        // Input region needed: [tile_start_out - (period-1), tile_start_out + n_outs - 1]
        int in_begin = tile_start_out - (period - 1);
        int in_end   = tile_start_out + n_outs - 1;
        // Clamp to [0, series_len-1] for loads; we guard outputs anyway
        int load_begin = max(in_begin, 0);
        int load_end   = min(in_end, series_len - 1);
        int load_len   = max(0, load_end - load_begin + 1);

        // Cooperative load of prices into shared memory
        for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
            tile[i] = prices[load_begin + i];
        }
        __syncthreads();

        // Compute outputs in the tile (each thread handles OUTS_PER_THREAD items)
#pragma unroll
        for (int u = 0; u < OUTS_PER_THREAD; ++u) {
            int local_idx = threadIdx.x + u * blockDim.x;
            if (local_idx < n_outs) {
                const int t = tile_start_out + local_idx;
                // original warm guard preserved
                if (t < warm || (t - period + 1) < first) {
                    out[out_base + t] = NAN;
                } else {
                    const int start = t - period + 1;
                    // local index into the shared tile buffer
                    const int tile_off = (start - load_begin);
                    float acc = 0.0f;

                    // small-period fast paths
                    if (period == 1) {
                        acc = tile[tile_off];
                    } else if (period == 2) {
                        acc = 0.5f * (tile[tile_off] + tile[tile_off + 1]);
                    } else {
#pragma unroll 4
                        for (int k = 0; k < period; ++k) {
                            acc = fmaf(tile[tile_off + k], w_sh[k], acc);
                        }
                    }
                    out[out_base + t] = acc;
                }
            }
        }
        __syncthreads();

        tile_start_out += GRID_TILE_STRIDE;
    }
}

// ===================================================================
//  Kernel 2: many-series × one-parameter, time-major input
//  - Processes multiple series per block (blockDim.y)
//  - Coalesced loads across series dimension for each time step
//  - Uses __constant__ weights by default (broadcast), falls back to
//    shared if SWMA_USE_CONST_WEIGHTS=0.
//  - out_tm and prices_tm are time-major: index = t*num_series + s.
// ===================================================================
extern "C" __global__
void swma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,  // ABI compatibility
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    // 2D block: X = time lanes, Y = series lanes within a block
    const int series_block_base = blockIdx.y * blockDim.y;
    const int s_local = threadIdx.y;
    const int s = series_block_base + s_local;
    if (s >= num_series) return;

    // Shared memory layout: [optional shared weights (period)] [tile of prices: (tileT + p - 1) * blockDim.y]
    extern __shared__ float smem[];
    float* sh = smem;
#if SWMA_USE_CONST_WEIGHTS
    float* tile = sh;                              // no shared weights
#else
    float* w_sh = sh;                              // copy weights to shared
    float* tile  = w_sh + period;
    for (int k = threadIdx.x; k < period; k += blockDim.x) { w_sh[k] = weights[k]; }
    __syncthreads();
#endif

    const int warm = first_valids[s] + period - 1;

    const int TILE_T = blockDim.x;                 // outputs along time per tile
    const int GRID_TILE_STRIDE = gridDim.x * TILE_T;

    int tile_t0 = blockIdx.x * TILE_T;             // first output t covered by this tile
    while (tile_t0 < series_len) {
        const int n_outs = min(TILE_T, series_len - tile_t0);
        // Input window needed along time:
        const int in_begin_t = tile_t0 - (period - 1);
        const int in_end_t   = tile_t0 + n_outs - 1;
        const int load_begin_t = max(in_begin_t, 0);
        const int load_end_t   = min(in_end_t, series_len - 1);
        const int load_len_t   = max(0, load_end_t - load_begin_t + 1);

        // Cooperative, coalesced load across the series dimension:
        // Layout in shared: tile[ dt * blockDim.y + s_local ]
        const int tile_span = load_len_t * blockDim.y;
        // Flatten 2D threads to cover tile
        int lin = threadIdx.x * blockDim.y + s_local;
        for (int idx = lin; idx < tile_span; idx += blockDim.x * blockDim.y) {
            int dt = idx / blockDim.y;
            int ss = idx % blockDim.y;
            int gs = series_block_base + ss;
            if (gs < num_series) {
                int g_index = (load_begin_t + dt) * num_series + gs;
                tile[idx] = prices_tm[g_index];    // contiguous across ss ⇒ coalesced
            }
        }
        __syncthreads();

        // Compute this thread's outputs (for its series 's', over up to one time)
        // Each thread in X handles one output time in the tile.
        int local_t = threadIdx.x;
        if (local_t < n_outs) {
            const int t = tile_t0 + local_t;
            const int out_idx = t * num_series + s;
            if (t < warm) {
                out_tm[out_idx] = NAN;
            } else {
                const int start_t = t - period + 1;
                const int base = (start_t - load_begin_t) * blockDim.y + s_local;
                float acc = 0.0f;

#if SWMA_USE_CONST_WEIGHTS
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    // successive times are stride blockDim.y apart in shared
                    acc = fmaf(tile[base + k * blockDim.y], c_swma_weights[k], acc);
                }
#else
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[base + k * blockDim.y], w_sh[k], acc);
                }
#endif
                out_tm[out_idx] = acc;
            }
        }
        __syncthreads();

        tile_t0 += GRID_TILE_STRIDE;
    }
}
