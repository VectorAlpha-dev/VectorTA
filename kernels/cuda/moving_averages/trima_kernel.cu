// CUDA kernel for Triangular Moving Average (TRIMA)
//
// This file provides:
//  - A drop-in tiled batch kernel (shared-memory reuse + optional async copy)
//  - A time-major many-series tiled kernel (coalesced across series)
//  - Plain 1-D kernels remain available for fallback/compat
//  - Optional primitives for the two-SMA path (prefix-sum based)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#endif
#include <math.h>

// Tunables (must match wrapper assumptions if changed)
#ifndef TRIMA_TILE
#define TRIMA_TILE 256
#endif
#ifndef TRIMA_TS
#define TRIMA_TS 128
#endif
#ifndef TRIMA_TT
#define TRIMA_TT 64
#endif

// ----------------------
// Plain 1-D batch kernel
// ----------------------
extern "C" __global__
void trima_batch_f32(const float* __restrict__ prices,
                     const int* __restrict__ periods,
                     const int* __restrict__ warm_indices,
                     int series_len,
                     int n_combos,
                     int max_period,
                     float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int warm = warm_indices[combo];

    extern __shared__ float weights[]; // sized to at least 'period'
    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const float inv_norm = 1.0f / float(m1 * m2);
    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        int w = (idx < m1) ? (idx + 1) : (idx < m2 ? m1 : (m1 + m2 - 1) - idx);
        if (w < 0) w = 0;
        weights[idx] = float(w) * inv_norm;
    }
    __syncthreads();

    const int base_out = combo * series_len;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}

// ---------------------------------------
// Tiled batch kernel (same signature)
// ---------------------------------------
extern "C" __global__
void trima_batch_f32_tiled(const float* __restrict__ prices,
                           const int* __restrict__ periods,
                           const int* __restrict__ warm_indices,
                           int series_len,
                           int n_combos,
                           int max_period,
                           float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int warm = warm_indices[combo];

    // Shared memory layout:
    // [0 .. max_period-1] -> weights (we only fill [0..period-1])
    // [max_period .. max_period + tile_load_len-1] -> tile window
    extern __shared__ float smem[];
    float* __restrict__ weights = smem;
    float* __restrict__ tile    = smem + max_period;

    // Precompute normalized triangular weights (once per CTA)
    const int m1 = (period + 1) / 2;
    const int m2 = period - m1 + 1;
    const float inv_norm = 1.0f / float(m1 * m2);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        int w;
        if (i < m1)       w = i + 1;
        else if (i < m2)  w = m1;
        else              w = (m1 + m2 - 1) - i;
        weights[i] = float(w > 0 ? w : 0) * inv_norm;
    }
    __syncthreads();

    // Tile coordinates along time
    const int TILE = blockDim.x; // blockDim.x configured to TRIMA_TILE at launch
    const int t0   = blockIdx.x * TILE;
    if (t0 >= series_len) return;
    const int t1   = min(t0 + TILE, series_len);

    const int tile_base = max(t0 - period + 1, 0);
    const int tile_end  = t1 - 1;
    const int tile_len  = tile_end - tile_base + 1; // in [1, TILE + period - 1]

#if __CUDA_ARCH__ >= 800
    namespace cg = cooperative_groups;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 1> pstate;
    auto block = cg::this_thread_block();
    auto pipe  = cuda::make_pipeline(block, &pstate);
    pipe.producer_acquire();
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        cuda::memcpy_async(block, &tile[i], &prices[tile_base + i], sizeof(float), pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();
#else
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        tile[i] = prices[tile_base + i];
    }
    __syncthreads();
#endif

    // Each thread computes one output in this tile (or none if out-of-range).
    const int t = t0 + threadIdx.x;
    if (t < t1) {
        const int out_idx = combo * series_len + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            const int start_global = t - period + 1;
            const int start_local  = start_global - tile_base; // into tile[]
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(tile[start_local + k], weights[k], acc);
            }
            out[out_idx] = acc;
        }
    }
}

// -----------------------------------------------------------------
// Many-series × one-parameter (time-major input) — 1D baseline
// -----------------------------------------------------------------
extern "C" __global__
void trima_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                      const float* __restrict__ weights,
                                      int period,
                                      int num_series,
                                      int series_len,
                                      const int* __restrict__ first_valids,
                                      float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[]; // sized to at least period
    for (int i = threadIdx.x; i < period; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int warm = first_valids[series_idx] + period - 1;
    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                acc = fmaf(prices_tm[in_idx], shared_weights[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}

// -----------------------------------------------------------------
// Many-series × one-parameter (time-major) — tiled & coalesced
// -----------------------------------------------------------------
extern "C" __global__
void trima_multi_series_one_param_f32_tm_tiled(const float* __restrict__ prices_tm,
                                               const float* __restrict__ weights_in,
                                               int period,
                                               int num_series,
                                               int series_len,
                                               const int* __restrict__ first_valids,
                                               float* __restrict__ out_tm) {
    // Shared memory: [0..period-1] weights, then time-major tile: rows*(TRIMA_TS)
    extern __shared__ float smem[];
    float* __restrict__ w    = smem;
    float* __restrict__ tile = smem + period;

    // Copy weights once per CTA
    for (int i = threadIdx.x; i < period; i += blockDim.x) w[i] = weights_in[i];
    __syncthreads();

    // This block covers series in [s0, s1) and time in [t0, t1)
    const int s0 = blockIdx.x * TRIMA_TS;
    const int s  = s0 + threadIdx.x; // each thread owns one series
    if (s >= num_series) return;

    const int t0 = blockIdx.y * TRIMA_TT;
    if (t0 >= series_len) return;
    const int t1 = min(t0 + TRIMA_TT, series_len);

    // Load rows in [base .. t1-1], backing off by (period-1)
    const int base  = max(t0 - period + 1, 0);
    const int rows  = t1 - base; // <= TRIMA_TT + period - 1

    // Cooperative row loads (coalesced across series for fixed time)
    for (int r = 0; r < rows; ++r) {
        const int t = base + r;
        if (s < num_series) {
            tile[r * TRIMA_TS + threadIdx.x] = prices_tm[t * num_series + s];
        }
    }
    __syncthreads();

    // Compute outputs for this (series, time) tile
    const int warm = first_valids[s] + period - 1;
    for (int t = t0; t < t1; ++t) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start_row = (t - period + 1) - base;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(tile[(start_row + k) * TRIMA_TS + threadIdx.x], w[k], acc);
            }
            out_tm[out_idx] = acc;
        }
    }
}

// -----------------------------------------------------------------
// Optional primitives: two-pass SMA path using prefix sums
// -----------------------------------------------------------------
extern "C" __global__
void sma_from_prefix_exclusive_f32(const float* __restrict__ P,
                                   int series_len,
                                   int m1,
                                   int warm_first_valid,
                                   float* __restrict__ A) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_first_valid + (m1 - 1);
    if (t < warm) {
        A[t] = NAN;
    } else {
        float sum = P[t + 1] - P[t + 1 - m1];
        A[t] = sum * (1.0f / float(m1));
    }
}

extern "C" __global__
void trima_from_prefix_exclusive_f32(const float* __restrict__ PA,
                                     int series_len,
                                     int m2,
                                     int warm_after_first_sma,
                                     float* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= series_len) return;
    int warm = warm_after_first_sma + (m2 - 1);
    if (t < warm) {
        out[t] = NAN;
    } else {
        float sum = PA[t + 1] - PA[t + 1 - m2];
        out[t] = sum * (1.0f / float(m2));
    }
}
