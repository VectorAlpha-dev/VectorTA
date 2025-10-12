// CUDA kernels for the Square Root Weighted Moving Average (SRWMA).
// Optimized: shared-memory tiling, reversed weights, FMA inner loop,
// optional async copy on SM80+ (Ampere/Ada).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SRWMA_USE_ASYNC_COPY
#define SRWMA_USE_ASYNC_COPY 0  // set to 1 to enable cuda::memcpy_async path on SM80+
#endif

#if SRWMA_USE_ASYNC_COPY
  #include <cuda/pipeline>
#endif

// -----------------------------------------
// Batch kernel: 1 price series × many (period,weight) combos
// Shared memory layout per block (combo):
// [0 .. max_wlen-1]               : reversed weights (only first wlen used)
// [max_wlen .. max_wlen+tile-1]   : price tile with halo (tile = blockDim.x + wlen - 1)
// -----------------------------------------
extern "C" __global__
void srwma_batch_f32(const float* __restrict__ prices,
                     const float* __restrict__ weights_flat,
                     const int*   __restrict__ periods,
                     const int*   __restrict__ warm_indices,
                     const float* __restrict__ inv_norms,
                     int max_wlen,
                     int series_len,
                     int n_combos,
                     float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const int wlen = period - 1;
    const int warm = warm_indices[combo];
    const int start_t = max(warm, wlen - 1);  // ensure full window exists
    const int row_offset = combo * series_len;
    const float inv_norm = inv_norms[combo];

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;                      // size: max_wlen
    float* __restrict__ tile  = smem + max_wlen;           // size: blockDim.x + wlen - 1

    // Stage reversed weights for this combo into shared memory
    const int wbase = combo * max_wlen;
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        // reverse: w_rev[0] = weights[wlen-1], ... w_rev[wlen-1] = weights[0]
        w_rev[k] = weights_flat[wbase + (wlen - 1 - k)];
    }
    __syncthreads();

    // Iterate tiles along time axis
    const int tile_span = blockDim.x + wlen - 1;   // halo on the left
    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        // Global index where this tile starts, including left halo
        const int t0 = base - (wlen - 1);

        // Load price tile with halo into shared memory (guard OOB)
        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src) < static_cast<unsigned>(series_len))
                v = prices[src];
            tile[i] = v;
        }

        // Optional: use async copy on SM80+ (Ampere/Ada) if enabled at compile time
        // Note: to keep this drop-in and robust at boundaries we keep the guarded path above.
#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)
        // Example of how to structure it; we do not duplicate actual copies here to avoid double stores.
        // For production, you can split the in-bounds contiguous region and issue cuda::memcpy_async
        // for that subrange while zero-filling OOB edges synchronously.
        // Using a pipeline can overlap the previous tile's compute with the next tile's load.
        // See CUDA Best Practices §10.2.3.4 and libcudacxx cuda::pipeline docs.
        // (left here as a template hook to avoid code duplication)
#endif

        __syncthreads();

        // Compute outputs for this tile
        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int out_idx = row_offset + t;
            if (t < start_t) {
                out[out_idx] = NAN;
            } else {
                // Window in shared memory starts at tile offset == threadIdx.x
                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out[out_idx] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}

// -----------------------------------------
// Many-series × one-period (time-major input).
// Each block.y == series index; block.x tiles along time for that series.
// Shared memory layout per block (per series):
// [0 .. wlen-1]                     : reversed weights (or read from __constant__)
// [wlen .. wlen+tile-1]             : price tile for this series (time-major stride)
// -----------------------------------------

// Optional constant-memory weights for the time-major kernel.
// When enabled from host, copy weights once per launch to this symbol.
// NOTE: leave disabled by default to keep your current host path unchanged.
#ifndef SRWMA_USE_CONST_WEIGHTS
#define SRWMA_USE_CONST_WEIGHTS 0
#endif
#if SRWMA_USE_CONST_WEIGHTS
__constant__ float srwma_const_w[4096];  // supports periods up to 4097; adjust if needed
#endif

extern "C" __global__
void srwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
#if SRWMA_USE_CONST_WEIGHTS
                                     const float* __restrict__ weights_unused, // kept for signature
#else
                                     const float* __restrict__ weights,
#endif
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;
    if (period <= 1) return;

    const int wlen = period - 1;
    const int first_valid = first_valids[series_idx];
    // Preserve your warm logic but also ensure full window exists
    const int warm = first_valid + period + 1;
    const int start_t = max(warm, wlen - 1);

    const int stride = num_series;

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;                     // size: wlen
    float* __restrict__ tile  = smem + wlen;              // size: blockDim.x + wlen - 1

    // Stage reversed weights (or rely on constant memory broadcast)
#if SRWMA_USE_CONST_WEIGHTS
    // With constant memory, threads will repeatedly read srwma_const_w[k] (broadcast-friendly).
    // We still place a reversed copy in shared once to enable contiguous win[k] * w_rev[k] access.
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = srwma_const_w[wlen - 1 - k];
    }
#else
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = weights[wlen - 1 - k];
    }
#endif
    __syncthreads();

    const int tile_span = blockDim.x + wlen - 1;

    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        const int t0 = base - (wlen - 1);

        // Load series tile for this block/series (guard OOB); accesses are strided by 'stride'
        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src_t = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src_t) < static_cast<unsigned>(series_len)) {
                v = prices_tm[src_t * stride + series_idx];
            }
            tile[i] = v;
        }

#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)
        // See comment in batch kernel: you can split the in-bounds contiguous region and use cuda::memcpy_async
        // with a pipeline to overlap loads and compute if beneficial for your wlen/tile size.
#endif

        __syncthreads();

        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int offset = t * stride + series_idx;
            if (t < start_t) {
                out_tm[offset] = NAN;
            } else {
                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out_tm[offset] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}
