// CUDA kernels for Fibonacci Weighted Moving Average (FWMA).
// Optimized for ADA+ (RTX 4090) and newer SMs by tiling the time dimension
// (batch path) and coalescing across series (time-major path).
// NOTE: Host-side wrapper checks for these exact symbol names:
//  - "fwma_batch_f32"
//  - "fwma_multi_series_one_param_f32"

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FWMA_TILE_T
// Tile width along time for the batch kernel. Choose 128 or 256 typically.
#define FWMA_TILE_T 256
#endif

// -----------------------------
// Batch path: many parameter combos, one price series
// Each block (grid.y == combo) owns one parameter set, and grid.x tiles time.
// Dynamic shared memory layout per block:
//   [0 .. max_period-1]                 -> weights for this combo
//   [max_period .. max_period + (FWMA_TILE_T + max_period - 1) - 1] -> price tile
// -----------------------------
extern "C" __global__
void fwma_batch_f32(const float* __restrict__ prices,          // [series_len]
                    const float* __restrict__ weights_flat,    // [n_combos * max_period]
                    const int*   __restrict__ periods,         // [n_combos]
                    const int*   __restrict__ warm_indices,    // [n_combos]
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {                 // [n_combos * series_len]
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    // Dynamic shared memory:
    extern __shared__ float smem[];
    float* __restrict__ s_w = smem;                 // length = max_period
    float* __restrict__ s_x = s_w + max_period;     // length = FWMA_TILE_T + max_period - 1

    // Stage this combo's weights once into SMEM (only first 'period' entries are needed)
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm     = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f  = __int_as_float(0x7fffffff);

    // Tile indices along time
    const int tile_t0 = blockIdx.x * blockDim.x;                 // tile start time
    const int tile_t1 = min(series_len, tile_t0 + blockDim.x);   // exclusive end

    // If the entire tile is before warm, just fill NaN and exit fast.
    if (tile_t1 <= warm) {
        const int t = tile_t0 + threadIdx.x;
        if (t < tile_t1) out[base_out + t] = nan_f;
        return;
    }

    // Load the input span that covers the whole tile's windows:
    // Need values from (tile_t0 - period + 1) .. (tile_t1 - 1)
    const int load_base = tile_t0 - period + 1;
    const int load_len  = (tile_t1 - tile_t0) + period - 1;  // == blockDim.x + period - 1 (except at series tail)

    // Cooperatively load with OOB guard (threads may read early negatives near t=0)
    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        const int g = load_base + i;
        s_x[i] = (unsigned(g) < (unsigned)series_len) ? prices[g] : 0.0f;
    }
    __syncthreads();

    // Compute each output in the tile
    const int t = tile_t0 + threadIdx.x;
    if (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {
            // Offset of the sliding window start within s_x
            const int offset = (t - period + 1) - load_base;
            float acc = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(s_x[offset + k], s_w[k], acc);
            }
            out[base_out + t] = acc;
        }
    }
}

// -----------------------------
// Many-series x one-parameter path (time-major input):
// Map threads across series for a fixed time index, making loads for each k coalesced.
// grid.x tiles time in chunks of TIME_STEPS_PER_BLOCK, grid.y tiles series.
// -----------------------------
#ifndef FWMA_TIME_STEPS_PER_BLOCK
#define FWMA_TIME_STEPS_PER_BLOCK 4   // small inner time loop per block to amortize weight staging
#endif

extern "C" __global__
void fwma_multi_series_one_param_f32(const float* __restrict__ prices_tm, // [series_len, num_series] time-major
                                     const float* __restrict__ weights,   // [period]
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,// [num_series]
                                     float* __restrict__ out_tm) {        // [series_len, num_series] time-major
    // Stage weights once per block
    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    // Threads cover series dimension for coalesced accesses
    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    // Small time loop inside the block
    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    // For fixed k, addresses across threads differ by +1 (series) -> coalesced
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}

// Optional alias retained for wrapper compatibility (same implementation).
extern "C" __global__
void fwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    // Reuse the same body as fwma_multi_series_one_param_f32
    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}
