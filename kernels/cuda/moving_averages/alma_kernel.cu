// CUDA kernel for ALMA (Arnaud Legoux Moving Average) batch operations
// This kernel processes multiple series with different ALMA parameters in parallel
//
// IMPORTANT: While this kernel uses double precision (f64), GPU computation may have
// different rounding behavior compared to CPU implementations. Users should validate
// that the precision meets their requirements, especially for financial calculations.
// CUDA is never auto-selected and must be explicitly requested via Kernel::CudaBatch.

#include <cuda_runtime.h>
#include <math.h>

// One series with multiple parameter combinations.
// Grid mapping: blockIdx.y = combo index, blockIdx.x/threadIdx.x cover time indices.
extern "C" __global__
void alma_batch_f32(const float*  __restrict__ prices,      // Input price data (length = series_len)
                    const float*  __restrict__ weights_flat, // Flattened weights for all parameter combos
                    const int*    __restrict__ periods,      // Period for each parameter combo
                    const float*  __restrict__ inv_norms,    // 1/sum(weights) for each combo
                    int           max_period,                // Maximum period across all combos
                    int           series_len,                // Length of the series
                    int           n_combos,                  // Number of parameter combinations
                    int           first_valid,               // First valid index in price data
                    float*        __restrict__ out)          // Output array (row-major: combos x series_len)
{
    extern __shared__ float shared_weights[]; // size = max_period

    const int combo_idx = blockIdx.y;
    if (combo_idx >= n_combos) return;

    const int period = periods[combo_idx];
    const float inv_norm = inv_norms[combo_idx];

    // cooperative load of weights for this combo
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_weights[i] = weights_flat[combo_idx * max_period + i];
    }
    __syncthreads();

    // Iterate time indices grid-stride across x dimension
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int t_stride = gridDim.x * blockDim.x;
    const int warm = first_valid + period - 1;
    const int base_out = combo_idx * series_len;

    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = NAN;
        } else {
            const int start_idx = t - period + 1;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < period; ++k) {
                sum += prices[start_idx + k] * shared_weights[k];
            }
            out[base_out + t] = sum * inv_norm;
        }
        t += t_stride;
    }
}


// Helper kernel to compute Gaussian weights on GPU
// Multiple series (time-major) with one parameter combination.
// Input layout: prices[t * num_series + series_idx]
extern "C" __global__
void alma_multi_series_one_param_f32(const float* __restrict__ prices_tm, // time-major input
                                     const float* __restrict__ weights,   // weights for the single param
                                     int          period,
                                     float        inv_norm,
                                     int          num_series,
                                     int          series_len,
                                     const int*   __restrict__ first_valids, // length = num_series
                                     float*       __restrict__ out_tm)    // time-major output
{
    extern __shared__ float shared_weights[]; // size = period

    // cooperative load of weights
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_weights[i] = weights[i];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int t_stride = gridDim.x * blockDim.x;
    const int warm = first_valids[series_idx] + period - 1;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start_idx = t - period + 1;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start_idx + k) * num_series + series_idx;
                sum += prices_tm[in_idx] * shared_weights[k];
            }
            out_tm[out_idx] = sum * inv_norm;
        }
        t += t_stride;
    }
}

// One series with many parameter combinations, with price tiling in shared memory to reduce
// redundant global loads.
extern "C" __global__
void alma_batch_tiled_f32(const float*  __restrict__ prices,      // Input price data (length = series_len)
                          const float*  __restrict__ weights_flat, // Flattened weights for all parameter combos
                          const int*    __restrict__ periods,      // Period for each parameter combo
                          const float*  __restrict__ inv_norms,    // 1/sum(weights) for each combo
                          int           max_period,                // Maximum period across all combos
                          int           series_len,                // Length of the series
                          int           n_combos,                  // Number of parameter combinations
                          int           first_valid,               // First valid index in price data
                          float*        __restrict__ out)          // Output array (row-major: combos x series_len)
{
    int combo_idx = blockIdx.y;
    if (combo_idx >= n_combos) return;

    const int period = periods[combo_idx];
    const float inv_norm = inv_norms[combo_idx];
    const int tile_len = blockDim.x;
    const int t0 = blockIdx.x * tile_len;
    if (t0 >= series_len) return;

    const int warm = first_valid + period - 1;

    extern __shared__ float sh[];
    float* w = sh;                 // period
    float* p = sh + period;        // tile_len + period - 1

    // Load weights cooperatively
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        w[i] = weights_flat[combo_idx * max_period + i];
    }
    __syncthreads();

    // Load price tile (with preceding window)
    const int p_base = t0 - (period - 1);
    const int total = tile_len + period - 1;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int idx = p_base + i;
        float v = 0.0f;
        if (idx >= 0 && idx < series_len) {
            v = prices[idx];
        }
        p[i] = v;
    }
    __syncthreads();

    int t = t0 + threadIdx.x;
    if (t >= series_len) return;

    int out_idx = combo_idx * series_len + t;
    if (t < warm) {
        out[out_idx] = NAN;
        return;
    }

    int start = threadIdx.x; // offset into p
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        sum += p[start + k] * w[k];
    }
    out[out_idx] = sum * inv_norm;
}

