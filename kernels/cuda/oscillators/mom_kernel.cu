// CUDA kernels for Momentum (MOM)
// Math: mom[t] = price[t] - price[t - period]
// Semantics:
// - Warmup: first_valid + period - 1 indices are NaN (i.e., valid starts at t = first_valid + period)
// - No special NaN handling beyond default FP rules; mid-stream NaNs propagate
// - FP32 compute

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void mom_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base   = combo * series_len;
    const int period = periods[combo];

    // Degenerate/invalid cases -> fill the entire row with NaN (matches original behavior)
    if (series_len <= 0 || first_valid >= series_len || period <= 0) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    int warm = first_valid + period;           // first valid output index
    if (warm >= series_len) {
        // Entire row is warmup → all NaN
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    // 1) Write exactly the warmup prefix as NaN (no full-row prefill, no __syncthreads())
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }

    // 2) Compute valid range; contiguous across warp for coalescing
    //    Hint read-only cache for immutable price stream
    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float cur  = __ldg(&prices[t]);
        const float prev = __ldg(&prices[t - period]);
        out[base + t] = cur - prev;
    }
}

// Many-series × one-param (time-major)
// prices_tm/out_tm layout: index = t * cols + s
extern "C" __global__
void mom_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    if (period <= 0) return; // match original behavior (no writes when period invalid)

    // Grid-stride across series to handle large 'cols' and saturate the GPU
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv   = first_valids[s];
        const int warm = fv + period;

        // Invalid first_valid or warmup spans entire column → fill column with NaN
        if (fv < 0 || fv >= rows || warm >= rows) {
            for (int t = 0; t < rows; ++t) {
                out_tm[t * cols + s] = NAN;
            }
            continue;
        }

        // Warmup prefix
        for (int t = 0; t < warm; ++t) {
            out_tm[t * cols + s] = NAN;
        }

        // Valid range: time-major layout means contiguous addresses for a warp at fixed 't'
        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_tm[idx] = __ldg(&prices_tm[idx]) - __ldg(&prices_tm[(t - period) * cols + s]);
        }
    }
}
