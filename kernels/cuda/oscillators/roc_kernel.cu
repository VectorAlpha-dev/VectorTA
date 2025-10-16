// CUDA kernels for Rate of Change (ROC)
//
// Math: roc[t] = (price[t] / price[t - period]) * 100 - 100
// Semantics:
// - Warmup: first_valid + period - 1 indices are NaN (match scalar batch semantics)
// - If previous value is 0.0 or NaN => output 0.0 (match scalar policy)
// - Mid-stream NaNs in current propagate naturally via division

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void roc_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int base = combo * series_len;

    // Initialize entire row to NaN in parallel (warmup handled this way)
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();
    if (first_valid >= series_len) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int warm = first_valid + period;
    if (warm >= series_len) return;

    // Compute valid region in parallel
    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const float cur = prices[t];
        const float prev = prices[t - period];
        if (prev == 0.0f || isnan(prev)) {
            out[base + t] = 0.0f;
        } else {
            // FMA: (cur/prev)*100 - 100
            out[base + t] = fmaf(cur / prev, 100.0f, -100.0f);
        }
    }
}

// Many-series × one-param (time-major)
// prices_tm/out_tm layout: index = t * cols + s
extern "C" __global__
void roc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;
    if (period <= 0) return;
    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        // Fill entire column with NaN (best-effort)
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }
    const int warm = fv + period;

    // Warmup prefix
    for (int t = 0; t < warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    // Compute valid range
    for (int t = warm; t < rows; ++t) {
        const int idx = t * cols + s;
        const float cur = prices_tm[idx];
        const float prev = prices_tm[(t - period) * cols + s];
        if (prev == 0.0f || isnan(prev)) {
            out_tm[idx] = 0.0f;
        } else {
            out_tm[idx] = fmaf(cur / prev, 100.0f, -100.0f);
        }
    }
}

