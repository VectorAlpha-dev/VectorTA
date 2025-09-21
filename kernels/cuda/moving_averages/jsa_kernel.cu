// CUDA kernels for the Jump Step Average (JSA).
//
// Each block handles a single parameter combination (period) or a single price
// series in the many-series entry point. The computation is embarrassingly
// parallel because each output sample depends only on two raw inputs, so threads
// cooperatively initialise the NaN prefix and then compute averages without
// sequential coupling.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void jsa_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   const int* __restrict__ warm_indices,
                   int first_valid,
                   int series_len,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0 || n_combos <= 0) {
        return;
    }

    const int period = periods[combo];
    const int warm = warm_indices[combo];

    if (period <= 0 || warm < first_valid || warm > series_len) {
        return;
    }

    const int row_offset = combo * series_len;

    // Fill entire row with NaNs cooperatively.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();

    if (warm >= series_len) {
        return;
    }

    // Compute the averages in parallel once the warm-up threshold is reached.
    for (int idx = warm + threadIdx.x; idx < series_len; idx += blockDim.x) {
        const float current = prices[idx];
        const float past = prices[idx - period];
        out[row_offset + idx] = 0.5f * (current + past);
    }
}

extern "C" __global__
void jsa_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
                                   const int* __restrict__ warm_indices,
                                   int period,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || num_series <= 0 || series_len <= 0) {
        return;
    }
    if (period <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];
    const int warm = warm_indices[series_idx];

    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (warm >= series_len) {
        return;
    }

    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        const int curr_offset = t * stride + series_idx;
        const int prev_offset = (t - period) * stride + series_idx;
        const float current = prices_tm[curr_offset];
        const float past = prices_tm[prev_offset];
        out_tm[curr_offset] = 0.5f * (current + past);
    }
}
