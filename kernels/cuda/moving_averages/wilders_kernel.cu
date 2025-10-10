// CUDA kernel for Wilder's Moving Average (Wilders).
//
// Each block computes one parameter combination (period). Threads collaborate to
// initialise the NaN prefix and to accumulate the first complete window. The
// sequential Wilder's recurrence is then evaluated by lane 0 to mirror the CPU
// semantics exactly.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void wilders_batch_f32(const float* __restrict__ prices,
                       const int* __restrict__ periods,
                       const float* __restrict__ alphas,
                       const int* __restrict__ warm_indices,
                       int series_len,
                       int first_valid,
                       int n_combos,
                       float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const float alpha = alphas[combo];
    const int warm = warm_indices[combo];

    if (period <= 0 || warm >= series_len || first_valid >= series_len) {
        return;
    }

    const int base = combo * series_len;

    // 1) Fill entire output row with NaNs cooperatively.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // 2) Cooperatively accumulate the first window sum; reuse shared storage.
    __shared__ float partial_sum[256];
    float local_sum = 0.0f;
    const int start = first_valid;
    const int window_end = start + period;
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const int idx = start + i;
        if (idx < series_len) {
            local_sum += prices[idx];
        }
    }
    partial_sum[threadIdx.x] = local_sum;
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float sum = 0.0f;
    const int lane_dim = blockDim.x;
    for (int lane = 0; lane < lane_dim; ++lane) {
        sum += partial_sum[lane];
    }

    if (window_end > series_len) {
        return;
    }

    float value = sum / static_cast<float>(period);
    out[base + warm] = value;

    for (int t = warm + 1; t < series_len; ++t) {
        const float price = prices[t];
        value = __fmaf_rn(price - value, alpha, value);
        out[base + t] = value;
    }
}

// Many-series × one-param (time-major) kernel for Wilder's MA.
//
// - prices are time-major: prices_tm[t * num_series + series]
// - first_valids[series] marks the first finite sample per series
// - warmup is the simple average over the first full `period` window
// - recurrence uses FMA and propagates non-finite per IEEE-754 semantics
extern "C" __global__
void wilders_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                       const int* __restrict__ first_valids,
                                       int period,
                                       float alpha,
                                       int num_series,
                                       int series_len,
                                       float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (period <= 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    // Initialize output with NaNs cooperatively
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const int warm_end = first_valid + period;
    if (warm_end > series_len) {
        return;
    }

    // Initial mean over the first full window [first_valid, warm_end)
    float sum = 0.0f;
    for (int k = 0; k < period; ++k) {
        const int idx = (first_valid + k) * stride + series_idx;
        sum += prices_tm[idx];
    }
    float y = sum / static_cast<float>(period);
    const int warm = warm_end - 1;
    out_tm[warm * stride + series_idx] = y;

    // Recurrence
    for (int t = warm + 1; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        y = __fmaf_rn(x - y, alpha, y);
        out_tm[t * stride + series_idx] = y;
    }
}
