// CUDA kernels for Exponential Moving Average (EMA).
//
// Both kernels operate in FP32 and mirror the scalar CPU implementation by
// using an initial running-mean warmup followed by the standard EMA recurrence.
// A block handles either a parameter combination (batch sweep) or a single
// series (many-series path); thread 0 performs the sequential recurrence while
// other threads initialize the NaN prefix.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void ema_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   const float* __restrict__ alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) {
        return;
    }

    const float alpha = alphas[combo];
    const float beta = 1.0f - alpha;
    const int base = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    int warm_end = first_valid + period;
    if (warm_end > series_len) {
        warm_end = series_len;
    }

    float mean = prices[first_valid];
    out[base + first_valid] = mean;
    int valid_count = 1;

    for (int i = first_valid + 1; i < warm_end; ++i) {
        const float x = prices[i];
        if (isfinite(x)) {
            ++valid_count;
            const float vc = static_cast<float>(valid_count);
            mean = ((vc - 1.0f) * mean + x) / vc;
        }
        out[base + i] = mean;
    }

    float prev = mean;
    for (int i = warm_end; i < series_len; ++i) {
        const float x = prices[i];
        if (isfinite(x)) {
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out[base + i] = prev;
    }
}

extern "C" __global__
void ema_many_series_one_param_f32(const float* __restrict__ prices_tm,
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
    if (first_valid >= series_len) {
        return;
    }

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    int warm_end = first_valid + period;
    if (warm_end > series_len) {
        warm_end = series_len;
    }

    float mean = prices_tm[first_valid * stride + series_idx];
    out_tm[first_valid * stride + series_idx] = mean;
    int valid_count = 1;

    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            ++valid_count;
            const float vc = static_cast<float>(valid_count);
            mean = ((vc - 1.0f) * mean + x) / vc;
        }
        out_tm[t * stride + series_idx] = mean;
    }

    float prev = mean;
    for (int t = warm_end; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out_tm[t * stride + series_idx] = prev;
    }
}
