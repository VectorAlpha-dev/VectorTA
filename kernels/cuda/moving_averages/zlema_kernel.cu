// CUDA kernel for ZLEMA (Zero Lag Exponential Moving Average) batch evaluation.
//
// Each parameter combination (period) runs on an independent thread, producing
// one row of the output matrix. Dependencies across time steps require serial
// evaluation per combination, but with many parameter sweeps we still expose
// ample parallelism across rows.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef ZLEMA_NAN
#define ZLEMA_NAN (__int_as_float(0x7fffffff))
#endif

extern "C" __global__
void zlema_batch_f32(const float* __restrict__ prices,
                     const int* __restrict__ periods,
                     const int* __restrict__ lags,
                     const float* __restrict__ alphas,
                     int series_len,
                     int n_combos,
                     int first_valid,
                     float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int lag = lags[combo];
    const float alpha = alphas[combo];
    const float one_minus_alpha = 1.0f - alpha;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    for (int i = 0; i < warm && i < series_len; ++i) {
        out[base + i] = ZLEMA_NAN;
    }

    if (first_valid >= series_len) {
        return;
    }

    float last_ema = prices[first_valid];

    // Store warm point if warm == first_valid (period == 1)
    if (warm <= first_valid) {
        out[base + first_valid] = last_ema;
    }

    for (int t = first_valid + 1; t < series_len; ++t) {
        float current = prices[t];
        float val;
        if (t < first_valid + lag) {
            val = current;
        } else {
            float lagged = prices[t - lag];
            val = 2.0f * current - lagged;
        }
        last_ema = alpha * val + one_minus_alpha * last_ema;
        if (t >= warm) {
            out[base + t] = last_ema;
        }
    }

    // Handle the very first valid index separately once loop above skips it.
    if (first_valid < series_len && warm <= first_valid) {
        // already set for period == 1
        return;
    }
}

// Many-series × one-param (time-major layout)
// prices_tm/out_tm are time-major: index = row * num_series + series
extern "C" __global__
void zlema_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int* __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int period,
                                     float alpha,
                                     float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    if (period <= 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int lag = (period - 1) / 2;
    const float one_minus_alpha = 1.0f - alpha;
    const int warm = first_valid + period - 1;

    // Initialize warmup prefix to NaN
    for (int row = 0; row < warm && row < series_len; ++row) {
        out_tm[row * stride + series] = ZLEMA_NAN;
    }

    float last_ema = prices_tm[first_valid * stride + series];

    // Handle period == 1 explicitly: warm == first_valid
    if (warm <= first_valid) {
        out_tm[first_valid * stride + series] = last_ema;
    }

    for (int t = first_valid + 1; t < series_len; ++t) {
        const float cur = prices_tm[t * stride + series];
        float val;
        if (t < first_valid + lag) {
            val = cur;
        } else {
            const float lagged = prices_tm[(t - lag) * stride + series];
            val = 2.0f * cur - lagged;
        }
        last_ema = alpha * val + one_minus_alpha * last_ema;
        if (t >= warm) {
            out_tm[t * stride + series] = last_ema;
        }
    }
}
