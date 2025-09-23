// CUDA kernel for DEMA (Double Exponential Moving Average).
//
// Each CUDA block processes a single parameter combination (period) and walks
// the input series sequentially. The implementation keeps the recurrence
// identical to the scalar Rust path and only writes outputs once the warm-up
// window (first_valid + period - 1) has been reached, leaving earlier samples
// as NaN to match CPU semantics.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Accuracy toggles for DEMA updates:
// 0 = plain FMA delta form (fast, tighter rounding)
// 1 = add Kahan-style error feedback (slightly more math, tighter still)
#ifndef USE_DEMA_COMPENSATION
#define USE_DEMA_COMPENSATION 0
#endif

extern "C" __global__
void dema_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int base = combo * series_len;

    // Initialise the entire output row with NaNs in parallel.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // Only lane 0 performs the sequential EMA recurrence for this combo.
    if (threadIdx.x != 0) {
        return;
    }

    if (first_valid >= series_len) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0) {
        return;
    }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);

    const int warm = first_valid + period - 1;
    const int start = first_valid;

    float ema = prices[start];
    float ema2 = ema;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif

    if (start >= warm && start < series_len) {
        out[base + start] = 2.0f * ema - ema2;
    }

    for (int t = start + 1; t < series_len; ++t) {
        const float x = prices[t];
#if USE_DEMA_COMPENSATION
        // ema += alpha * (x - ema) with one-float error feedback
        float inc1 = fmaf(alpha, x - ema, -c1);
        float tmp1 = ema + inc1;
        c1 = (tmp1 - ema) - inc1;
        ema = tmp1;

        // ema2 += alpha * (ema - ema2) with compensation
        float inc2 = fmaf(alpha, ema - ema2, -c2);
        float tmp2 = ema2 + inc2;
        c2 = (tmp2 - ema2) - inc2;
        ema2 = tmp2;
#else
        // Plain FMA delta form: one rounding per update
        ema  = fmaf(alpha, x - ema,    ema);
        ema2 = fmaf(alpha, ema - ema2, ema2);
#endif
        if (t >= warm) {
            out[base + t] = fmaf(2.0f, ema, -ema2);
        }
    }
}

// Many-series (time-major) kernel: each block handles one series.
// Thread 0 in the block performs the sequential DEMA recurrence for that
// series/parameter while other threads initialize the NaN prefix in parallel.
extern "C" __global__
void dema_many_series_one_param_time_major_f32(const float* __restrict__ prices_tm,
                                               const int* __restrict__ first_valids,
                                               int period,
                                               int num_series,
                                               int series_len,
                                               float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) { return; }
    if (period <= 0 || series_len <= 0) { return; }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];
    if (first_valid >= series_len) { return; }

    // Initialize NaN prefix/time in parallel
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) { return; }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const int warm = first_valid + period - 1;

    // Seed EMAs from first finite sample for this series
    float ema = prices_tm[first_valid * stride + series_idx];
    float ema2 = ema;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif

    if (first_valid >= warm && first_valid < series_len) {
        out_tm[first_valid * stride + series_idx] = 2.0f * ema - ema2;
    }

    for (int t = first_valid + 1; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
#if USE_DEMA_COMPENSATION
        float inc1 = fmaf(alpha, x - ema, -c1);
        float tmp1 = ema + inc1;
        c1 = (tmp1 - ema) - inc1;
        ema = tmp1;

        float inc2 = fmaf(alpha, ema - ema2, -c2);
        float tmp2 = ema2 + inc2;
        c2 = (tmp2 - ema2) - inc2;
        ema2 = tmp2;
#else
        ema  = fmaf(alpha, x - ema,    ema);
        ema2 = fmaf(alpha, ema - ema2, ema2);
#endif
        if (t >= warm) {
            out_tm[t * stride + series_idx] = fmaf(2.0f, ema, -ema2);
        }
    }
}
