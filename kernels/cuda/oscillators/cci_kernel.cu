// CUDA kernels for Commodity Channel Index (CCI)
//
// Semantics mirror the scalar Rust implementation in src/indicators/cci.rs:
// - Warmup prefix of NaNs at indices [0 .. first_valid + period - 1)
// - NaN inputs propagate (any NaN in the active window yields NaN via FP ops)
// - Denominator zero -> 0.0 (when mean absolute deviation is exactly 0)
// - FP32 arithmetic for throughput and interoperability with existing wrappers

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// One-series × many-params (batch). Each block processes a single period row.
extern "C" __global__ void cci_batch_f32(const float* __restrict__ prices,
                                          const int*   __restrict__ periods,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    // Fill row with NaN up-front (parallel across threads).
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    // Single thread performs the sequential scan per row (simple and robust).
    if (threadIdx.x != 0) return;

    if (UNLIKELY(period <= 0 || period > series_len)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) return;

    const float inv_p   = 1.0f / static_cast<float>(period);
    const float cci_mul = (static_cast<float>(period)) * (1.0f / 0.015f);

    const int warm = first_valid + period - 1;
    // Initial rolling sum for SMA on [first_valid .. first_valid+period-1]
    float sum = 0.0f;
    const float* p0 = prices + first_valid;
    for (int k = 0; k < period; ++k) sum += p0[k];
    float sma = sum * inv_p;

    // Compute first MAD and CCI at index = warm
    {
        float sum_abs = 0.0f;
        const float* wptr = prices + (warm - period + 1);
        for (int k = 0; k < period; ++k) {
            float d = wptr[k] - sma;
            sum_abs += fabsf(d);
        }
        float denom = 0.015f * (sum_abs * inv_p);
        float px = prices[warm];
        out[base + warm] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }

    // Roll forward
    for (int t = warm + 1; t < series_len; ++t) {
        sum += prices[t];
        sum -= prices[t - period];
        sma = sum * inv_p;

        float sum_abs = 0.0f;
        const float* wptr = prices + (t - period + 1);
        for (int k = 0; k < period; ++k) {
            float d = wptr[k] - sma;
            sum_abs += fabsf(d);
        }

        float denom = 0.015f * (sum_abs * inv_p);
        float px = prices[t];
        out[base + t] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }
}

// Many-series × one-param (time-major). Each thread handles one series (column).
extern "C" __global__ void cci_many_series_one_param_f32(
    const float* __restrict__ prices_tm,   // [row * num_series + series]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)            // time-major output
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* col_in  = prices_tm + series;
    float*       col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    // Warmup prefix
    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) col_out[r * num_series] = NAN;

    // Initial rolling sum for SMA
    float sum = 0.0f;
    const float inv_p = 1.0f / static_cast<float>(period);
    const float* p = col_in + static_cast<size_t>(first_valid) * num_series;
    for (int k = 0; k < period; ++k, p += num_series) sum += *p;
    float sma = sum * inv_p;

    // First MAD / CCI at warm
    {
        float sum_abs = 0.0f;
        const float* w = col_in + static_cast<size_t>(warm - period + 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            float d = *w - sma;
            sum_abs += fabsf(d);
        }
        float denom = 0.015f * (sum_abs * inv_p);
        float px = *(col_in + static_cast<size_t>(warm) * num_series);
        *(col_out + static_cast<size_t>(warm) * num_series) = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }

    // Rolling
    const float* cur = col_in + static_cast<size_t>(warm + 1) * num_series;
    const float* old = col_in + static_cast<size_t>(first_valid) * num_series;
    float* dst       = col_out + static_cast<size_t>(warm + 1) * num_series;
    for (int r = warm + 1; r < series_len; ++r) {
        sum += *cur;
        sum -= *old;
        sma = sum * inv_p;

        float sum_abs = 0.0f;
        const float* w = cur - static_cast<size_t>(period - 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            float d = *w - sma;
            sum_abs += fabsf(d);
        }
        float denom = 0.015f * (sum_abs * inv_p);
        *dst = (denom == 0.0f) ? 0.0f : ((*cur) - sma) / denom;
        cur += num_series;
        old += num_series;
        dst += num_series;
    }
}

