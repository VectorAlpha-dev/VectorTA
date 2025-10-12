// Optimized CUDA kernels for Simple Moving Average (SMA).
// CUDA 13, targeting Ada (SM 8.9) and later. Drop-in compatible with wrapper.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef SMA_NAN
#define SMA_NAN (__int_as_float(0x7fffffff))
#endif

// Optional: enable compensated summation for improved numerical stability.
// #define SMA_USE_KAHAN 1

// Branch prediction hints
#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void sma_batch_f32(const float* __restrict__ prices,
                                         const int*   __restrict__ periods,
                                         int series_len,
                                         int n_combos,
                                         int first_valid,
                                         float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int  period  = periods[combo];
    const int  base    = combo * series_len;
    float*     out_ptr = out + base;

    // Invalid inputs: write full NaN and return (preserve behavior)
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = SMA_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = SMA_NAN;
        return;
    }

    const int   warm = first_valid + period - 1;  // first valid SMA index
    const float inv  = 1.0f / static_cast<float>(period);

    // Prefill only [0, warm) with NaN
    for (int i = 0; i < warm; ++i) out_ptr[i] = SMA_NAN;

    if (period == 1) {
        // Copy tail directly
        const float* src = prices + first_valid;
        float*       dst = out_ptr + first_valid;
        for (int i = first_valid; i < series_len; ++i) *dst++ = *src++;
        return;
    }

    // Initial window sum
    float sum = 0.0f;
#ifdef SMA_USE_KAHAN
    float c = 0.0f;
#endif
    const float* p0 = prices + first_valid;
    for (int k = 0; k < period; ++k) {
#ifdef SMA_USE_KAHAN
        float y = p0[k] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
#else
        sum += p0[k];
#endif
    }
    out_ptr[warm] = sum * inv;

    // Rolling updates using pointer bumping
    const float* cur = prices + (warm + 1);
    const float* old = prices + first_valid;
    float*       dst = out_ptr + (warm + 1);
    for (int i = warm + 1; i < series_len; ++i) {
#ifdef SMA_USE_KAHAN
        float delta = (*cur++) - (*old++);
        float y = delta - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        *dst++ = sum * inv;
#else
        sum += *cur++;
        sum -= *old++;
        *dst++ = sum * inv;
#endif
    }
}

extern "C" __global__ void sma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,   // time-major: [row * num_series + series]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)            // time-major
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    // Column pointers for this series in time-major layout.
    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;

    // Invalid cases: write full NaN and return
    if (UNLIKELY(period <= 0 || period > series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = SMA_NAN;
        return;
    }

    const int   warm = first_valid + period - 1;
    const float inv  = 1.0f / static_cast<float>(period);

    // Prefill NaN up to warm-1
    {
        float* o = col_out;
        for (int row = 0; row < warm; ++row, o += num_series) *o = SMA_NAN;
    }

    if (period == 1) {
        // Copy tail for this series (time-major pointer bump)
        const float* src = col_in  + static_cast<size_t>(first_valid) * num_series;
        float*       dst = col_out + static_cast<size_t>(first_valid) * num_series;
        for (int row = first_valid; row < series_len; ++row, src += num_series, dst += num_series)
            *dst = *src;
        return;
    }

    // Initial window sum (strided by num_series)
    float sum = 0.0f;
#ifdef SMA_USE_KAHAN
    float c = 0.0f;
#endif
    const float* p = col_in + static_cast<size_t>(first_valid) * num_series;
    for (int k = 0; k < period; ++k, p += num_series) {
#ifdef SMA_USE_KAHAN
        float y = *p - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
#else
        sum += *p;
#endif
    }

    // Store first average at warm
    *(col_out + static_cast<size_t>(warm) * num_series) = sum * inv;

    // Rolling averages
    const float* cur = col_in  + static_cast<size_t>(warm + 1)    * num_series;
    const float* old = col_in  + static_cast<size_t>(first_valid) * num_series;
    float*       dst = col_out + static_cast<size_t>(warm + 1)    * num_series;

    for (int row = warm + 1; row < series_len; ++row) {
#ifdef SMA_USE_KAHAN
        float delta = (*cur) - (*old);
        float y = delta - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        *dst = sum * inv;
#else
        sum += *cur;
        sum -= *old;
        *dst = sum * inv;
#endif
        cur += num_series;
        old += num_series;
        dst += num_series;
    }
}
