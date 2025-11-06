// CUDA kernels for Exponential Moving Average (EMA).
//
// Both kernels operate in FP32 and mirror the scalar CPU implementation by
// using an initial running-mean warmup followed by the standard EMA recurrence.
// These implementations reduce global memory traffic by initializing only the
// [0, first_valid) prefix to NaN (no full-series memset) and avoid unnecessary
// block-wide barriers. The warmup loop uses Welford’s update form with FMA and
// single-precision division to tighten the instruction mix while preserving
// numerical behavior.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Enable a light-touch L2 prefetch hint by default on newer toolchains.
// This stays file-local (no build.rs changes) and can be disabled by defining
// EMA_DISABLE_L2_PREFETCH at compile time if ever needed.
#if !defined(EMA_USE_L2_PREFETCH) && !defined(EMA_DISABLE_L2_PREFETCH)
#  if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
#    define EMA_USE_L2_PREFETCH 1
#  endif
#endif

#if defined(EMA_USE_L2_PREFETCH)
__device__ __forceinline__ void prefetch_L2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

extern "C" __global__
void ema_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) return;

    const float alpha = alphas[combo];
    const int   base  = combo * series_len;

    // Initialize only [0, first_valid) to NaN (no overlap with subsequent writes).
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[base + idx] = NAN;
    }

    // Only thread 0 runs the sequential recurrence.
    if (threadIdx.x != 0) return;

    // Warmup window end (clamped).
    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    // Running-mean warmup (Welford update form).
    float mean = prices[first_valid];
    out[base + first_valid] = mean;
    int valid_count = 1;
    for (int i = first_valid + 1; i < warm_end; ++i) {
        const float x = prices[i];
        if (isfinite(x)) {
            ++valid_count;
            const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
            mean = __fmaf_rn(x - mean, inv, mean);
        }
        out[base + i] = mean;
    }

    // EMA recurrence
    float prev = mean;
#if defined(EMA_USE_L2_PREFETCH)
    constexpr int PREFETCH_DIST = 64; // 64 floats (~256B)
#endif
    for (int i = warm_end; i < series_len; ++i) {
#if defined(EMA_USE_L2_PREFETCH)
        if (i + PREFETCH_DIST < series_len) prefetch_L2(&prices[i + PREFETCH_DIST]);
#endif
        const float x = prices[i];
        if (isfinite(x)) {
            // prev = prev + alpha*(x - prev)
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out[base + i] = prev;
    }
}

extern "C" __global__
void ema_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x; // one block per series (compat mode)
    if (series_idx >= num_series || period <= 0 || series_len <= 0) return;

    const int stride = num_series; // time-major
    int first_valid  = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len)    return;

    // Initialize only [0, first_valid) to NaN for this series.
    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }

    if (threadIdx.x != 0) return;

    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    float mean = prices_tm[first_valid * stride + series_idx];
    out_tm[first_valid * stride + series_idx] = mean;

    int valid_count = 1;
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            ++valid_count;
            const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
            mean = __fmaf_rn(x - mean, inv, mean);
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

// Optional: warp-coalesced mapping (one thread per series). This kernel
// expects grid.x * blockDim.x >= num_series and typically benefits time-major
// layouts by issuing contiguous loads/stores per warp at each timestep.
extern "C" __global__
void ema_many_series_one_param_f32_coalesced(const float* __restrict__ prices_tm,
                                             const int*   __restrict__ first_valids,
                                             int period,
                                             float alpha,
                                             int num_series,
                                             int series_len,
                                             float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series || period <= 0 || series_len <= 0) return;

    const int stride      = num_series; // time-major
    const int first_valid = max(0, first_valids[series_idx]);
    const int warm_end    = min(series_len, first_valid + period);

    // Prefix NaN init (no sync required)
    for (int t = 0; t < first_valid; ++t) {
        out_tm[t * stride + series_idx] = NAN;
    }

    float mean = prices_tm[first_valid * stride + series_idx];
    out_tm[first_valid * stride + series_idx] = mean;

    int valid_count = 1;
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            ++valid_count;
            const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
            mean = __fmaf_rn(x - mean, inv, mean);
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
