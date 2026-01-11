// CUDA kernels for Exponential Moving Average (EMA).
//
// Both kernels operate in FP32 and mirror the scalar CPU implementation by
// using an initial running-mean warmup followed by the standard EMA recurrence.
// These implementations reduce global memory traffic by initializing only the
// [0, first_valid) prefix to NaN (no full-series memset) and avoid unnecessary
// block-wide barriers. The warmup loop uses Welford’s update form with FMA and
// single-precision division to tighten the instruction mix while preserving
// numerical behavior.
//
// NOTE: Kernel symbol names are part of the wrapper contract. The Rust wrapper
// resolves the following symbols and will return a readable MissingKernelSymbol
// error if one is absent:
//   - "ema_batch_f32"
//   - "ema_many_series_one_param_f32"
//   - "ema_many_series_one_param_f32_coalesced" (optional fast path)
// Keep names stable unless coordinated with the wrapper.

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
    const float one_minus_alpha = 1.0f - alpha;
    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    // Initialize only [0, first_valid) to NaN (no overlap with subsequent writes).
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[base + static_cast<size_t>(idx)] = NAN;
    }

    // Compatibility: if launched with < 1 warp, fall back to the sequential path.
    if (blockDim.x < 32) {
        if (threadIdx.x != 0) return;

        // Warmup window end (clamped).
        int warm_end = first_valid + period;
        if (warm_end > series_len) warm_end = series_len;

        // Running-mean warmup (Welford update form).
        float mean = prices[first_valid];
        out[base + static_cast<size_t>(first_valid)] = mean;
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float x = prices[i];
            if (isfinite(x)) {
                ++valid_count;
                const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
                mean = __fmaf_rn(x - mean, inv, mean);
            }
            out[base + static_cast<size_t>(i)] = mean;
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
            out[base + static_cast<size_t>(i)] = prev;
        }
        return;
    }

    // Warp-cooperative scan: one warp per combo, emitting 32 outputs per iteration.
    // Launch with blockDim.x == 32 for best occupancy (extra threads early-out below).
    if (threadIdx.x >= 32) return;

    const unsigned lane = static_cast<unsigned>(threadIdx.x); // 0..31
    const unsigned mask = 0xffffffffu;

    // Warmup window end (clamped).
    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    float prev = 0.0f;
    if (lane == 0) {
        // Running-mean warmup (Welford update form).
        float mean = prices[first_valid];
        out[base + static_cast<size_t>(first_valid)] = mean;
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float x = prices[i];
            if (isfinite(x)) {
                ++valid_count;
                const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
                mean = __fmaf_rn(x - mean, inv, mean);
            }
            out[base + static_cast<size_t>(i)] = mean;
        }
        prev = mean;
    }
    // Broadcast the warmup terminal value to all lanes.
    prev = __shfl_sync(mask, prev, 0);

#if defined(EMA_USE_L2_PREFETCH)
    constexpr int PREFETCH_DIST = 256; // floats (~1KB)
#endif
    for (int t0 = warm_end; t0 < series_len; t0 += 32) {
#if defined(EMA_USE_L2_PREFETCH)
        if (lane == 0) {
            const int pf = t0 + PREFETCH_DIST;
            if (pf < series_len) prefetch_L2(&prices[pf]);
        }
#endif
        const int t = t0 + static_cast<int>(lane);

        // Per-step affine transform y = A*y_prev + B.
        float A = 1.0f;
        float B = 0.0f;
        if (t < series_len) {
            const float x = prices[t];
            if (isfinite(x)) {
                A = one_minus_alpha;
                B = alpha * x;
            }
        }

        // Inclusive warp scan: compose transforms (A,B) left-to-right.
        // Composition: (A1,B1) ∘ (A2,B2) = (A1*A2, A1*B2 + B1).
        // We want prefix up to lane: T_lane ∘ ... ∘ T_0.
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }

        const float y = __fmaf_rn(A, prev, B);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = y;
        }

        // Advance to next tile using the last valid lane.
        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        prev = __shfl_sync(mask, y, last_lane);
    }
}

// FP64-internal EMA surfaces (one series x many params), written as f32.
// This exists to support composite indicators that are numerically sensitive
// to EMA drift (e.g., PPO ratio when slow EMA is near zero) while keeping a
// warp-cooperative implementation for performance.
extern "C" __global__
void ema_batch_f64_to_f32(const float* __restrict__ prices,
                          const int*   __restrict__ periods,
                          int series_len,
                          int first_valid,
                          int n_combos,
                          float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) return;

    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    const double one_minus_alpha = 1.0 - alpha;
    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    // Initialize only [0, first_valid) to NaN.
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[base + static_cast<size_t>(idx)] = NAN;
    }

    // Compatibility: < 1 warp -> sequential.
    if (blockDim.x < 32) {
        if (threadIdx.x != 0) return;

        int warm_end = first_valid + period;
        if (warm_end > series_len) warm_end = series_len;

        double mean = static_cast<double>(prices[first_valid]);
        out[base + static_cast<size_t>(first_valid)] = static_cast<float>(mean);

        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                ++valid_count;
                const double x = static_cast<double>(xf);
                const double vc = static_cast<double>(valid_count);
                // Match scalar EMA warmup: mean = ((vc - 1) * mean + x) / vc
                mean = ((vc - 1.0) * mean + x) / vc;
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(mean);
        }

        double prev = mean;
        for (int i = warm_end; i < series_len; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                const double x = static_cast<double>(xf);
                // prev = (1 - alpha) * prev + alpha * x
                prev = (one_minus_alpha * prev) + (alpha * x);
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(prev);
        }
        return;
    }

    // Warp-cooperative scan (one warp per combo).
    if (threadIdx.x >= 32) return;
    const unsigned lane = static_cast<unsigned>(threadIdx.x);
    const unsigned mask = 0xffffffffu;

    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    double prev = 0.0;
    if (lane == 0) {
        double mean = static_cast<double>(prices[first_valid]);
        out[base + static_cast<size_t>(first_valid)] = static_cast<float>(mean);
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                ++valid_count;
                const double x = static_cast<double>(xf);
                const double vc = static_cast<double>(valid_count);
                mean = ((vc - 1.0) * mean + x) / vc;
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(mean);
        }
        prev = mean;
    }
    prev = __shfl_sync(mask, prev, 0);

    for (int t0 = warm_end; t0 < series_len; t0 += 32) {
        const int t = t0 + static_cast<int>(lane);

        // Affine step y = A*y_prev + B in double.
        double A = 1.0;
        double B = 0.0;
        if (t < series_len) {
            const float xf = prices[t];
            if (isfinite(xf)) {
                A = one_minus_alpha;
                B = alpha * static_cast<double>(xf);
            }
        }

        // Inclusive warp scan composing (A,B) left-to-right.
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double A_prev = __shfl_up_sync(mask, A, offset);
            const double B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const double A_cur = A;
                const double B_cur = B;
                A = A_cur * A_prev;
                B = fma(A_cur, B_prev, B_cur);
            }
        }

        const double y = fma(A, prev, B);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = static_cast<float>(y);
        }

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        prev = __shfl_sync(mask, y, last_lane);
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
