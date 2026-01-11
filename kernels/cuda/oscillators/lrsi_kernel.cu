// CUDA kernels for Laguerre RSI (LRSI)
//
// Semantics (unchanged):
// - Warmup: first_valid + 3 elements are NaN
// - NaN inputs do not advance state; post-warm outputs become NaN for those slots
// - sum_abs <= FLT_EPSILON => output 0.0
// - FP32 math; FMA ordering mirrors scalar for numerical stability
//
// This version optimizes the one-series-many-params path and improves the
// many-series-one-param path. It relies only on standard CUDA C++ (no special
// build flags). CUDA 13 / cc>=8.6+ friendly (Ada+).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LRDG(ptr) __ldg(ptr)
#else
  #define LRDG(ptr) (*(ptr))
#endif

// Warp-scope broadcast of a single global value with one load per active warp.
// Works for partial warps: elects a live lane as leader.
static __device__ __forceinline__ float warp_broadcast_ldg(const float* addr) {
    unsigned mask = __activemask();
    int leader   = __ffs(mask) - 1;                       // lane id of first active
    int lane     = threadIdx.x & 31;
    float v = 0.0f;
    if (lane == leader) v = LRDG(addr);
    return __shfl_sync(mask, v, leader);
}

// --- Helper: one step of 4-stage Laguerre filter (FMA ordering preserved)
static __device__ __forceinline__
void laguerre4_step(float p, float alpha, float gamma, float mgamma,
                    float &l0, float &l1, float &l2, float &l3,
                    float &t0, float &t1, float &t2, float &t3) {
    // 4-stage Laguerre filter with FMAs, mirrors scalar ordering
    t0 = fmaf(alpha, (p - l0), l0);
    t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
    t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
    t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));
    l0 = t0; l1 = t1; l2 = t2; l3 = t3;
}

// =========================================================================
// One price series, many parameter combos (main throughput kernel)
// Layout: out is [n_combos x series_len] row-major (combo-major).
// =========================================================================

extern "C" __global__
void lrsi_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ alphas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    // Grid-stride over parameter combos so launches can scale across SMs.
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x) {
        const int base = combo * series_len;

        // Validate inputs for this combo
        if (first_valid < 0 || first_valid >= series_len) {
            // Fill with NaNs to preserve semantics and continue
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }

        const float alpha = alphas[combo];
        if (!(alpha > 0.0f && alpha < 1.0f)) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }
        const float gamma  = 1.0f - alpha;
        const float mgamma = -gamma;

        const int warm = first_valid + 3; // need 4 valid values total
        if (warm >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = NAN;
            continue;
        }

        // Warmup prefix: ensure NaNs up to warm-1
        for (int t = 0; t < warm; ++t) out[base + t] = NAN;

        // Initialize filter state from first valid price
        const float p0 = prices[first_valid];
        float l0 = p0, l1 = p0, l2 = p0, l3 = p0;

        // Phase 1: advance to warm-1 without output
        for (int t = first_valid + 1; t < warm; ++t) {
            const float p = warp_broadcast_ldg(prices + t);
            if (isnan(p)) continue;
            const float t0 = fmaf(alpha, (p - l0), l0);
            const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
            const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
            const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));
            l0 = t0; l1 = t1; l2 = t2; l3 = t3;
        }

        // Phase 2: output path
        for (int t = warm; t < series_len; ++t) {
            const float p = warp_broadcast_ldg(prices + t);
            if (isnan(p)) { out[base + t] = NAN; continue; }

            const float t0 = fmaf(alpha, (p - l0), l0);
            const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
            const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
            const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));

            l0 = t0; l1 = t1; l2 = t2; l3 = t3;

            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;
            if (sum_abs <= FLT_EPSILON) {
                out[base + t] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out[base + t] = cu / sum_abs;
            }
        }
    }
}

// =========================================================================
// Many series (time-major), one parameter.
// In/out are time-major [series_len rows x num_series cols].
// =========================================================================

extern "C" __global__
void lrsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    float alpha,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    // Grid-stride across series (columns)
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_series;
         s += blockDim.x * gridDim.x) {
        if (!(alpha > 0.0f && alpha < 1.0f)) {
            // Invalid alpha => fill column with NaN and continue
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const float gamma  = 1.0f - alpha;
        const float mgamma = -gamma;

        const int first = max(0, first_valids[s]);
        const int warm  = first + 3;

        // If there is no valid warm window, fill column with NaN (makes behavior explicit)
        if (first >= series_len || warm >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + s] = NAN;
            continue;
        }

        const int cols = num_series;

        // Warmup prefix
        for (int t = 0; t < warm; ++t) out_tm[t * cols + s] = NAN;

        // Initialize filter state from first valid price
        const int idx0 = first * cols + s;
        float l0 = prices_tm[idx0];
        float l1 = l0, l2 = l0, l3 = l0;

        // Phase 1: advance to warm-1 (no output)
        for (int t = first + 1; t < warm; ++t) {
            const float p = prices_tm[t * cols + s]; // coalesced across s at fixed t
            if (isnan(p)) continue;
            float t0, t1, t2, t3;
            laguerre4_step(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);
        }

        // Phase 2: output path
        for (int t = warm; t < series_len; ++t) {
            const int idx = t * cols + s;
            const float p = prices_tm[idx]; // coalesced
            if (isnan(p)) { out_tm[idx] = NAN; continue; }

            float t0, t1, t2, t3;
            laguerre4_step(p, alpha, gamma, mgamma, l0, l1, l2, l3, t0, t1, t2, t3);

            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;

            if (sum_abs <= FLT_EPSILON) {
                out_tm[idx] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out_tm[idx]   = cu / sum_abs;
            }
        }
    }
}
