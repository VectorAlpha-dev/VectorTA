// CUDA kernels for Elder's Force Index (EFI)
//
// Math pattern: recurrence/IIR over time where the input per-step is
// diff = (price[t] - price[t-1]) * volume[t]. The output is an EMA of this
// diff stream. Warmup semantics match the scalar implementation:
// - Let `warm` be the first index t where price[t], price[t-1], and volume[t]
//   are all finite.
// - For [0, warm) we write NaN.
// - At t = warm, seed prev = diff(warm) and write it.
// - For t > warm: if the triple at t is finite, update
//     prev = prev + alpha * (diff(t) - prev)
//   else carry `prev` forward unchanged.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
#define EFI_USE_L2_PREFETCH 1
#endif

#if defined(EFI_USE_L2_PREFETCH)
__device__ __forceinline__ void prefetch_L2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

extern "C" __global__
void efi_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ volumes,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   int series_len,
                   int warm,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    if (warm >= series_len) return;
    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    if (period <= 0) return;

    const int base = combo * series_len;

    // Prefix NaNs (only [0, warm))
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = NAN;
    }

    // Single-thread sequential scan per combo
    if (threadIdx.x != 0) return;

    // Seed at `warm` using finite triple (by construction of warm)
    float prev = (prices[warm] - prices[warm - 1]) * volumes[warm];
    out[base + warm] = prev;

    // Main scan
#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PREFETCH_DIST = 64; // floats (~256B)
#endif
    for (int t = warm + 1; t < series_len; ++t) {
#if defined(EFI_USE_L2_PREFETCH)
        if (t + PREFETCH_DIST < series_len) prefetch_L2(&prices[t + PREFETCH_DIST]);
#endif
        const float pc = prices[t];
        const float pp = prices[t - 1];
        const float vc = volumes[t];
        if (isfinite(pc) && isfinite(pp) && isfinite(vc)) {
            const float diff = (pc - pp) * vc;
            // prev = prev + alpha * (diff - prev)
            prev = __fmaf_rn(diff - prev, alpha, prev);
        }
        out[base + t] = prev;
    }
}

// Variant that consumes precomputed diffs with NaNs for invalid indices.
// This avoids recomputing (p[t]-p[t-1])*v[t] per combo and shares that work
// across all parameter rows.
extern "C" __global__
void efi_batch_from_diff_f32(const float* __restrict__ diffs,
                             const int*   __restrict__ periods,
                             const float* __restrict__ alphas,
                             int series_len,
                             int warm,
                             int n_combos,
                             float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;
    if (warm >= series_len) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    if (period <= 0) return;

    const int base = combo * series_len;

    // Prefix NaNs
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = NAN;
    }
    if (threadIdx.x != 0) return;

    float prev = diffs[warm];
    out[base + warm] = prev;
    for (int t = warm + 1; t < series_len; ++t) {
        const float x = diffs[t];
        if (isfinite(x)) {
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out[base + t] = prev;
    }
}

extern "C" __global__
void efi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const float* __restrict__ volumes_tm,
                                   const int*   __restrict__ first_valids_diff,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x; // one block per series (compat mode)
    if (series_idx >= num_series || series_len <= 0 || period <= 0) return;

    const int stride = num_series; // time-major layout
    int warm = first_valids_diff[series_idx];
    if (warm < 0) warm = 0;
    if (warm >= series_len) return;

    // Prefix NaNs for this series
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }

    if (threadIdx.x != 0) return;

    // Seed at `warm`
    float prev = (prices_tm[warm * stride + series_idx] -
                  prices_tm[(warm - 1) * stride + series_idx]) *
                 volumes_tm[warm * stride + series_idx];
    out_tm[warm * stride + series_idx] = prev;

    // Sequential scan
    for (int t = warm + 1; t < series_len; ++t) {
        const float pc = prices_tm[t * stride + series_idx];
        const float pp = prices_tm[(t - 1) * stride + series_idx];
        const float vc = volumes_tm[t * stride + series_idx];
        if (isfinite(pc) && isfinite(pp) && isfinite(vc)) {
            const float diff = (pc - pp) * vc;
            prev = __fmaf_rn(diff - prev, alpha, prev);
        }
        out_tm[t * stride + series_idx] = prev;
    }
}
