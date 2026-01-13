












#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
#define EFI_USE_L2_PREFETCH 1
#endif

#if defined(EFI_USE_L2_PREFETCH)
__device__ __forceinline__ void prefetch_L2(const void* p) {

    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif


__device__ __forceinline__ bool finite_f32(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}


__device__ __forceinline__ void kahan_add(float& sum, float y, float& c) {
    float z = y - c;
    float t = sum + z;
    c = (t - sum) - z;
    sum = t;
}




extern "C" __global__
void efi_precompute_diffs_f32(const float* __restrict__ prices,
                              const float* __restrict__ volumes,
                              int series_len,
                              float* __restrict__ diffs,
                              int* __restrict__ warm_out ) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) diffs[0] = NAN;

    int local_warm = series_len;

#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PDIST = 128;
#endif


    for (int t = gid + 1; t < series_len; t += blockDim.x * gridDim.x) {
#if defined(EFI_USE_L2_PREFETCH)
        if (t + PDIST < series_len) {
            prefetch_L2(&prices[t + PDIST]);
            prefetch_L2(&volumes[t + PDIST]);
        }
#endif
        const float pp = prices[t - 1];
        const float pc = prices[t];
        const float vc = volumes[t];

        float d = NAN;
        if (finite_f32(pc) && finite_f32(pp) && finite_f32(vc)) {
            d = __fmaf_rn(pc - pp, vc, 0.0f);
            if (t < local_warm) local_warm = t;
        }
        diffs[t] = d;
    }

    if (warm_out) {
        if (local_warm < series_len) atomicMin(warm_out, local_warm);
    }
}

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


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out[base + i] = NAN;
    }


    if (threadIdx.x != 0) return;


    float prev = (prices[warm] - prices[warm - 1]) * volumes[warm];
    out[base + warm] = prev;


#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PREFETCH_DIST = 64;
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

            prev = __fmaf_rn(diff - prev, alpha, prev);
        }
        out[base + t] = prev;
    }
}




extern "C" __global__
void efi_batch_from_diff_f32(const float* __restrict__ diffs,
                             const int*   __restrict__ periods,
                             const float* __restrict__ alphas,
                             int series_len,
                             int warm,
                             int n_combos,
                             float* __restrict__ out) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos) && (periods[combo] > 0) && (series_len > 0) && (warm < series_len);
    const unsigned warp_mask = __ballot_sync(0xffffffff, active);
    if (warp_mask == 0) return;


    if (active) {
        const int base = combo * series_len;

        for (int t = 0; t < warm; ++t) { out[base + t] = NAN; }
    }

    const int lane = threadIdx.x & 31;
    const int src_lane = __ffs(warp_mask) - 1;

    if (!active) return;

    const int base = combo * series_len;
    float prev = diffs[warm];
    float c = 0.0f;
    const float alpha = alphas[combo];
    out[base + warm] = prev;

#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PDIST = 128;
#endif

    for (int t = warm + 1; t < series_len; ++t) {
        float d = 0.0f;
#if defined(EFI_USE_L2_PREFETCH)
        if (lane == src_lane && (t + PDIST) < series_len) prefetch_L2(&diffs[t + PDIST]);
#endif
        if (lane == src_lane) d = diffs[t];
        d = __shfl_sync(warp_mask, d, src_lane);

        if (finite_f32(d)) {

            const float y = __fmaf_rn(alpha, (d - prev), 0.0f);
            kahan_add(prev, y, c);
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

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series || series_len <= 0 || period <= 0) return;

    const int stride = num_series;
    int warm = first_valids_diff[s];
    if (warm < 1 || warm >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * stride + s] = NAN;
        }
        return;
    }


    for (int t = 0; t < warm; ++t) {
        out_tm[t * stride + s] = NAN;
    }


    const float pcw = prices_tm[warm * stride + s];
    const float ppw = prices_tm[(warm - 1) * stride + s];
    const float vcw = volumes_tm[warm * stride + s];
    float prev = __fmaf_rn(pcw - ppw, vcw, 0.0f);
    float c = 0.0f;
    out_tm[warm * stride + s] = prev;

#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PDIST = 64;
#endif

    for (int t = warm + 1; t < series_len; ++t) {
#if defined(EFI_USE_L2_PREFETCH)
        if ((t + PDIST) < series_len) {
            prefetch_L2(&prices_tm[(t + PDIST) * stride + s]);
            prefetch_L2(&volumes_tm[(t + PDIST) * stride + s]);
        }
#endif
        const float pc = prices_tm[t * stride + s];
        const float pp = prices_tm[(t - 1) * stride + s];
        const float vc = volumes_tm[t * stride + s];

        if (finite_f32(pc) && finite_f32(pp) && finite_f32(vc)) {
            const float diff = __fmaf_rn(pc - pp, vc, 0.0f);
            const float y = __fmaf_rn(alpha, (diff - prev), 0.0f);
            kahan_add(prev, y, c);
        }
        out_tm[t * stride + s] = prev;
    }
}



extern "C" __global__
void efi_one_series_many_params_from_diff_tm_f32(
        const float* __restrict__ diffs,
        const int*   __restrict__ periods,
        const float* __restrict__ alphas,
        int series_len,
        int warm,
        int n_combos,
        float* __restrict__ out_tm) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos) && (periods[combo] > 0);
    const unsigned warp_mask = __ballot_sync(0xffffffff, active);
    if (warp_mask == 0) return;

    if (active) {
        for (int t = 0; t < warm; ++t) { out_tm[t * n_combos + combo] = NAN; }
    }
    const int lane = threadIdx.x & 31;
    const int src_lane = __ffs(warp_mask) - 1;

    if (!active) return;

    float prev = diffs[warm];
    float c = 0.0f;
    const float alpha = alphas[combo];
    out_tm[warm * n_combos + combo] = prev;

#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PDIST = 128;
#endif

    for (int t = warm + 1; t < series_len; ++t) {
        float d = 0.0f;
#if defined(EFI_USE_L2_PREFETCH)
        if (lane == src_lane && (t + PDIST) < series_len) prefetch_L2(&diffs[t + PDIST]);
#endif
        if (lane == src_lane) d = diffs[t];
        d = __shfl_sync(warp_mask, d, src_lane);

        if (finite_f32(d)) {
            const float y = __fmaf_rn(alpha, (d - prev), 0.0f);
            kahan_add(prev, y, c);
        }
        out_tm[t * n_combos + combo] = prev;
    }
}


extern "C" __global__
void efi_one_series_many_params_from_diff_rm_f32(
        const float* __restrict__ diffs,
        const int*   __restrict__ periods,
        const float* __restrict__ alphas,
        int series_len,
        int warm,
        int n_combos,
        float* __restrict__ out_rm) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos) && (periods[combo] > 0);
    const unsigned warp_mask = __ballot_sync(0xffffffff, active);
    if (warp_mask == 0) return;

    if (active) {
        const int base = combo * series_len;
        for (int t = 0; t < warm; ++t) { out_rm[base + t] = NAN; }
    }

    const int lane = threadIdx.x & 31;
    const int src_lane = __ffs(warp_mask) - 1;

    if (!active) return;

    const int base = combo * series_len;
    float prev = diffs[warm];
    float c = 0.0f;
    const float alpha = alphas[combo];
    out_rm[base + warm] = prev;

#if defined(EFI_USE_L2_PREFETCH)
    constexpr int PDIST = 128;
#endif

    for (int t = warm + 1; t < series_len; ++t) {
        float d = 0.0f;
#if defined(EFI_USE_L2_PREFETCH)
        if (lane == src_lane && (t + PDIST) < series_len) prefetch_L2(&diffs[t + PDIST]);
#endif
        if (lane == src_lane) d = diffs[t];
        d = __shfl_sync(warp_mask, d, src_lane);

        if (finite_f32(d)) {
            const float y = __fmaf_rn(alpha, (d - prev), 0.0f);
            kahan_add(prev, y, c);
        }
        out_rm[base + t] = prev;
    }
}
