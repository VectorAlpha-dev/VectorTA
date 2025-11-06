// CUDA kernels for the Midway Weighted Exponential (MWDX) indicator.
// Optimized for Ada (SM 89, RTX 4090) and newer while preserving exact scalar semantics.
//
// Category: Recurrence/IIR (EMA-like). Each parameter combo or series is
// processed sequentially in time by a single thread (lane 0) to preserve the
// exact scalar semantics. Other threads in the CTA help initialize outputs to
// NaN only up to the warm region, avoiding unnecessary global stores and
// removing the CTA-wide barrier.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Optional device-side access property hint (CUDA 12+/13).
#if !defined(__CUDACC_VER_MAJOR__)
#define __CUDACC_VER_MAJOR__ 0
#endif
#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda/annotated_ptr>
#endif

// Canonical single-precision quiet NaN for NVIDIA GPUs.
static __device__ __forceinline__ float qnan() {
    // Canonical QNaN used by NVIDIA GPUs. Faster than calling nanf("") in a loop.
    return __int_as_float(0x7fffffff);
}

// Very small helper to nudge future cachelines into L2 when useful.
// Guard for SM80+ where the instruction exists.
static __device__ __forceinline__ void prefetch_L2(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile ("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}

// Batch: one price series × many parameters (grid.y = combos)
extern "C" __global__
void mwdx_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ facs,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.y; // combos live on grid.y (ALMA convention)
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    // Optional device-side hint: mark 'prices' as persisting to L2 on CUDA 12+.
#if __CUDACC_VER_MAJOR__ >= 12
    const float* __restrict__ prices_persist =
        cuda::associate_access_property(prices, cuda::access_property::persisting{});
#else
    const float* __restrict__ prices_persist = prices;
#endif

    const float fac = facs[combo];
    const float beta = 1.0f - fac;
    const int row_offset = combo * series_len;

    // If first_valid is invalid, write NaN to the entire row and return.
    if (first_valid < 0 || first_valid >= series_len) {
        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[row_offset + idx] = qnan();
        }
        return; // no barrier needed
    }

    // Initialize only the warm region [0, first_valid).
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[row_offset + idx] = qnan();
    }
    // No __syncthreads(): lane 0 writes t >= first_valid only.

    // Sequential recurrence (lane 0)
    if (threadIdx.x == 0) {
        float prev = prices_persist[first_valid];
        out[row_offset + first_valid] = prev;

        // Modest prefetch distance ~ 64 floats (256B) ahead is a safe default.
        const int PDIST = 64;
        for (int t = first_valid + 1; t < series_len; ++t) {
#if __CUDA_ARCH__ >= 800
            int pf = t + PDIST;
            if (pf < series_len) prefetch_L2(prices_persist + pf);
#endif
            const float price = prices_persist[t];
            prev = __fmaf_rn(price, fac, beta * prev);
            out[row_offset + t] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    float fac,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    // Invalid fv => fill entire row with NaN and return.
    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * stride + series_idx] = qnan();
        }
        return;
    }

    // Initialize only [0, first_valid)
    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = qnan();
    }

    // Sequential lane
    if (threadIdx.x == 0) {
        int offset = first_valid * stride + series_idx;
        float prev = prices_tm[offset];
        out_tm[offset] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            offset = t * stride + series_idx;
            const float price = prices_tm[offset];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[offset] = prev;
        }
    }
}

// Many-series, one-parameter: 2D tiled variant (time-major)
// Threads only help with NaN initialization; lane 0 of each series does the
// sequential recurrence. Provided for launch-geometry symmetry with ALMA.
template<int TX, int TY>
__device__ void mwdx_many_series_one_param_tiled2d_f32_core(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    const int s_base = blockIdx.y * TY;
    const int s_local = s_base + threadIdx.y;
    if (s_local >= num_series || series_len <= 0) return;

    const float beta = 1.0f - fac;
    const int stride = num_series;
    const int first_valid = first_valids[s_local];

    if (first_valid < 0 || first_valid >= series_len) {
        // Entire row => NaN
        for (int t = threadIdx.x; t < series_len; t += TX) {
            out_tm[t * stride + s_local] = qnan();
        }
        return;
    }

    // Only warm region
    for (int t = threadIdx.x; t < first_valid; t += TX) {
        out_tm[t * stride + s_local] = qnan();
    }

    if (threadIdx.x == 0) {
        int off0 = first_valid * stride + s_local;
        float prev = prices_tm[off0];
        out_tm[off0] = prev;
        for (int t = first_valid + 1; t < series_len; ++t) {
            const int off = t * stride + s_local;
            const float price = prices_tm[off];
            prev = __fmaf_rn(price, fac, beta * prev);
            out_tm[off] = prev;
        }
    }
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty2(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 2>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}

extern "C" __global__
void mwdx_many_series_one_param_tiled2d_f32_tx128_ty4(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    float fac,
    int num_series,
    int series_len,
    float* __restrict__ out_tm) {
    mwdx_many_series_one_param_tiled2d_f32_core<128, 4>(
        prices_tm, first_valids, fac, num_series, series_len, out_tm);
}
