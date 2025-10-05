// CUDA kernels for Pascal Weighted Moving Average (PWMA).
//
// The batch kernel assigns one block per parameter combination. Each block
// stages the pre-normalized Pascal weights for its period into shared memory
// and has threads stride across the timeline applying the weighted sum. A
// second kernel handles the many-series × one-parameter path using time-major
// input with shared weights. Optional 2D tiled variants are provided to mirror
// the ALMA/CWMA naming and launch geometry when available.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__
void pwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int* __restrict__ periods,
                    const int* __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights_flat[combo * max_period + idx];
    }
    __syncthreads();

    const int warm = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], shared_weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}

extern "C" __global__
void pwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     // retained for signature symmetry with ALMA/CWMA; weights are
                                     // pre-normalized on host for PWMA, so this parameter is unused
                                     float /*inv_norm*/,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
            const int in_idx = (start + k) * num_series + series_idx;
            acc = fmaf(prices_tm[in_idx], shared_weights[k], acc);
        }
        out_tm[out_idx] = acc;
    }
    t += stride;
}
}

// -------------------- 2D tiled many-series variants ------------------------
// Mirrors ALMA/CWMA tiled mapping for time-major inputs with shared weights.

__device__ __forceinline__ size_t pwma_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

template<int TX, int TY>
__device__ __forceinline__
void pwma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float /*inv_norm_unused*/, // kept for symmetry
                          int num_series,
                          int series_len,
                          const int* __restrict__ first_valids,
                          float* __restrict__ out_tm) {
    const int t0 = blockIdx.x * TX;
    const int s0 = blockIdx.y * TY;
    if (t0 >= series_len || s0 >= num_series) return;

    // Shared: weights + tile [ (TX+period-1) x TY ]
    const int total = TX + period - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = pwma_align_up(off + size_t(period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);

    // Load weights into shared (vectorized if aligned)
    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2; // period / 4
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += blockDim.x * blockDim.y) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += blockDim.x * blockDim.y) {
            w[i] = weights[i];
        }
    }
    __syncthreads();

    // Cooperative load of tile across series and time
    const bool vec_ok = (TY == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);
    const int p0 = t0 - (period - 1);
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            if (vec_ok && threadIdx.y == 0) {
                const float4* src4 = reinterpret_cast<const float4*>(&prices_tm[t * num_series + s0]);
                float4 v = src4[0];
                tile[dt * TY + 0] = v.x;
                tile[dt * TY + 1] = v.y;
                tile[dt * TY + 2] = v.z;
                tile[dt * TY + 3] = v.w;
            } else {
                int s = s0 + threadIdx.y;
                float val = 0.f;
                if (s < num_series) val = prices_tm[t * num_series + s];
                tile[dt * TY + threadIdx.y] = val;
            }
        } else {
            int idx = dt * TY + threadIdx.y;
            if (idx < total * TY) tile[idx] = 0.f;
        }
    }
    __syncthreads();

    // Compute outputs for this CTA
    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;

    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) {
        out_tm[out_idx] = __int_as_float(0x7fffffff);
        return;
    }

    int start = threadIdx.x; // within tile
    const float* xptr = &tile[start * TY + threadIdx.y];
    float acc = 0.f;
#pragma unroll 4
    for (int i = 0; i < period; ++i) {
        acc = fmaf(xptr[i * TY], w[i], acc);
    }
    // Weights are pre-normalized on host for PWMA
    out_tm[out_idx] = acc;
}

#define DEFINE_PWMA_MS1P_TILED(NAME, TX, TY)                                    \
extern "C" __global__ void NAME(                                                \
  const float* __restrict__ prices_tm,                                          \
  const float* __restrict__ weights,                                            \
  int period, float inv_norm, int num_series, int series_len,                   \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {           \
  pwma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,            \
                               num_series, series_len, first_valids, out_tm);   \
}

// Expose the two tiled variants used by wrappers
DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty4, 128, 4)
