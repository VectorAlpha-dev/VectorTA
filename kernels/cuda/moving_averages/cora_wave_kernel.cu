// CUDA kernels for the Compound Ratio Moving Average (CoRa Wave).
//
// Batch (one-series × many-params) and many-series (time-major, one param)
// variants, plus a simple per-combo WMA smoothing pass used by the batch path.
//
// Semantics:
// - FP32 throughout.
// - Warm-up: outputs prior to warm = first_valid + (period-1) are NaN for the
//   CoRa stage. When smoothing is enabled (period_smooth > 1), final warm-up is
//   warm + (period_smooth-1).
// - Weights are provided per-combo as oldest→newest of length = period and an
//   inv_norm (1/sum(weights)).
//
// Notes:
// - WMA smoothing in the batch path uses a dedicated kernel so each combo can
//   have a different first-valid frontier and smoothing period.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Quiet NaN helper
static __device__ __forceinline__ float f32_qnan() {
    return __int_as_float(0x7fffffff);
}

// ---------------------- 1) Plain batched kernel ----------------------------

extern "C" __global__
void cora_wave_batch_f32(const float* __restrict__ prices,
                         const float* __restrict__ weights_flat,
                         const int*   __restrict__ periods,
                         const float* __restrict__ inv_norms,
                         int max_period,
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    extern __shared__ float shared_weights[]; // [period]
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_weights[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = f32_qnan();
        } else {
            const int start = t - period + 1;
            float s = 0.f;
            // Kahan-Neumaier compensated accumulation for numerical stability
            float c = 0.f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                float term = __fmaf_rn(prices[start + k], shared_weights[k], 0.f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
            out[out_idx] = __fmul_rn(s, inv_norms[combo]);
        }
        t += stride;
    }
}

// -------- 2) Batch WMA smoothing pass over CoRa outputs --------------------
// Each combo can have its own smoothing period m[combo] and its own CoRa
// frontier warm0[combo].

extern "C" __global__
void cora_wave_batch_wma_from_y_f32(const float* __restrict__ y,
                                     const int*   __restrict__ smooth_periods,
                                     const int*   __restrict__ warm0s,
                                     int series_len,
                                     int n_combos,
                                     float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int m = smooth_periods[combo];
    if (m <= 1) {
        // No smoothing: copy y → out with NaN preserved
        const int base = combo * series_len;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            out[base + t] = y[base + t];
            t += stride;
        }
        return;
    }

    const float inv_norm = 2.0f / (float(m) * (float(m) + 1.0f));
    const int warm = warm0s[combo] + (m - 1);
    const int base = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base + t;
        if (t < warm) {
            out[out_idx] = f32_qnan();
        } else {
            const int start = t - m + 1;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < m; ++k) {
                acc = __fmaf_rn(y[base + start + k], float(k + 1), acc);
            }
            out[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}

// -------- 3) Many-series, one param, time-major ----------------------------

extern "C" __global__
void cora_wave_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ weights, // length = period, oldest→newest
    int period,
    float inv_norm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    if (period <= 0) return;

    const int s = blockIdx.y; // series index
    if (s >= num_series) return;

    const int warm = first_valids[s] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = f32_qnan();
        } else {
            const int start = (t - period + 1) * num_series + s;
            float sacc = 0.f, c = 0.f;
#pragma unroll 4
            for (int k = 0; k < period; ++k) {
                float x = prices_tm[start + k * num_series];
                float term = __fmaf_rn(x, weights[k], 0.f);
                float y = term - c;
                float u = sacc + y;
                c = (u - sacc) - y;
                sacc = u;
            }
            out_tm[out_idx] = sacc * inv_norm;
        }
        t += stride;
    }
}

// -------- 4) Many-series WMA smoothing over CoRa outputs -------------------

extern "C" __global__
void cora_wave_ms1p_wma_time_major_f32(const float* __restrict__ y_tm,
                                       int wma_period,
                                       int num_series,
                                       int series_len,
                                       const int* __restrict__ warm0s,
                                       float* __restrict__ out_tm) {
    if (wma_period <= 1) {
        // No smoothing: copy
        int s = blockIdx.y;
        if (s >= num_series) return;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < series_len) {
            out_tm[t * num_series + s] = y_tm[t * num_series + s];
            t += stride;
        }
        return;
    }
    const float inv_norm = 2.0f / (float(wma_period) * (float(wma_period) + 1.0f));
    const int s = blockIdx.y;
    if (s >= num_series) return;
    const int warm = warm0s[s] + (wma_period - 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = f32_qnan();
        } else {
            const int start = (t - wma_period + 1) * num_series + s;
            float acc = 0.0f;
#pragma unroll 4
            for (int k = 0; k < wma_period; ++k) {
                float y = y_tm[start + k * num_series];
                acc = __fmaf_rn(y, float(k + 1), acc);
            }
            out_tm[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}

