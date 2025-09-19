// CUDA kernels for ALMA (Arnaud Legoux Moving Average) operations.
//
// These variants compute Gaussian weights on-device to avoid host-side
// allocations/transfers and operate exclusively in FP32. Two one-series
// kernels are provided: a straightforward grid-stride version and an
// async-tiled variant that leverages cuda::memcpy_async for long series.
// A many-series × one-parameter kernel is also supplied, all following the
// VRAM-first design used by the Rust wrapper.

#include <cuda_runtime.h>
#include <math.h>

#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 0
#endif
#ifndef __CUDACC_VER_MINOR__
#define __CUDACC_VER_MINOR__ 0
#endif

#define ALMA_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)

#if ALMA_CUDA_VERSION >= 12040
#define ALMA_HAS_PIPELINE 1
#include <cuda/barrier>
#include <cuda/pipeline>
#else
#define ALMA_HAS_PIPELINE 0
#endif

// Grid: blockIdx.y = parameter combo, blockIdx.x/threadIdx.x = time indices
extern "C" __global__
void alma_batch_f32_onthefly(const float* __restrict__ prices,
                             const int* __restrict__ periods,
                             const float* __restrict__ offsets,
                             const float* __restrict__ sigmas,
                             int series_len,
                             int n_combos,
                             int first_valid,
                             float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float offset = offsets[combo];
    const float sigma = sigmas[combo];

    const float m = offset * float(period - 1);
    const float s = float(period) / sigma;
    const float s2 = 2.0f * s * s;

    extern __shared__ float sh[];
    float* weights = sh; // [period]

    __shared__ float norm;
    if (threadIdx.x == 0) norm = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        float d = float(i) - m;
        float wi = __expf(-(d * d) / s2);
        weights[i] = wi;
        atomicAdd(&norm, wi);
    }
    __syncthreads();

    const float inv_norm = 1.0f / norm;
    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < period; ++k) {
                sum += prices[start + k] * weights[k];
            }
            out[base_out + t] = sum * inv_norm;
        }
        t += stride;
    }
}

// Async-tiled variant for long series. Grid/block mapping identical to the
// on-the-fly kernel but tiles price data into shared memory with cp.async.
extern "C" __global__
void alma_batch_tiled_async_f32(const float* __restrict__ prices,
                                const int* __restrict__ periods,
                                const float* __restrict__ offsets,
                                const float* __restrict__ sigmas,
                                int series_len,
                                int n_combos,
                                int first_valid,
                                float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float offset = offsets[combo];
    const float sigma = sigmas[combo];
    const int tile = blockDim.x;
    const int t0 = blockIdx.x * tile;
    if (t0 >= series_len) return;

    const float m = offset * float(period - 1);
    const float s = float(period) / sigma;
    const float s2 = 2.0f * s * s;
    const int warm = first_valid + period - 1;

    extern __shared__ float sh[];
    float* weights = sh;              // [period]
    float* price_tile = sh + period;  // [tile + period - 1]

    __shared__ float norm;
    if (threadIdx.x == 0) norm = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        float d = float(i) - m;
        float wi = __expf(-(d * d) / s2);
        weights[i] = wi;
        atomicAdd(&norm, wi);
    }
    __syncthreads();

    const float inv_norm = 1.0f / norm;
    const int total = tile + period - 1;
    const int p_base = t0 - (period - 1);

#if ALMA_HAS_PIPELINE
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();
#endif

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int idx = p_base + i;
#if ALMA_HAS_PIPELINE
        if (idx >= 0 && idx < series_len) {
            cuda::memcpy_async(price_tile + i, prices + idx, sizeof(float), bar);
        } else {
            price_tile[i] = 0.0f;
        }
#else
        float v = 0.0f;
        if (idx >= 0 && idx < series_len) {
            v = prices[idx];
        }
        price_tile[i] = v;
#endif
    }
#if ALMA_HAS_PIPELINE
    cuda::pipeline_commit();
    cuda::pipeline_wait_prior(0);
#endif
    __syncthreads();

    const int t = t0 + threadIdx.x;
    if (t >= series_len) return;

    const int out_idx = combo * series_len + t;
    if (t < warm) {
        out[out_idx] = NAN;
        return;
    }

    const int start = threadIdx.x;
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < period; ++k) {
        sum += price_tile[start + k] * weights[k];
    }
    out[out_idx] = sum * inv_norm;
}

// Many-series × one-parameter kernel, time-major input layout.
extern "C" __global__
void alma_multi_series_one_param_onthefly_f32(const float* __restrict__ prices_tm,
                                              const int* __restrict__ first_valids,
                                              int period,
                                              float offset,
                                              float sigma,
                                              int num_series,
                                              int series_len,
                                              float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const float m = offset * float(period - 1);
    const float s = float(period) / sigma;
    const float s2 = 2.0f * s * s;

    extern __shared__ float sh[];
    float* weights = sh; // [period]

    __shared__ float norm;
    if (threadIdx.x == 0) norm = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        float d = float(i) - m;
        float wi = __expf(-(d * d) / s2);
        weights[i] = wi;
        atomicAdd(&norm, wi);
    }
    __syncthreads();

    const float inv_norm = 1.0f / norm;
    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                sum += prices_tm[in_idx] * weights[k];
            }
            out_tm[out_idx] = sum * inv_norm;
        }
        t += stride;
    }
}
