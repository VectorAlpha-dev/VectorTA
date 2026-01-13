








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef WMA_MAX_PERIOD
#define WMA_MAX_PERIOD 8192
#endif



__constant__ __align__(16) float C_WMA_RAMP[WMA_MAX_PERIOD];


static __device__ __forceinline__ const float* ramp_ptr_or_shared(
    const float* __restrict__ sh, int period) {
    return (period <= WMA_MAX_PERIOD) ? C_WMA_RAMP : sh;
}


static __device__ __forceinline__ float wma_inv_norm(int period) {
    float p = (float)period;
    return 2.0f / (p * (p + 1.0f));
}


static __device__ __forceinline__ float f32_qnan() {
    return __int_as_float(0x7fffffff);
}

extern "C" __global__
void wma_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 1) {
        return;
    }


    extern __shared__ float shmem[];
    float* sh_ramp = shmem;
    const bool need_sh = (period > WMA_MAX_PERIOD);
    if (need_sh) {
        for (int i = threadIdx.x; i < period; i += blockDim.x) {
            sh_ramp[i] = float(i + 1);
        }
        __syncthreads();
    }
    const float* __restrict__ ramp = ramp_ptr_or_shared(sh_ramp, period);

    const float inv_norm = wma_inv_norm(period);
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
            float acc = 0.0f;
        #pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], ramp[k], acc);
            }
            out[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}


#ifndef WMA_ROLLING_CHUNK
#define WMA_ROLLING_CHUNK 256
#endif


extern "C" __global__
void wma_batch_rolling_f32(const float* __restrict__ prices,
                           const int* __restrict__ periods,
                           int series_len,
                           int n_combos,
                           int first_valid,
                           float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const float inv_norm = wma_inv_norm(period);
    const int   warm     = first_valid + period - 1;
    const int   base_out = combo * series_len;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int end_nan = min(warm, series_len);
        while (t < end_nan) {
            out[base_out + t] = f32_qnan();
            t += stride;
        }
    }
    __syncthreads();


    const int thread_linear = blockIdx.x * blockDim.x + threadIdx.x;
    int seg0 = warm + thread_linear * WMA_ROLLING_CHUNK;
    if (seg0 >= series_len) return;
    const int segN = min(series_len, seg0 + WMA_ROLLING_CHUNK);


    int t = seg0;
    int start = t - period + 1;
    const float* __restrict__ p0 = prices + start;
    float S1 = 0.0f;
    float S2 = 0.0f;
#pragma unroll 8
    for (int k = 0; k < period; ++k) {
        float x = p0[k];
        S1 += x;
        S2 = fmaf(float(k + 1), x, S2);
    }
    out[base_out + t] = S2 * inv_norm;

    for (++t; t < segN; ++t) {
        const float x_new = prices[t];
        const float x_old = prices[t - period];
        const float S1_prev = S1;
        S1 = S1 - x_old + x_new;
        S2 = (S2 - S1_prev) + float(period) * x_new;
        out[base_out + t] = S2 * inv_norm;
    }
}


extern "C" __global__
void wma_batch_prefix_f32(const float* __restrict__ pref_a,
                          const float* __restrict__ pref_b,
                          const int* __restrict__ periods,
                          int series_len,
                          int n_combos,
                          int first_valid,
                          float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const float inv_norm = wma_inv_norm(period);
    const int warm = first_valid + period - 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = f32_qnan();
        } else {

            const int start = t + 1 - period;
            const float s_a = pref_a[t + 1] - pref_a[start];
            const float s_b = pref_b[t + 1] - pref_b[start];
            const float wsum = fmaf(-float(t - period), s_a, s_b);
            out[out_idx] = wsum * inv_norm;
        }
        t += stride;
    }
}

extern "C" __global__
void wma_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    if (period <= 1) {
        return;
    }


    extern __shared__ float shmem[];
    float* sh_ramp = shmem;
    const bool need_sh = (period > WMA_MAX_PERIOD);
    if (need_sh) {
        for (int i = threadIdx.x; i < period; i += blockDim.x) {
            sh_ramp[i] = float(i + 1);
        }
        __syncthreads();
    }
    const float* __restrict__ ramp = ramp_ptr_or_shared(sh_ramp, period);

    const float inv_norm = wma_inv_norm(period);

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int warm = first_valids[series_idx] + period - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = f32_qnan();
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                acc = fmaf(prices_tm[in_idx], ramp[k], acc);
            }
            out_tm[out_idx] = acc * inv_norm;
        }
        t += stride;
    }
}
