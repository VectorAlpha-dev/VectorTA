





#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef SINWMA_BLOCK_X
#define SINWMA_BLOCK_X 256
#endif



static __device__ __forceinline__ float sinwma_inv_norm(int period) {


    const double theta = CUDART_PI / (double(period) + 1.0);
    const double shalf = sin(0.5 * theta);
    const double sn    = sin(0.5 * theta * double(period));
    const double denom = (fabs(shalf) > 1e-20) ? (sn / shalf) : double(period);
    const double inv   = (denom > 0.0) ? (1.0 / denom) : 0.0;
    return (float)inv;
}



static __device__ __forceinline__
void compute_weights_pre_normalized(float* __restrict__ weights, int period) {
    const float theta = CUDART_PI_F / (float(period) + 1.0f);
    const float inv_norm = sinwma_inv_norm(period);
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float angle = (float(i + 1)) * theta;
        weights[i] = sinf(angle) * inv_norm;
    }
}



extern "C" __global__
void sinwma_batch_f32(const float* __restrict__ prices,
                      const int* __restrict__ periods,
                      int series_len,
                      int n_combos,
                      int first_valid,
                      float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;


    extern __shared__ float shmem[];
    float* __restrict__ weights = shmem;
    float* __restrict__ tile    = weights + period;

    const int warm     = first_valid + period - 1;
    const int base_out = combo * series_len;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out[base_out + t] = NAN;
        }
    }


    compute_weights_pre_normalized(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {


        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                tile[i] = prices[tile_in_start + i];
            }
            __syncthreads();


            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                float acc = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[start_in_tile + k], weights[k], acc);
                }
                out[base_out + t] = acc;
            }
            __syncthreads();
        }
    }
}



extern "C" __global__
void sinwma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    if (period <= 0) return;

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    extern __shared__ float shmem[];
    float* __restrict__ weights = shmem;
    float* __restrict__ tile    = weights + period;

    const int warm = first_valids[series_idx] + period - 1;


    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        const int stop   = min(warm, series_len);
        for (; t < stop; t += stride) {
            out_tm[t * num_series + series_idx] = NAN;
        }
    }


    compute_weights_pre_normalized(weights, period);
    __syncthreads();


    const int stride = gridDim.x * blockDim.x;
    for (int base_t = blockIdx.x * blockDim.x; base_t < series_len; base_t += stride) {
        const int t_begin = max(base_t, warm);
        const int t_end   = min(base_t + blockDim.x - 1, series_len - 1);

        if (t_begin <= t_end) {
            const int tile_in_start = t_begin - (period - 1);
            const int tile_len      = (t_end - t_begin + 1) + (period - 1);


            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                const int tt = tile_in_start + i;
                tile[i] = prices_tm[tt * num_series + series_idx];
            }
            __syncthreads();

            const int t = base_t + threadIdx.x;
            if (t >= t_begin && t <= t_end) {
                const int start_in_tile = t - t_begin;
                float acc = 0.0f;
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[start_in_tile + k], weights[k], acc);
                }
                out_tm[t * num_series + series_idx] = acc;
            }
            __syncthreads();
        }
    }
}
