






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FWMA_TILE_T

#define FWMA_TILE_T 256
#endif








extern "C" __global__
void fwma_batch_f32(const float* __restrict__ prices,          
                    const float* __restrict__ weights_flat,    
                    const int*   __restrict__ periods,         
                    const int*   __restrict__ warm_indices,    
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {                 
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    
    extern __shared__ float smem[];
    float* __restrict__ s_w = smem;                 
    float* __restrict__ s_x = s_w + max_period;     

    
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm     = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f  = __int_as_float(0x7fffffff);

    
    const int tile_t0 = blockIdx.x * blockDim.x;                 
    const int tile_t1 = min(series_len, tile_t0 + blockDim.x);   

    
    if (tile_t1 <= warm) {
        const int t = tile_t0 + threadIdx.x;
        if (t < tile_t1) out[base_out + t] = nan_f;
        return;
    }

    
    
    const int load_base = tile_t0 - period + 1;
    const int load_len  = (tile_t1 - tile_t0) + period - 1;  

    
    for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
        const int g = load_base + i;
        s_x[i] = (unsigned(g) < (unsigned)series_len) ? prices[g] : 0.0f;
    }
    __syncthreads();

    
    const int t = tile_t0 + threadIdx.x;
    if (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {
            
            const int offset = (t - period + 1) - load_base;
            float acc = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(s_x[offset + k], s_w[k], acc);
            }
            out[base_out + t] = acc;
        }
    }
}






#ifndef FWMA_TIME_STEPS_PER_BLOCK
#define FWMA_TIME_STEPS_PER_BLOCK 4   
#endif

extern "C" __global__
void fwma_multi_series_one_param_f32(const float* __restrict__ prices_tm, 
                                     const float* __restrict__ weights,   
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {        
    
    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    
    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    
    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}


extern "C" __global__
void fwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const float* __restrict__ weights,
                                    int period,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    
    extern __shared__ float s_w[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        s_w[i] = weights[i];
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);

    const int series = blockIdx.y * blockDim.x + threadIdx.x;
    const int t_tile0 = blockIdx.x * FWMA_TIME_STEPS_PER_BLOCK;

    #pragma unroll
    for (int dt = 0; dt < FWMA_TIME_STEPS_PER_BLOCK; ++dt) {
        const int t = t_tile0 + dt;
        if (t >= series_len) break;

        if (series < num_series) {
            const int warm = first_valids[series] + period - 1;
            const int out_idx = t * num_series + series;

            if (t < warm) {
                out_tm[out_idx] = nan_f;
            } else {
                const int base_in = (t - period + 1) * num_series + series;
                float acc = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(prices_tm[base_in + k * num_series], s_w[k], acc);
                }
                out_tm[out_idx] = acc;
            }
        }
    }
}
