







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef SWMA_BLOCK_X
#define SWMA_BLOCK_X 128           
#endif
#ifndef SWMA_OUTS_PER_THREAD
#define SWMA_OUTS_PER_THREAD 2     
#endif
#ifndef SWMA_SERIES_PER_BLOCK
#define SWMA_SERIES_PER_BLOCK 8    
#endif
#ifndef SWMA_MAX_PERIOD
#define SWMA_MAX_PERIOD 4096       
#endif
#ifndef SWMA_USE_CONST_WEIGHTS
#define SWMA_USE_CONST_WEIGHTS 1   
#endif

#if SWMA_USE_CONST_WEIGHTS
__constant__ float c_swma_weights[SWMA_MAX_PERIOD];
#endif


static __device__ __forceinline__ float swma_norm_inv(int period) {
    if (period <= 2) return (period == 1) ? 1.0f : 0.5f;
    if ((period & 1) == 0) {
        float m = float(period >> 1);
        return 1.0f / (m * (m + 1.0f));
    } else {
        float m = float((period + 1) >> 1);
        return 1.0f / (m * m);
    }
}

static __device__ __forceinline__ void fill_tri_weights(float* __restrict__ w_sh, int period, float inv_norm) {
    int half = period >> 1;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float v = float(i + 1) * inv_norm;
        w_sh[i] = v;
        w_sh[period - 1 - i] = v;
    }
    if ((period & 1) && threadIdx.x == 0) {
        w_sh[half] = float(half + 1) * inv_norm;
    }
}






extern "C" __global__
void swma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    
    extern __shared__ float smem[];
    float* const w_sh = smem;
    float* const tile = smem + max_period;

    
    const float inv_norm = swma_norm_inv(period);
    __syncthreads();
    fill_tri_weights(w_sh, period, inv_norm);
    __syncthreads();

    const int warm  = warm_indices[combo];
    const int first = warm - period + 1;      
    const int out_base = combo * series_len;

    
    constexpr int OUTS_PER_THREAD = SWMA_OUTS_PER_THREAD;
    const int TILE_OUT = blockDim.x * OUTS_PER_THREAD;       
    const int GRID_TILE_STRIDE = gridDim.x * TILE_OUT;

    
    int tile_start_out = blockIdx.x * TILE_OUT;
    while (tile_start_out < series_len) {
        
        const int n_outs = min(TILE_OUT, series_len - tile_start_out);

        
        int in_begin = tile_start_out - (period - 1);
        int in_end   = tile_start_out + n_outs - 1;
        
        int load_begin = max(in_begin, 0);
        int load_end   = min(in_end, series_len - 1);
        int load_len   = max(0, load_end - load_begin + 1);

        
        for (int i = threadIdx.x; i < load_len; i += blockDim.x) {
            tile[i] = prices[load_begin + i];
        }
        __syncthreads();

        
#pragma unroll
        for (int u = 0; u < OUTS_PER_THREAD; ++u) {
            int local_idx = threadIdx.x + u * blockDim.x;
            if (local_idx < n_outs) {
                const int t = tile_start_out + local_idx;
                
                if (t < warm || (t - period + 1) < first) {
                    out[out_base + t] = NAN;
                } else {
                    const int start = t - period + 1;
                    
                    const int tile_off = (start - load_begin);
                    float acc = 0.0f;

                    
                    if (period == 1) {
                        acc = tile[tile_off];
                    } else if (period == 2) {
                        acc = 0.5f * (tile[tile_off] + tile[tile_off + 1]);
                    } else {
#pragma unroll 4
                        for (int k = 0; k < period; ++k) {
                            acc = fmaf(tile[tile_off + k], w_sh[k], acc);
                        }
                    }
                    out[out_base + t] = acc;
                }
            }
        }
        __syncthreads();

        tile_start_out += GRID_TILE_STRIDE;
    }
}









extern "C" __global__
void swma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,  
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    
    const int series_block_base = blockIdx.y * blockDim.y;
    const int s_local = threadIdx.y;
    const int s = series_block_base + s_local;
    if (s >= num_series) return;

    
    extern __shared__ float smem[];
    float* sh = smem;
#if SWMA_USE_CONST_WEIGHTS
    float* tile = sh;                              
#else
    float* w_sh = sh;                              
    float* tile  = w_sh + period;
    for (int k = threadIdx.x; k < period; k += blockDim.x) { w_sh[k] = weights[k]; }
    __syncthreads();
#endif

    const int warm = first_valids[s] + period - 1;

    const int TILE_T = blockDim.x;                 
    const int GRID_TILE_STRIDE = gridDim.x * TILE_T;

    int tile_t0 = blockIdx.x * TILE_T;             
    while (tile_t0 < series_len) {
        const int n_outs = min(TILE_T, series_len - tile_t0);
        
        const int in_begin_t = tile_t0 - (period - 1);
        const int in_end_t   = tile_t0 + n_outs - 1;
        const int load_begin_t = max(in_begin_t, 0);
        const int load_end_t   = min(in_end_t, series_len - 1);
        const int load_len_t   = max(0, load_end_t - load_begin_t + 1);

        
        
        const int tile_span = load_len_t * blockDim.y;
        
        int lin = threadIdx.x * blockDim.y + s_local;
        for (int idx = lin; idx < tile_span; idx += blockDim.x * blockDim.y) {
            int dt = idx / blockDim.y;
            int ss = idx % blockDim.y;
            int gs = series_block_base + ss;
            if (gs < num_series) {
                int g_index = (load_begin_t + dt) * num_series + gs;
                tile[idx] = prices_tm[g_index];    
            }
        }
        __syncthreads();

        
        
        int local_t = threadIdx.x;
        if (local_t < n_outs) {
            const int t = tile_t0 + local_t;
            const int out_idx = t * num_series + s;
            if (t < warm) {
                out_tm[out_idx] = NAN;
            } else {
                const int start_t = t - period + 1;
                const int base = (start_t - load_begin_t) * blockDim.y + s_local;
                float acc = 0.0f;

#if SWMA_USE_CONST_WEIGHTS
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    
                    acc = fmaf(tile[base + k * blockDim.y], c_swma_weights[k], acc);
                }
#else
#pragma unroll 4
                for (int k = 0; k < period; ++k) {
                    acc = fmaf(tile[base + k * blockDim.y], w_sh[k], acc);
                }
#endif
                out_tm[out_idx] = acc;
            }
        }
        __syncthreads();

        tile_t0 += GRID_TILE_STRIDE;
    }
}
