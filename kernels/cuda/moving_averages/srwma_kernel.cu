



#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SRWMA_USE_ASYNC_COPY
#define SRWMA_USE_ASYNC_COPY 0  
#endif

#if SRWMA_USE_ASYNC_COPY
  #include <cuda/pipeline>
#endif







extern "C" __global__
void srwma_batch_f32(const float* __restrict__ prices,
                     const float* __restrict__ weights_flat,
                     const int*   __restrict__ periods,
                     const int*   __restrict__ warm_indices,
                     const float* __restrict__ inv_norms,
                     int max_wlen,
                     int series_len,
                     int n_combos,
                     float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 1) return;

    const int wlen = period - 1;
    const int warm = warm_indices[combo];
    const int start_t = max(warm, wlen - 1);  
    const int row_offset = combo * series_len;
    const float inv_norm = inv_norms[combo];

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;                      
    float* __restrict__ tile  = smem + max_wlen;           

    
    const int wbase = combo * max_wlen;
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        
        w_rev[k] = weights_flat[wbase + (wlen - 1 - k)];
    }
    __syncthreads();

    
    const int tile_span = blockDim.x + wlen - 1;   
    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        
        const int t0 = base - (wlen - 1);

        
        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src) < static_cast<unsigned>(series_len))
                v = prices[src];
            tile[i] = v;
        }

        
        
#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)
        
        
        
        
        
        
#endif

        __syncthreads();

        
        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int out_idx = row_offset + t;
            if (t < start_t) {
                out[out_idx] = NAN;
            } else {
                
                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out[out_idx] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}












#ifndef SRWMA_USE_CONST_WEIGHTS
#define SRWMA_USE_CONST_WEIGHTS 0
#endif
#if SRWMA_USE_CONST_WEIGHTS
__constant__ float srwma_const_w[4096];  
#endif

extern "C" __global__
void srwma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const int*   __restrict__ first_valids,
#if SRWMA_USE_CONST_WEIGHTS
                                     const float* __restrict__ weights_unused, 
#else
                                     const float* __restrict__ weights,
#endif
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series || series_len <= 0) return;
    if (period <= 1) return;

    const int wlen = period - 1;
    const int first_valid = first_valids[series_idx];
    
    const int warm = first_valid + period + 1;
    const int start_t = max(warm, wlen - 1);

    const int stride = num_series;

    extern __shared__ float smem[];
    float* __restrict__ w_rev = smem;                     
    float* __restrict__ tile  = smem + wlen;              

    
#if SRWMA_USE_CONST_WEIGHTS
    
    
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = srwma_const_w[wlen - 1 - k];
    }
#else
    for (int k = threadIdx.x; k < wlen; k += blockDim.x) {
        w_rev[k] = weights[wlen - 1 - k];
    }
#endif
    __syncthreads();

    const int tile_span = blockDim.x + wlen - 1;

    for (int base = blockIdx.x * blockDim.x; base < series_len; base += gridDim.x * blockDim.x) {
        const int t0 = base - (wlen - 1);

        
        for (int i = threadIdx.x; i < tile_span; i += blockDim.x) {
            const int src_t = t0 + i;
            float v = 0.0f;
            if (static_cast<unsigned>(src_t) < static_cast<unsigned>(series_len)) {
                v = prices_tm[src_t * stride + series_idx];
            }
            tile[i] = v;
        }

#if SRWMA_USE_ASYNC_COPY && (__CUDA_ARCH__ >= 800)
        
        
#endif

        __syncthreads();

        const int t = base + threadIdx.x;
        if (t < series_len) {
            const int offset = t * stride + series_idx;
            if (t < start_t) {
                out_tm[offset] = NAN;
            } else {
                const float* __restrict__ win = tile + threadIdx.x;
                float acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < wlen; ++k) {
                    acc = __fmaf_rn(win[k], w_rev[k], acc);
                }
                out_tm[offset] = acc * inv_norm;
            }
        }
        __syncthreads();
    }
}
