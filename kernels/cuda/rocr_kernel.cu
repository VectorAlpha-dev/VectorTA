










#include <cuda_runtime.h>

#ifndef ROCR_NAN
#define ROCR_NAN (__int_as_float(0x7fffffff))  
#endif


static __device__ __forceinline__ bool rocr_isnan(float x) { return x != x; }



extern "C" __global__ void rocr_prepare_inv_f32(
    const float* __restrict__ data,  
    int len,
    float* __restrict__ inv_out      
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < len) {
        float x = data[i];
        inv_out[i] = (x == 0.0f || rocr_isnan(x)) ? 0.0f : (1.0f / x);
        i += stride;
    }
}



extern "C" __global__ void rocr_batch_f32(
    const float* __restrict__ data,     
    const float* __restrict__ inv_opt,  
    int len,
    int first_valid,
    const int* __restrict__ periods,    
    int n_combos,
    float* __restrict__ out             
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period;
    const int row_off = combo * len;

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    
    for (int i = tid; i < warm && i < len; i += stride) {
        out[row_off + i] = ROCR_NAN;
    }

    
    int start = tid;
    if (start < warm) {
        int delta = warm - start;
        int steps = (delta + stride - 1) / stride;  
        start += steps * stride;
    }

    
    if (inv_opt) {
        int i = start;
        const int stride4 = stride << 2;
        const int end4 = len - 3 * stride;
        for (; i < end4; i += stride4) {
            
            {
                const int d = i - period;
                const float inv = inv_opt[d];
                out[row_off + i] = (inv == 0.0f || rocr_isnan(inv)) ? 0.0f : (data[i] * inv);
            }
            
            {
                const int i1 = i + stride;
                const int d1 = i1 - period;
                const float inv1 = inv_opt[d1];
                out[row_off + i1] = (inv1 == 0.0f || rocr_isnan(inv1)) ? 0.0f : (data[i1] * inv1);
            }
            
            {
                const int i2 = i + 2 * stride;
                const int d2 = i2 - period;
                const float inv2 = inv_opt[d2];
                out[row_off + i2] = (inv2 == 0.0f || rocr_isnan(inv2)) ? 0.0f : (data[i2] * inv2);
            }
            
            {
                const int i3 = i + 3 * stride;
                const int d3 = i3 - period;
                const float inv3 = inv_opt[d3];
                out[row_off + i3] = (inv3 == 0.0f || rocr_isnan(inv3)) ? 0.0f : (data[i3] * inv3);
            }
        }
        for (; i < len; i += stride) {
            const int d = i - period;
            const float inv = inv_opt[d];
            out[row_off + i] = (inv == 0.0f || rocr_isnan(inv)) ? 0.0f : (data[i] * inv);
        }
    } else {
        int i = start;
        const int stride4 = stride << 2;
        const int end4 = len - 3 * stride;
        for (; i < end4; i += stride4) {
            
            {
                const int d = i - period;
                const float denom = data[d];
                out[row_off + i] = (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data[i] / denom);
            }
            
            {
                const int i1 = i + stride;
                const int d1 = i1 - period;
                const float denom1 = data[d1];
                out[row_off + i1] = (denom1 == 0.0f || rocr_isnan(denom1)) ? 0.0f : (data[i1] / denom1);
            }
            
            {
                const int i2 = i + 2 * stride;
                const int d2 = i2 - period;
                const float denom2 = data[d2];
                out[row_off + i2] = (denom2 == 0.0f || rocr_isnan(denom2)) ? 0.0f : (data[i2] / denom2);
            }
            
            {
                const int i3 = i + 3 * stride;
                const int d3 = i3 - period;
                const float denom3 = data[d3];
                out[row_off + i3] = (denom3 == 0.0f || rocr_isnan(denom3)) ? 0.0f : (data[i3] / denom3);
            }
        }
        for (; i < len; i += stride) {
            const int d = i - period;
            const float denom = data[d];
            out[row_off + i] = (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data[i] / denom);
        }
    }
}






extern "C" __global__ void rocr_many_series_one_param_f32(
    const float* __restrict__ data_tm,  
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,  
    float* __restrict__ out_tm             
) {
    if (period <= 0) return;

    const int stride_series = num_series; 

    if (blockDim.y == 1 && gridDim.y == (unsigned)num_series) {
        
        const int series = blockIdx.y;
        if (series >= num_series) return;
        const int warm = first_valids[series] + period;
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int step = gridDim.x * blockDim.x;

        
        for (int tt = t; tt < series_len && tt < warm; tt += step) {
            out_tm[tt * stride_series + series] = ROCR_NAN;
        }
        
        int t_start = t;
        if (t_start < warm) {
            int delta = warm - t_start;
            int steps = (delta + step - 1) / step;
            t_start += steps * step;
        }
        for (int tt = t_start; tt < series_len; tt += step) {
            const float denom = data_tm[(tt - period) * stride_series + series];
            out_tm[tt * stride_series + series] =
                (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data_tm[tt * stride_series + series] / denom);
        }
        return;
    }

    
    const int s0 = blockIdx.x * blockDim.x + threadIdx.x; 
    const int t0 = blockIdx.y * blockDim.y + threadIdx.y; 
    const int s_step = blockDim.x * gridDim.x;
    const int t_step = blockDim.y * gridDim.y;

    for (int s = s0; s < num_series; s += s_step) {
        const int warm = first_valids[s] + period;
        
        for (int t = t0; t < series_len && t < warm; t += t_step) {
            out_tm[t * stride_series + s] = ROCR_NAN;
        }
        
        int t_start = t0;
        if (t_start < warm) {
            int delta = warm - t_start;
            int steps = (delta + t_step - 1) / t_step; 
            t_start += steps * t_step;
        }
        for (int t = t_start; t < series_len; t += t_step) {
            const float denom = data_tm[(t - period) * stride_series + s];
            out_tm[t * stride_series + s] =
                (denom == 0.0f || rocr_isnan(denom)) ? 0.0f : (data_tm[t * stride_series + s] / denom);
        }
    }
}

