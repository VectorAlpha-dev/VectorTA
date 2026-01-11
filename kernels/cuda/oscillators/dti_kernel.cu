






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>       

#ifndef DTI_QNAN
#define DTI_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ void ema_kahan_step(const float alpha,
                                                      const float x,
                                                      float &e, float &c)
{
    
    const float diff   = x - e;
    const float delta  = fmaf(alpha, diff, 0.0f);  
    const float y      = delta - c;
    const float t      = e + y;
    c                  = (t - e) - y;
    e                  = t;
}



extern "C" __global__ void dti_batch_f32(
    const float* __restrict__ x,          
    const float* __restrict__ ax,         
    const int*   __restrict__ r_arr,      
    const int*   __restrict__ s_arr,      
    const int*   __restrict__ u_arr,      
    int series_len,
    int n_combos,
    int start,                             
    float* __restrict__ out               
){
    
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_combos;
         row += blockDim.x * gridDim.x)
    {
        const int r = r_arr[row];
        const int s = s_arr[row];
        const int u = u_arr[row];
        float* out_row = out + (size_t)row * series_len;

        if (UNLIKELY(r <= 0 || s <= 0 || u <= 0 || start < 1 || start > series_len)) {
            for (int i = 0; i < series_len; ++i) out_row[i] = DTI_QNAN;
            continue;
        }

        
        for (int i = 0; i < start; ++i) out_row[i] = DTI_QNAN;

        
        const float ar = 2.0f / (float(r) + 1.0f);
        const float as_ = 2.0f / (float(s) + 1.0f);
        const float au = 2.0f / (float(u) + 1.0f);

        
        float e0_r = 0.0f, e0_s = 0.0f, e0_u = 0.0f;
        float e1_r = 0.0f, e1_s = 0.0f, e1_u = 0.0f;
        float c0_r = 0.0f, c0_s = 0.0f, c0_u = 0.0f;
        float c1_r = 0.0f, c1_s = 0.0f, c1_u = 0.0f;

        
        for (int i = start; i < series_len; ++i) {
            const float xi  = x[i];
            const float axi = ax[i];

            
            ema_kahan_step(ar,  xi,   e0_r, c0_r);
            ema_kahan_step(as_, e0_r, e0_s, c0_s);
            ema_kahan_step(au,  e0_s, e0_u, c0_u);

            ema_kahan_step(ar,  axi,  e1_r, c1_r);
            ema_kahan_step(as_, e1_r, e1_s, c1_s);
            ema_kahan_step(au,  e1_s, e1_u, c1_u);

            const float den = e1_u;
            out_row[i] = (den == den && den != 0.0f) ? (100.0f * (e0_u / den)) : 0.0f;
        }
    }
}



extern "C" __global__ void dti_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids, 
    int num_series,
    int series_len,
    int r,
    int s,
    int u,
    float* __restrict__ out_tm 
){
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(fv < 0 || fv >= series_len || r <= 0 || s <= 0 || u <= 0)) {
        
        for (int t = 0; t < series_len; ++t)
            out_tm[(size_t)t * num_series + series] = DTI_QNAN;
        return;
    }

    const int start = fv + 1;
    if (UNLIKELY(start >= series_len)) {
        for (int t = 0; t < series_len; ++t)
            out_tm[(size_t)t * num_series + series] = DTI_QNAN;
        return;
    }

    
    for (int t = 0; t < start; ++t)
        out_tm[(size_t)t * num_series + series] = DTI_QNAN;

    
    const float ar  = 2.0f / (float(r) + 1.0f);
    const float as_ = 2.0f / (float(s) + 1.0f);
    const float au  = 2.0f / (float(u) + 1.0f);

    
    float e0_r = 0.0f, e0_s = 0.0f, e0_u = 0.0f;
    float e1_r = 0.0f, e1_s = 0.0f, e1_u = 0.0f;
    float c0_r = 0.0f, c0_s = 0.0f, c0_u = 0.0f;
    float c1_r = 0.0f, c1_s = 0.0f, c1_u = 0.0f;

    const size_t stride = (size_t)num_series;

    
    size_t idx_prev = (size_t)fv * stride + series;
    float prev_h = high_tm[idx_prev];
    float prev_l = low_tm [idx_prev];

    
    size_t idx = (size_t)start * stride + series;

    for (int t = start; t < series_len; ++t, idx += stride) {
        const float h  = high_tm[idx];
        const float l  = low_tm[idx];
        const float dh = h - prev_h;
        const float dl = l - prev_l;
        prev_h = h;
        prev_l = l;

        
        const float up  = fmaxf(dh, 0.0f);
        const float dn  = fmaxf(-dl, 0.0f);
        const float xi  = up - dn;
        const float axi = fabsf(xi);

        
        ema_kahan_step(ar,  xi,   e0_r, c0_r);
        ema_kahan_step(as_, e0_r, e0_s, c0_s);
        ema_kahan_step(au,  e0_s, e0_u, c0_u);

        ema_kahan_step(ar,  axi,  e1_r, c1_r);
        ema_kahan_step(as_, e1_r, e1_s, c1_s);
        ema_kahan_step(au,  e1_s, e1_u, c1_u);

        const float den = e1_u;
        out_tm[idx] = (den == den && den != 0.0f) ? (100.0f * (e0_u / den)) : 0.0f;
    }
}
