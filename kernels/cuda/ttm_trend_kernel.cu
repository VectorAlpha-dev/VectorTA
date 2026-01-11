








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <stdint.h>






__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


__device__ __forceinline__ float2 ff2_sub(const float2 a, const float2 b) {
    float s, e;
    two_sum(a.x, -b.x, s, e);      
    e += (a.y - b.y);              
    float hi, lo;
    two_sum(s, e, hi, lo);         
    return make_float2(hi, lo);
}


__device__ __forceinline__ float2 ff2_scale(const float2 a, const float s) {
    float hi = a.x * s;
    
    float err = fmaf(a.x, s, -hi) + a.y * s;
    float rhi, rlo;
    two_sum(hi, err, rhi, rlo);
    return make_float2(rhi, rlo);
}

















#ifndef TTM_TILE_TIME
#define TTM_TILE_TIME 256     
#endif
#ifndef TTM_TILE_PARAMS
#define TTM_TILE_PARAMS 4     
#endif

extern "C" __global__
void ttm_trend_batch_prefix_ff2_tiled(
    const float2* __restrict__ prefix_ff2, 
    const float*  __restrict__ close,      
    const int*    __restrict__ periods,    
    const int*    __restrict__ warm_idx,   
    int series_len,
    int n_combos,
    float* __restrict__ out)               
{
    const int tx = threadIdx.x;                       
    const int ty = threadIdx.y;                       
    const int t0 = blockIdx.x * TTM_TILE_TIME;        
    const int p0 = blockIdx.y * TTM_TILE_PARAMS;      
    const int t  = t0 + tx;
    const int row = p0 + ty;

    
    __shared__ float  sh_close[TTM_TILE_TIME];
    __shared__ float2 sh_pref [TTM_TILE_TIME];
    __shared__ int    sh_period[TTM_TILE_PARAMS];
    __shared__ int    sh_warm  [TTM_TILE_PARAMS];

    
    if (ty == 0 && t < series_len) {
        sh_close[tx] = close[t];
        sh_pref[tx]  = prefix_ff2[t];
    }

    
    if (tx == 0) {
        if (row < n_combos) {
            sh_period[ty] = periods[row];
            sh_warm  [ty] = warm_idx[row];
        } else {
            sh_period[ty] = 0;
            sh_warm  [ty] = INT_MAX;
        }
    }
    __syncthreads();

    if (row >= n_combos || t >= series_len) return;

    const int p    = sh_period[ty];
    const int warm = sh_warm  [ty];
    if (p <= 0) return;

    
    if (t < warm) {
        out[(size_t)row * series_len + t] = 0.0f;
        return;
    }

    const float invp = 1.0f / (float)p;
    float avg;

    if (t == warm) {
        
        float2 scaled = ff2_scale(sh_pref[tx], invp);
        avg = scaled.x + scaled.y;
    } else {
        
        const int j = t - p;
        float2 pref_t  = sh_pref[tx];
        float2 pref_j  = prefix_ff2[j];         
        float2 diff    = ff2_sub(pref_t, pref_j);
        float2 scaled  = ff2_scale(diff, invp);
        avg = scaled.x + scaled.y;
    }

    const float cv = sh_close[tx];
    out[(size_t)row * series_len + t] = (cv > avg) ? 1.0f : 0.0f;
}









extern "C" __global__
void ttm_trend_many_series_one_param_time_major_f32(
    const float* __restrict__ source_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series || period <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int fv     = first_valids[series];
    if (fv < 0 || fv >= series_len) return;

    const int warm = fv + period - 1;
    if (warm >= series_len) return;

    
    float s = 0.0f, c = 0.0f;
    for (int k = fv; k <= warm; ++k) {
        const float x = source_tm[(size_t)k * stride + series];
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }

    const float invp = 1.0f / (float)period;
    float avg = s * invp;
    out_tm[(size_t)warm * stride + series] =
        (close_tm[(size_t)warm * stride + series] > avg) ? 1.0f : 0.0f;

    
    for (int t = warm + 1; t < series_len; ++t) {
        const float add = source_tm[(size_t)t * stride + series];
        const float sub = source_tm[(size_t)(t - period) * stride + series];

        
        float y1 = add - c;
        float u1 = s + y1;
        c = (u1 - s) - y1;
        s = u1;

        
        float y2 = -sub - c;
        float u2 = s + y2;
        c = (u2 - s) - y2;
        s = u2;

        avg = s * invp;
        out_tm[(size_t)t * stride + series] =
            (close_tm[(size_t)t * stride + series] > avg) ? 1.0f : 0.0f;
    }
}

