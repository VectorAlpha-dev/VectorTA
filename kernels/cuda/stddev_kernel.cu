







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>





struct ds {
    float hi, lo;
    __device__ __forceinline__ ds() {}
    __device__ __forceinline__ ds(float h, float l=0.f): hi(h), lo(l) {}
};

__device__ __forceinline__ ds ds_from_f(float x) { return ds(x, 0.f); }
__device__ __forceinline__ float ds_to_f(ds a)  { return a.hi + a.lo; }
__device__ __forceinline__ ds ds_neg(ds a)      { return ds(-a.hi, -a.lo); }



__device__ __forceinline__ ds ds_from_d(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return ds(hi, lo);
}


__device__ __forceinline__ ds ds_add(ds a, ds b) {
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float e  = (a.hi - (s - bb)) + (b.hi - bb);
    e += a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return ds(hi, lo);
}
__device__ __forceinline__ ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }


__device__ __forceinline__ ds ds_mul(ds a, ds b) {
    float p   = a.hi * b.hi;
    float err = fmaf(a.hi, b.hi, -p); 
    err += a.hi * b.lo + a.lo * b.hi;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_scale(ds a, float s) {
    float p   = a.hi * s;
    float err = fmaf(a.hi, s, -p) + a.lo * s;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_square(ds a) { return ds_mul(a, a); }







#ifndef STDDEV_COMBO_TILE
#define STDDEV_COMBO_TILE 4
#endif
extern "C" __global__ void stddev_batch_f32(
    const float2* __restrict__ ps_x,    
    const float2* __restrict__ ps_x2,   
    const int*    __restrict__ ps_nan,  
    int len,
    int first_valid,
    const int* __restrict__ periods,    
    const float* __restrict__ nbdevs,   
    int n_combos,
    float* __restrict__ out             
) {
    const int group = blockIdx.y;
    const int co_base = group * STDDEV_COMBO_TILE;

    __shared__ int s_period[STDDEV_COMBO_TILE];
    __shared__ int s_warm[STDDEV_COMBO_TILE];
    __shared__ float s_nb[STDDEV_COMBO_TILE];
    __shared__ double s_inv_n[STDDEV_COMBO_TILE];

    if (threadIdx.x < STDDEV_COMBO_TILE) {
        const int c = co_base + threadIdx.x;
        if (c < n_combos) {
            const int p = periods[c];
            s_period[threadIdx.x] = p;
            s_warm[threadIdx.x] = first_valid + p - 1;
            s_nb[threadIdx.x] = nbdevs[c];
            s_inv_n[threadIdx.x] = (p > 0) ? (1.0 / (double)p) : 0.0;
        } else {
            s_period[threadIdx.x] = 0;
            s_warm[threadIdx.x] = INT_MAX;
            s_nb[threadIdx.x] = 0.0f;
            s_inv_n[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nan_f = __int_as_float(0x7fffffff);

    while (t < len) {
        const int end = t + 1;
        const float2 ex = ps_x[end];
        const float2 ex2 = ps_x2[end];
        const int end_bad = ps_nan[end];

#pragma unroll
        for (int k = 0; k < STDDEV_COMBO_TILE; ++k) {
            const int combo = co_base + k;
            if (combo >= n_combos) break;

            const int period = s_period[k];
            float outv = nan_f;
            if (period > 0) {
                const int warm = s_warm[k];
                const float nb = s_nb[k];
                const double inv_n = s_inv_n[k];

                if (nb == 0.0f) {
                    outv = (t >= warm) ? 0.0f : nan_f;
                } else if (t >= warm) {
                    int start = end - period;
                    if (start < 0) start = 0;
                    const int nan_count = end_bad - ps_nan[start];
                    if (nan_count == 0) {
                        float2 sx = ps_x[start];
                        float2 sx2 = ps_x2[start];
                        const double s1 =
                            ((double)ex.x + (double)ex.y) - ((double)sx.x + (double)sx.y);
                        const double s2 =
                            ((double)ex2.x + (double)ex2.y) - ((double)sx2.x + (double)sx2.y);
                        const double mean = s1 * inv_n;
                        const double var = (s2 * inv_n) - (mean * mean);
                        outv = (var > 0.0) ? (float)(sqrt(var) * (double)nb) : 0.0f;
                    }
                }
            }
            out[combo * len + t] = outv;
        }

        t += stride;
    }
}





extern "C" __global__ void stddev_many_series_one_param_f32(
    const float* __restrict__ data_tm,    
    const int*  __restrict__ first_valids,
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm            
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;

    
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm     = first_valid + period - 1;
    const double inv_n = 1.0 / (double)period;

    
    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else { double d = (double)v; s1 += d; s2 += d * d; }
    }

    if (warm < rows) {
        if (nan_in_win == 0) {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[warm * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        } else {
            out_tm[warm * stride + series] = __int_as_float(0x7fffffff);
        }
    }

    
    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        
        if (!isnan(old_v)) { double od = (double)old_v; s1 -= od; s2 -= od * od; }
        else { nan_in_win--; }
        if (!isnan(new_v)) { double nd = (double)new_v; s1 += nd; s2 += nd * nd; }
        else { nan_in_win++; }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[t * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        }
    }
}
