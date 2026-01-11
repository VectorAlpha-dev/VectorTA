





















#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}


struct dsf { float hi, lo; };

__device__ __forceinline__ dsf ds_make(float hi, float lo) { dsf r; r.hi = hi; r.lo = lo; return r; }
__device__ __forceinline__ dsf ds_from_f(float a) { return ds_make(a, 0.0f); }


__device__ __forceinline__ dsf ds_from_double(double a) {
    float hi = (float)a;
    float lo = (float)(a - (double)hi);
    return ds_make(hi, lo);
}


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void two_prod(float a, float b, float &p, float &err) {
    p = a * b; err = __fmaf_rn(a, b, -p);
}
__device__ __forceinline__ dsf ds_add(dsf a, dsf b) {
    float s, e; two_sum(a.hi, b.hi, s, e); e += a.lo + b.lo; float t, lo; two_sum(s, e, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ dsf ds_sub(dsf a, dsf b) { return ds_add(a, ds_make(-b.hi, -b.lo)); }
__device__ __forceinline__ dsf ds_mul_f(dsf a, float b) {
    float p, err; two_prod(a.hi, b, p, err); err += a.lo * b; float t, lo; two_sum(p, err, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ dsf ds_mul(dsf a, dsf b) {
    float p, err; two_prod(a.hi, b.hi, p, err); err += a.hi * b.lo + a.lo * b.hi; err += a.lo * b.lo; float t, lo; two_sum(p, err, t, lo); return ds_make(t, lo);
}
__device__ __forceinline__ float ds_to_f(dsf a) { return a.hi + a.lo; }


__device__ __forceinline__ dsf load_dsf(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}

extern "C" __global__ void bollinger_bands_sma_prefix_f32(
    const float* __restrict__ data,
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ devups,
    const float* __restrict__ devdns,
    int n_combos,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float devup = devups[combo];
    const float devdn = devdns[combo];

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nanf = qnan32();
    const float invP = 1.0f / (float)period;

    
    
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float u = nanf, m = nanf, l = nanf;
        if (t >= warm) {
            bool ok = true;
            if (any_nan_since_first) {
                const int nan_since_first = prefix_nan[t + 1] - nan_base;
                ok = (nan_since_first == 0);
            }
            if (ok) {
                
                const int t1 = t + 1;
                const int s = t1 - period;

                
                const float2 ps_e  = prefix_sum[t1];
                const float2 ps_s  = prefix_sum[s];
                const float2 ps2_e = prefix_sum_sq[t1];
                const float2 ps2_s = prefix_sum_sq[s];

                const float sum  = (ps_e.x  - ps_s.x)  + (ps_e.y  - ps_s.y);
                const float sum2 = (ps2_e.x - ps2_s.x) + (ps2_e.y - ps2_s.y);

                const float mean = sum * invP;
                const float ex2  = sum2 * invP;
                float var = fmaf(-mean, mean, ex2);
                if (var < 0.0f) var = 0.0f;
                const float sd = sqrtf(var);

                m = mean;
                u = mean + devup * sd;
                l = mean - devdn * sd;
            }
        }
        out_upper[row_off + t]  = u;
        out_middle[row_off + t] = m;
        out_lower[row_off + t]  = l;
        t += stride;
    }
}


extern "C" __global__ void bollinger_bands_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,   
    const float2* __restrict__ prefix_sum_sq_tm,
    const int* __restrict__ prefix_nan_tm,      
    int period,
    float devup,
    float devdn,
    int num_series,  
    int series_len,  
    const int* __restrict__ first_valids,       
    float* __restrict__ out_upper_tm,           
    float* __restrict__ out_middle_tm,          
    float* __restrict__ out_lower_tm) {         
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (period <= 0) return;
    const int fv = first_valids[s];
    const int warm = fv + period - 1;
    const int stride = num_series; 
    const float invP = 1.0f / (float)period;
    const int nan_base = prefix_nan_tm[fv * stride + s];
    const bool any_nan_since_first = (prefix_nan_tm[series_len * stride + s] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + s;
        float u = qnan32(), m = qnan32(), l = qnan32();
        if (t >= warm) {
            bool ok = true;
            if (any_nan_since_first) {
                const int p_idx_t1 = (t + 1) * stride + s;
                const int nan_since_first = prefix_nan_tm[p_idx_t1] - nan_base;
                ok = (nan_since_first == 0);
            }
            if (ok) {
                const int t1 = t + 1;
                const int p_idx = t1 * stride + s;
                const int s_idx = (t1 - period) * stride + s;

                const dsf sum_ds  = ds_sub(load_dsf(prefix_sum_tm,    p_idx), load_dsf(prefix_sum_tm,    s_idx));
                const dsf sum2_ds = ds_sub(load_dsf(prefix_sum_sq_tm, p_idx), load_dsf(prefix_sum_sq_tm, s_idx));

                
                
                const double sum_d  = (double)sum_ds.hi  + (double)sum_ds.lo;
                const double sum2_d = (double)sum2_ds.hi + (double)sum2_ds.lo;
                const double invPd = 1.0 / (double)period;
                const double mean_d = sum_d * invPd;
                double var_d = (sum2_d * invPd) - mean_d * mean_d;
                if (var_d < 0.0) var_d = 0.0;
                const float sd = (float)sqrt(var_d);

                const float mean_f = (float)mean_d;
                m = mean_f;
                u = mean_f + devup * sd;
                l = mean_f - devdn * sd;
            }
        }
        out_upper_tm[out_idx]  = u;
        out_middle_tm[out_idx] = m;
        out_lower_tm[out_idx]  = l;
        t += step;
    }
}
