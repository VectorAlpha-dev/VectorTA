









#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

extern "C" __global__ void bbw_sma_prefix_f32(
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    const float*  __restrict__ uplusd,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float k = uplusd[combo]; 
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f32();
        if (t >= warm) {
            const int start = t + 1 - period; 
            const int nan_count = prefix_nan[t + 1] - prefix_nan[start];
            if (nan_count == 0) {
                const double sum  = prefix_sum[t + 1]    - prefix_sum[start];
                const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[start];
                const double den = static_cast<double>(period);
                const double mean = sum / den;
                double var = sum2 / den - mean * mean;
                if (var < 0.0) var = 0.0;
                const double std = (var > 0.0) ? sqrt(var) : 0.0;
                
                out_val = __double2float_rn((static_cast<double>(k) * std) / mean);
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}



extern "C" __global__ void bbw_multi_series_one_param_tm_f32(
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float u_plus_d,
    float*        __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series_idx;
        float out_val = nan_f32();
        if (t >= warm) {
            const int start = (t + 1 - period) * num_series + series_idx;
            const int nan_count = prefix_nan_tm[idx + 1] - prefix_nan_tm[start];
            if (nan_count == 0) {
                const double sum  = prefix_sum_tm[idx + 1]    - prefix_sum_tm[start];
                const double sum2 = prefix_sum_sq_tm[idx + 1] - prefix_sum_sq_tm[start];
                const double den = static_cast<double>(period);
                const double mean = sum / den;
                double var = sum2 / den - mean * mean;
                if (var < 0.0) var = 0.0;
                const double std = (var > 0.0) ? sqrt(var) : 0.0;
                out_val = __double2float_rn((static_cast<double>(u_plus_d) * std) / mean);
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}










#include <stdint.h>

#ifndef CUDA_FORCEINLINE
#define CUDA_FORCEINLINE __forceinline__
#endif


CUDA_FORCEINLINE __device__ float2 ff_make(float hi, float lo) {
    float s = hi + lo;
    float e = lo - (s - hi);
    return make_float2(s, e);
}


CUDA_FORCEINLINE __device__ float2 ff_two_sum(float a, float b) {
    float s  = a + b;
    float bb = s - a;
    float e  = (a - (s - bb)) + (b - bb);
    float t  = s + e;
    float e2 = e - (t - s);
    return make_float2(t, e2);
}


CUDA_FORCEINLINE __device__ float2 ff_two_prod_fma(float a, float b) {
    float p = a * b;
    float e = fmaf(a, b, -p);
    float t = p + e;
    float e2 = e - (t - p);
    return make_float2(t, e2);
}


CUDA_FORCEINLINE __device__ float2 ff_add(float2 x, float2 y) {
    float s  = x.x + y.x;
    float bb = s - x.x;
    float e  = (x.x - (s - bb)) + (y.x - bb);
    e += x.y + y.y;
    float t = s + e;
    float e2 = e - (t - s);
    return make_float2(t, e2);
}


CUDA_FORCEINLINE __device__ float2 ff_sub(float2 x, float2 y) {
    float s  = x.x - y.x;
    float bb = s - x.x;
    float e  = (x.x - (s - bb)) - (y.x + bb);
    e += x.y - y.y;
    float t = s + e;
    float e2 = e - (t - s);
    return make_float2(t, e2);
}


CUDA_FORCEINLINE __device__ float2 ff_mul(float2 x, float2 y) {
    float2 p = ff_two_prod_fma(x.x, y.x);
    float e  = fmaf(x.x, y.y, 0.f);
    e        = fmaf(x.y, y.x, e);
    float hi = p.x + (p.y + e);
    float err = (p.y + e) - (hi - p.x);
    return make_float2(hi, err);
}


CUDA_FORCEINLINE __device__ float2 ff_div_scalar(float2 a, float b) {
    float y  = a.x / b;
    float2 yb = ff_two_prod_fma(y, b);
    float2 r  = ff_sub(a, yb);
    float y2  = r.x / b;
    float s   = y + y2;
    float e   = y2 - (s - y);
    return make_float2(s, e);
}

CUDA_FORCEINLINE __device__ float clamp_nonneg(float x) { return x < 0.f ? 0.f : x; }


CUDA_FORCEINLINE __device__ float bbw_base_from_prefix_ff(
    const float2* __restrict__ ps,
    const float2* __restrict__ ps2,
    const int*    __restrict__ pnan,
    int t, int period, int warm)
{
    if (t < warm) return nan_f32();
    const int a = t + 1;
    const int b = t + 1 - period;
    const int nan_count = pnan[a] - pnan[b];
    if (nan_count != 0) return nan_f32();

    float2 sum  = ff_sub(ps [a], ps [b]);
    float2 sum2 = ff_sub(ps2[a], ps2[b]);

    const float den = (float)period;
    float2 mean  = ff_div_scalar(sum,  den);
    float2 m2    = ff_div_scalar(sum2, den);
    float2 var2  = ff_sub(m2, ff_mul(mean, mean));
    float var    = clamp_nonneg(var2.x + var2.y);
    float stdv   = (var > 0.f) ? sqrtf(var) : 0.f;
    float m      = mean.x + mean.y;
    return stdv / m; 
}


CUDA_FORCEINLINE __device__ float bbw_base_from_prefix_ff_no_nan(
    const float2* __restrict__ ps,
    const float2* __restrict__ ps2,
    int t, int period, int warm)
{
    if (t < warm) return nan_f32();
    const int a = t + 1;
    const int b = t + 1 - period;

    float2 sum  = ff_sub(ps [a], ps [b]);
    float2 sum2 = ff_sub(ps2[a], ps2[b]);

    const float den = (float)period;
    float2 mean  = ff_div_scalar(sum,  den);
    float2 m2    = ff_div_scalar(sum2, den);
    float2 var2  = ff_sub(m2, ff_mul(mean, mean));
    float var    = clamp_nonneg(var2.x + var2.y);
    float stdv   = (var > 0.f) ? sqrtf(var) : 0.f;
    float m      = mean.x + mean.y;
    return stdv / m; 
}


extern "C" __global__ void bbw_sma_prefix_ff_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    const float*  __restrict__ uplusd,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float k = uplusd[combo];
    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float base = any_nan_since_first
            ? bbw_base_from_prefix_ff(prefix_sum, prefix_sum_sq, prefix_nan, t, period, warm)
            : bbw_base_from_prefix_ff_no_nan(prefix_sum, prefix_sum_sq, t, period, warm);
        out[row_off + t] = k * base; 
        t += stride;
    }
}


extern "C" __global__ void bbw_sma_prefix_grouped_ff_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*   __restrict__ unique_periods,  
    const int*   __restrict__ offsets,         
    const float* __restrict__ uplusd_sorted,   
    const int*   __restrict__ combo_index,     
    int num_unique,
    float*       __restrict__ out)
{
    const int up = blockIdx.y;
    if (up >= num_unique) return;

    const int period = unique_periods[up];
    if (period <= 0) return;
    const int warm = first_valid + period - 1;
    const int begin = offsets[up];
    const int end   = offsets[up + 1];
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float base = any_nan_since_first
            ? bbw_base_from_prefix_ff(prefix_sum, prefix_sum_sq, prefix_nan, t, period, warm)
            : bbw_base_from_prefix_ff_no_nan(prefix_sum, prefix_sum_sq, t, period, warm);
        for (int j = begin; j < end; ++j) {
            const int row = combo_index[j];
            out[row * len + t] = uplusd_sorted[j] * base;
        }
        t += stride;
    }
}


extern "C" __global__ void bbw_multi_series_one_param_tm_ff_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float u_plus_d,
    float*        __restrict__ out_tm)
{
    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx   = t * num_series + s;
        const int a_idx = (t + 1) * num_series + s;
        const int b_idx = (t + 1 - period) * num_series + s;

        float out_val = nan_f32();
        if (t >= warm) {
            const int nan_count = prefix_nan_tm[a_idx] - prefix_nan_tm[b_idx];
            if (nan_count == 0) {
                float2 sum  = ff_sub(prefix_sum_tm[a_idx],    prefix_sum_tm[b_idx]);
                float2 sum2 = ff_sub(prefix_sum_sq_tm[a_idx], prefix_sum_sq_tm[b_idx]);
                const float den = (float)period;
                float2 mean  = ff_div_scalar(sum,  den);
                float2 m2    = ff_div_scalar(sum2, den);
                float2 var2  = ff_sub(m2, ff_mul(mean, mean));
                float var    = clamp_nonneg(var2.x + var2.y);
                float stdv   = (var > 0.f) ? sqrtf(var) : 0.f;
                float m      = mean.x + mean.y;
                out_val = (u_plus_d * stdv) / m; 
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}






extern "C" __global__ void bbw_sma_streaming_f64(
    const float* __restrict__ data, 
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ uplusd,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const double k = (double)uplusd[combo];
    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    
    for (int t = threadIdx.x; t < min(len, warm); t += blockDim.x) {
        out[row_off + t] = nan_f32();
    }

    if (threadIdx.x == 0) {
        if (warm < len) {
            
            int start = warm + 1 - period;
            double sum = 0.0;
            double sum2 = 0.0;
            for (int i = start; i <= warm; ++i) {
                double v = (double)data[i];
                sum += v;
                sum2 = fma(v, v, sum2);
            }
            double mean = sum / (double)period;
            double var = (sum2 / (double)period) - mean * mean;
            if (var < 0.0) var = 0.0;
            double stdv = sqrt(var);
            out[row_off + warm] = (float)(k * stdv / mean);

            
            for (int t = warm + 1; t < len; ++t) {
                double vin = (double)data[t];
                double vout = (double)data[t - period];
                sum += vin - vout;
                sum2 = fma(vin, vin, sum2 - vout * vout);
                mean = sum / (double)period;
                var = (sum2 / (double)period) - mean * mean;
                if (var < 0.0) var = 0.0;
                stdv = sqrt(var);
                out[row_off + t] = (float)(k * stdv / mean);
            }
        }
    }
}

extern "C" __global__ void bbw_multi_series_one_param_tm_streaming_f64(
    const float* __restrict__ data_tm, 
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float u_plus_d,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y;
    if (s >= num_series) return;
    if (period <= 0) return;

    const int warm = first_valids[s] + period - 1;
    const double k = (double)u_plus_d;

    
    for (int t = threadIdx.x; t < min(series_len, warm); t += blockDim.x) {
        out_tm[t * num_series + s] = nan_f32();
    }

    if (threadIdx.x == 0) {
        if (warm < series_len) {
            
            int start = warm + 1 - period;
            double sum = 0.0;
            double sum2 = 0.0;
            for (int i = start; i <= warm; ++i) {
                double v = (double)data_tm[i * num_series + s];
                sum += v;
                sum2 = fma(v, v, sum2);
            }
            double mean = sum / (double)period;
            double var = (sum2 / (double)period) - mean * mean;
            if (var < 0.0) var = 0.0;
            double stdv = sqrt(var);
            out_tm[warm * num_series + s] = (float)(k * stdv / mean);

            
            for (int t = warm + 1; t < series_len; ++t) {
                double vin = (double)data_tm[t * num_series + s];
                double vout = (double)data_tm[(t - period) * num_series + s];
                sum += vin - vout;
                sum2 = fma(vin, vin, sum2 - vout * vout);
                mean = sum / (double)period;
                var = (sum2 / (double)period) - mean * mean;
                if (var < 0.0) var = 0.0;
                stdv = sqrt(var);
                out_tm[t * num_series + s] = (float)(k * stdv / mean);
            }
        }
    }
}
