








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ float sanitize_nan(float x) { return isnan(x) ? 0.f : x; }







extern "C" __global__ void correlation_cycle_batch_f32_ria(
    const float* __restrict__ prices,
    const float* __restrict__ cos_flat,
    const float* __restrict__ sin_flat,
    const int*   __restrict__ periods,
    const float* __restrict__ sum_cos_arr,
    const float* __restrict__ sum_sin_arr,
    const float* __restrict__ sqrt_t2_arr,
    const float* __restrict__ sqrt_t4_arr,
    int max_period,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_offset,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    float* __restrict__ out_angle)
{
    const int combo = combo_offset + blockIdx.y;
    if (combo >= n_combos) return;

    const int period   = periods[combo];
    const float n      = (float)period;
    const int warm_ria = first_valid + period; 

    
    const float sum_cos = sum_cos_arr[combo];
    const float sum_sin = sum_sin_arr[combo];
    const float sqrt_t2 = sqrt_t2_arr[combo];
    const float sqrt_t4 = sqrt_t4_arr[combo];
    const int   base    = combo * series_len;

    extern __shared__ float sh[];
    float* wcos = sh;             
    float* wsin = sh + period;    
    
    const float* wcos_src = cos_flat + combo * max_period;
    const float* wsin_src = sin_flat + combo * max_period;
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        wcos[i] = wcos_src[i];
        wsin[i] = wsin_src[i];
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        float r_out = NAN, i_out = NAN, ang_out = NAN;
        if (t >= warm_ria) {
            float mean = 0.f, m2 = 0.f;
            float sum_xc = 0.f, sum_xs = 0.f;
            int k = 0;
            
            #pragma unroll 4
            for (int j = 0; j < period; ++j) {
                int idx = t - (j + 1);
                float x = sanitize_nan(prices[idx]);
                float c = wcos[j];
                float s = wsin[j];
                
                ++k;
                float delta = x - mean;
                mean += delta / (float)k;
                float delta2 = x - mean;
                m2 = fmaf(delta, delta2, m2);
                sum_xc  = fmaf(x, c, sum_xc);
                sum_xs  = fmaf(x, s, sum_xs);
            }
            float sum_x = mean * n;
            float t1 = n * m2;
            if (t1 < 0.f) t1 = 0.f;
            float r_val = 0.f, i_val = 0.f;
            if (t1 > 0.f) {
                float root = sqrtf(t1);
                if (sqrt_t2 > 0.f) {
                    float denom_r = root * sqrt_t2;
                    if (denom_r > 0.f)
                        r_val = (fmaf(n, sum_xc, -(sum_x * sum_cos))) / denom_r;
                }
                if (sqrt_t4 > 0.f) {
                    float denom_i = root * sqrt_t4;
                    if (denom_i > 0.f)
                        i_val = (fmaf(n, sum_xs, -(sum_x * sum_sin))) / denom_i;
                }
            }
            r_out = r_val;
            i_out = i_val;
            ang_out = (i_val == 0.f) ? 0.f : atan2f(-i_val, r_val) * (180.f / (float)M_PI);
        }
        out_real[base + t]  = r_out;
        out_imag[base + t]  = i_out;
        out_angle[base + t] = ang_out;
        t += stride;
    }
}


extern "C" __global__ void correlation_cycle_state_batch_f32(
    const float* __restrict__ angle_flat, 
    const float* __restrict__ thresholds, 
    const int*   __restrict__ periods,    
    int series_len,
    int n_combos,
    int first_valid,
    int combo_offset,
    float* __restrict__ out_state)        
{
    const int combo = combo_offset + blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float thr  = thresholds[combo];
    const int warm_s = first_valid + period + 1; 
    const int base   = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < series_len) {
        float st = NAN;
        if (t >= warm_s) {
            float a  = angle_flat[base + t];
            float ap = angle_flat[base + t - 1];
            if (!isnan(ap) && fabsf(a - ap) < thr) {
                st = (a >= 0.f) ? 1.f : -1.f;
            } else {
                st = 0.f;
            }
        }
        out_state[base + t] = st;
        t += stride;
    }
}



extern "C" __global__ void correlation_cycle_many_series_one_param_f32_ria(
    const float* __restrict__ prices_tm,   
    const float* __restrict__ wcos,        
    const float* __restrict__ wsin,        
    const float  sum_cos,
    const float  sum_sin,
    const float  sqrt_t2,
    const float  sqrt_t4,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,  
    float* __restrict__ out_real_tm,       
    float* __restrict__ out_imag_tm,
    float* __restrict__ out_angle_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; 
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= cols || t0 >= rows) return;

    const int warm_ria = first_valids[s] + period;
    const float n = (float)period;
    const int stride_t = gridDim.x * blockDim.x;
    const int stride_s = gridDim.y * blockDim.y; 

    for (int t = t0; t < rows; t += stride_t) {
        const int out_idx = t * cols + s;
        float r_out = NAN, i_out = NAN, ang_out = NAN;
        if (t >= warm_ria) {
            float mean = 0.f, m2 = 0.f;
            float sum_xc = 0.f, sum_xs = 0.f;
            int k = 0;
            #pragma unroll 4
            for (int j = 0; j < period; ++j) {
                int tt = t - (j + 1);
                float x = sanitize_nan(prices_tm[tt * cols + s]);
                float c = wcos[j];
                float si = wsin[j];
                ++k;
                float delta = x - mean;
                mean += delta / (float)k;
                float delta2 = x - mean;
                m2 = fmaf(delta, delta2, m2);
                sum_xc  = fmaf(x, c, sum_xc);
                sum_xs  = fmaf(x, si, sum_xs);
            }
            float sum_x = mean * n;
            float t1 = n * m2;
            if (t1 < 0.f) t1 = 0.f;
            float r_val = 0.f, i_val = 0.f;
            if (t1 > 0.f) {
                float root = sqrtf(t1);
                if (sqrt_t2 > 0.f) {
                    float denom_r = root * sqrt_t2;
                    if (denom_r > 0.f) r_val = (fmaf(n, sum_xc, -(sum_x * sum_cos))) / denom_r;
                }
                if (sqrt_t4 > 0.f) {
                    float denom_i = root * sqrt_t4;
                    if (denom_i > 0.f) i_val = (fmaf(n, sum_xs, -(sum_x * sum_sin))) / denom_i;
                }
            }
            r_out = r_val;
            i_out = i_val;
            ang_out = (i_val == 0.f) ? 0.f : atan2f(-i_val, r_val) * (180.f / (float)M_PI);
        }
        out_real_tm[out_idx]  = r_out;
        out_imag_tm[out_idx]  = i_out;
        out_angle_tm[out_idx] = ang_out;
    }
}


extern "C" __global__ void correlation_cycle_state_many_series_one_param_f32(
    const float* __restrict__ angle_tm,  
    const float  threshold,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float* __restrict__ out_state_tm)    
{
    const int s  = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || t0 >= rows) return;

    const int warm_s = first_valids[s] + period + 1;
    const int stride_t = gridDim.x * blockDim.x;
    for (int t = t0; t < rows; t += stride_t) {
        int idx = t * cols + s;
        float st = NAN;
        if (t >= warm_s) {
            float a  = angle_tm[idx];
            float ap = angle_tm[idx - cols];
            if (!isnan(ap) && fabsf(a - ap) < threshold) {
                st = (a >= 0.f) ? 1.f : -1.f;
            } else {
                st = 0.f;
            }
        }
        out_state_tm[idx] = st;
    }
}
