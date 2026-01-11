






#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


namespace {

__device__ __forceinline__ bool is_finite_f(float x) {
    return !(isnan(x) || isinf(x));
}


__device__ __forceinline__ float ema_update(float state, float x, float alpha) {
    return fmaf(alpha, x - state, state);
}


struct KahanSumF {
    float s;  
    float c;  
    __device__ KahanSumF() : s(0.0f), c(0.0f) {}
    __device__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ void sub(float x) { add(-x); }
};

} 

extern "C" __global__ void wavetrend_batch_f32(
    const float* __restrict__ prices,   
    int len,
    int first_valid,
    int n_combos,
    const int* __restrict__ channel_lengths,
    const int* __restrict__ average_lengths,
    const int* __restrict__ ma_lengths,
    const float* __restrict__ factors,
    float* __restrict__ wt1_out,        
    float* __restrict__ wt2_out,        
    float* __restrict__ wt_diff_out     
){
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride  = blockDim.x * gridDim.x;

    for (int row = tid; row < n_combos; row += stride) {
        const int ch  = channel_lengths[row];
        const int avg = average_lengths[row];
        const int ma  = ma_lengths[row];
        const float factor = factors[row];

        float* __restrict__ wt1_row  = wt1_out     + (size_t)row * (size_t)len;
        float* __restrict__ wt2_row  = wt2_out     + (size_t)row * (size_t)len;
        float* __restrict__ diff_row = wt_diff_out + (size_t)row * (size_t)len;

        
        if (len <= 0 || ch <= 0 || avg <= 0 || ma <= 0) {
            for (int i = 0; i < len; ++i) {
                wt1_row[i]  = CUDART_NAN_F;
                wt2_row[i]  = CUDART_NAN_F;
                diff_row[i] = CUDART_NAN_F;
            }
            continue;
        }

        const float alpha_ch  = 2.0f / (float(ch) + 1.0f);
        const float alpha_avg = 2.0f / (float(avg) + 1.0f);
        const float inv_ma    = 1.0f / (float)ma;

        
        int warmup = first_valid + (ch - 1) + (avg - 1) + (ma - 1);
        if (warmup < 0)       warmup = 0;
        if (warmup > len)     warmup = len;

        
        int prefill = first_valid;
        if (prefill < 0) prefill = 0;
        if (prefill > len) prefill = len;
        for (int i = 0; i < prefill; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }

        
        bool esa_init = false, de_init = false, wt1_init = false;
        float esa = 0.0f, de = 0.0f, wt1_state = 0.0f;

        
        KahanSumF acc;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int i = start; i < len; ++i) {
            const float price = prices[i];
            const bool price_ok = is_finite_f(price);

            
            if (!esa_init) {
                if (price_ok) {
                    esa = price;
                    esa_init = true;
                }
            } else if (price_ok) {
                esa = ema_update(esa, price, alpha_ch);
            }

            
            if (esa_init && price_ok) {
                const float absdiff = fabsf(price - esa);
                if (!de_init) {
                    de = absdiff;
                    de_init = true;
                } else {
                    de = ema_update(de, absdiff, alpha_ch);
                }
            }

            
            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const float denom = factor * de;
                if (denom != 0.0f && is_finite_f(denom)) {
                    const float ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (is_finite_f(ci)) {
                            wt1_state = ci;
                            wt1_init = true;
                        }
                    } else if (is_finite_f(ci)) {
                        wt1_state = ema_update(wt1_state, ci, alpha_avg);
                    }
                }
            }
            if (wt1_init) wt1_val = wt1_state;

            
            wt1_row[i] = wt1_val;

            
            if (is_finite_f(wt1_val)) { acc.add(wt1_val); ++window_count; }

            if (i >= ma) {
                const float old = wt1_row[i - ma];
                if (is_finite_f(old)) { acc.sub(old); --window_count; }
            }

            
            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma) {
                wt2_val = acc.s * inv_ma;
            }
            wt2_row[i] = wt2_val;

            
            if (i >= warmup && is_finite_f(wt2_val) && is_finite_f(wt1_val)) {
                diff_row[i] = wt2_val - wt1_val;
            } else {
                diff_row[i] = CUDART_NAN_F;
            }
        }

        
        for (int i = 0; i < warmup; ++i) {
            wt1_row[i]  = CUDART_NAN_F;
            wt2_row[i]  = CUDART_NAN_F;
            diff_row[i] = CUDART_NAN_F;
        }
    }
}


extern "C" __global__ void wavetrend_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, 
    int cols,
    int rows,
    int channel_length,
    int average_length,
    int ma_length,
    float factor,
    const int* __restrict__ first_valids, 
    float* __restrict__ wt1_tm,           
    float* __restrict__ wt2_tm,           
    float* __restrict__ wt_diff_tm        
){
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (rows <= 0 || cols <= 0 || channel_length <= 0 || average_length <= 0 || ma_length <= 0) return;

    const float alpha_ch  = 2.0f / (float(channel_length) + 1.0f);
    const float alpha_avg = 2.0f / (float(average_length) + 1.0f);
    const float inv_ma    = 1.0f / (float)ma_length;

    for (int series = tid; series < cols; series += stride) {
        float* __restrict__ wt1_col  = wt1_tm     + series;
        float* __restrict__ wt2_col  = wt2_tm     + series;
        float* __restrict__ diff_col = wt_diff_tm + series;

        const int first_valid = first_valids[series];
        int warmup = first_valid + (channel_length - 1) + (average_length - 1) + (ma_length - 1);
        if (warmup < 0) warmup = 0;
        if (warmup > rows) warmup = rows;

        
        int pre = first_valid;
        if (pre < 0) pre = 0;
        if (pre > rows) pre = rows;
        for (int t = 0; t < pre; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }

        
        bool esa_init = false, de_init = false, wt1_init = false;
        double esa = 0.0, de = 0.0, wt1_state = 0.0;

        
        double sum_wt1 = 0.0;
        int window_count = 0;

        int start = first_valid > 0 ? first_valid : 0;
        for (int t = start; t < rows; ++t) {
            const int idx = t * cols;
            const double price = static_cast<double>(prices_tm[idx + series]);
            const bool price_ok = isfinite(price);

            
            if (!esa_init) {
                if (price_ok) { esa = price; esa_init = true; }
            } else if (price_ok) {
                const double alpha_ch_d = static_cast<double>(alpha_ch);
                const double beta_ch_d  = 1.0 - alpha_ch_d;
                esa = fma(alpha_ch_d, price, beta_ch_d * esa);
            }

            
            if (esa_init && price_ok) {
                const double absdiff = fabs(price - esa);
                if (!de_init) { de = absdiff; de_init = isfinite(de); }
                else if (isfinite(absdiff)) {
                    const double alpha_ch_d = static_cast<double>(alpha_ch);
                    const double beta_ch_d  = 1.0 - alpha_ch_d;
                    de = fma(alpha_ch_d, absdiff, beta_ch_d * de);
                }
            }

            
            float wt1_val = CUDART_NAN_F;
            if (esa_init && de_init && price_ok) {
                const double denom = static_cast<double>(factor) * de;
                if (denom != 0.0 && isfinite(denom)) {
                    const double ci = (price - esa) / denom;
                    if (!wt1_init) {
                        if (isfinite(ci)) { wt1_state = ci; wt1_init = true; }
                    } else if (isfinite(ci)) {
                        const double alpha_avg_d = static_cast<double>(alpha_avg);
                        const double beta_avg_d  = 1.0 - alpha_avg_d;
                        wt1_state = fma(alpha_avg_d, ci, beta_avg_d * wt1_state);
                    }
                }
            }
            if (wt1_init) wt1_val = static_cast<float>(wt1_state);
            wt1_col[idx] = wt1_val;

            
            if (isfinite(static_cast<double>(wt1_val))) { sum_wt1 += wt1_state; ++window_count; }
            if (t >= ma_length) {
                const float old = wt1_col[(t - ma_length) * cols];
                if (isfinite(static_cast<double>(old))) { sum_wt1 -= static_cast<double>(old); --window_count; }
            }

            float wt2_val = CUDART_NAN_F;
            if (window_count >= ma_length) wt2_val = static_cast<float>(sum_wt1 * inv_ma);
            wt2_col[idx] = wt2_val;

            
            if (t >= warmup && isfinite(static_cast<double>(wt1_val)) && isfinite(static_cast<double>(wt2_val))) {
                diff_col[idx] = wt2_val - wt1_val;
            } else {
                diff_col[idx] = CUDART_NAN_F;
            }
        }

        
        for (int t = 0; t < rows && t < warmup; ++t) {
            const int idx = t * cols;
            wt1_col[idx]  = CUDART_NAN_F;
            wt2_col[idx]  = CUDART_NAN_F;
            diff_col[idx] = CUDART_NAN_F;
        }
    }
}
