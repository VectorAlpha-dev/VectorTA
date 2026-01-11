










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>


#ifndef AT_USE_F64_SUM

#define AT_USE_F64_SUM 0
#endif

#ifndef AT_BLOCK_SIZE
#define AT_BLOCK_SIZE 256
#endif


__device__ __forceinline__ void kahan_add(float& sum, float& c, float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

namespace {
__device__ inline bool is_finite(float x) { return !isnan(x) && !isinf(x); }
}

extern "C" __global__ void alphatrend_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ tr,                 
    const float* __restrict__ momentum_table,     
    const int*   __restrict__ mrow_for_combo,     
    const float* __restrict__ coeffs,             
    const int*   __restrict__ periods,            
    int len,
    int first_valid,
    int n_combos,
    int n_mrows,
    float* __restrict__ k1_out,                   
    float* __restrict__ k2_out)                   
{
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid0; row < n_combos; row += stride) {
        const int period = periods[row];
        const float coeff = coeffs[row];
        float* __restrict__ k1_row = k1_out + (size_t)row * len;
        float* __restrict__ k2_row = k2_out + (size_t)row * len;

        
        if (period <= 0 || period > len) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue;
        }

        const int warm = first_valid + period - 1;
        if (warm >= len) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue; 
        }

        const int mrow = mrow_for_combo[row];
        if (mrow < 0 || mrow >= n_mrows) {
            for (int i = 0; i < len; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }
            continue;
        }
        const float* __restrict__ mom = momentum_table + (size_t)mrow * len;

        
        for (int i = 0; i < warm; ++i) { k1_row[i] = CUDART_NAN_F; k2_row[i] = CUDART_NAN_F; }

        
        const float p_inv = 1.0f / (float)period;
#if AT_USE_F64_SUM
        double s = 0.0;
        for (int j = first_valid; j <= warm; ++j) s += (double)tr[j];
#else
        float s = 0.0f, c = 0.0f;
        for (int j = first_valid; j <= warm; ++j) kahan_add(s, c, tr[j]);
#endif

        
        float prev_alpha = CUDART_NAN_F;
        float prev1 = CUDART_NAN_F;
        float prev2 = CUDART_NAN_F;

        #pragma unroll 1
        for (int i = warm; i < len; ++i) {
            
#if AT_USE_F64_SUM
            const float a = (float)(s * (double)p_inv);
#else
            const float a = s * p_inv;
#endif
            const float up = fmaf(-coeff, a, low[i]);   
            const float dn = fmaf( coeff, a, high[i]);  

            const float m = mom[i];
            const bool m_ge_50 = is_finite(m) ? (m >= 50.0f) : false;

            
            const float up_clamped = fmaxf(up, prev_alpha);
            const float dn_clamped = fminf(dn, prev_alpha);
            const float cur = m_ge_50 ? up_clamped : dn_clamped;

            k1_row[i] = cur;
            k2_row[i] = prev2;     

            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;

            const int nxt = i + 1;
            if (nxt < len) {
#if AT_USE_F64_SUM
                s += (double)tr[nxt] - (double)tr[nxt - period];
#else
                kahan_add(s, c, tr[nxt]);
                kahan_add(s, c, -tr[nxt - period]);
#endif
            }
        }
    }
}



extern "C" __global__ void alphatrend_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ tr_tm,
    const float* __restrict__ momentum_tm,
    const int*   __restrict__ first_valids, 
    int num_series,
    int series_len,
    float coeff,
    int period,
    float* __restrict__ k1_tm_out, 
    float* __restrict__ k2_tm_out) 
{
    int s = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= num_series) return;
    const int fv = first_valids[s];

    if (period <= 0 || fv >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + s;
            k1_tm_out[idx] = CUDART_NAN_F;
            k2_tm_out[idx] = CUDART_NAN_F;
        }
        return;
    }

    const int warm = fv + period - 1;
    const float p_inv = 1.0f / (float)period;

    if (warm >= series_len) {
        for (int t = 0; t < series_len; ++t) {
            const int idx = t * num_series + s;
            k1_tm_out[idx] = CUDART_NAN_F;
            k2_tm_out[idx] = CUDART_NAN_F;
        }
        return;
    }

    
    for (int t = 0; t < warm; ++t) {
        const int idx = t * num_series + s;
        k1_tm_out[idx] = CUDART_NAN_F;
        k2_tm_out[idx] = CUDART_NAN_F;
    }

    
#if AT_USE_F64_SUM
    double ssum = 0.0;
    for (int t = fv; t <= warm; ++t) ssum += (double)tr_tm[t * num_series + s];
#else
    float ssum = 0.0f, csum = 0.0f;
    for (int t = fv; t <= warm; ++t) kahan_add(ssum, csum, tr_tm[t * num_series + s]);
#endif

    float prev_alpha = CUDART_NAN_F, prev1 = CUDART_NAN_F, prev2 = CUDART_NAN_F;

    #pragma unroll 1
    for (int t = warm; t < series_len; ++t) {
        const int idx = t * num_series + s;

#if AT_USE_F64_SUM
        const float a = (float)(ssum * (double)p_inv);
#else
        const float a = ssum * p_inv;
#endif
        const float up = fmaf(-coeff, a, low_tm[idx]);
        const float dn = fmaf( coeff, a, high_tm[idx]);
        const float m  = momentum_tm[idx];
        const bool m_ge_50 = is_finite(m) ? (m >= 50.0f) : false;

        const float up_clamped = fmaxf(up, prev_alpha);
        const float dn_clamped = fminf(dn, prev_alpha);
        const float cur = m_ge_50 ? up_clamped : dn_clamped;

        k1_tm_out[idx] = cur;
        k2_tm_out[idx] = prev2;

        prev2 = prev1;
        prev1 = cur;
        prev_alpha = cur;

        const int nxt = t + 1;
        if (nxt < series_len) {
#if AT_USE_F64_SUM
            ssum += (double)tr_tm[nxt * num_series + s] - (double)tr_tm[(nxt - period) * num_series + s];
#else
            kahan_add(ssum, csum,  tr_tm[nxt * num_series + s]);
            kahan_add(ssum, csum, -tr_tm[(nxt - period) * num_series + s]);
#endif
        }
    }
}



extern "C" __global__ void atr_table_from_tr_f32(
    const float* __restrict__ tr,          
    int len,
    int first_valid,
    const int* __restrict__ periods_unique,
    int n_u,
    float* __restrict__ atr_table          
){
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_u) return;

    const int period = periods_unique[u];
    float* __restrict__ out = atr_table + (size_t)u * len;

    if (period <= 0 || period > len) {
        for (int i=0;i<len;++i) out[i] = CUDART_NAN_F;
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i=0;i<warm;++i) out[i] = CUDART_NAN_F;

#if AT_USE_F64_SUM
    double s = 0.0;
    for (int j = first_valid; j <= warm; ++j) s += (double)tr[j];
#else
    float s = 0.0f, c = 0.0f;
    for (int j = first_valid; j <= warm; ++j) kahan_add(s, c, tr[j]);
#endif

    const float p_inv = 1.0f / (float)period;

    #pragma unroll 1
    for (int i = warm; i < len; ++i) {
#if AT_USE_F64_SUM
        out[i] = (float)(s * (double)p_inv);
#else
        out[i] = s * p_inv;
#endif
        const int nxt = i + 1;
        if (nxt < len) {
#if AT_USE_F64_SUM
            s += (double)tr[nxt] - (double)tr[nxt - period];
#else
            kahan_add(s, c, tr[nxt]);
            kahan_add(s, c, -tr[nxt - period]);
#endif
        }
    }
}

extern "C" __global__ void momentum_to_mask_bits(
    const float* __restrict__ momentum_table, 
    int len, int n_mrows,
    unsigned* __restrict__ mask_bits          
){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_mrows) return;

    const float* __restrict__ mrow = momentum_table + (size_t)row * len;
    const int n_words = (len + 31) >> 5;
    unsigned* __restrict__ out = mask_bits + (size_t)row * n_words;

    for (int w = 0; w < n_words; ++w) {
        unsigned word = 0u;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            const int i = (w << 5) + b;
            if (i >= len) break;
            const float m = mrow[i];
            const unsigned bit = (is_finite(m) && m >= 50.0f) ? 1u : 0u;
            word |= (bit << b);
        }
        out[w] = word;
    }
}

__device__ __forceinline__ bool mask_test(const unsigned* __restrict__ row, int i){
    const unsigned w = row[i >> 5];
    return (w >> (i & 31)) & 1u;
}

extern "C" __global__ void alphatrend_batch_from_precomputed_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ atr_table,       
    const unsigned* __restrict__ mask_bits,    
    const int* __restrict__ period_row_for_combo, 
    const int* __restrict__ mrow_for_combo,       
    const float* __restrict__ coeffs,             
    const int*   __restrict__ periods,            
    int len,
    int first_valid,
    int n_combos,
    int n_pr, int n_mrows,
    float* __restrict__ k1_out, float* __restrict__ k2_out)
{
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid0; row < n_combos; row += stride) {
        const int period = periods[row];
        float* __restrict__ k1_row = k1_out + (size_t)row * len;
        float* __restrict__ k2_row = k2_out + (size_t)row * len;

        if (period <= 0 || period > len) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int warm = first_valid + period - 1;
        if (warm >= len) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int pr = period_row_for_combo[row];
        if (pr < 0 || pr >= n_pr) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const int mrow = mrow_for_combo[row];
        if (mrow < 0 || mrow >= n_mrows) {
            for (int i=0;i<len;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }
            continue;
        }
        const float* __restrict__ arow = atr_table + (size_t)pr * len;
        const unsigned* __restrict__ mask_row = mask_bits + (size_t)mrow * ((len + 31) >> 5);
        const float coeff = coeffs[row];

        for (int i=0;i<warm;++i){ k1_row[i]=CUDART_NAN_F; k2_row[i]=CUDART_NAN_F; }

        float prev_alpha = CUDART_NAN_F, prev1 = CUDART_NAN_F, prev2 = CUDART_NAN_F;

        #pragma unroll 1
        for (int i = warm; i < len; ++i){
            const float a = arow[i];
            const float up = fmaf(-coeff, a, low[i]);
            const float dn = fmaf( coeff, a, high[i]);

            const bool m_ge_50 = mask_test(mask_row, i);
            const float up_clamped = fmaxf(up, prev_alpha);
            const float dn_clamped = fminf(dn, prev_alpha);
            const float cur = m_ge_50 ? up_clamped : dn_clamped;

            k1_row[i] = cur;
            k2_row[i] = prev2;

            prev2 = prev1;
            prev1 = cur;
            prev_alpha = cur;
        }
    }
}

