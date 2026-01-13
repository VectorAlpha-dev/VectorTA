


#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>








#ifdef JMA_INTERNAL_F64
  using JMA_T = double;
  #define JMA_FMA  fma
  #define JMA_NAN  CUDART_NAN
  __device__ __forceinline__ JMA_T cvt(float x){ return static_cast<double>(x); }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return static_cast<float>(x); }
#else
  using JMA_T = float;
  #define JMA_FMA  __fmaf_rn
  #define JMA_NAN  CUDART_NAN_F
  __device__ __forceinline__ JMA_T cvt(float x){ return x; }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return x; }
#endif



extern "C" __global__
void jma_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ alphas,
                   const float* __restrict__ one_minus_betas,
                   const float* __restrict__ phase_ratios,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out)
{

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* __restrict__ out_row = out + combo * series_len;

    if (series_len <= 0) return;

    int fv = first_valid;
    if (fv < 0) fv = 0;


    if (fv >= series_len) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < series_len; ++i) out_row[i] = nanv;
        return;
    }


    if (fv > 0) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < fv; ++i) out_row[i] = nanv;
    }


    const JMA_T alpha           = cvt(alphas[combo]);
    const JMA_T one_minus_beta  = cvt(one_minus_betas[combo]);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratios[combo]);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;


    JMA_T e0 = cvt(prices[fv]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_row[fv] = cvt_back(j_prev);


    for (int i = fv + 1; i < series_len; ++i) {
        const JMA_T price = cvt(prices[i]);


        e0 = JMA_FMA(alpha, e0, one_minus_alpha * price);


        const JMA_T diff_price = price - e0;
        e1 = JMA_FMA(beta, e1, one_minus_beta * diff_price);


        const JMA_T diff = JMA_FMA(phase_ratio, e1, e0) - j_prev;


        e2 = JMA_FMA(alpha_sq, e2, oma_sq * diff);

        j_prev += e2;
        out_row[i] = cvt_back(j_prev);
    }
}

extern "C" __global__
void jma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   float alpha_f,
                                   float one_minus_beta_f,
                                   float phase_ratio_f,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_tm)
{

    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    if (series_len <= 0) return;

    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;

    const float nanv = JMA_NAN;


    if (fv >= series_len) {
        int idx = series_idx;
        for (int t = 0; t < series_len; ++t, idx += num_series) out_tm[idx] = nanv;
        return;
    }


    if (fv > 0) {
        int idx = series_idx;
        for (int t = 0; t < fv; ++t, idx += num_series) out_tm[idx] = nanv;
    }


    const JMA_T alpha           = cvt(alpha_f);
    const JMA_T one_minus_beta  = cvt(one_minus_beta_f);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratio_f);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;


    int idx = fv * num_series + series_idx;
    JMA_T e0 = cvt(prices_tm[idx]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_tm[idx] = cvt_back(j_prev);


    for (int t = fv + 1; t < series_len; ++t) {
        idx += num_series;
        const JMA_T price = cvt(prices_tm[idx]);

        e0 = JMA_FMA(alpha, e0, one_minus_alpha * price);
        const JMA_T diff_price = price - e0;
        e1 = JMA_FMA(beta,  e1, one_minus_beta * diff_price);
        const JMA_T diff = JMA_FMA(phase_ratio, e1, e0) - j_prev;
        e2 = JMA_FMA(alpha_sq, e2, oma_sq * diff);

        j_prev += e2;
        out_tm[idx] = cvt_back(j_prev);
    }
}
