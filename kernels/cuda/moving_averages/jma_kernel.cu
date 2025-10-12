// Optimized JMA kernels (CUDA 13 / Ada+), FP32 hot path with FMAs.
// Drop-in replacements for your two kernels.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>   // CUDART_NAN_F

// -----------------------------------------------------------------------------
// Toggle this to fall back to FP64 internal math if you must A/B compare.
// On RTX 4090 and most Ada GeForce parts, FP64 is 1/64 rate vs FP32, so keep
// this undefined for performance.
// -----------------------------------------------------------------------------
/* #define JMA_INTERNAL_F64 */

#ifdef JMA_INTERNAL_F64
  using JMA_T = double;
  #define JMA_FMA  fma
  #define JMA_NAN  CUDART_NAN
  __device__ __forceinline__ JMA_T cvt(float x){ return static_cast<double>(x); }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return static_cast<float>(x); }
#else
  using JMA_T = float;
  #define JMA_FMA  __fmaf_rn   // fused multiply-add, round-to-nearest-even.
  #define JMA_NAN  CUDART_NAN_F
  __device__ __forceinline__ JMA_T cvt(float x){ return x; }
  __device__ __forceinline__ float cvt_back(JMA_T x){ return x; }
#endif

// -----------------------------------------------------------------------------

extern "C" __global__
void jma_batch_f32(const float* __restrict__ prices,          // [series_len]
                   const float* __restrict__ alphas,          // [n_combos]
                   const float* __restrict__ one_minus_betas, // [n_combos]
                   const float* __restrict__ phase_ratios,    // [n_combos]
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out)                   // [n_combos * series_len]
{
    // One thread per combo for full warp utilization.
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* __restrict__ out_row = out + combo * series_len;

    if (series_len <= 0) return;

    int fv = first_valid;
    if (fv < 0) fv = 0;

    // Degenerate: if first_valid is beyond the series, just fill NaNs for compatibility.
    if (fv >= series_len) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < series_len; ++i) out_row[i] = nanv;
        return;
    }

    // Pre-prefix NaNs only where needed (avoid full-row prefill).
    if (fv > 0) {
        const float nanv = JMA_NAN;
        for (int i = 0; i < fv; ++i) out_row[i] = nanv;
    }

    // Load per-combo constants into registers (FP32 path by default).
    const JMA_T alpha           = cvt(alphas[combo]);
    const JMA_T one_minus_beta  = cvt(one_minus_betas[combo]);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratios[combo]);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;

    // Seed state from the first valid price.
    JMA_T e0 = cvt(prices[fv]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_row[fv] = cvt_back(j_prev);

    // Time-domain walk (sequential by definition).
    for (int i = fv + 1; i < series_len; ++i) {
        const JMA_T price = cvt(prices[i]);

        // e0 = (1-α)*price + α*e0  -> FMA
        e0 = JMA_FMA(alpha, e0, one_minus_alpha * price);

        // e1 = (price - e0)*(1-β) + β*e1
        const JMA_T diff_price = price - e0;
        e1 = JMA_FMA(beta, e1, one_minus_beta * diff_price);

        // diff = (e0 + phase_ratio*e1) - j_prev
        const JMA_T diff = JMA_FMA(phase_ratio, e1, e0) - j_prev;

        // e2 = diff*(1-α)^2 + α^2*e2
        e2 = JMA_FMA(alpha_sq, e2, oma_sq * diff);

        j_prev += e2;
        out_row[i] = cvt_back(j_prev);
    }
}

extern "C" __global__
void jma_many_series_one_param_f32(const float* __restrict__ prices_tm, // [series_len * num_series], time-major
                                   float alpha_f,
                                   float one_minus_beta_f,
                                   float phase_ratio_f,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids, // [num_series]
                                   float* __restrict__ out_tm)            // [series_len * num_series], time-major
{
    // One thread per series index for coalesced accesses each timestep.
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    if (series_len <= 0) return;

    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;

    const float nanv = JMA_NAN;

    // Degenerate: entire column is NaN.
    if (fv >= series_len) {
        int idx = series_idx;
        for (int t = 0; t < series_len; ++t, idx += num_series) out_tm[idx] = nanv;
        return;
    }

    // Prefix NaNs up to first_valid.
    if (fv > 0) {
        int idx = series_idx;
        for (int t = 0; t < fv; ++t, idx += num_series) out_tm[idx] = nanv;
    }

    // Constants in registers.
    const JMA_T alpha           = cvt(alpha_f);
    const JMA_T one_minus_beta  = cvt(one_minus_beta_f);
    const JMA_T beta            = JMA_T(1) - one_minus_beta;
    const JMA_T phase_ratio     = cvt(phase_ratio_f);
    const JMA_T one_minus_alpha = JMA_T(1) - alpha;
    const JMA_T alpha_sq        = alpha * alpha;
    const JMA_T oma_sq          = one_minus_alpha * one_minus_alpha;

    // Seed at t = fv.
    int idx = fv * num_series + series_idx;
    JMA_T e0 = cvt(prices_tm[idx]);
    JMA_T e1 = JMA_T(0);
    JMA_T e2 = JMA_T(0);
    JMA_T j_prev = e0;

    out_tm[idx] = cvt_back(j_prev);

    // Walk forward in time; coalesced across threads at each step.
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
