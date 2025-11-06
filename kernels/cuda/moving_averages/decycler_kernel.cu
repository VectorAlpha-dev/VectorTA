// CUDA kernels for Ehlers Decycler (input minus 2‑pole high‑pass output).
//
// Batch (one series × many params): uses host-precomputed second-difference `diff[i] = x[i]-2x[i-1]+x[i-2]`
// and per-row coefficients (c, two_1m, neg_oma_sq). Each thread processes grid-strided combos.
//
// Many-series × one-param (time-major): processes each series independently; writes NaN during warmup
// (first_valid .. first_valid+1) and computes out[t] = x[t] - hp[t] for t >= first_valid+2.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void decycler_batch_f32(const float* __restrict__ prices,          // [series_len]
                        const int*   __restrict__ periods,         // [n_combos] (kept for ABI)
                        const float* __restrict__ c_vals,          // [n_combos]
                        const float* __restrict__ two_1m_vals,     // [n_combos]
                        const float* __restrict__ neg_oma_sq_vals, // [n_combos]
                        const float* __restrict__ diff,            // [series_len] second difference
                        int series_len,
                        int n_combos,
                        int first_valid,
                        float* __restrict__ out)                   // [n_combos * series_len]
{
    (void)periods; // not used in recurrence, retained for a stable ABI
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int combo = tid; combo < n_combos; combo += blockDim.x * gridDim.x) {
        float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

        if (series_len <= 0) continue;

        // If first_valid invalid, fill full row with NaNs
        if (first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = CUDART_NAN_F;
            continue;
        }

        const float c          = c_vals[combo];
        const float two_1m     = two_1m_vals[combo];
        const float neg_oma_sq = neg_oma_sq_vals[combo];

        // Warmup semantics: NaN for indices < first_valid+2
        const int warm = min(series_len, first_valid + 2);
        for (int i = 0; i < warm; ++i) out_row[i] = CUDART_NAN_F;

        if (first_valid + 1 >= series_len) continue; // nothing beyond warmup

        // Seed high-pass internal state from inputs (as in scalar):
        float hp_im2 = prices[first_valid];
        float hp_im1 = prices[first_valid + 1];

        for (int t = first_valid + 2; t < series_len; ++t) {
            // hp[t] = two_1m*hp[t-1] + neg_oma_sq*hp[t-2] + c * diff[t]
            const float s3 = __fmaf_rn(two_1m, hp_im1, c * diff[t]);
            const float hp = __fmaf_rn(neg_oma_sq, hp_im2, s3);
            // decycler = x[t] - hp[t]
            out_row[t] = prices[t] - hp;
            // advance state
            hp_im2 = hp_im1;
            hp_im1 = hp;
        }
    }
}

// Many-series × one-param, time-major layout
extern "C" __global__
void decycler_many_series_one_param_f32(const float* __restrict__ prices_tm, // [series_len * num_series], time-major
                                        const int*   __restrict__ first_valids, // [num_series]
                                        int period,    // kept for ABI
                                        float c,
                                        float two_1m,
                                        float neg_oma_sq,
                                        int num_series,
                                        int series_len,
                                        float* __restrict__ out_tm)
{
    (void)period;
    const int stride = num_series; // time-major stride (contiguous in time)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = tid; s < num_series; s += blockDim.x * gridDim.x) {
        if (series_len <= 0) continue;
        const int fv = first_valids[s];
        if (fv < 0 || fv >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
            continue;
        }

        // Warmup: NaNs through fv+1
        const int warm = min(series_len, fv + 2);
        for (int t = 0; t < warm; ++t) {
            out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
        }
        if (fv + 1 >= series_len) continue;

        // Seed hp state
        float hp_im2 = prices_tm[(size_t)fv * (size_t)stride + s];
        float hp_im1 = prices_tm[(size_t)(fv + 1) * (size_t)stride + s];

        for (int t = fv + 2; t < series_len; ++t) {
            const int idx = (size_t)t * (size_t)stride + s;
            const float x = prices_tm[idx];
            const float s3 = __fmaf_rn(two_1m, hp_im1, c * (x - 2.0f * prices_tm[idx - stride] + prices_tm[idx - 2 * stride]));
            const float hp = __fmaf_rn(neg_oma_sq, hp_im2, s3);
            out_tm[idx] = x - hp;
            hp_im2 = hp_im1;
            hp_im1 = hp;
        }
    }
}

