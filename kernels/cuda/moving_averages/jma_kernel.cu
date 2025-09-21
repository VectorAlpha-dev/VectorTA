// CUDA kernels for the Jurik Moving Average (JMA).
//
// Each parameter combination is evaluated sequentially because the recurrence
// depends on the previous output (j_prev). Blocks are mapped one-to-one with
// parameter combos or series, while thread 0 performs the time-domain walk.
// Host helpers precompute the per-combo constants (alpha, 1-beta, phase_ratio)
// so the kernels only execute fused multiply-add style arithmetic.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void jma_batch_f32(const float* __restrict__ prices,
                   const float* __restrict__ alphas,
                   const float* __restrict__ one_minus_betas,
                   const float* __restrict__ phase_ratios,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    float* out_row = out + combo * series_len;
    const float nanf32 = nanf("");
    for (int i = 0; i < series_len; ++i) {
        out_row[i] = nanf32;
    }

    if (series_len <= 0) {
        return;
    }
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    double alpha = static_cast<double>(alphas[combo]);
    double one_minus_beta = static_cast<double>(one_minus_betas[combo]);
    double beta = 1.0 - one_minus_beta;
    double phase_ratio = static_cast<double>(phase_ratios[combo]);
    double one_minus_alpha = 1.0 - alpha;
    double alpha_sq = alpha * alpha;
    double oma_sq = one_minus_alpha * one_minus_alpha;

    double seed = static_cast<double>(prices[first_valid]);
    double e0 = seed;
    double e1 = 0.0;
    double e2 = 0.0;
    double j_prev = seed;

    out_row[first_valid] = static_cast<float>(j_prev);

    for (int i = first_valid + 1; i < series_len; ++i) {
        double price = static_cast<double>(prices[i]);
        e0 = one_minus_alpha * price + alpha * e0;
        double diff_price = price - e0;
        e1 = diff_price * one_minus_beta + beta * e1;
        double diff = e0 + phase_ratio * e1 - j_prev;
        e2 = diff * oma_sq + alpha_sq * e2;
        j_prev += e2;
        out_row[i] = static_cast<float>(j_prev);
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
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const float nanf32 = nanf("");
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series_idx;
        out_tm[idx] = nanf32;
    }

    if (series_len <= 0) {
        return;
    }

    int first_valid = first_valids[series_idx];
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    double alpha = static_cast<double>(alpha_f);
    double one_minus_beta = static_cast<double>(one_minus_beta_f);
    double beta = 1.0 - one_minus_beta;
    double phase_ratio = static_cast<double>(phase_ratio_f);
    double one_minus_alpha = 1.0 - alpha;
    double alpha_sq = alpha * alpha;
    double oma_sq = one_minus_alpha * one_minus_alpha;

    int first_idx = first_valid * num_series + series_idx;
    double seed = static_cast<double>(prices_tm[first_idx]);
    double e0 = seed;
    double e1 = 0.0;
    double e2 = 0.0;
    double j_prev = seed;

    out_tm[first_idx] = static_cast<float>(j_prev);

    for (int t = first_valid + 1; t < series_len; ++t) {
        int idx = t * num_series + series_idx;
        double price = static_cast<double>(prices_tm[idx]);
        e0 = one_minus_alpha * price + alpha * e0;
        double diff_price = price - e0;
        e1 = diff_price * one_minus_beta + beta * e1;
        double diff = e0 + phase_ratio * e1 - j_prev;
        e2 = diff * oma_sq + alpha_sq * e2;
        j_prev += e2;
        out_tm[idx] = static_cast<float>(j_prev);
    }
}
