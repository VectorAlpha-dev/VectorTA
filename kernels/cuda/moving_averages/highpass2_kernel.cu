// CUDA kernels for the two-pole high-pass filter (highpass_2_pole).
//
// Mirrors the ALMA-style API surface by providing one-series × many-parameter
// and many-series × one-parameter entry points. Kernels operate entirely in
// FP32 and rely on host-precomputed coefficients so each output row reuses the
// shared constants rather than recomputing trig-heavy terms.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void highpass2_batch_f32(const float* __restrict__ prices,
                         const int* __restrict__ periods,
                         const float* __restrict__ c_vals,
                         const float* __restrict__ cm2_vals,
                         const float* __restrict__ two_1m_vals,
                         const float* __restrict__ neg_oma_sq_vals,
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (series_len <= 0) {
        return;
    }

    const int period = periods[combo];
    const float c = c_vals[combo];
    const float cm2 = cm2_vals[combo];
    const float two_1m = two_1m_vals[combo];
    const float neg_oma_sq = neg_oma_sq_vals[combo];
    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int first_idx = row_offset + first_valid;
    float x_im2 = prices[first_valid];
    out[first_idx] = x_im2;
    if (first_valid + 1 >= series_len) {
        return;
    }

    const int second_idx = row_offset + first_valid + 1;
    float x_im1 = prices[first_valid + 1];
    out[second_idx] = x_im1;

    float y_im2 = x_im2;
    float y_im1 = x_im1;

    for (int t = first_valid + 2; t < series_len; ++t) {
        const float x_i = prices[t];
        const float t0 = __fmaf_rn(cm2, x_im1, c * x_i);
        const float t1 = __fmaf_rn(c, x_im2, t0);
        const float y_i = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, t1));

        out[row_offset + t] = y_i;

        x_im2 = x_im1;
        x_im1 = x_i;
        y_im2 = y_im1;
        y_im1 = y_i;
    }
}

extern "C" __global__
void highpass2_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                         const int* __restrict__ first_valids,
                                         int period,
                                         float c,
                                         float cm2,
                                         float two_1m,
                                         float neg_oma_sq,
                                         int num_series,
                                         int series_len,
                                         float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (series_len <= 0) {
        return;
    }

    const int stride = num_series;
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int offset = first_valid * stride + series_idx;
    float x_im2 = prices_tm[offset];
    out_tm[offset] = x_im2;
    if (first_valid + 1 >= series_len) {
        return;
    }

    offset = (first_valid + 1) * stride + series_idx;
    float x_im1 = prices_tm[offset];
    out_tm[offset] = x_im1;

    float y_im2 = x_im2;
    float y_im1 = x_im1;

    for (int t = first_valid + 2; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float x_i = prices_tm[idx];
        const float t0 = __fmaf_rn(cm2, x_im1, c * x_i);
        const float t1 = __fmaf_rn(c, x_im2, t0);
        const float y_i = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, t1));

        out_tm[idx] = y_i;

        x_im2 = x_im1;
        x_im1 = x_i;
        y_im2 = y_im1;
        y_im1 = y_i;
    }
}
