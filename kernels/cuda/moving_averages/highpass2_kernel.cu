// Optimized CUDA kernels for the two-pole high-pass filter (highpass_2_pole).
// - Thread-per-work-item + grid-stride loops.
// - No inter-thread dependency; no __syncthreads().
// - Only write NaNs where required; preserve "all-NaN if first_valid invalid".
// - Hot loop does pointer/index hoisting + 2-step unroll to raise ILP.
// - FP32 throughout with __fmaf_rn for accuracy.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef HP2_UNROLL2
#define HP2_UNROLL2 1  // set to 0 to disable 2-step unroll
#endif

// ---------- one series × many parameter combos ----------
extern "C" __global__
void highpass2_batch_f32(const float* __restrict__ prices,          // [series_len]
                         const int*   __restrict__ periods,         // [n_combos] (unused here, kept for ABI)
                         const float* __restrict__ c_vals,          // [n_combos]
                         const float* __restrict__ cm2_vals,        // [n_combos]
                         const float* __restrict__ two_1m_vals,     // [n_combos]
                         const float* __restrict__ neg_oma_sq_vals, // [n_combos]
                         int series_len,
                         int n_combos,
                         int first_valid,
                         float* __restrict__ out)                   // [n_combos * series_len]
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride over independent combos
    for (int combo = tid; combo < n_combos; combo += blockDim.x * gridDim.x) {

        // Load per-combo constants into registers once
        // (period kept for ABI; not used in the recurrence below)
        const int   period     = periods[combo];
        (void)period;
        const float c          = c_vals[combo];
        const float cm2        = cm2_vals[combo];
        const float two_1m     = two_1m_vals[combo];
        const float neg_oma_sq = neg_oma_sq_vals[combo];

        float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

        if (series_len <= 0) {
            // nothing to write
            continue;
        }

        // If first_valid is invalid, fill entire row with NaNs (preserving original semantics)
        if (first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = CUDART_NAN_F;
            continue;
        }

        // Prefix NaNs (only where needed)
        for (int i = 0; i < first_valid; ++i) out_row[i] = CUDART_NAN_F;

        // Seed outputs = raw inputs for first two valid points (as in original)
        float x_im2 = prices[first_valid];
        out_row[first_valid] = x_im2;
        if (first_valid + 1 >= series_len) continue;

        float x_im1 = prices[first_valid + 1];
        out_row[first_valid + 1] = x_im1;

        float y_im2 = x_im2;
        float y_im1 = x_im1;

        // Main recursive pass (t >= first_valid+2)
        int t = first_valid + 2;

#if HP2_UNROLL2
        // 2-step unroll to increase ILP and reduce loop overhead
        for (; t + 1 < series_len; t += 2) {
            // step 0
            const float x0   = prices[t];
            const float c_x0 = c * x0;
            const float t0   = __fmaf_rn(cm2, x_im1, c_x0);
            const float t1   = __fmaf_rn(c,    x_im2, t0);
            const float y0   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, t1));
            out_row[t] = y0;

            // advance state
            x_im2 = x_im1;
            x_im1 = x0;
            y_im2 = y_im1;
            y_im1 = y0;

            // step 1
            const float x1   = prices[t + 1];
            const float c_x1 = c * x1;
            const float u0   = __fmaf_rn(cm2, x_im1, c_x1);
            const float u1   = __fmaf_rn(c,    x_im2, u0);
            const float y1   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, u1));
            out_row[t + 1] = y1;

            // advance state
            x_im2 = x_im1;
            x_im1 = x1;
            y_im2 = y_im1;
            y_im1 = y1;
        }
#endif
        // tail
        for (; t < series_len; ++t) {
            const float x_i   = prices[t];
            const float c_xi  = c * x_i;
            const float t0    = __fmaf_rn(cm2, x_im1, c_xi);
            const float t1    = __fmaf_rn(c,    x_im2, t0);
            const float y_i   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, t1));
            out_row[t] = y_i;

            x_im2 = x_im1;
            x_im1 = x_i;
            y_im2 = y_im1;
            y_im1 = y_i;
        }
    }
}

// ---------- many series × one parameter ----------
extern "C" __global__
void highpass2_many_series_one_param_f32(const float* __restrict__ prices_tm,  // [series_len * num_series], time-major
                                         const int*   __restrict__ first_valids,// [num_series]
                                         int period,      // kept for ABI; unused
                                         float c,
                                         float cm2,
                                         float two_1m,
                                         float neg_oma_sq,
                                         int num_series,
                                         int series_len,
                                         float* __restrict__ out_tm)           // same layout as prices_tm
{
    (void)period;
    const int stride = num_series; // time-major stride

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid-stride over independent series
    for (int series_idx = tid; series_idx < num_series; series_idx += blockDim.x * gridDim.x) {

        if (series_len <= 0) continue;

        const int fv = first_valids[series_idx];

        // If fv invalid, fill whole column with NaNs to match original behavior
        if (fv < 0 || fv >= series_len) {
            for (int t = 0; t < series_len; ++t) {
                out_tm[(size_t)t * (size_t)stride + series_idx] = CUDART_NAN_F;
            }
            continue;
        }

        // Prefix NaNs only up to fv-1
        for (int t = 0; t < fv; ++t) {
            out_tm[(size_t)t * (size_t)stride + series_idx] = CUDART_NAN_F;
        }

        // Seed two points = raw inputs
        int idx0 = fv * stride + series_idx;
        float x_im2 = prices_tm[idx0];
        out_tm[idx0] = x_im2;

        if (fv + 1 >= series_len) continue;

        int idx1 = (fv + 1) * stride + series_idx;
        float x_im1 = prices_tm[idx1];
        out_tm[idx1] = x_im1;

        float y_im2 = x_im2;
        float y_im1 = x_im1;

        // Main recursive pass (t >= fv+2). Keep indexes linear in 't'.
        int t = fv + 2;

#if HP2_UNROLL2
        // 2-step unroll across time dimension
        for (; t + 1 < series_len; t += 2) {
            int i0 = t * stride + series_idx;
            const float x0   = prices_tm[i0];
            const float c_x0 = c * x0;
            const float t0   = __fmaf_rn(cm2, x_im1, c_x0);
            const float t1   = __fmaf_rn(c,    x_im2, t0);
            const float y0   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, t1));
            out_tm[i0] = y0;

            // advance
            x_im2 = x_im1;
            x_im1 = x0;
            y_im2 = y_im1;
            y_im1 = y0;

            int i1 = (t + 1) * stride + series_idx;
            const float x1   = prices_tm[i1];
            const float c_x1 = c * x1;
            const float u0   = __fmaf_rn(cm2, x_im1, c_x1);
            const float u1   = __fmaf_rn(c,    x_im2, u0);
            const float y1   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, u1));
            out_tm[i1] = y1;

            // advance
            x_im2 = x_im1;
            x_im1 = x1;
            y_im2 = y_im1;
            y_im1 = y1;
        }
#endif
        for (; t < series_len; ++t) {
            int i = t * stride + series_idx;
            const float x_i   = prices_tm[i];
            const float c_xi  = c * x_i;
            const float r0    = __fmaf_rn(cm2, x_im1, c_xi);
            const float r1    = __fmaf_rn(c,    x_im2, r0);
            const float y_i   = __fmaf_rn(two_1m, y_im1, __fmaf_rn(neg_oma_sq, y_im2, r1));
            out_tm[i] = y_i;

            x_im2 = x_im1;
            x_im1 = x_i;
            y_im2 = y_im1;
            y_im1 = y_i;
        }
    }
}
