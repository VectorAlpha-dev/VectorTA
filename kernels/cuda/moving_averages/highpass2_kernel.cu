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

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the 2x2 affine state update:
//
//   y_t = a1*y_{t-1} + a2*y_{t-2} + u_t
//   u_t = c*x_t + cm2*x_{t-1} + c*x_{t-2}
//
// State is s_t = [y_t, y_{t-1}] and:
//   s_t = M*s_{t-1} + [u_t, 0]
//   M = [[a1, a2],
//        [ 1,  0]]
//
// - blockDim.x must be exactly 32
// - outputs:
//   - [0, first_valid) = NaN
//   - y[first_valid] = x[first_valid], y[first_valid+1] = x[first_valid+1]
//   - t >= first_valid+2 computed by recurrence
extern "C" __global__
void highpass2_batch_warp_scan_f32(const float* __restrict__ prices,
                                   const int*   __restrict__ periods,         // unused (ABI)
                                   const float* __restrict__ c_vals,
                                   const float* __restrict__ cm2_vals,
                                   const float* __restrict__ two_1m_vals,
                                   const float* __restrict__ neg_oma_sq_vals,
                                   int series_len,
                                   int n_combos,
                                   int first_valid,
                                   float* __restrict__ out) {
    (void)periods;
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0) return;
    if (threadIdx.x >= 32) return;

    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;

    float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

    if (first_valid < 0 || first_valid >= series_len) {
        for (int t = lane; t < series_len; t += 32) out_row[t] = CUDART_NAN_F;
        return;
    }

    // Prefix NaNs only where needed
    for (int t = lane; t < first_valid; t += 32) out_row[t] = CUDART_NAN_F;

    // Seed two points = raw inputs
    if (lane == 0) {
        out_row[first_valid] = prices[first_valid];
        if (first_valid + 1 < series_len) {
            out_row[first_valid + 1] = prices[first_valid + 1];
        }
    }
    if (first_valid + 1 >= series_len) return;

    const float c      = c_vals[combo];
    const float cm2    = cm2_vals[combo];
    const float a1     = two_1m_vals[combo];
    const float a2     = neg_oma_sq_vals[combo];

    // Previous state at t0-1 = first_valid+1
    float s0_prev = 0.0f;
    float s1_prev = 0.0f;
    if (lane == 0) {
        s1_prev = prices[first_valid];     // y_{t0-2}
        s0_prev = prices[first_valid + 1]; // y_{t0-1}
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);

    // Constant state-transition matrix M
    const float m00 = a1;
    const float m01 = a2;
    const float m10 = 1.0f;
    const float m11 = 0.0f;

    int t0 = first_valid + 2;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        // Per-step u_t term (scalar)
        float u = 0.0f;
        if (valid) {
            const float x0  = prices[t];
            const float x1  = prices[t - 1];
            const float x2  = prices[t - 2];
            // u = c*x0 + cm2*x1 + c*x2
            u = fmaf(c, x2, fmaf(cm2, x1, c * x0));
        }

        // Each lane starts with its step transform: (P, v), where
        // P = M and v = [u, 0]. For inactive lanes, use identity.
        float p00 = valid ? m00 : 1.0f;
        float p01 = valid ? m01 : 0.0f;
        float p10 = valid ? m10 : 0.0f;
        float p11 = valid ? m11 : 1.0f;
        float v0  = valid ? u   : 0.0f;
        float v1  = 0.0f;

        // Inclusive scan over composed transforms: (P_cur, v_cur) ∘ (P_prev, v_prev)
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float p00_prev = __shfl_up_sync(mask, p00, offset);
            const float p01_prev = __shfl_up_sync(mask, p01, offset);
            const float p10_prev = __shfl_up_sync(mask, p10, offset);
            const float p11_prev = __shfl_up_sync(mask, p11, offset);
            const float v0_prev  = __shfl_up_sync(mask, v0,  offset);
            const float v1_prev  = __shfl_up_sync(mask, v1,  offset);
            if (lane >= offset) {
                const float c00 = p00, c01 = p01, c10 = p10, c11 = p11;
                const float cv0 = v0,  cv1 = v1;

                const float n00 = fmaf(c00, p00_prev, c01 * p10_prev);
                const float n01 = fmaf(c00, p01_prev, c01 * p11_prev);
                const float n10 = fmaf(c10, p00_prev, c11 * p10_prev);
                const float n11 = fmaf(c10, p01_prev, c11 * p11_prev);

                const float nv0 = fmaf(c00, v0_prev, fmaf(c01, v1_prev, cv0));
                const float nv1 = fmaf(c10, v0_prev, fmaf(c11, v1_prev, cv1));

                p00 = n00; p01 = n01; p10 = n10; p11 = n11;
                v0  = nv0; v1  = nv1;
            }
        }

        // Apply prefix transform to previous state s_prev
        const float s0 = fmaf(p00, s0_prev, fmaf(p01, s1_prev, v0));
        const float s1 = fmaf(p10, s0_prev, fmaf(p11, s1_prev, v1));

        if (valid) {
            out_row[t] = s0;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        s0_prev = __shfl_sync(mask, s0, last_lane);
        s1_prev = __shfl_sync(mask, s1, last_lane);
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
