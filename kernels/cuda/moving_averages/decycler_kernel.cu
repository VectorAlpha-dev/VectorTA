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

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the 2x2 affine state update:
//
//   hp[t] = a1*hp[t-1] + a2*hp[t-2] + u[t]
//   u[t]  = c * diff[t]
//
// State is s[t] = [hp[t], hp[t-1]] and:
//   s[t] = M*s[t-1] + [u[t], 0]
//   M = [[a1, a2],
//        [ 1,  0]]
//
// Output: out[t] = x[t] - hp[t], with warmup NaNs for indices < first_valid+2.
//
// - blockDim.x must be exactly 32
extern "C" __global__
void decycler_batch_warp_scan_f32(const float* __restrict__ prices,
                                 const int*   __restrict__ periods,         // unused (ABI)
                                 const float* __restrict__ c_vals,
                                 const float* __restrict__ two_1m_vals,
                                 const float* __restrict__ neg_oma_sq_vals,
                                 const float* __restrict__ diff,
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

    const int warm = min(series_len, first_valid + 2);
    for (int t = lane; t < warm; t += 32) out_row[t] = CUDART_NAN_F;

    if (first_valid + 1 >= series_len) return;

    const float c      = c_vals[combo];
    const float a1     = two_1m_vals[combo];
    const float a2     = neg_oma_sq_vals[combo];

    // Previous state at t0-1 = first_valid+1
    float s0_prev = 0.0f;
    float s1_prev = 0.0f;
    if (lane == 0) {
        s1_prev = prices[first_valid];     // hp_{t0-2}
        s0_prev = prices[first_valid + 1]; // hp_{t0-1}
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);

    // Constant state-transition matrix M
    const float m00 = a1;
    const float m01 = a2;
    const float m10 = 1.0f;
    const float m11 = 0.0f;

    const int t0 = first_valid + 2;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        float u = 0.0f;
        if (valid) {
            u = c * diff[t];
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
        const float hp0 = fmaf(p00, s0_prev, fmaf(p01, s1_prev, v0));
        const float hp1 = fmaf(p10, s0_prev, fmaf(p11, s1_prev, v1));

        if (valid) {
            out_row[t] = prices[t] - hp0;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        s0_prev = __shfl_sync(mask, hp0, last_lane);
        s1_prev = __shfl_sync(mask, hp1, last_lane);
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

