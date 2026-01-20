







#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void decycler_batch_f32(const float* __restrict__ prices,
                        const int*   __restrict__ periods,
                        const float* __restrict__ c_vals,
                        const float* __restrict__ two_1m_vals,
                        const float* __restrict__ neg_oma_sq_vals,
                        const float* __restrict__ diff,
                        int series_len,
                        int n_combos,
                        int first_valid,
                        float* __restrict__ out)
{
    (void)periods;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int combo = tid; combo < n_combos; combo += blockDim.x * gridDim.x) {
        float* __restrict__ out_row = out + (size_t)combo * (size_t)series_len;

        if (series_len <= 0) continue;


        if (first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out_row[i] = CUDART_NAN_F;
            continue;
        }

        const float c          = c_vals[combo];
        const float two_1m     = two_1m_vals[combo];
        const float neg_oma_sq = neg_oma_sq_vals[combo];


        const int warm = min(series_len, first_valid + 2);
        for (int i = 0; i < warm; ++i) out_row[i] = CUDART_NAN_F;

        if (first_valid + 1 >= series_len) continue;


        float hp_im2 = prices[first_valid];
        float hp_im1 = prices[first_valid + 1];

        for (int t = first_valid + 2; t < series_len; ++t) {

            const float s3 = __fmaf_rn(two_1m, hp_im1, c * diff[t]);
            const float hp = __fmaf_rn(neg_oma_sq, hp_im2, s3);

            out_row[t] = prices[t] - hp;

            hp_im2 = hp_im1;
            hp_im1 = hp;
        }
    }
}















extern "C" __global__
void decycler_batch_warp_scan_f32(const float* __restrict__ prices,
                                 const int*   __restrict__ periods,
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


    float s0_prev = 0.0f;
    float s1_prev = 0.0f;
    if (lane == 0) {
        s1_prev = prices[first_valid];
        s0_prev = prices[first_valid + 1];
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);


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



        float p00 = valid ? m00 : 1.0f;
        float p01 = valid ? m01 : 0.0f;
        float p10 = valid ? m10 : 0.0f;
        float p11 = valid ? m11 : 1.0f;
        float v0  = valid ? u   : 0.0f;
        float v1  = 0.0f;


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


extern "C" __global__
void decycler_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                        const int*   __restrict__ first_valids,
                                        int period,
                                        float c,
                                        float two_1m,
                                        float neg_oma_sq,
                                        int num_series,
                                        int series_len,
                                        float* __restrict__ out_tm)
{
    (void)period;
    const int stride = num_series;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = tid; s < num_series; s += blockDim.x * gridDim.x) {
        if (series_len <= 0) continue;
        const int fv = first_valids[s];
        if (fv < 0 || fv >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
            continue;
        }


        const int warm = min(series_len, fv + 2);
        for (int t = 0; t < warm; ++t) {
            out_tm[(size_t)t * (size_t)stride + s] = CUDART_NAN_F;
        }
        if (fv + 1 >= series_len) continue;


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

