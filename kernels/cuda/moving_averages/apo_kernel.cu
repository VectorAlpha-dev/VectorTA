









#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void apo_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ short_periods,
                   const float* __restrict__ short_alphas,
                   const int*   __restrict__ long_periods,
                   const float* __restrict__ long_alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int  sp    = short_periods[combo];
    const int  lp    = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const float a_s  = short_alphas[combo];
    const float a_l  = long_alphas[combo];
    const float oma_s= 1.0f - a_s;
    const float oma_l= 1.0f - a_l;

    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);


    for (int i = threadIdx.x; i < first_valid; i += blockDim.x) {
        out[base + static_cast<size_t>(i)] = NAN;
    }


    if (blockDim.x < 32) {

        if (threadIdx.x != 0) return;


        float se = prices[first_valid];
        float le = se;
        out[base + static_cast<size_t>(first_valid)] = 0.0f;


        for (int i = first_valid + 1; i < series_len; ++i) {
            const float x = prices[i];

            se = a_s * x + oma_s * se;
            le = a_l * x + oma_l * le;
            out[base + static_cast<size_t>(i)] = se - le;
        }
        return;
    }



    if (threadIdx.x >= 32) return;

    const unsigned lane = static_cast<unsigned>(threadIdx.x);
    const unsigned mask = 0xffffffffu;

    float se_prev = prices[first_valid];
    float le_prev = se_prev;
    if (lane == 0) {
        out[base + static_cast<size_t>(first_valid)] = 0.0f;
    }

    for (int t0 = first_valid + 1; t0 < series_len; t0 += 32) {
        const int t = t0 + static_cast<int>(lane);

        float A_s = 1.0f;
        float B_s = 0.0f;
        float A_l = 1.0f;
        float B_l = 0.0f;
        if (t < series_len) {
            const float x = prices[t];
            A_s = oma_s;
            B_s = a_s * x;
            A_l = oma_l;
            B_l = a_l * x;
        }




        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_s_prev = __shfl_up_sync(mask, A_s, offset);
            const float B_s_prev = __shfl_up_sync(mask, B_s, offset);
            const float A_l_prev = __shfl_up_sync(mask, A_l, offset);
            const float B_l_prev = __shfl_up_sync(mask, B_l, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const float A_s_cur = A_s;
                const float B_s_cur = B_s;
                const float A_l_cur = A_l;
                const float B_l_cur = B_l;
                A_s = A_s_cur * A_s_prev;
                B_s = __fmaf_rn(A_s_cur, B_s_prev, B_s_cur);
                A_l = A_l_cur * A_l_prev;
                B_l = __fmaf_rn(A_l_cur, B_l_prev, B_l_cur);
            }
        }

        const float se = __fmaf_rn(A_s, se_prev, B_s);
        const float le = __fmaf_rn(A_l, le_prev, B_l);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = se - le;
        }


        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        se_prev = __shfl_sync(mask, se, last_lane);
        le_prev = __shfl_sync(mask, le, last_lane);
    }
}


extern "C" __global__
void apo_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int short_period,
                                   float short_alpha,
                                   int long_period,
                                   float long_alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || series_len <= 0) return;
    if (short_period <= 0 || long_period <= 0 || short_period >= long_period) return;

    const int stride = num_series;
    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    const float a_s   = short_alpha;
    const float a_l   = long_alpha;
    const float oma_s = 1.0f - a_s;
    const float oma_l = 1.0f - a_l;


    for (int t = threadIdx.x; t < fv; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    if (threadIdx.x != 0) return;


    float se = prices_tm[fv * stride + series_idx];
    float le = se;
    out_tm[fv * stride + series_idx] = 0.0f;

    for (int t = fv + 1; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        se = a_s * x + oma_s * se;
        le = a_l * x + oma_l * le;
        out_tm[t * stride + series_idx] = se - le;
    }
}

