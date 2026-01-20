








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>



static __forceinline__ __device__
float sc_from_er_poly(float er, float a, float b, float c) {
    float er2 = er * er;
    return fmaf(a, er2, fmaf(b, er, c));
}

extern "C" __global__
void maaq_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const float* __restrict__ fast_scs,
                    const float* __restrict__ slow_scs,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0 || max_period <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const float fast_sc = fast_scs[combo];
    const float slow_sc = slow_scs[combo];

    int first = first_valid;
    if (first < 0) first = 0;
    if (first >= series_len) return;

    const int warm = first + period - 1;
    if (warm >= series_len) return;

    extern __shared__ float diffs[];

    const int base_out = combo * series_len;
    const float nan_f = CUDART_NAN_F;
    const float anchor = prices[first];
    const float EPS = 1.0e-12f;


    if (threadIdx.x == 0) {

        for (int idx = 0; idx < first; ++idx) {
            out[base_out + idx] = nan_f;
        }

        for (int idx = first; idx <= warm; ++idx) {
            out[base_out + idx] = prices[idx];
        }
        if (warm + 1 >= series_len) return;


        float vol_sum = 0.0f;
        for (int j = 1; j < period; ++j) {
            const int cur = first + j;
            const float diff = fabsf(prices[cur] - prices[cur - 1]);
            diffs[j] = diff;
            vol_sum += diff;
        }


        const int i0 = warm + 1;
        float prev = prices[warm];
        float prev_input = prices[warm];

        const float newest = prices[i0];
        const float newest_diff = fabsf(newest - prev_input);
        diffs[0] = newest_diff;
        vol_sum += newest_diff;
        prev_input = newest;


        const float a = fast_sc * fast_sc;
        const float b = 2.0f * fast_sc * slow_sc;
        const float c = slow_sc * slow_sc;

        float er = 0.0f;
        if (vol_sum > EPS) {
            er = fabsf(newest - anchor) / vol_sum;
        }
        float sc = sc_from_er_poly(er, a, b, c);
        prev = fmaf(sc, newest - prev, prev);
        out[base_out + i0] = prev;


        int head = 1;
        for (int t = i0 + 1; t < series_len; ++t) {

            vol_sum -= diffs[head];

            const float cur_price = prices[t];
            const float nd = fabsf(cur_price - prev_input);
            diffs[head] = nd;
            vol_sum += nd;
            prev_input = cur_price;

            ++head; if (head == period) head = 0;

            float er_t = 0.0f;
            if (vol_sum > EPS) {
                er_t = fabsf(cur_price - prices[t - period]) / vol_sum;
            }
            const float sc_t = sc_from_er_poly(er_t, a, b, c);
            prev = fmaf(sc_t, cur_price - prev, prev);
            out[base_out + t] = prev;
        }
    }
}

extern "C" __global__
void maaq_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     float fast_sc,
                                     float slow_sc,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (period <= 0 || series_len <= 0) return;

    extern __shared__ float diffs[];

    int first = first_valids[series_idx];
    if (first < 0) first = 0;
    if (first >= series_len) return;

    const int warm = first + period - 1;
    if (warm >= series_len) return;

    const int stride = num_series;
    const float nan_f = CUDART_NAN_F;
    const float EPS = 1.0e-12f;

    if (threadIdx.x == 0) {

        for (int t = 0; t < warm; ++t) {
            out_tm[t * stride + series_idx] = nan_f;
        }

        const int warm_idx = warm * stride + series_idx;
        out_tm[warm_idx] = prices_tm[warm_idx];

        if (warm + 1 >= series_len) return;


        float vol_sum = 0.0f;
        for (int j = 1; j < period; ++j) {
            const int cur = first + j;
            const int idx = cur * stride + series_idx;
            const int prev_idx = (cur - 1) * stride + series_idx;
            const float diff = fabsf(prices_tm[idx] - prices_tm[prev_idx]);
            diffs[j] = diff;
            vol_sum += diff;
        }

        const int i0 = warm + 1;
        const int prev_idx = warm * stride + series_idx;
        float prev       = prices_tm[prev_idx];
        float prev_input = prices_tm[prev_idx];

        const int cur_idx = i0 * stride + series_idx;
        const float newest = prices_tm[cur_idx];
        const float newest_diff = fabsf(newest - prev_input);
        diffs[0] = newest_diff;
        vol_sum += newest_diff;
        prev_input = newest;

        const float anchor = prices_tm[first * stride + series_idx];


        const float a = fast_sc * fast_sc;
        const float b = 2.0f * fast_sc * slow_sc;
        const float c = slow_sc * slow_sc;

        float er = 0.0f;
        if (vol_sum > EPS) {
            er = fabsf(newest - anchor) / vol_sum;
        }
        float sc = sc_from_er_poly(er, a, b, c);
        prev = fmaf(sc, newest - prev, prev);
        out_tm[cur_idx] = prev;

        int head = 1;
        for (int t = i0 + 1; t < series_len; ++t) {
            vol_sum -= diffs[head];

            const int idx_curr = t * stride + series_idx;
            const float cur_price = prices_tm[idx_curr];
            const float nd = fabsf(cur_price - prev_input);
            diffs[head] = nd;
            vol_sum += nd;
            prev_input = cur_price;

            ++head; if (head == period) head = 0;

            float er_t = 0.0f;
            if (vol_sum > EPS) {
                const int idx_old = (t - period) * stride + series_idx;
                er_t = fabsf(cur_price - prices_tm[idx_old]) / vol_sum;
            }
            const float sc_t = sc_from_er_poly(er_t, a, b, c);
            prev = fmaf(sc_t, cur_price - prev, prev);
            out_tm[idx_curr] = prev;
        }
    }
}
