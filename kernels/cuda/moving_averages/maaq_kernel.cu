// CUDA kernels for Moving Average Adaptive Q (MAAQ).
//
// The batch kernel assigns one block per parameter combination and walks the
// time series sequentially while keeping the adaptive smoothing factors in
// registers. A second kernel covers the many-series × one-parameter case with
// time-major input layout.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

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
    if (combo >= n_combos) {
        return;
    }

    if (series_len <= 0 || max_period <= 0) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    const float fast_sc = fast_scs[combo];
    const float slow_sc = slow_scs[combo];

    int first = first_valid;
    if (first < 0) {
        first = 0;
    }
    if (first >= series_len) {
        return;
    }

    const int warm = first + period - 1;
    if (warm >= series_len) {
        return;
    }

    extern __shared__ float shared_diffs[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        shared_diffs[i] = 0.0f;
    }
    __syncthreads();

    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    for (int idx = 0; idx < first; ++idx) {
        out[base_out + idx] = nan_f;
    }

    for (int idx = first; idx <= warm; ++idx) {
        out[base_out + idx] = prices[idx];
    }

    if (warm + 1 >= series_len) {
        return;
    }

    float vol_sum = 0.0f;
    for (int j = 1; j < period; ++j) {
        const int cur = first + j;
        const float diff = fabsf(prices[cur] - prices[cur - 1]);
        shared_diffs[j] = diff;
        vol_sum += diff;
    }

    const int i0 = warm + 1;
    float prev = prices[warm];
    const float newest = prices[i0];
    const float newest_diff = fabsf(newest - prices[i0 - 1]);
    shared_diffs[0] = newest_diff;
    vol_sum += newest_diff;

    const float anchor = prices[first];
    const float eps = 1.0e-12f;
    float er = 0.0f;
    if (vol_sum > eps) {
        er = fabsf(newest - anchor) / vol_sum;
    }
    float sc = fast_sc * er + slow_sc;
    sc *= sc;
    prev = fmaf(sc, newest - prev, prev);
    out[base_out + i0] = prev;

    int head = 1;
    for (int t = i0 + 1; t < series_len; ++t) {
        vol_sum -= shared_diffs[head];
        const float nd = fabsf(prices[t] - prices[t - 1]);
        shared_diffs[head] = nd;
        vol_sum += nd;
        ++head;
        if (head == period) {
            head = 0;
        }

        float er_t = 0.0f;
        if (vol_sum > eps) {
            er_t = fabsf(prices[t] - prices[t - period]) / vol_sum;
        }
        float sc_t = fast_sc * er_t + slow_sc;
        sc_t *= sc_t;
        prev = fmaf(sc_t, prices[t] - prev, prev);
        out[base_out + t] = prev;
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
    if (series_idx >= num_series) {
        return;
    }

    if (period <= 0 || series_len <= 0) {
        return;
    }

    extern __shared__ float diffs[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        diffs[i] = 0.0f;
    }
    __syncthreads();

    int first = first_valids[series_idx];
    if (first < 0) {
        first = 0;
    }
    if (first >= series_len) {
        return;
    }

    const int warm = first + period - 1;
    if (warm >= series_len) {
        return;
    }

    const int stride = num_series;
    const float nan_f = __int_as_float(0x7fffffff);

    for (int t = 0; t < warm; ++t) {
        out_tm[t * stride + series_idx] = nan_f;
    }

    const int warm_idx = warm * stride + series_idx;
    out_tm[warm_idx] = prices_tm[warm_idx];

    if (warm + 1 >= series_len) {
        return;
    }

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
    float prev = prices_tm[prev_idx];
    const int cur_idx = i0 * stride + series_idx;
    const float newest = prices_tm[cur_idx];
    const float newest_diff = fabsf(newest - prices_tm[prev_idx]);
    diffs[0] = newest_diff;
    vol_sum += newest_diff;

    const float anchor = prices_tm[first * stride + series_idx];
    const float eps = 1.0e-12f;
    float er = 0.0f;
    if (vol_sum > eps) {
        er = fabsf(newest - anchor) / vol_sum;
    }
    float sc = fast_sc * er + slow_sc;
    sc *= sc;
    prev = fmaf(sc, newest - prev, prev);
    out_tm[cur_idx] = prev;

    int head = 1;
    for (int t = i0 + 1; t < series_len; ++t) {
        vol_sum -= diffs[head];
        const int idx_curr = t * stride + series_idx;
        const int idx_prev = (t - 1) * stride + series_idx;
        const float nd = fabsf(prices_tm[idx_curr] - prices_tm[idx_prev]);
        diffs[head] = nd;
        vol_sum += nd;
        ++head;
        if (head == period) {
            head = 0;
        }

        float er_t = 0.0f;
        if (vol_sum > eps) {
            const int idx_old = (t - period) * stride + series_idx;
            er_t = fabsf(prices_tm[idx_curr] - prices_tm[idx_old]) / vol_sum;
        }
        float sc_t = fast_sc * er_t + slow_sc;
        sc_t *= sc_t;
        prev = fmaf(sc_t, prices_tm[idx_curr] - prev, prev);
        out_tm[idx_curr] = prev;
    }
}
