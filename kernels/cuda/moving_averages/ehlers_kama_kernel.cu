// CUDA kernels for Ehlers Kaufman Adaptive Moving Average (KAMA).
//
// Mirrors the scalar implementation: each parameter combination is handled by
// a single thread block that walks the time series sequentially while
// maintaining the rolling volatility sum used to derive the adaptive smoothing
// factor. A second kernel covers the many-series × one-parameter path operating
// on time-major input.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void ehlers_kama_batch_f32(const float* __restrict__ prices,
                           const int* __restrict__ periods,
                           int first_valid,
                           int series_len,
                           int n_combos,
                           float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    if (series_len <= 0) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0) {
        return;
    }

    int first = first_valid;
    if (first < 0) {
        first = 0;
    }
    if (first >= series_len) {
        return;
    }

    const int warm = first + period - 1;
    const int base = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int idx = 0; idx < warm_clamped; ++idx) {
        out[base + idx] = nan_f;
    }

    if (warm >= series_len) {
        return;
    }

    const int start = warm;
    int delta_start = (start >= period) ? (start - period + 1) : (first + 1);
    if (delta_start < first + 1) {
        delta_start = first + 1;
    }

    float delta_sum = 0.0f;
    for (int k = delta_start; k <= start; ++k) {
        if (k > first) {
            delta_sum += fabsf(prices[k] - prices[k - 1]);
        }
    }

    float prev;
    if (start > 0) {
        prev = prices[start - 1];
    } else {
        prev = prices[start];
    }

    const float current = prices[start];
    const int dir_idx = start - (period - 1);
    float direction = 0.0f;
    if (dir_idx >= 0) {
        direction = fabsf(current - prices[dir_idx]);
    }

    float ef = 0.0f;
    if (delta_sum > 0.0f) {
        ef = direction / delta_sum;
        if (ef > 1.0f) {
            ef = 1.0f;
        }
    }

    float sc = (0.6667f * ef) + 0.0645f;
    sc *= sc;
    prev = fmaf(sc, current - prev, prev);
    out[base + start] = prev;

    for (int i = start + 1; i < series_len; ++i) {
        const int drop_idx = i - period;
        if (drop_idx > first) {
            const float drop = fabsf(prices[drop_idx] - prices[drop_idx - 1]);
            delta_sum -= drop;
            if (delta_sum < 0.0f) {
                delta_sum = 0.0f;
            }
        }

        const float newest = prices[i];
        const float newest_diff = fabsf(newest - prices[i - 1]);
        delta_sum += newest_diff;

        const int anchor_idx = i - (period - 1);
        float dir = 0.0f;
        if (anchor_idx >= 0) {
            dir = fabsf(newest - prices[anchor_idx]);
        }

        float ef_i = 0.0f;
        if (delta_sum > 0.0f) {
            ef_i = dir / delta_sum;
            if (ef_i > 1.0f) {
                ef_i = 1.0f;
            }
        }

        float sc_i = (0.6667f * ef_i) + 0.0645f;
        sc_i *= sc_i;
        prev = fmaf(sc_i, newest - prev, prev);
        out[base + i] = prev;
    }
}

extern "C" __global__
void ehlers_kama_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                            int period,
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

    int first = first_valids[series_idx];
    if (first < 0) {
        first = 0;
    }
    if (first >= series_len) {
        return;
    }

    const int stride = num_series;
    const int warm = first + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[t * stride + series_idx] = nan_f;
    }

    if (warm >= series_len) {
        return;
    }

    const int start = warm;
    int delta_start = (start >= period) ? (start - period + 1) : (first + 1);
    if (delta_start < first + 1) {
        delta_start = first + 1;
    }

    float delta_sum = 0.0f;
    for (int k = delta_start; k <= start; ++k) {
        if (k > first) {
            const int idx_cur = k * stride + series_idx;
            const int idx_prev = (k - 1) * stride + series_idx;
            delta_sum += fabsf(prices_tm[idx_cur] - prices_tm[idx_prev]);
        }
    }

    float prev;
    if (start > 0) {
        prev = prices_tm[(start - 1) * stride + series_idx];
    } else {
        prev = prices_tm[start * stride + series_idx];
    }
    const float current = prices_tm[start * stride + series_idx];
    const int dir_idx = start - (period - 1);
    float direction = 0.0f;
    if (dir_idx >= 0) {
        const int anchor = dir_idx * stride + series_idx;
        direction = fabsf(current - prices_tm[anchor]);
    }

    float ef = 0.0f;
    if (delta_sum > 0.0f) {
        ef = direction / delta_sum;
        if (ef > 1.0f) {
            ef = 1.0f;
        }
    }

    float sc = (0.6667f * ef) + 0.0645f;
    sc *= sc;
    prev = fmaf(sc, current - prev, prev);
    out_tm[start * stride + series_idx] = prev;

    for (int t = start + 1; t < series_len; ++t) {
        const int drop_idx = t - period;
        if (drop_idx > first) {
            const int idx_drop = drop_idx * stride + series_idx;
            const int idx_drop_prev = (drop_idx - 1) * stride + series_idx;
            const float drop = fabsf(prices_tm[idx_drop] - prices_tm[idx_drop_prev]);
            delta_sum -= drop;
            if (delta_sum < 0.0f) {
                delta_sum = 0.0f;
            }
        }

        const int cur_idx = t * stride + series_idx;
        const int prev_idx = (t - 1) * stride + series_idx;
        const float newest = prices_tm[cur_idx];
        const float newest_diff = fabsf(newest - prices_tm[prev_idx]);
        delta_sum += newest_diff;

        const int anchor_idx = t - (period - 1);
        float dir = 0.0f;
        if (anchor_idx >= 0) {
            const int anchor = anchor_idx * stride + series_idx;
            dir = fabsf(newest - prices_tm[anchor]);
        }

        float ef_t = 0.0f;
        if (delta_sum > 0.0f) {
            ef_t = dir / delta_sum;
            if (ef_t > 1.0f) {
                ef_t = 1.0f;
            }
        }

        float sc_t = (0.6667f * ef_t) + 0.0645f;
        sc_t *= sc_t;
        prev = fmaf(sc_t, newest - prev, prev);
        out_tm[cur_idx] = prev;
    }
}
