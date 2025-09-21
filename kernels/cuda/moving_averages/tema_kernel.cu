// CUDA kernels for TEMA (Triple Exponential Moving Average).
//
// Each parameter combination (period) is assigned to a block. Because TEMA is
// built from three cascaded EMAs, the recurrence is inherently sequential in
// time. The kernels therefore run the series sequentially per combo/series
// using thread 0 while keeping data resident on device so large parameter sweeps
// still avoid repeated host transfers.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void tema_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
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

    const int period = periods[combo];
    if (period <= 0 || series_len <= 0) {
        return;
    }

    const int base_out = combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        out[base_out + i] = NAN;
    }

    if (first_valid >= series_len) {
        return;
    }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const float alpha1 = 1.0f - alpha;
    const int lookback = (period - 1) * 3;
    const int ema2_start = first_valid + (period - 1);
    const int ema3_start = first_valid + 2 * (period - 1);
    const int warm = first_valid + lookback;

    float ema1 = prices[first_valid];
    float ema2 = 0.0f;
    float ema3 = 0.0f;
    bool ema2_init = false;
    bool ema3_init = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices[i];
        ema1 = ema1 * alpha1 + price * alpha;

        if (!ema2_init) {
            if (i >= ema2_start) {
                ema2 = ema1;
                ema2_init = true;
            }
        }
        if (ema2_init) {
            ema2 = ema2 * alpha1 + ema1 * alpha;
        }

        if (!ema3_init) {
            if (i >= ema3_start) {
                ema3 = ema2;
                ema3_init = true;
            }
        }
        if (ema3_init) {
            ema3 = ema3 * alpha1 + ema2 * alpha;
        }

        if (i >= warm && i < series_len) {
            out[base_out + i] = 3.0f * ema1 - 3.0f * ema2 + ema3;
        }
    }
}

extern "C" __global__
void tema_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
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

    if (period <= 0 || series_len <= 0) {
        return;
    }

    for (int i = 0; i < series_len; ++i) {
        const int out_idx = i * num_series + series_idx;
        out_tm[out_idx] = NAN;
    }

    int first_valid = first_valids[series_idx];
    if (first_valid < 0) {
        first_valid = 0;
    }
    if (first_valid >= series_len) {
        return;
    }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const float alpha1 = 1.0f - alpha;
    const int lookback = (period - 1) * 3;
    const int ema2_start = first_valid + (period - 1);
    const int ema3_start = first_valid + 2 * (period - 1);
    const int warm = first_valid + lookback;

    const int first_idx = first_valid * num_series + series_idx;
    float ema1 = prices_tm[first_idx];
    float ema2 = 0.0f;
    float ema3 = 0.0f;
    bool ema2_init = false;
    bool ema3_init = false;

    for (int t = first_valid; t < series_len; ++t) {
        const int idx = t * num_series + series_idx;
        const float price = prices_tm[idx];

        ema1 = ema1 * alpha1 + price * alpha;

        if (!ema2_init) {
            if (t >= ema2_start) {
                ema2 = ema1;
                ema2_init = true;
            }
        }
        if (ema2_init) {
            ema2 = ema2 * alpha1 + ema1 * alpha;
        }

        if (!ema3_init) {
            if (t >= ema3_start) {
                ema3 = ema2;
                ema3_init = true;
            }
        }
        if (ema3_init) {
            ema3 = ema3 * alpha1 + ema2 * alpha;
        }

        if (t >= warm) {
            out_tm[idx] = 3.0f * ema1 - 3.0f * ema2 + ema3;
        }
    }
}
