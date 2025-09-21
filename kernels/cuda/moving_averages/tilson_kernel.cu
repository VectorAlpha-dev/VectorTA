#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void tilson_batch_f32(const float* __restrict__ prices,
                      const int* __restrict__ periods,
                      const float* __restrict__ ks,
                      const float* __restrict__ c1s,
                      const float* __restrict__ c2s,
                      const float* __restrict__ c3s,
                      const float* __restrict__ c4s,
                      const int* __restrict__ lookbacks,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int lookback = lookbacks[combo];
    if (period <= 0 || lookback < 0 || series_len <= 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int warm_index = first_valid + lookback;
    if (warm_index >= series_len) {
        return;
    }

    const float k = ks[combo];
    const float one_minus_k = 1.0f - k;
    const float c1 = c1s[combo];
    const float c2 = c2s[combo];
    const float c3 = c3s[combo];
    const float c4 = c4s[combo];

    const int base = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    int today = 0;
    float temp_real = 0.0f;

    if (first_valid + period > series_len) {
        return;
    }

    for (int i = 0; i < period; ++i) {
        const int idx = first_valid + today + i;
        if (idx >= series_len) {
            return;
        }
        temp_real += prices[idx];
    }
    float e1 = temp_real / static_cast<float>(period);
    today += period;

    temp_real = e1;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices[idx];
        e1 = k * price + one_minus_k * e1;
        temp_real += e1;
        today += 1;
    }
    float e2 = temp_real / static_cast<float>(period);

    temp_real = e2;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices[idx];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        temp_real += e2;
        today += 1;
    }
    float e3 = temp_real / static_cast<float>(period);

    temp_real = e3;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices[idx];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        temp_real += e3;
        today += 1;
    }
    float e4 = temp_real / static_cast<float>(period);

    temp_real = e4;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices[idx];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        temp_real += e4;
        today += 1;
    }
    float e5 = temp_real / static_cast<float>(period);

    temp_real = e5;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices[idx];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        temp_real += e5;
        today += 1;
    }
    float e6 = temp_real / static_cast<float>(period);

    int out_idx = warm_index;
    const int end_idx = series_len - 1;
    if (out_idx < series_len) {
        out[base + out_idx] =
            c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }
    out_idx += 1;

    while ((first_valid + today) <= end_idx) {
        const int price_idx = first_valid + today;
        const float price = prices[price_idx];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        e6 = k * e5 + one_minus_k * e6;

        if (out_idx < series_len) {
            out[base + out_idx] =
                c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
        }
        today += 1;
        out_idx += 1;
    }
}

extern "C" __global__
void tilson_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                      const int* __restrict__ first_valids,
                                      int period,
                                      float k,
                                      float c1,
                                      float c2,
                                      float c3,
                                      float c4,
                                      int lookback,
                                      int num_series,
                                      int series_len,
                                      float* __restrict__ out_tm) {
    const int series = blockIdx.y;
    if (series >= num_series) {
        return;
    }
    if (period <= 0 || lookback < 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        const int idx = t * stride + series;
        out_tm[idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const float one_minus_k = 1.0f - k;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }
    const int warm_index = first_valid + lookback;
    if (warm_index >= series_len) {
        return;
    }

    int today = 0;
    float temp_real = 0.0f;

    if (first_valid + period > series_len) {
        return;
    }

    for (int i = 0; i < period; ++i) {
        const int idx = first_valid + today + i;
        if (idx >= series_len) {
            return;
        }
        temp_real += prices_tm[idx * stride + series];
    }
    float e1 = temp_real / static_cast<float>(period);
    today += period;

    temp_real = e1;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices_tm[idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        temp_real += e1;
        today += 1;
    }
    float e2 = temp_real / static_cast<float>(period);

    temp_real = e2;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices_tm[idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        temp_real += e2;
        today += 1;
    }
    float e3 = temp_real / static_cast<float>(period);

    temp_real = e3;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices_tm[idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        temp_real += e3;
        today += 1;
    }
    float e4 = temp_real / static_cast<float>(period);

    temp_real = e4;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices_tm[idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        temp_real += e4;
        today += 1;
    }
    float e5 = temp_real / static_cast<float>(period);

    temp_real = e5;
    for (int i = 1; i < period; ++i) {
        const int idx = first_valid + today;
        if (idx >= series_len) {
            return;
        }
        const float price = prices_tm[idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        temp_real += e5;
        today += 1;
    }
    float e6 = temp_real / static_cast<float>(period);

    int out_idx = warm_index;
    const int end_idx = series_len - 1;
    if (out_idx < series_len) {
        out_tm[out_idx * stride + series] =
            c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }
    out_idx += 1;

    while ((first_valid + today) <= end_idx) {
        const int price_idx = first_valid + today;
        const float price = prices_tm[price_idx * stride + series];
        e1 = k * price + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        e6 = k * e5 + one_minus_k * e6;

        if (out_idx < series_len) {
            out_tm[out_idx * stride + series] =
                c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
        }
        today += 1;
        out_idx += 1;
    }
}
