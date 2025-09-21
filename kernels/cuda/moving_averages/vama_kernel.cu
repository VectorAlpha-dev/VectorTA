#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void vama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ base_periods,
                    const int* __restrict__ vol_periods,
                    const float* __restrict__ alphas,
                    const float* __restrict__ betas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ ema_buf,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    if (series_len <= 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int base_period = base_periods[combo];
    const int vol_period = vol_periods[combo];
    if (base_period <= 0 || vol_period <= 0) {
        return;
    }

    const float alpha = alphas[combo];
    const float beta = betas[combo];
    const int base_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        ema_buf[base_offset + idx] = NAN;
        out[base_offset + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const float first_price = prices[first_valid];
    float mean = first_price;
    int valid_count = 1;
    ema_buf[base_offset + first_valid] = mean;

    int warm_base_end = first_valid + base_period;
    if (warm_base_end > series_len) {
        warm_base_end = series_len;
    }

    for (int i = first_valid + 1; i < warm_base_end; ++i) {
        const float price = prices[i];
        if (isfinite(price)) {
            const float prev_total = mean * static_cast<float>(valid_count);
            ++valid_count;
            mean = (prev_total + price) / static_cast<float>(valid_count);
        }
        ema_buf[base_offset + i] = mean;
    }

    float prev = mean;
    for (int i = warm_base_end; i < series_len; ++i) {
        const float price = prices[i];
        if (isfinite(price)) {
            prev = beta * prev + alpha * price;
        }
        ema_buf[base_offset + i] = prev;
    }

    const int max_period = (base_period > vol_period) ? base_period : vol_period;
    const int warm = first_valid + max_period - 1;
    if (warm >= series_len) {
        return;
    }

    for (int i = warm; i < series_len; ++i) {
        const float mid = ema_buf[base_offset + i];
        if (!isfinite(mid)) {
            out[base_offset + i] = NAN;
            continue;
        }

        int window_len = vol_period;
        const int available = i + 1 - first_valid;
        if (available < window_len) {
            window_len = available;
        }
        if (window_len <= 0) {
            out[base_offset + i] = NAN;
            continue;
        }
        const int start = i + 1 - window_len;

        float vol_up = -CUDART_INF_F;
        float vol_down = CUDART_INF_F;
        for (int j = start; j <= i; ++j) {
            const float ema_j = ema_buf[base_offset + j];
            if (!isfinite(ema_j)) {
                continue;
            }
            const float price_j = prices[j];
            if (!isfinite(price_j)) {
                continue;
            }
            const float dev = price_j - ema_j;
            if (dev > vol_up) {
                vol_up = dev;
            }
            if (dev < vol_down) {
                vol_down = dev;
            }
        }

        if (!isfinite(vol_up) || !isfinite(vol_down)) {
            out[base_offset + i] = mid;
        } else {
            out[base_offset + i] = mid + 0.5f * (vol_up + vol_down);
        }
    }
}

extern "C" __global__
void vama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int base_period,
                                    int vol_period,
                                    float alpha,
                                    float beta,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ ema_tm,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        const int offset = t * num_series + series_idx;
        ema_tm[offset] = NAN;
        out_tm[offset] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (base_period <= 0 || vol_period <= 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int first_valid = first_valids[series_idx];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int stride = num_series;
    const int first_idx = first_valid * stride + series_idx;
    float first_price = prices_tm[first_idx];
    if (!isfinite(first_price)) {
        for (int t = first_valid + 1; t < series_len; ++t) {
            const float candidate = prices_tm[t * stride + series_idx];
            if (isfinite(candidate)) {
                first_price = candidate;
                break;
            }
        }
    }
    if (!isfinite(first_price)) {
        return;
    }

    float mean = first_price;
    int valid_count = 1;
    ema_tm[first_idx] = mean;

    int warm_base_end = first_valid + base_period;
    if (warm_base_end > series_len) {
        warm_base_end = series_len;
    }

    for (int t = first_valid + 1; t < warm_base_end; ++t) {
        const float price = prices_tm[t * stride + series_idx];
        if (isfinite(price)) {
            const float prev_total = mean * static_cast<float>(valid_count);
            ++valid_count;
            mean = (prev_total + price) / static_cast<float>(valid_count);
        }
        ema_tm[t * stride + series_idx] = mean;
    }

    float prev = mean;
    for (int t = warm_base_end; t < series_len; ++t) {
        const float price = prices_tm[t * stride + series_idx];
        if (isfinite(price)) {
            prev = beta * prev + alpha * price;
        }
        ema_tm[t * stride + series_idx] = prev;
    }

    const int max_period = (base_period > vol_period) ? base_period : vol_period;
    const int warm = first_valid + max_period - 1;
    if (warm >= series_len) {
        return;
    }

    for (int t = warm; t < series_len; ++t) {
        const float mid = ema_tm[t * stride + series_idx];
        if (!isfinite(mid)) {
            continue;
        }

        int window_len = vol_period;
        const int available = t + 1 - first_valid;
        if (available < window_len) {
            window_len = available;
        }
        if (window_len <= 0) {
            continue;
        }

        const int start = t + 1 - window_len;
        float vol_up = -CUDART_INF_F;
        float vol_down = CUDART_INF_F;
        for (int k = start; k <= t; ++k) {
            const float ema_val = ema_tm[k * stride + series_idx];
            if (!isfinite(ema_val)) {
                continue;
            }
            const float price = prices_tm[k * stride + series_idx];
            if (!isfinite(price)) {
                continue;
            }
            const float dev = price - ema_val;
            if (dev > vol_up) {
                vol_up = dev;
            }
            if (dev < vol_down) {
                vol_down = dev;
            }
        }

        const int out_idx = t * stride + series_idx;
        if (!isfinite(vol_up) || !isfinite(vol_down)) {
            out_tm[out_idx] = mid;
        } else {
            out_tm[out_idx] = mid + 0.5f * (vol_up + vol_down);
        }
    }
}
