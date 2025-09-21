// CUDA kernel for UMA (Ultimate Moving Average) computations.
//
// Each parameter combination is evaluated within a dedicated block using a
// sequential walk identical to the scalar Rust implementation. Sliding sums
// provide the max-length mean/std window while the adaptive power weights and
// optional smoothing mirror the CPU path. This kernel operates in FP32 to
// minimise bandwidth and aligns with the CUDA scaffolding used by ALMA/DMA.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

static __device__ __forceinline__ bool is_nan(float v) {
    return isnan(v);
}

static __device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

static __device__ __forceinline__ float compute_rsi(
    const float* __restrict__ data,
    int start,
    int end,
    int period
) {
    if (period <= 1) {
        return 50.0f;
    }
    int len = end - start;
    if (len <= period) {
        return 50.0f;
    }

    float inv_period = 1.0f / static_cast<float>(period);
    float beta = 1.0f - inv_period;
    float avg_gain = 0.0f;
    float avg_loss = 0.0f;

    int warm_end = start + period;
    if (warm_end >= end) {
        return 50.0f;
    }

    for (int idx = start + 1; idx <= warm_end; ++idx) {
        float cur = data[idx];
        float prev = data[idx - 1];
        if (!isfinite(cur) || !isfinite(prev)) {
            return 50.0f;
        }
        float delta = cur - prev;
        if (delta > 0.0f) {
            avg_gain += delta;
        } else if (delta < 0.0f) {
            avg_loss -= delta;
        }
    }

    avg_gain *= inv_period;
    avg_loss *= inv_period;
    float denom = avg_gain + avg_loss;
    float rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;

    for (int idx = warm_end + 1; idx < end; ++idx) {
        float cur = data[idx];
        float prev = data[idx - 1];
        if (!isfinite(cur) || !isfinite(prev)) {
            return 50.0f;
        }
        float delta = cur - prev;
        float gain = delta > 0.0f ? delta : 0.0f;
        float loss = delta < 0.0f ? -delta : 0.0f;
        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;
        denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;
    }
    return rsi;
}

static __device__ __forceinline__ float load_tm(
    const float* __restrict__ data,
    int num_series,
    int series_idx,
    int t
) {
    return data[t * num_series + series_idx];
}

static __device__ __forceinline__ float compute_rsi_tm(
    const float* __restrict__ data_tm,
    int num_series,
    int series_idx,
    int start,
    int end,
    int period
) {
    if (period <= 1) {
        return 50.0f;
    }
    int len = end - start;
    if (len <= period) {
        return 50.0f;
    }

    float inv_period = 1.0f / static_cast<float>(period);
    float beta = 1.0f - inv_period;
    float avg_gain = 0.0f;
    float avg_loss = 0.0f;

    int warm_end = start + period;
    if (warm_end >= end) {
        return 50.0f;
    }

    for (int idx = start + 1; idx <= warm_end; ++idx) {
        float cur = load_tm(data_tm, num_series, series_idx, idx);
        float prev = load_tm(data_tm, num_series, series_idx, idx - 1);
        if (!isfinite(cur) || !isfinite(prev)) {
            return 50.0f;
        }
        float delta = cur - prev;
        if (delta > 0.0f) {
            avg_gain += delta;
        } else if (delta < 0.0f) {
            avg_loss -= delta;
        }
    }

    avg_gain *= inv_period;
    avg_loss *= inv_period;
    float denom = avg_gain + avg_loss;
    float rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;

    for (int idx = warm_end + 1; idx < end; ++idx) {
        float cur = load_tm(data_tm, num_series, series_idx, idx);
        float prev = load_tm(data_tm, num_series, series_idx, idx - 1);
        if (!isfinite(cur) || !isfinite(prev)) {
            return 50.0f;
        }
        float delta = cur - prev;
        float gain = delta > 0.0f ? delta : 0.0f;
        float loss = delta < 0.0f ? -delta : 0.0f;
        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;
        denom = avg_gain + avg_loss;
        rsi = (denom == 0.0f) ? 50.0f : 100.0f * avg_gain / denom;
    }
    return rsi;
}

extern "C" __global__ void uma_batch_f32(
    const float* __restrict__ prices,
    const float* __restrict__ volumes,
    int has_volume,
    const float* __restrict__ accelerators,
    const int* __restrict__ min_lengths,
    const int* __restrict__ max_lengths,
    const int* __restrict__ smooth_lengths,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ raw_out,
    float* __restrict__ final_out
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    if (threadIdx.x != 0) {
        return; // single-thread execution per parameter set
    }

    const float accelerator = accelerators[combo];
    const int min_len = min_lengths[combo];
    const int max_len = max_lengths[combo];
    const int smooth_len = smooth_lengths[combo];

    if (series_len <= 0 || max_len <= 0 || min_len <= 0) {
        return;
    }

    const int base = combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        raw_out[base + i] = NAN;
        final_out[base + i] = NAN;
    }

    if (first_valid >= series_len) {
        return;
    }

    float length_f = static_cast<float>(max_len);
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    for (int i = first_valid; i < series_len; ++i) {
        float price_now = prices[i];
        if (!is_nan(price_now)) {
            sum += price_now;
            sum_sq += price_now * price_now;
            ++count;
        }

        if (i >= first_valid + max_len) {
            float price_old = prices[i - max_len];
            if (!is_nan(price_old)) {
                sum -= price_old;
                sum_sq -= price_old * price_old;
                --count;
            }
        }

        const int warm_raw = first_valid + max_len - 1;
        if (i < warm_raw) {
            continue;
        }

        if (count != max_len) {
            continue; // window polluted by NaN
        }

        const float mean = sum / static_cast<float>(max_len);
        float var = sum_sq / static_cast<float>(max_len) - mean * mean;
        if (var < 0.0f) {
            var = 0.0f;
        }
        const float std = sqrtf(var);
        if (isnan(std) || isnan(mean)) {
            continue;
        }

        const float a = mean - 1.75f * std;
        const float b = mean - 0.25f * std;
        const float c = mean + 0.25f * std;
        const float d = mean + 1.75f * std;

        if (!isfinite(price_now)) {
            continue;
        }

        if (price_now >= b && price_now <= c) {
            length_f += 1.0f;
        } else if (price_now < a || price_now > d) {
            length_f -= 1.0f;
        }
        length_f = clampf(length_f, static_cast<float>(min_len), static_cast<float>(max_len));

        int len_r = static_cast<int>(floorf(length_f + 0.5f));
        if (len_r < min_len) {
            len_r = min_len;
        }
        if (len_r > max_len) {
            len_r = max_len;
        }
        if (len_r < 1) {
            len_r = 1;
        }
        if (i + 1 < len_r) {
            continue;
        }

        float mf = 50.0f;
        bool used_volume = false;

        if (has_volume && volumes != nullptr) {
            float vol_now = volumes[i];
            if (!is_nan(vol_now) && vol_now != 0.0f) {
                int len_mf = len_r;
                int available = i + 1 - first_valid;
                if (len_mf > available) {
                    len_mf = available;
                }
                if (len_mf >= 2) {
                    int window_start = i + 1 - len_mf;
                    float up_vol = 0.0f;
                    float down_vol = 0.0f;
                    bool volume_valid = true;
                    for (int j = window_start + 1; j <= i; ++j) {
                        float px_cur = prices[j];
                        float px_prev = prices[j - 1];
                        float vol_j = volumes[j];
                        if (!isfinite(px_cur) || !isfinite(px_prev) || !isfinite(vol_j)) {
                            volume_valid = false;
                            break;
                        }
                        float delta = px_cur - px_prev;
                        if (delta > 0.0f) {
                            up_vol += vol_j;
                        } else if (delta < 0.0f) {
                            down_vol += vol_j;
                        }
                    }
                    if (volume_valid) {
                        float denom_vol = up_vol + down_vol;
                        if (denom_vol > 0.0f) {
                            mf = 100.0f * up_vol / denom_vol;
                            used_volume = true;
                        }
                    }
                }
            }
        }

        if (!used_volume) {
            int window_start = i + 1 - (len_r * 2);
            if (window_start < 0) {
                window_start = 0;
            }
            int window_end = i + 1;
            mf = compute_rsi(prices, window_start, window_end, len_r);
        }

        float mf_scaled = mf * 2.0f - 100.0f;
        float p = accelerator + fabsf(mf_scaled) / 25.0f;

        int window_start = i + 1 - len_r;
        float weighted_sum = 0.0f;
        float weight_total = 0.0f;
        for (int j = 0; j < len_r; ++j) {
            int idx = window_start + j;
            float val = prices[idx];
            if (is_nan(val)) {
                continue;
            }
            float base = static_cast<float>(len_r - j);
            float w = powf(base, p);
            weighted_sum += val * w;
            weight_total += w;
        }

        float result = price_now;
        if (weight_total > 0.0f) {
            result = weighted_sum / weight_total;
        }
        raw_out[base + i] = result;
    }

    if (smooth_len <= 1) {
        for (int i = 0; i < series_len; ++i) {
            final_out[base + i] = raw_out[base + i];
        }
        return;
    }

    const int warm_raw = first_valid + max_len - 1;
    const int warm_smooth = warm_raw + smooth_len - 1;
    if (warm_smooth >= series_len) {
        return; // smoothing never produces a value; keep NaNs
    }

    const int lookback = smooth_len - 1;
    const float denom = 0.5f * static_cast<float>(smooth_len) * static_cast<float>(smooth_len + 1);

    float weighted_sum = 0.0f;
    float plain_sum = 0.0f;
    for (int j = 0; j < lookback; ++j) {
        float val = raw_out[base + warm_raw + j];
        weighted_sum += (static_cast<float>(j) + 1.0f) * val;
        plain_sum += val;
    }

    int first_idx = warm_raw + lookback;
    float val = raw_out[base + first_idx];
    weighted_sum += static_cast<float>(smooth_len) * val;
    plain_sum += val;
    final_out[base + first_idx] = weighted_sum / denom;

    weighted_sum -= plain_sum;
    plain_sum -= raw_out[base + warm_raw];

    for (int i = first_idx + 1; i < series_len; ++i) {
        float v_new = raw_out[base + i];
        weighted_sum += static_cast<float>(smooth_len) * v_new;
        plain_sum += v_new;
        final_out[base + i] = weighted_sum / denom;
        weighted_sum -= plain_sum;
        float v_old = raw_out[base + i - lookback];
        plain_sum -= v_old;
    }
}

extern "C" __global__
void uma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const float* __restrict__ volumes_tm,
                                   int has_volume,
                                   float accelerator,
                                   int min_length,
                                   int max_length,
                                   int smooth_length,
                                   int num_series,
                                   int series_len,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ raw_out_tm,
                                   float* __restrict__ final_out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (threadIdx.x != 0) return;

    int fv = first_valids[series_idx];
    if (fv < 0) fv = 0;
    if (fv >= series_len) return;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + series_idx;
        raw_out_tm[idx] = NAN;
        final_out_tm[idx] = NAN;
    }

    if (max_length <= 0 || min_length <= 0 || smooth_length <= 0) {
        return;
    }

    float length_f = static_cast<float>(max_length);
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    const int warm_raw = fv + max_length - 1;

    for (int i = fv; i < series_len; ++i) {
        const float price_now = load_tm(prices_tm, num_series, series_idx, i);
        if (!is_nan(price_now)) {
            sum += price_now;
            sum_sq += price_now * price_now;
            ++count;
        }

        if (i >= fv + max_length) {
            const float price_old = load_tm(prices_tm, num_series, series_idx, i - max_length);
            if (!is_nan(price_old)) {
                sum -= price_old;
                sum_sq -= price_old * price_old;
                --count;
            }
        }

        if (i < warm_raw) {
            continue;
        }

        if (count != max_length) {
            continue;
        }

        const float mean = sum / static_cast<float>(max_length);
        float var = sum_sq / static_cast<float>(max_length) - mean * mean;
        if (var < 0.0f) {
            var = 0.0f;
        }
        const float std = sqrtf(var);
        if (isnan(std) || isnan(mean)) {
            continue;
        }

        const float a = mean - 1.75f * std;
        const float b = mean - 0.25f * std;
        const float c = mean + 0.25f * std;
        const float d = mean + 1.75f * std;

        if (!isfinite(price_now)) {
            continue;
        }

        if (price_now >= b && price_now <= c) {
            length_f += 1.0f;
        } else if (price_now < a || price_now > d) {
            length_f -= 1.0f;
        }
        length_f = clampf(length_f, static_cast<float>(min_length), static_cast<float>(max_length));

        int len_r = static_cast<int>(floorf(length_f + 0.5f));
        if (len_r < min_length) {
            len_r = min_length;
        }
        if (len_r > max_length) {
            len_r = max_length;
        }
        if (len_r < 1) {
            len_r = 1;
        }
        if (i + 1 < len_r) {
            continue;
        }

        float mf = 50.0f;
        bool used_volume = false;

        if (has_volume && volumes_tm != nullptr) {
            const float vol_now = load_tm(volumes_tm, num_series, series_idx, i);
            if (!is_nan(vol_now) && vol_now != 0.0f) {
                int len_mf = len_r;
                int available = i + 1 - fv;
                if (len_mf > available) {
                    len_mf = available;
                }
                if (len_mf >= 2) {
                    const int window_start = i + 1 - len_mf;
                    float up_vol = 0.0f;
                    float down_vol = 0.0f;
                    bool volume_valid = true;
                    for (int j = window_start + 1; j <= i; ++j) {
                        const float px_cur = load_tm(prices_tm, num_series, series_idx, j);
                        const float px_prev = load_tm(prices_tm, num_series, series_idx, j - 1);
                        const float vol_j = load_tm(volumes_tm, num_series, series_idx, j);
                        if (!isfinite(px_cur) || !isfinite(px_prev) || !isfinite(vol_j)) {
                            volume_valid = false;
                            break;
                        }
                        const float delta = px_cur - px_prev;
                        if (delta > 0.0f) {
                            up_vol += vol_j;
                        } else if (delta < 0.0f) {
                            down_vol += vol_j;
                        }
                    }
                    if (volume_valid) {
                        const float denom_vol = up_vol + down_vol;
                        if (denom_vol > 0.0f) {
                            mf = 100.0f * up_vol / denom_vol;
                            used_volume = true;
                        }
                    }
                }
            }
        }

        if (!used_volume) {
            int window_start = i + 1 - (len_r * 2);
            if (window_start < 0) {
                window_start = 0;
            }
            const int window_end = i + 1;
            mf = compute_rsi_tm(prices_tm, num_series, series_idx, window_start, window_end, len_r);
        }

        const float mf_scaled = mf * 2.0f - 100.0f;
        const float p = accelerator + fabsf(mf_scaled) / 25.0f;

        const int window_start = i + 1 - len_r;
        float weighted_sum = 0.0f;
        float weight_total = 0.0f;
        for (int j = 0; j < len_r; ++j) {
            const int t = window_start + j;
            const float val = load_tm(prices_tm, num_series, series_idx, t);
            if (is_nan(val)) {
                continue;
            }
            const float base = static_cast<float>(len_r - j);
            const float w = powf(base, p);
            weighted_sum += val * w;
            weight_total += w;
        }

        float result = price_now;
        if (weight_total > 0.0f) {
            result = weighted_sum / weight_total;
        }
        raw_out_tm[i * num_series + series_idx] = result;
    }

    if (smooth_length <= 1) {
        for (int t = fv; t < series_len; ++t) {
            const int idx = t * num_series + series_idx;
            final_out_tm[idx] = raw_out_tm[idx];
        }
        return;
    }

    const int warm_smooth = warm_raw + smooth_length - 1;
    if (warm_smooth >= series_len) {
        return;
    }

    const int lookback = smooth_length - 1;
    const float denom = 0.5f * static_cast<float>(smooth_length)
        * static_cast<float>(smooth_length + 1);

    float weighted_sum = 0.0f;
    float plain_sum = 0.0f;
    for (int j = 0; j < lookback; ++j) {
        const int t = warm_raw + j;
        const float val = raw_out_tm[t * num_series + series_idx];
        weighted_sum += (static_cast<float>(j) + 1.0f) * val;
        plain_sum += val;
    }

    int first_idx = warm_raw + lookback;
    const float first_val = raw_out_tm[first_idx * num_series + series_idx];
    weighted_sum += static_cast<float>(smooth_length) * first_val;
    plain_sum += first_val;
    final_out_tm[first_idx * num_series + series_idx] = weighted_sum / denom;

    weighted_sum -= plain_sum;
    plain_sum -= raw_out_tm[warm_raw * num_series + series_idx];

    for (int t = first_idx + 1; t < series_len; ++t) {
        const float v_new = raw_out_tm[t * num_series + series_idx];
        weighted_sum += static_cast<float>(smooth_length) * v_new;
        plain_sum += v_new;
        final_out_tm[t * num_series + series_idx] = weighted_sum / denom;
        weighted_sum -= plain_sum;
        const float v_old = raw_out_tm[(t - lookback) * num_series + series_idx];
        plain_sum -= v_old;
    }
}
