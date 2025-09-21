// CUDA kernels for the Volume Adjusted Moving Average (VAMA).
//
// These kernels evaluate a single price/volume series across a grid of
// parameter combinations. The implementation follows the VRAM-first approach
// used throughout the project: host code prepares prefix-sum helpers and the
// kernel keeps all computation in FP32.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Shared helper: fetch prefix-sum difference in a safe manner.
static __device__ __forceinline__ float prefix_window(
    const float* __restrict__ prefix,
    int end_idx,
    int start_idx) {
    if (start_idx <= 0) {
        return prefix[end_idx];
    }
    return prefix[end_idx] - prefix[start_idx - 1];
}

static __device__ __forceinline__ float prefix_at_tm(
    const float* __restrict__ prefix,
    int time,
    int series,
    int num_series) {
    return prefix[time * num_series + series];
}

static __device__ __forceinline__ float prefix_window_tm(
    const float* __restrict__ prefix,
    int end_time,
    int start_time,
    int series,
    int num_series) {
    if (start_time <= 0) {
        return prefix_at_tm(prefix, end_time, series, num_series);
    }
    return prefix_at_tm(prefix, end_time, series, num_series)
        - prefix_at_tm(prefix, start_time - 1, series, num_series);
}

extern "C" __global__
void volume_adjusted_ma_batch_f32(
    const float* __restrict__ prices,
    const float* __restrict__ volumes,
    const float* __restrict__ prefix_volumes,
    const float* __restrict__ prefix_price_volumes,
    const int* __restrict__ lengths,
    const float* __restrict__ vi_factors,
    const int* __restrict__ sample_periods,
    const unsigned char* __restrict__ strict_flags,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    if (length <= 0 || length > series_len) {
        return;
    }

    const float vi_factor = vi_factors[combo];
    const int sample_period = sample_periods[combo];
    const bool strict = strict_flags[combo] != 0;

    const int warm = first_valid + length - 1;
    const int base_out = combo * series_len;

    const int stride = gridDim.x * blockDim.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
            t += stride;
            continue;
        }

        float avg_volume;
        if (sample_period == 0) {
            avg_volume = prefix_volumes[t] / float(t + 1);
        } else {
            if (t + 1 < sample_period) {
                out[out_idx] = NAN;
                t += stride;
                continue;
            }
            const int start = t + 1 - sample_period;
            const float window_sum = prefix_window(prefix_volumes, t, start);
            avg_volume = window_sum / float(sample_period);
        }

        const float vi_threshold = avg_volume * vi_factor;
        float weighted_sum = 0.0f;
        float v2i_sum = 0.0f;
        int nmb = 0;

        if (!strict && vi_threshold > 0.0f) {
            const int window = length;
            const int start = t - window + 1;
            const float sum_vol = prefix_window(prefix_volumes, t, start);
            const float sum_price_vol = prefix_window(prefix_price_volumes, t, start);
            const float inv = 1.0f / vi_threshold;
            v2i_sum = sum_vol * inv;
            weighted_sum = sum_price_vol * inv;
            nmb = window;
        } else {
            int cap = strict ? length * 10 : length;
            if (cap > t + 1) {
                cap = t + 1;
            }

            int idx = t;
            for (int j = 0; j < cap; ++j) {
                const float vol_val = volumes[idx];
                float v2i_nz = 0.0f;
                if (vi_threshold > 0.0f && isfinite(vol_val)) {
                    const float ratio = vol_val / vi_threshold;
                    if (isfinite(ratio)) {
                        v2i_nz = ratio;
                    }
                }

                if (v2i_nz != 0.0f) {
                    const float price_val = prices[idx];
                    if (isfinite(price_val)) {
                        weighted_sum = fmaf(price_val, v2i_nz, weighted_sum);
                    }
                }

                v2i_sum += v2i_nz;
                nmb = j + 1;

                if (strict) {
                    if (v2i_sum >= float(length)) {
                        break;
                    }
                } else {
                    if (nmb >= length) {
                        break;
                    }
                }

                if (idx == 0) {
                    break;
                }
                --idx;
            }
        }

        if (nmb > 0 && t >= nmb) {
            const int idx_nmb = t - nmb;
            const float p0 = prices[idx_nmb];
            if (isfinite(p0)) {
                const float numer = weighted_sum - (v2i_sum - float(length)) * p0;
                out[out_idx] = numer / float(length);
            } else {
                out[out_idx] = NAN;
            }
        } else {
            out[out_idx] = NAN;
        }

        t += stride;
    }
}

extern "C" __global__
void volume_adjusted_ma_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ volumes_tm,
    const float* __restrict__ prefix_volumes_tm,
    const float* __restrict__ prefix_price_volumes_tm,
    int period,
    float vi_factor,
    int sample_period,
    unsigned char strict_flag,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    if (period <= 0 || series_len <= 0) {
        return;
    }

    const bool strict = strict_flag != 0;
    const int warm = first_valids[series_idx] + period - 1;
    const int stride = gridDim.x * blockDim.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
            t += stride;
            continue;
        }

        float avg_volume;
        if (sample_period == 0) {
            avg_volume = prefix_at_tm(prefix_volumes_tm, t, series_idx, num_series)
                / float(t + 1);
        } else {
            if (t + 1 < sample_period) {
                out_tm[out_idx] = NAN;
                t += stride;
                continue;
            }
            const int start = t + 1 - sample_period;
            const float window_sum = prefix_window_tm(
                prefix_volumes_tm,
                t,
                start,
                series_idx,
                num_series);
            avg_volume = window_sum / float(sample_period);
        }

        const float vi_threshold = avg_volume * vi_factor;
        float weighted_sum = 0.0f;
        float v2i_sum = 0.0f;
        int nmb = 0;

        if (!strict && vi_threshold > 0.0f) {
            const int window = period;
            const int start = t - window + 1;
            const float sum_vol = prefix_window_tm(
                prefix_volumes_tm,
                t,
                start,
                series_idx,
                num_series);
            const float sum_price_vol = prefix_window_tm(
                prefix_price_volumes_tm,
                t,
                start,
                series_idx,
                num_series);
            const float inv = 1.0f / vi_threshold;
            v2i_sum = sum_vol * inv;
            weighted_sum = sum_price_vol * inv;
            nmb = window;
        } else {
            int cap = strict ? period * 10 : period;
            if (cap > t + 1) {
                cap = t + 1;
            }

            int idx = t;
            for (int j = 0; j < cap; ++j) {
                const int in_idx = idx * num_series + series_idx;
                const float vol_val = volumes_tm[in_idx];
                float v2i_nz = 0.0f;
                if (vi_threshold > 0.0f && isfinite(vol_val)) {
                    const float ratio = vol_val / vi_threshold;
                    if (isfinite(ratio)) {
                        v2i_nz = ratio;
                    }
                }

                if (v2i_nz != 0.0f) {
                    const float price_val = prices_tm[in_idx];
                    if (isfinite(price_val)) {
                        weighted_sum = fmaf(price_val, v2i_nz, weighted_sum);
                    }
                }

                v2i_sum += v2i_nz;
                nmb = j + 1;

                if (strict) {
                    if (v2i_sum >= float(period)) {
                        break;
                    }
                } else if (nmb >= period) {
                    break;
                }

                if (idx == 0) {
                    break;
                }
                --idx;
            }
        }

        if (nmb > 0 && t >= nmb) {
            const int idx_nmb = (t - nmb) * num_series + series_idx;
            const float p0 = prices_tm[idx_nmb];
            if (isfinite(p0)) {
                const float numer = weighted_sum - (v2i_sum - float(period)) * p0;
                out_tm[out_idx] = numer / float(period);
            } else {
                out_tm[out_idx] = NAN;
            }
        } else {
            out_tm[out_idx] = NAN;
        }

        t += stride;
    }
}
