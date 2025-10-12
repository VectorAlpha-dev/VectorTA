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

        double avg_volume_d;
        if (sample_period == 0) {
            avg_volume_d = (double)prefix_volumes[t] / (double)(t + 1);
        } else {
            if (t + 1 < sample_period) {
                out[out_idx] = NAN;
                t += stride;
                continue;
            }
            const int start = t + 1 - sample_period;
            const double window_sum = (double)prefix_window(prefix_volumes, t, start);
            avg_volume_d = window_sum / (double)sample_period;
        }

        const double vi_threshold_d = avg_volume_d * (double)vi_factor;
        const float  vi_threshold   = (float)vi_threshold_d;
        float weighted_sum = 0.0f;
        float v2i_sum = 0.0f;
        int nmb = 0;

        if (!strict) {
            // Non-strict: fixed-width window with explicit accumulation (matches scalar ordering)
            int cap = length;
            if (cap > t + 1) cap = t + 1;

            double wsum_d = 0.0;
            double v2i_sum_d = 0.0;
            int idx = t;
            for (int j = 0; j < cap; ++j) {
                const float vol_val = volumes[idx];
                if (isfinite(vol_val)) {
                    const double v2i = ((double)vol_val) / vi_threshold_d;
                    if (isfinite(v2i)) {
                        v2i_sum_d += v2i;
                        const float price_val = prices[idx];
                        if (isfinite(price_val)) {
                            wsum_d = fma((double)price_val, v2i, wsum_d);
                        }
                    }
                }
                nmb = j + 1;
                if (nmb >= length) break;
                if (idx == 0) break;
                --idx;
            }
            v2i_sum = (float)v2i_sum_d;
            weighted_sum = (float)wsum_d;
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
    float* __restrict__ out_tm)
{
    if (period <= 0 || series_len <= 0) return;
    const bool strict = (strict_flag != 0);
    const float inv_period = 1.0f / float(period);

    // Grid-stride loop over time; threads iterate over series for coalesced access
    for (int t = blockIdx.x; t < series_len; t += gridDim.x) {
        for (int s = threadIdx.x; s < num_series; s += blockDim.x) {
            const int warm = first_valids[s] + period - 1;
            const int out_idx = t * num_series + s;

            if (t < warm) {
                out_tm[out_idx] = NAN;
                continue;
            }

            // Average volume for (t, s)
            float avg_volume;
            if (sample_period == 0) {
                const float pref = prefix_volumes_tm[t * num_series + s];
                avg_volume = pref / float(t + 1);
            } else {
                if (t + 1 < sample_period) {
                    out_tm[out_idx] = NAN;
                    continue;
                }
                const int start = t + 1 - sample_period;
                const float end_sum = prefix_volumes_tm[t * num_series + s];
                const float start_sum = (start <= 0)
                    ? 0.0f
                    : prefix_volumes_tm[(start - 1) * num_series + s];
                avg_volume = (end_sum - start_sum) / float(sample_period);
            }

            const float vi_threshold = avg_volume * vi_factor;
            const float inv_th = (vi_threshold > 0.0f) ? (1.0f / vi_threshold) : 0.0f;

            float weighted_sum = 0.0f;
            float v2i_sum      = 0.0f;
            int   nmb          = 0;

            if (!strict && inv_th > 0.0f) {
                const int start = t - period + 1;
                const float sum_vol = (start <= 0)
                    ? prefix_volumes_tm[t * num_series + s]
                    : (prefix_volumes_tm[t * num_series + s] -
                       prefix_volumes_tm[(start - 1) * num_series + s]);
                const float sum_price_vol = (start <= 0)
                    ? prefix_price_volumes_tm[t * num_series + s]
                    : (prefix_price_volumes_tm[t * num_series + s] -
                       prefix_price_volumes_tm[(start - 1) * num_series + s]);

                v2i_sum      = sum_vol * inv_th;
                weighted_sum = sum_price_vol * inv_th;
                nmb          = period;
            } else {
                int cap = strict ? (period * 10) : period;
                if (cap > t + 1) cap = t + 1;

                int idx = t;
                #pragma unroll 4
                for (int j = 0; j < cap; ++j, --idx) {
                    const float vol_val = volumes_tm[idx * num_series + s];
                    const float v2i_nz  = (isfinite(vol_val)) ? (vol_val * inv_th) : 0.0f;

                    if (v2i_nz != 0.0f) {
                        const float price_val = prices_tm[idx * num_series + s];
                        if (isfinite(price_val)) {
                            weighted_sum = fmaf(price_val, v2i_nz, weighted_sum);
                        }
                    }

                    v2i_sum += v2i_nz;
                    nmb = j + 1;

                    if (strict) {
                        if (v2i_sum >= float(period)) break;
                    } else if (nmb >= period) {
                        break;
                    }
                    if (idx == 0) break;
                }
            }

            if (nmb > 0 && t >= nmb) {
                const float p0 = prices_tm[(t - nmb) * num_series + s];
                if (isfinite(p0)) {
                    const float numer = weighted_sum - (v2i_sum - float(period)) * p0;
                    out_tm[out_idx] = numer * inv_period;
                } else {
                    out_tm[out_idx] = NAN;
                }
            } else {
                out_tm[out_idx] = NAN;
            }
        }
    }
}
