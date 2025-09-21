// CUDA kernels for the Ehlers Error Correcting Exponential Moving Average (ECEMA).
//
// Kernels operate on FP32 buffers to match the public API but promote the
// recurrence to FP64 so the numerical behaviour mirrors the CPU reference.
// Two variants are exposed:
//   * ehlers_ecema_batch_f32:            single price series × many (length, gain) pairs
//   * ehlers_ecema_many_series_one_param_time_major_f32: time-major matrix where every
//       column shares the same parameter pair.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

namespace {
__device__ inline double compute_alpha(int length) {
    return 2.0 / (static_cast<double>(length) + 1.0);
}

__device__ inline double compute_beta(double alpha) {
    return 1.0 - alpha;
}

__device__ inline double to_double(float v) {
    return static_cast<double>(v);
}

__device__ inline double clamp_prev_ec(bool pine_mode, double ema_value) {
    return pine_mode ? 0.0 : ema_value;
}

__device__ inline double pick_src(const float* prices, int idx, bool confirmed) {
    int source_idx = confirmed && idx > 0 ? idx - 1 : idx;
    return static_cast<double>(prices[source_idx]);
}

__device__ inline double pick_src_tm(
    const float* prices_tm,
    int idx,
    int series,
    int num_series,
    bool confirmed
) {
    int row = confirmed && idx > 0 ? idx - 1 : idx;
    return static_cast<double>(prices_tm[row * num_series + series]);
}
}

extern "C" __global__
void ehlers_ecema_batch_f32(const float* __restrict__ prices,
                            const int* __restrict__ lengths,
                            const int* __restrict__ gain_limits,
                            const unsigned char* __restrict__ pine_flags,
                            const unsigned char* __restrict__ confirmed_flags,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const int gain_limit = gain_limits[combo];
    const bool pine_mode = pine_flags[combo] != 0;
    const bool confirmed = confirmed_flags[combo] != 0;

    const int base = combo * series_len;
    const float nan_f = NAN;
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (length <= 0 || gain_limit <= 0) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) {
        return;
    }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        return;
    }

    const double alpha = compute_alpha(length);
    const double beta = compute_beta(alpha);

    double ema = 0.0;
    double running_mean = 0.0;
    int mean_count = 0;
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;
    double prev_ec = 0.0;

    for (int i = first_valid; i < series_len; ++i) {
        const double price = to_double(prices[i]);
        double ema_value;

        if (pine_mode) {
            if (isfinite(price)) {
                ema = alpha * price + beta * ema;
            }
            ema_value = ema;
        } else {
            if (i == first_valid) {
                running_mean = price;
                mean_count = 1;
                ema = price;
                ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    mean_count += 1;
                    running_mean = ((static_cast<double>(mean_count) - 1.0) * running_mean + price) /
                                    static_cast<double>(mean_count);
                }
                ema = running_mean;
                ema_value = running_mean;
            } else {
                if (isfinite(price)) {
                    ema = beta * ema + alpha * price;
                }
                ema_value = ema;
            }
        }

        if (i < warm) {
            continue;
        }

        const double src = pick_src(prices, i, confirmed);
        const double prev = has_prev ? prev_ec : clamp_prev_ec(pine_mode, ema_value);

        double least_error = DBL_MAX;
        double best_gain = 0.0;
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const double gain = static_cast<double>(g) / 10.0;
            const double candidate = alpha * (ema_value + gain * (src - prev)) + beta * prev;
            const double err = fabs(src - candidate);
            if (err < least_error) {
                least_error = err;
                best_gain = gain;
            }
        }

        const double ec = alpha * (ema_value + best_gain * (src - prev)) + beta * prev;
        out[base + i] = static_cast<float>(ec);
        prev_ec = ec;
        has_prev = true;
    }
}

extern "C" __global__
void ehlers_ecema_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    const float nan_f = NAN;
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + series] = nan_f;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    if (length <= 0 || gain_limit <= 0) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;

    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) {
        return;
    }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        return;
    }

    const double alpha = compute_alpha(length);
    const double beta = compute_beta(alpha);

    double ema = 0.0;
    double running_mean = 0.0;
    int mean_count = 0;
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;
    double prev_ec = 0.0;

    for (int i = first_valid; i < series_len; ++i) {
        const double price = static_cast<double>(prices_tm[i * num_series + series]);
        double ema_value;

        if (pine_mode) {
            if (isfinite(price)) {
                ema = alpha * price + beta * ema;
            }
            ema_value = ema;
        } else {
            if (i == first_valid) {
                running_mean = price;
                mean_count = 1;
                ema = price;
                ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    mean_count += 1;
                    running_mean = ((static_cast<double>(mean_count) - 1.0) * running_mean + price) /
                                    static_cast<double>(mean_count);
                }
                ema = running_mean;
                ema_value = running_mean;
            } else {
                if (isfinite(price)) {
                    ema = beta * ema + alpha * price;
                }
                ema_value = ema;
            }
        }

        if (i < warm) {
            continue;
        }

        const double src = pick_src_tm(prices_tm, i, series, num_series, confirmed);
        const double prev = has_prev ? prev_ec : clamp_prev_ec(pine_mode, ema_value);

        double least_error = DBL_MAX;
        double best_gain = 0.0;
        for (int g = -gain_limit; g <= gain_limit; ++g) {
            const double gain = static_cast<double>(g) / 10.0;
            const double candidate = alpha * (ema_value + gain * (src - prev)) + beta * prev;
            const double err = fabs(src - candidate);
            if (err < least_error) {
                least_error = err;
                best_gain = gain;
            }
        }

        const double ec = alpha * (ema_value + best_gain * (src - prev)) + beta * prev;
        out_tm[i * num_series + series] = static_cast<float>(ec);
        prev_ec = ec;
        has_prev = true;
    }
}
