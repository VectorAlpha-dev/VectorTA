#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void dual_ulcer_index_build_squares_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out_long_sq,
    double* __restrict__ out_short_sq
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row_long_sq = out_long_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_short_sq = out_short_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_long_sq[t] = CUDART_NAN;
        row_short_sq[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    int close_count = 0;

    for (int t = 0; t < len; ++t) {
        double close = data[t];
        if (!isfinite(close) || close <= 0.0) {
            close_count = 0;
            continue;
        }

        if (close_count < period) {
            close_count += 1;
        }
        if (close_count < period) {
            continue;
        }

        int window_start = t + 1 - period;
        double highest = -CUDART_INF;
        double lowest = CUDART_INF;
        bool valid = true;

        for (int i = window_start; i <= t; ++i) {
            double value = data[i];
            if (!isfinite(value) || value <= 0.0) {
                valid = false;
                break;
            }
            if (value > highest) {
                highest = value;
            }
            if (value < lowest) {
                lowest = value;
            }
        }

        if (!valid) {
            close_count = 0;
            continue;
        }

        double long_ret = 100.0 * (close - highest) / highest;
        double short_ret = 100.0 * (close - lowest) / lowest;
        row_long_sq[t] = long_ret * long_ret;
        row_short_sq[t] = short_ret * short_ret;
    }
}

extern "C" __global__ void dual_ulcer_index_finalize_f64(
    const double* __restrict__ long_sq,
    const double* __restrict__ short_sq,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ thresholds,
    int auto_threshold,
    int n_combos,
    double* __restrict__ out_long_ulcer,
    double* __restrict__ out_short_ulcer,
    double* __restrict__ out_threshold
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double custom_threshold = thresholds[combo_idx];
    const double* row_long_sq = long_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    const double* row_short_sq = short_sq + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_long = out_long_ulcer + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_short = out_short_ulcer + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_threshold = out_threshold + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_long[t] = CUDART_NAN;
        row_short[t] = CUDART_NAN;
        row_threshold[t] = CUDART_NAN;
    }

    if (period <= 0 || !isfinite(custom_threshold) || custom_threshold < 0.0) {
        return;
    }

    int sq_count = 0;
    double long_sq_sum = 0.0;
    double short_sq_sum = 0.0;
    double diff_sum = 0.0;
    int diff_count = 0;

    for (int t = 0; t < len; ++t) {
        double current_long_sq = row_long_sq[t];
        double current_short_sq = row_short_sq[t];
        if (!isfinite(current_long_sq) || !isfinite(current_short_sq)) {
            sq_count = 0;
            long_sq_sum = 0.0;
            short_sq_sum = 0.0;
            continue;
        }

        if (sq_count == period) {
            long_sq_sum -= row_long_sq[t - period];
            short_sq_sum -= row_short_sq[t - period];
        } else {
            sq_count += 1;
        }

        long_sq_sum += current_long_sq;
        short_sq_sum += current_short_sq;

        if (sq_count < period) {
            continue;
        }

        double denom = static_cast<double>(period);
        double long_ulcer = sqrt(long_sq_sum) / denom;
        double short_ulcer = sqrt(short_sq_sum) / denom;
        double threshold_value;

        if (auto_threshold != 0) {
            double diff = fabs(long_ulcer - short_ulcer);
            diff_sum += diff;
            diff_count += 1;
            threshold_value = diff_sum / static_cast<double>(diff_count);
        } else {
            threshold_value = custom_threshold;
        }

        row_long[t] = long_ulcer;
        row_short[t] = short_ulcer;
        row_threshold[t] = threshold_value;
    }
}
