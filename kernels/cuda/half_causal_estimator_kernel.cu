#include <cmath>
#include <cstddef>

namespace {

constexpr int KERNEL_GAUSSIAN = 0;
constexpr int KERNEL_EPANECHNIKOV = 1;
constexpr int KERNEL_TRIANGULAR = 2;
constexpr int KERNEL_SINC = 3;

constexpr int CONFIDENCE_SYMMETRIC = 0;
constexpr int CONFIDENCE_LINEAR = 1;
constexpr int CONFIDENCE_NONE = 2;

__device__ inline bool finite_value(double value) {
    return isfinite(value);
}

__device__ inline double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

__device__ bool bucket_mean_icv(
    const double* data,
    int history_end,
    int target_slot,
    int slots_per_day,
    int data_period,
    double maximum_confidence_adjust_factor,
    double* out_mean,
    double* out_icv
) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    for (int idx = history_end; idx >= 0; --idx) {
        if ((idx % slots_per_day) != target_slot) {
            continue;
        }
        const double value = data[idx];
        if (!finite_value(value)) {
            continue;
        }
        sum += value;
        sum_sq += value * value;
        count += 1;
        if (data_period > 0 && count >= data_period) {
            break;
        }
    }

    if (count == 0) {
        return false;
    }

    const double mean = sum / static_cast<double>(count);
    double icv = 1.0;
    if (fabs(mean) > DBL_EPSILON) {
        const double variance = sum_sq / static_cast<double>(count) - mean * mean;
        const double stdev = sqrt(fmax(variance, 0.0));
        const double ratio = clamp01(stdev / mean);
        icv = 1.0 - ratio * maximum_confidence_adjust_factor;
    }

    *out_mean = mean;
    *out_icv = icv;
    return true;
}

__device__ bool collect_future(
    const double* data,
    int history_end,
    int slot,
    int slots_per_day,
    int data_period,
    int future_len,
    double maximum_confidence_adjust_factor,
    double* future_values,
    double* future_weights
) {
    if (future_len <= 0) {
        return true;
    }
    if (slots_per_day <= 0) {
        return false;
    }

    int found = 0;
    int offset = 1;
    bool saw_valid = false;
    while (found < future_len) {
        const int next_slot = (slot + offset) % slots_per_day;
        double mean = NAN;
        double icv = 1.0;
        if (bucket_mean_icv(
                data,
                history_end,
                next_slot,
                slots_per_day,
                data_period,
                maximum_confidence_adjust_factor,
                &mean,
                &icv
            )) {
            saw_valid = true;
            future_values[found] = mean;
            future_weights[found] = fmax(icv, 0.0);
            found += 1;
        }
        offset += 1;
        if (offset > slots_per_day * 4 && !saw_valid) {
            return false;
        }
    }

    for (int left = 0, right = future_len - 1; left < right; ++left, --right) {
        const double tmp_value = future_values[left];
        future_values[left] = future_values[right];
        future_values[right] = tmp_value;

        const double tmp_weight = future_weights[left];
        future_weights[left] = future_weights[right];
        future_weights[right] = tmp_weight;
    }
    return true;
}

__device__ double compute_estimate_window(
    const double* data,
    int len,
    int index,
    int slots_per_day,
    int data_period,
    int real_filter_length,
    int window_size,
    int confidence_adjust,
    double maximum_confidence_adjust_factor,
    const double* kernel_row,
    double* future_values,
    double* future_weights
) {
    if (real_filter_length <= 0 || window_size <= 0 || index < 0 || index >= len) {
        return NAN;
    }

    const int future_len = real_filter_length - 1;
    if (!collect_future(
            data,
            index - 1,
            index % slots_per_day,
            slots_per_day,
            data_period,
            future_len,
            maximum_confidence_adjust_factor,
            future_values,
            future_weights
        )) {
        return NAN;
    }

    double acc = 0.0;
    for (int j = 0; j < future_len; ++j) {
        const double value = future_values[j];
        if (!finite_value(value)) {
            return NAN;
        }
        const double confidence =
            confidence_adjust == CONFIDENCE_NONE ? 1.0 : future_weights[j];
        acc += value * confidence * kernel_row[j];
    }

    double future_weight_sum = 0.0;
    for (int j = 0; j < future_len; ++j) {
        future_weight_sum += future_weights[j];
    }

    for (int j = 0; j < real_filter_length; ++j) {
        const int source_index = index - j;
        if (source_index < 0) {
            return NAN;
        }
        const double value = data[source_index];
        if (!finite_value(value)) {
            return NAN;
        }

        double confidence = 1.0;
        if (confidence_adjust == CONFIDENCE_SYMMETRIC) {
            if (j == 0) {
                confidence = 1.0;
            } else {
                confidence = 2.0 - future_weights[future_len - j];
            }
        } else if (confidence_adjust == CONFIDENCE_LINEAR) {
            confidence =
                real_filter_length > 1
                ? 2.0 - future_weight_sum / static_cast<double>(real_filter_length - 1)
                : 1.0;
        }

        acc += value * confidence * kernel_row[future_len + j];
    }

    return acc;
}

__device__ double compute_expected_window(
    const double* data,
    int len,
    int index,
    int slots_per_day,
    int data_period,
    int real_filter_length,
    int window_size,
    double maximum_confidence_adjust_factor,
    const double* kernel_row,
    double* future_values,
    double* future_weights
) {
    if (real_filter_length <= 0 || window_size <= 0 || index < 0 || index >= len) {
        return NAN;
    }

    const int future_len = real_filter_length - 1;
    if (!collect_future(
            data,
            index - 1,
            index % slots_per_day,
            slots_per_day,
            data_period,
            future_len,
            maximum_confidence_adjust_factor,
            future_values,
            future_weights
        )) {
        return NAN;
    }

    double acc = 0.0;
    for (int j = 0; j < future_len; ++j) {
        const double value = future_values[j];
        if (!finite_value(value)) {
            return NAN;
        }
        acc += value * kernel_row[j];
    }

    for (int j = 0; j < real_filter_length; ++j) {
        const int source_index = index - j;
        if (source_index < 0) {
            return NAN;
        }
        const int history_end = source_index - 1;
        double mean = NAN;
        double icv = 1.0;
        const bool ok = bucket_mean_icv(
            data,
            history_end,
            source_index % slots_per_day,
            slots_per_day,
            data_period,
            maximum_confidence_adjust_factor,
            &mean,
            &icv
        );
        (void)icv;
        if (!ok || !finite_value(mean)) {
            return NAN;
        }
        acc += mean * kernel_row[future_len + j];
    }

    return acc;
}

__device__ double wma_update(
    double raw_value,
    int wma_length,
    double* history,
    int* count,
    double* first,
    bool* has_first
) {
    if (!finite_value(raw_value)) {
        return NAN;
    }

    if (!(*has_first)) {
        *first = raw_value;
        *has_first = true;
    }

    if (wma_length <= 1) {
        history[0] = raw_value;
        *count = 1;
        return raw_value;
    }

    const int prev_count = *count;
    const int next_count = prev_count < wma_length ? prev_count + 1 : wma_length;
    for (int i = next_count - 1; i >= 1; --i) {
        history[i] = history[i - 1];
    }
    history[0] = raw_value;
    *count = next_count;

    const double denominator = static_cast<double>(wma_length * (wma_length + 1) / 2);
    double sum = 0.0;
    for (int i = 0; i < wma_length; ++i) {
        const double sample = i < next_count ? history[i] : *first;
        sum += sample * static_cast<double>(wma_length - i);
    }
    return sum / denominator;
}

}

extern "C" __global__ void half_causal_estimator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ slots_per_days,
    const int* __restrict__ data_periods,
    const int* __restrict__ filter_lengths,
    const int* __restrict__ real_filter_lengths,
    const int* __restrict__ window_sizes,
    const double* __restrict__ maximum_confidence_adjust_factors,
    const int* __restrict__ enable_expected_values,
    const int* __restrict__ confidence_adjusts,
    const int* __restrict__ wma_lengths,
    int rows,
    int future_cap,
    int window_cap,
    int wma_cap,
    const double* __restrict__ kernel_matrix,
    double* __restrict__ future_values_scratch,
    double* __restrict__ future_weights_scratch,
    double* __restrict__ wma_history_scratch,
    double* __restrict__ out_estimate,
    double* __restrict__ out_expected_value
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int slots_per_day = slots_per_days[row];
    const int data_period = data_periods[row];
    const int real_filter_length = real_filter_lengths[row];
    const int window_size = window_sizes[row];
    const double maximum_confidence_adjust_factor = maximum_confidence_adjust_factors[row];
    const int enable_expected_value = enable_expected_values[row];
    const int confidence_adjust = confidence_adjusts[row];
    const int wma_length = wma_lengths[row];

    double* row_estimate = out_estimate + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_expected =
        out_expected_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* future_values =
        future_values_scratch + static_cast<size_t>(row) * static_cast<size_t>(future_cap);
    double* future_weights =
        future_weights_scratch + static_cast<size_t>(row) * static_cast<size_t>(future_cap);
    double* wma_history =
        wma_history_scratch + static_cast<size_t>(row) * static_cast<size_t>(wma_cap);
    const double* kernel_row =
        kernel_matrix + static_cast<size_t>(row) * static_cast<size_t>(window_cap);

    for (int i = 0; i < len; ++i) {
        row_estimate[i] = NAN;
        row_expected[i] = NAN;
    }

    if (slots_per_day < 2 || real_filter_length < 2 || window_size <= 0 ||
        !finite_value(maximum_confidence_adjust_factor)) {
        return;
    }

    int wma_count = 0;
    double wma_first = NAN;
    bool wma_has_first = false;
    bool ready = false;

    for (int i = 0; i < len; ++i) {
        const int slot = i % slots_per_day;
        const bool session_start = slot == 0;
        if (!ready && i > window_size && session_start) {
            ready = true;
        }

        if (ready && i + 1 >= real_filter_length) {
            const double estimate_raw = compute_estimate_window(
                data,
                len,
                i,
                slots_per_day,
                data_period,
                real_filter_length,
                window_size,
                confidence_adjust,
                maximum_confidence_adjust_factor,
                kernel_row,
                future_values,
                future_weights
            );
            if (finite_value(estimate_raw)) {
                row_estimate[i] = wma_update(
                    estimate_raw,
                    wma_length,
                    wma_history,
                    &wma_count,
                    &wma_first,
                    &wma_has_first
                );
            }

            if (enable_expected_value != 0) {
                row_expected[i] = compute_expected_window(
                    data,
                    len,
                    i,
                    slots_per_day,
                    data_period,
                    real_filter_length,
                    window_size,
                    maximum_confidence_adjust_factor,
                    kernel_row,
                    future_values,
                    future_weights
                );
            }
        }
    }
}
