#include <cmath>
#include <cstddef>

extern "C" __global__ void regression_slope_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ min_ranges,
    const int* __restrict__ max_ranges,
    const int* __restrict__ steps,
    const int* __restrict__ signal_lines,
    int rows,
    double* __restrict__ out_value,
    double* __restrict__ out_signal,
    double* __restrict__ out_bullish_reversal,
    double* __restrict__ out_bearish_reversal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int min_range = min_ranges[row];
    int max_range = max_ranges[row];
    int step = steps[row];
    int signal_line = signal_lines[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish =
        out_bullish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish =
        out_bearish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_signal[i] = NAN;
        row_bullish[i] = NAN;
        row_bearish[i] = NAN;
    }

    if (min_range < 2 || max_range < 2 || step <= 0 || signal_line <= 0 || min_range > max_range) {
        return;
    }

    double* signal_queue = new double[signal_line];
    if (signal_queue == nullptr) {
        return;
    }

    for (int i = max_range - 1; i < len; ++i) {
        int max_start = i + 1 - max_range;
        bool max_window_valid = true;
        for (int j = max_start; j <= i; ++j) {
            double sample = data[j];
            if (!(isfinite(sample) && sample > 0.0)) {
                max_window_valid = false;
                break;
            }
        }
        if (!max_window_valid) {
            continue;
        }

        double sum_slopes = 0.0;
        int slope_count = 0;
        bool valid = true;

        for (int length = min_range; length <= max_range; length += step) {
            int start = i + 1 - length;
            double length_f64 = static_cast<double>(length);
            double sum_x = length_f64 * (length_f64 + 1.0) * 0.5;
            double sum_x_sqr =
                length_f64 * (length_f64 + 1.0) * (2.0 * length_f64 + 1.0) / 6.0;
            double denom = length_f64 * sum_x_sqr - sum_x * sum_x;
            double sum_y = 0.0;
            double sum_xy = 0.0;

            for (int j = 0; j < length; ++j) {
                double sample = data[start + j];
                if (!(isfinite(sample) && sample > 0.0)) {
                    valid = false;
                    break;
                }
                double x = static_cast<double>(j + 1);
                double logged = log(sample);
                sum_y += logged;
                sum_xy += x * logged;
            }

            if (!valid) {
                break;
            }

            sum_slopes += (length_f64 * sum_xy - sum_x * sum_y) / denom;
            slope_count += 1;
        }

        if (valid && slope_count > 0) {
            row_value[i] = sum_slopes / static_cast<double>(slope_count);
        }
    }

    double signal_sum = 0.0;
    int signal_count = 0;
    int signal_head = 0;

    for (int i = 0; i < len; ++i) {
        double value = row_value[i];
        if (isfinite(value)) {
            if (signal_count < signal_line) {
                signal_queue[(signal_head + signal_count) % signal_line] = value;
                signal_sum += value;
                signal_count += 1;
            } else {
                signal_sum -= signal_queue[signal_head];
                signal_queue[signal_head] = value;
                signal_sum += value;
                signal_head += 1;
                if (signal_head == signal_line) {
                    signal_head = 0;
                }
            }
        }

        if (signal_count == signal_line) {
            row_signal[i] = signal_sum / static_cast<double>(signal_line);
        }

        if (isfinite(value) && isfinite(row_signal[i])) {
            double prev_value = i > 0 ? row_value[i - 1] : NAN;
            double prev_signal = i > 0 ? row_signal[i - 1] : NAN;
            row_bearish[i] = isfinite(prev_value) && isfinite(prev_signal) &&
                    value < row_signal[i] && prev_value >= prev_signal && value > 0.0
                ? 1.0
                : 0.0;
            row_bullish[i] = isfinite(prev_value) && isfinite(prev_signal) &&
                    value > row_signal[i] && prev_value <= prev_signal && value < 0.0
                ? 1.0
                : 0.0;
        }
    }

    delete[] signal_queue;
}
