#include <cmath>
#include <cstddef>

static __device__ inline double adaptive_macd_corr_sq_window(
    const double* data,
    int end_idx,
    int length
) {
    int start = end_idx + 1 - length;
    double sum_x = static_cast<double>((length - 1) * length) * 0.5;
    double sum_x2 = static_cast<double>((length - 1) * length * (2 * length - 1)) / 6.0;
    double n = static_cast<double>(length);
    double denom_x = n * sum_x2 - sum_x * sum_x;
    double sum_y = 0.0;
    double sum_y2 = 0.0;
    double sum_xy = 0.0;

    for (int i = 0; i < length; ++i) {
        double value = data[start + i];
        if (!isfinite(value)) {
            return NAN;
        }
        sum_y += value;
        sum_y2 += value * value;
        sum_xy += static_cast<double>(i) * value;
    }

    double denom_y = n * sum_y2 - sum_y * sum_y;
    if (denom_y <= 1e-12) {
        return 0.0;
    }
    double num = n * sum_xy - sum_x * sum_y;
    double corr_sq = (num * num) / (denom_x * denom_y);
    if (corr_sq < 0.0) {
        return 0.0;
    }
    if (corr_sq > 1.0) {
        return 1.0;
    }
    return corr_sq;
}

extern "C" __global__ void adaptive_macd_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const int* __restrict__ signal_periods,
    int rows,
    double* __restrict__ out_macd,
    double* __restrict__ out_signal,
    double* __restrict__ out_hist
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int fast_period = fast_periods[row];
    int slow_period = slow_periods[row];
    int signal_period = signal_periods[row];

    double* row_out_macd = out_macd + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_hist = out_hist + static_cast<size_t>(row) * static_cast<size_t>(len);

    if (length < 2 ||
        fast_period < 2 ||
        slow_period < 2 ||
        signal_period < 2 ||
        length > len ||
        fast_period > len ||
        slow_period > len ||
        signal_period > len) {
        for (int i = 0; i < len; ++i) {
            row_out_macd[i] = NAN;
            row_out_signal[i] = NAN;
            row_out_hist[i] = NAN;
        }
        return;
    }

    double a1 = 2.0 / (static_cast<double>(fast_period) + 1.0);
    double a2 = 2.0 / (static_cast<double>(slow_period) + 1.0);
    double delta_coeff = a1 - a2;
    double recur_coeff = 2.0 - a1 - a2;
    double trend_coeff = (1.0 - a1) * (1.0 - a2);
    double cycle_coeff = (1.0 - a1) / (1.0 - a2);
    double alpha = 2.0 / (static_cast<double>(signal_period) + 1.0);
    double beta = 1.0 - alpha;

    bool signal_started = false;
    int signal_count = 0;
    double signal_sum = 0.0;
    double signal_value = NAN;
    double prev_close = NAN;
    double prev_macd1 = NAN;
    double prev_macd2 = NAN;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        double current_macd = NAN;
        if (isfinite(value) && i + 1 >= length) {
            bool valid_window = true;
            for (int j = i + 1 - length; j <= i; ++j) {
                if (!isfinite(data[j])) {
                    valid_window = false;
                    break;
                }
            }
            if (valid_window && isfinite(prev_close)) {
                double corr_sq = adaptive_macd_corr_sq_window(data, i, length);
                if (isfinite(corr_sq)) {
                    double r2 = 0.5 * corr_sq + 0.5;
                    double k = r2 * trend_coeff + (1.0 - r2) * cycle_coeff;
                    double prev1 = isfinite(prev_macd1) ? prev_macd1 : 0.0;
                    double prev2 = isfinite(prev_macd2) ? prev_macd2 : 0.0;
                    current_macd =
                        (value - prev_close) * delta_coeff + recur_coeff * prev1 - k * prev2;
                }
            }
        }

        prev_close = value;
        prev_macd2 = prev_macd1;
        prev_macd1 = current_macd;

        double signal = NAN;
        if (isfinite(current_macd)) {
            if (!signal_started) {
                signal_started = true;
                signal_count = 1;
                signal_sum = current_macd;
                signal_value = current_macd;
            } else if (signal_count < signal_period) {
                signal_count += 1;
                signal_sum += current_macd;
                signal_value = signal_sum / static_cast<double>(signal_count);
            } else {
                signal_value = beta * signal_value + alpha * current_macd;
            }
            signal = signal_value;
        } else if (signal_started) {
            signal = signal_value;
        }

        row_out_macd[i] = current_macd;
        row_out_signal[i] = signal;
        row_out_hist[i] =
            (isfinite(current_macd) && isfinite(signal)) ? current_macd - signal : NAN;
    }
}
