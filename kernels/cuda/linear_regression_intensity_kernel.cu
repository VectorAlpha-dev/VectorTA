#include <cmath>
#include <cstddef>
#include <cstdint>

static __device__ inline void push_shift(double* buffer, int* count, int capacity, double value) {
    if (*count < capacity) {
        buffer[*count] = value;
        *count += 1;
        return;
    }
    for (int i = 1; i < capacity; ++i) {
        buffer[i - 1] = buffer[i];
    }
    buffer[capacity - 1] = value;
}

static __device__ inline double linreg_value(const double* window, int length) {
    double period_f = static_cast<double>(length);
    double x_sum = static_cast<double>(length * (length + 1) / 2);
    double x2_sum = static_cast<double>(length * (length + 1) * (2 * length + 1) / 6);
    double denom = period_f * x2_sum - x_sum * x_sum;
    if (!(denom > 0.0) || !isfinite(denom)) {
        return NAN;
    }

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int i = 0; i < length; ++i) {
        double value = window[i];
        double x = static_cast<double>(i + 1);
        y_sum += value;
        xy_sum += value * x;
    }

    double b = (period_f * xy_sum - x_sum * y_sum) / denom;
    double a = (y_sum - b * x_sum) / period_f;
    return a + b * period_f;
}

static __device__ inline double trend_to_intensity(const double* window, int lookback) {
    int total = lookback * (lookback - 1) / 2;
    if (total == 0) {
        return 0.0;
    }

    int64_t trend = 0;
    for (int i = 0; i < lookback - 1; ++i) {
        double a = window[i];
        for (int j = i + 1; j < lookback; ++j) {
            double b = window[j];
            if (a != b) {
                trend += b > a ? 1 : -1;
            }
        }
    }
    return static_cast<double>(trend) / static_cast<double>(total);
}

extern "C" __global__ void linear_regression_intensity_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ linreg_lengths,
    int rows,
    int max_lookback_period,
    int max_linreg_length,
    double* __restrict__ linreg_input_buf,
    double* __restrict__ linreg_window_buf,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int lookback_period = lookback_periods[row];
    int linreg_length = linreg_lengths[row];
    double* linreg_input =
        linreg_input_buf + static_cast<size_t>(row) * static_cast<size_t>(max_linreg_length);
    double* linreg_window =
        linreg_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_lookback_period);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (lookback_period <= 0 ||
        linreg_length <= 0 ||
        lookback_period > max_lookback_period ||
        linreg_length > max_linreg_length) {
        return;
    }

    int linreg_input_count = 0;
    int linreg_window_count = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            linreg_input_count = 0;
            linreg_window_count = 0;
            continue;
        }

        push_shift(linreg_input, &linreg_input_count, linreg_length, value);
        if (linreg_input_count < linreg_length) {
            continue;
        }

        double lr = linreg_value(linreg_input, linreg_length);
        if (!isfinite(lr)) {
            linreg_window_count = 0;
            continue;
        }

        push_shift(linreg_window, &linreg_window_count, lookback_period, lr);
        if (linreg_window_count < lookback_period) {
            continue;
        }

        row_out[i] = trend_to_intensity(linreg_window, lookback_period);
    }
}
