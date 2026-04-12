#include <cmath>
#include <cstdint>

static __device__ inline void push_shift(double* buf, int* count, int cap, double value) {
    if (*count < cap) {
        buf[*count] = value;
        *count += 1;
        return;
    }
    for (int i = 1; i < cap; ++i) {
        buf[i - 1] = buf[i];
    }
    buf[cap - 1] = value;
}

static __device__ inline double linreg_slope(const double* window, int n) {
    if (n <= 1) {
        return 0.0;
    }
    double nf = static_cast<double>(n);
    double sum_x = static_cast<double>(n * (n - 1) / 2);
    double sum_x2 = static_cast<double>((n - 1) * n * (2 * n - 1) / 6);
    double denom = nf * sum_x2 - sum_x * sum_x;
    if (fabs(denom) <= 2.2204460492503131e-16) {
        return 0.0;
    }
    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double value = window[i];
        sum_y += value;
        sum_xy += x * value;
    }
    return (nf * sum_xy - sum_x * sum_y) / denom;
}

static __device__ inline double wma_value(const double* window, int period) {
    if (period <= 1) {
        return window[period - 1];
    }
    double weighted_sum = 0.0;
    double denom = static_cast<double>(period * (period + 1) / 2);
    for (int i = 0; i < period; ++i) {
        weighted_sum += window[i] * static_cast<double>(i + 1);
    }
    return weighted_sum / denom;
}

extern "C" __global__ void projection_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* source,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* high_window_buf,
    double* low_window_buf,
    double* high_slopes_buf,
    double* low_slopes_buf,
    double* pbo_window_buf,
    double* signal_window_buf,
    double* out_pbo,
    double* out_signal
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length <= 0 || smooth_length <= 0) {
        return;
    }

    const double nan = NAN;
    const double inf = 1.7976931348623157e308;
    double* high_window = high_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_window = low_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* high_slopes = high_slopes_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* low_slopes = low_slopes_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* pbo_window = pbo_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* signal_window =
        signal_window_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_out_pbo = out_pbo + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int window_count = 0;
    int slope_count = 0;
    int pbo_count = 0;
    int signal_count = 0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double s = source[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(s)) {
            window_count = 0;
            slope_count = 0;
            pbo_count = 0;
            signal_count = 0;
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }

        if (window_count < length) {
            high_window[window_count] = h;
            low_window[window_count] = l;
            window_count += 1;
        } else {
            for (int j = 1; j < length; ++j) {
                high_window[j - 1] = high_window[j];
                low_window[j - 1] = low_window[j];
            }
            high_window[length - 1] = h;
            low_window[length - 1] = l;
        }

        double high_slope = nan;
        double low_slope = nan;
        if (window_count == length) {
            high_slope = linreg_slope(high_window, length);
            low_slope = linreg_slope(low_window, length);
        }

        if (slope_count < length) {
            high_slopes[slope_count] = high_slope;
            low_slopes[slope_count] = low_slope;
            slope_count += 1;
        } else {
            for (int j = 1; j < length; ++j) {
                high_slopes[j - 1] = high_slopes[j];
                low_slopes[j - 1] = low_slopes[j];
            }
            high_slopes[length - 1] = high_slope;
            low_slopes[length - 1] = low_slope;
        }

        bool slopes_ready = window_count == length && slope_count == length;
        if (slopes_ready) {
            for (int j = 0; j < length; ++j) {
                if (!isfinite(high_slopes[j]) || !isfinite(low_slopes[j])) {
                    slopes_ready = false;
                    break;
                }
            }
        }
        if (!slopes_ready) {
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }

        double upper = -inf;
        double lower = inf;
        int last = length - 1;
        for (int age = 0; age < length; ++age) {
            int idx = last - age;
            double projected_high = high_window[idx] + high_slopes[idx] * static_cast<double>(age);
            double projected_low = low_window[idx] + low_slopes[idx] * static_cast<double>(age);
            if (projected_high > upper) {
                upper = projected_high;
            }
            if (projected_low < lower) {
                lower = projected_low;
            }
        }

        double range = upper - lower;
        double raw = fabs(range) <= 2.2204460492503131e-16 ? 0.0 : (100.0 * (s - lower) / range);

        push_shift(pbo_window, &pbo_count, smooth_length, raw);
        if (pbo_count < smooth_length) {
            row_out_pbo[i] = nan;
            row_out_signal[i] = nan;
            continue;
        }
        double pbo = smooth_length == 1 ? raw : wma_value(pbo_window, smooth_length);

        push_shift(signal_window, &signal_count, smooth_length, pbo);
        row_out_pbo[i] = pbo;
        row_out_signal[i] =
            signal_count < smooth_length ? nan
                                         : (smooth_length == 1 ? pbo
                                                               : wma_value(signal_window, smooth_length));
    }
}
