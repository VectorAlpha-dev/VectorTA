#include <cfloat>
#include <cmath>
#include <cstdint>

extern "C" __global__ void nonlinear_regression_zero_lag_moving_average_batch_f64(
    const double* data,
    int len,
    const int* zlma_periods,
    const int* regression_periods,
    int rows,
    int max_zlma_period,
    int max_regression_period,
    double* first_wma_rings,
    double* second_wma_rings,
    double* regression_rings,
    double* out_value,
    double* out_signal,
    double* out_long_signal,
    double* out_short_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int zlma_period = zlma_periods[row];
    int regression_period = regression_periods[row];
    if (zlma_period <= 0 || regression_period <= 0) {
        return;
    }

    const double nan = NAN;
    double* first_ring = first_wma_rings + static_cast<size_t>(row) * static_cast<size_t>(max_zlma_period);
    double* second_ring = second_wma_rings + static_cast<size_t>(row) * static_cast<size_t>(max_zlma_period);
    double* regression_ring =
        regression_rings + static_cast<size_t>(row) * static_cast<size_t>(max_regression_period);
    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_signal = out_long_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_signal = out_short_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    double zlma_period_f = static_cast<double>(zlma_period);
    double wma_denominator = zlma_period_f * static_cast<double>(zlma_period + 1) * 0.5;

    double regression_period_f = static_cast<double>(regression_period);
    double sx = 0.0;
    double sx2 = 0.0;
    double sx3 = 0.0;
    double sx4 = 0.0;
    for (int x = 0; x < regression_period; ++x) {
        double xf = static_cast<double>(x);
        double x2 = xf * xf;
        sx += xf;
        sx2 += x2;
        sx3 += x2 * xf;
        sx4 += x2 * x2;
    }
    double avg_x = sx / regression_period_f;
    double avg_x2 = avg_x * avg_x;
    double sxx = sx2 - regression_period_f * avg_x2;
    double sxx2 = sx3 - avg_x * sx2 - avg_x2 * sx + regression_period_f * avg_x2 * avg_x;
    double sx2x2 = sx4 - 2.0 * avg_x2 * sx2 + regression_period_f * avg_x2 * avg_x2;
    double denom = sxx * sx2x2 - sxx2 * sxx2;

    int first_head = 0;
    int first_count = 0;
    double first_sum = 0.0;
    double first_weighted_sum = 0.0;

    int second_head = 0;
    int second_count = 0;
    double second_sum = 0.0;
    double second_weighted_sum = 0.0;

    int regression_head = 0;
    int regression_count = 0;
    double sy = 0.0;
    double sxy = 0.0;
    double sx2y = 0.0;

    double prev_value = 0.0;
    double prev_signal = 0.0;
    bool has_prev_value = false;
    bool has_prev_signal = false;

    for (int i = 0; i < len; ++i) {
        row_value[i] = nan;
        row_signal[i] = nan;
        row_long_signal[i] = 0.0;
        row_short_signal[i] = 0.0;

        double value = data[i];
        if (isnan(value)) {
            first_head = 0;
            first_count = 0;
            first_sum = 0.0;
            first_weighted_sum = 0.0;
            second_head = 0;
            second_count = 0;
            second_sum = 0.0;
            second_weighted_sum = 0.0;
            regression_head = 0;
            regression_count = 0;
            sy = 0.0;
            sxy = 0.0;
            sx2y = 0.0;
            has_prev_value = false;
            has_prev_signal = false;
            continue;
        }

        double first_value = 0.0;
        bool first_ready = false;
        if (first_count < zlma_period) {
            int index = (first_head + first_count) % zlma_period;
            first_ring[index] = value;
            first_count += 1;
            first_sum += value;
            first_weighted_sum += static_cast<double>(first_count) * value;
            if (first_count == zlma_period) {
                first_value = first_weighted_sum / wma_denominator;
                first_ready = true;
            }
        } else {
            double oldest = first_ring[first_head];
            double old_sum = first_sum;
            first_weighted_sum = first_weighted_sum - old_sum + zlma_period_f * value;
            first_sum = old_sum - oldest + value;
            first_ring[first_head] = value;
            first_head += 1;
            if (first_head == zlma_period) {
                first_head = 0;
            }
            first_value = first_weighted_sum / wma_denominator;
            first_ready = true;
        }
        if (!first_ready) {
            continue;
        }

        double second_value = 0.0;
        bool second_ready = false;
        if (second_count < zlma_period) {
            int index = (second_head + second_count) % zlma_period;
            second_ring[index] = first_value;
            second_count += 1;
            second_sum += first_value;
            second_weighted_sum += static_cast<double>(second_count) * first_value;
            if (second_count == zlma_period) {
                second_value = second_weighted_sum / wma_denominator;
                second_ready = true;
            }
        } else {
            double oldest = second_ring[second_head];
            double old_sum = second_sum;
            second_weighted_sum = second_weighted_sum - old_sum + zlma_period_f * first_value;
            second_sum = old_sum - oldest + first_value;
            second_ring[second_head] = first_value;
            second_head += 1;
            if (second_head == zlma_period) {
                second_head = 0;
            }
            second_value = second_weighted_sum / wma_denominator;
            second_ready = true;
        }
        if (!second_ready) {
            continue;
        }

        double zl_value = 2.0 * first_value - second_value;
        double reg_value = 0.0;
        bool regression_ready = false;
        if (regression_count < regression_period) {
            int index = (regression_head + regression_count) % regression_period;
            regression_ring[index] = zl_value;
            regression_count += 1;
            if (regression_count == regression_period) {
                sy = 0.0;
                sxy = 0.0;
                sx2y = 0.0;
                for (int k = 0; k < regression_period; ++k) {
                    int idx = (regression_head + regression_period - 1 - k) % regression_period;
                    double ring_value = regression_ring[idx];
                    double kf = static_cast<double>(k);
                    sy += ring_value;
                    sxy += kf * ring_value;
                    sx2y += kf * kf * ring_value;
                }
                regression_ready = true;
            }
        } else {
            double oldest = regression_ring[regression_head];
            double old_sy = sy;
            double old_sxy = sxy;
            double carry = old_sy - oldest;

            regression_ring[regression_head] = zl_value;
            regression_head += 1;
            if (regression_head == regression_period) {
                regression_head = 0;
            }

            sy = old_sy - oldest + zl_value;
            sxy = old_sxy + carry;
            sx2y = sx2y + 2.0 * old_sxy + carry;
            regression_ready = true;
        }
        if (!regression_ready) {
            continue;
        }

        double avg_y = sy / regression_period_f;
        if (fabs(denom) <= DBL_EPSILON) {
            reg_value = avg_y;
        } else {
            double sxy_centered = sxy - avg_x * sy;
            double syx2 = sx2y - avg_y * sx2;
            double b = (sxy_centered * sx2x2 - syx2 * sxx2) / denom;
            double c = (sxx * syx2 - sxx2 * sxy_centered) / denom;
            double a = avg_y - b * avg_x - c * avg_x2;
            reg_value = a + c;
        }

        double signal_value = has_prev_value ? prev_value : nan;
        double long_signal = 0.0;
        double short_signal = 0.0;
        if (has_prev_value && has_prev_signal) {
            if (reg_value > signal_value && prev_value <= prev_signal) {
                long_signal = 1.0;
            }
            if (reg_value < signal_value && prev_value >= prev_signal) {
                short_signal = 1.0;
            }
        }

        prev_signal = signal_value;
        prev_value = reg_value;
        has_prev_signal = true;
        has_prev_value = true;

        row_value[i] = reg_value;
        row_signal[i] = signal_value;
        row_long_signal[i] = long_signal;
        row_short_signal[i] = short_signal;
    }
}
