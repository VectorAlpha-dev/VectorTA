#include <cmath>
#include <cstddef>

static __device__ inline double lco_compute_correlation_from_sums(
    double sum_y,
    double sum_y2,
    double weighted_sum,
    int period
) {
    double period_f = static_cast<double>(period);
    double inv_period = 1.0 / period_f;
    double mean_x = 0.5 * (period_f + 1.0);
    double var_x = static_cast<double>(period * period - 1) / 12.0;
    if (!(var_x > 0.0) || !isfinite(var_x)) {
        return NAN;
    }

    double centered = weighted_sum - mean_x * sum_y;
    double mean_y = sum_y * inv_period;
    double var_y = sum_y2 * inv_period - mean_y * mean_y;
    if (var_y < 0.0 && var_y > -1e-12) {
        var_y = 0.0;
    }
    if (!(var_y > 0.0) || !isfinite(var_y)) {
        return NAN;
    }

    double denom = sqrt(var_y * var_x);
    if (!(denom > 0.0) || !isfinite(denom)) {
        return NAN;
    }

    double corr = centered * inv_period / denom;
    if (!isfinite(corr)) {
        return NAN;
    }
    if (corr > 1.0) {
        return 1.0;
    }
    if (corr < -1.0) {
        return -1.0;
    }
    return corr;
}

extern "C" __global__ void linear_correlation_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    int rows,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int period = periods[row];
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (period <= 0 || period > len) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (!isnan(data[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    int warm = first + period + 1;
    if (warm >= len) {
        return;
    }

    for (int end = warm; end < len; ++end) {
        int start = end + 1 - period;
        double sum_y = 0.0;
        double sum_y2 = 0.0;
        double weighted_sum = 0.0;
        bool has_nan = false;

        for (int offset = 0; offset < period; ++offset) {
            double value = data[start + offset];
            if (isnan(value)) {
                has_nan = true;
                break;
            }
            double weight = static_cast<double>(offset + 1);
            sum_y += value;
            sum_y2 += value * value;
            weighted_sum += weight * value;
        }

        if (!has_nan) {
            row_out[end] = lco_compute_correlation_from_sums(sum_y, sum_y2, weighted_sum, period);
        }
    }
}
