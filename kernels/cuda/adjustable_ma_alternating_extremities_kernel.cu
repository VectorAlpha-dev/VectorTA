#include <cmath>
#include <cstddef>

namespace {

constexpr double TWO_PI = 6.28318530717958647692528676655900577;
constexpr double WEIGHT_SUM_EPS = 1e-12;

__device__ inline bool valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool pine_cross(double prev_a, double prev_b, double curr_a, double curr_b) {
    if (!(isfinite(prev_a) && isfinite(prev_b) && isfinite(curr_a) && isfinite(curr_b))) {
        return false;
    }
    return (curr_a > curr_b && prev_a <= prev_b) || (curr_a < curr_b && prev_a >= prev_b);
}

__device__ inline double raw_weight(int index, int length, double alpha, double beta) {
    const double denom = static_cast<double>(length - 1);
    const double x = static_cast<double>(index) / denom;
    return sin(TWO_PI * pow(x, alpha)) * (1.0 - pow(x, beta));
}

}

extern "C" __global__ void adjustable_ma_alternating_extremities_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ mults,
    const double* __restrict__ alphas,
    const double* __restrict__ betas,
    int rows,
    double* __restrict__ out_ma,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower,
    double* __restrict__ out_extremity,
    double* __restrict__ out_state,
    double* __restrict__ out_changed,
    double* __restrict__ out_smoothed_open,
    double* __restrict__ out_smoothed_high,
    double* __restrict__ out_smoothed_low,
    double* __restrict__ out_smoothed_close
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];
    const double alpha = alphas[row];
    const double beta = betas[row];

    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extremity = out_extremity + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_open =
        out_smoothed_open + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_high =
        out_smoothed_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_low =
        out_smoothed_low + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_smoothed_close =
        out_smoothed_close + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_ma[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_extremity[i] = NAN;
        row_state[i] = NAN;
        row_changed[i] = NAN;
        row_smoothed_open[i] = NAN;
        row_smoothed_high[i] = NAN;
        row_smoothed_low[i] = NAN;
        row_smoothed_close[i] = NAN;
    }

    if (length < 2 || length > len || !isfinite(mult) || mult < 1.0 || !isfinite(alpha) ||
        alpha < 0.0 || !isfinite(beta) || beta < 0.0) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (valid_bar(high[i], low[i], close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    const int needed = (length * 2) - 1;
    if (len - first < needed) {
        return;
    }

    double weight_sum = 0.0;
    for (int j = 0; j < length; ++j) {
        weight_sum += raw_weight(j, length, alpha, beta);
    }
    if (!isfinite(weight_sum) || fabs(weight_sum) <= WEIGHT_SUM_EPS) {
        return;
    }
    const double inv_weight_sum = 1.0 / weight_sum;

    const int ma_start = first + length - 1;
    for (int i = ma_start; i < len; ++i) {
        double ma_acc = 0.0;
        double high_acc = 0.0;
        double low_acc = 0.0;
        for (int j = 0; j < length; ++j) {
            const double w = raw_weight(j, length, alpha, beta) * inv_weight_sum;
            ma_acc += close[i - j] * w;
            high_acc += high[i - j] * w;
            low_acc += low[i - j] * w;
        }
        row_ma[i] = ma_acc;
        row_smoothed_close[i] = ma_acc;
        row_smoothed_high[i] = high_acc;
        row_smoothed_low[i] = low_acc;
    }

    const int open_start = ma_start + 2;
    for (int i = open_start; i < len; ++i) {
        row_smoothed_open[i] = 0.5 * (row_ma[i - 1] + row_ma[i - 2]);
    }

    const int band_start = first + (length * 2) - 2;
    double rolling = 0.0;
    for (int i = ma_start; i <= band_start; ++i) {
        rolling += fabs(close[i] - row_ma[i]);
    }
    const double first_dev = (rolling / static_cast<double>(length)) * mult;
    row_upper[band_start] = row_ma[band_start] + first_dev;
    row_lower[band_start] = row_ma[band_start] - first_dev;

    for (int i = band_start + 1; i < len; ++i) {
        rolling += fabs(close[i] - row_ma[i]);
        rolling -= fabs(close[i - length] - row_ma[i - length]);
        const double dev = (rolling / static_cast<double>(length)) * mult;
        row_upper[i] = row_ma[i] + dev;
        row_lower[i] = row_ma[i] - dev;
    }

    row_state[band_start] = 0.0;
    row_changed[band_start] = 0.0;
    row_extremity[band_start] = row_lower[band_start];

    for (int i = band_start + 1; i < len; ++i) {
        const double prev_state = row_state[i - 1];
        const bool cross_high = pine_cross(high[i - 1], row_upper[i - 1], high[i], row_upper[i]);
        const bool cross_low = pine_cross(low[i - 1], row_lower[i - 1], low[i], row_lower[i]);
        const double next_state = cross_high ? 1.0 : (cross_low ? 0.0 : prev_state);
        row_state[i] = next_state;
        row_changed[i] = fabs(next_state - prev_state) > 0.0 ? 1.0 : 0.0;
        row_extremity[i] = next_state >= 0.5 ? row_upper[i] : row_lower[i];
    }
}
