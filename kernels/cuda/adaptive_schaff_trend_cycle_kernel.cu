#include <cmath>
#include <cstddef>

namespace {

constexpr int HISTOGRAM_EMA_PERIOD = 9;
constexpr double SCALE_100 = 100.0;
constexpr double CENTER = 50.0;
constexpr double EPS = 1.0e-12;

__device__ inline bool valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close) && high >= low;
}

struct EmaStateDevice {
    double alpha;
    bool initialized;
    double value;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        initialized = false;
        value = NAN;
    }

    __device__ void reset() {
        initialized = false;
        value = NAN;
    }

    __device__ double update(double next) {
        if (!initialized) {
            value = next;
            initialized = true;
        } else {
            value += alpha * (next - value);
        }
        return value;
    }
};

struct RollingCorrelationTimeDevice {
    int period;
    double* values;
    int head;
    int count;
    double sum_x;
    double sum_x2;
    double sum_xy;
    double sum_y;
    double n_sum_y2_minus_sum_y_sq;

    __device__ void init(int period_value, double* values_storage) {
        period = period_value;
        values = values_storage;
        const double n = static_cast<double>(period);
        sum_y = n * (n - 1.0) * 0.5;
        const double sum_y2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
        n_sum_y2_minus_sum_y_sq = n * sum_y2 - sum_y * sum_y;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum_x = 0.0;
        sum_x2 = 0.0;
        sum_xy = 0.0;
    }

    __device__ double compute() const {
        if (period <= 1) {
            return 0.0;
        }

        const double n = static_cast<double>(period);
        const double numerator = n * sum_xy - sum_x * sum_y;
        const double denom_x = n * sum_x2 - sum_x * sum_x;
        if (denom_x <= EPS || n_sum_y2_minus_sum_y_sq <= EPS) {
            return 0.0;
        }
        const double corr = numerator / sqrt(denom_x * n_sum_y2_minus_sum_y_sq);
        return fmin(1.0, fmax(-1.0, corr));
    }

    __device__ bool update(double next, double* out_corr) {
        if (count < period) {
            const double idx = static_cast<double>(count);
            values[count] = next;
            count += 1;
            sum_x += next;
            sum_x2 += next * next;
            sum_xy += idx * next;
            if (count == period) {
                *out_corr = compute();
                return true;
            }
            return false;
        }

        const double old_sum_x = sum_x;
        const double old_first = values[head];
        values[head] = next;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum_x = old_sum_x - old_first + next;
        sum_x2 = sum_x2 - old_first * old_first + next * next;
        sum_xy = sum_xy - (old_sum_x - old_first) + (static_cast<double>(period) - 1.0) * next;
        *out_corr = compute();
        return true;
    }
};

__device__ inline void shift_left_pair(int* indices, double* values, int count) {
    for (int i = 1; i < count; ++i) {
        indices[i - 1] = indices[i];
        values[i - 1] = values[i];
    }
}

__device__ inline bool rolling_minmax_update(
    int period,
    int* next_index,
    int* min_indices,
    double* min_values,
    int* min_count,
    int* max_indices,
    double* max_values,
    int* max_count,
    double value,
    double* out_min,
    double* out_max
) {
    const int idx = *next_index;
    *next_index += 1;

    while (*min_count > 0) {
        if (min_values[*min_count - 1] <= value) {
            break;
        }
        *min_count -= 1;
    }
    min_indices[*min_count] = idx;
    min_values[*min_count] = value;
    *min_count += 1;

    while (*max_count > 0) {
        if (max_values[*max_count - 1] >= value) {
            break;
        }
        *max_count -= 1;
    }
    max_indices[*max_count] = idx;
    max_values[*max_count] = value;
    *max_count += 1;

    const int window_start = idx + 1 > period ? idx + 1 - period : 0;
    while (*min_count > 0 && min_indices[0] < window_start) {
        shift_left_pair(min_indices, min_values, *min_count);
        *min_count -= 1;
    }
    while (*max_count > 0 && max_indices[0] < window_start) {
        shift_left_pair(max_indices, max_values, *max_count);
        *max_count -= 1;
    }

    if (idx + 1 < period) {
        return false;
    }

    *out_min = *min_count > 0 ? min_values[0] : value;
    *out_max = *max_count > 0 ? max_values[0] : value;
    return true;
}

}

extern "C" __global__ void adaptive_schaff_trend_cycle_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ adaptive_lengths,
    const int* __restrict__ stc_lengths,
    const double* __restrict__ smoothing_factors,
    const int* __restrict__ fast_lengths,
    const int* __restrict__ slow_lengths,
    int rows,
    int adaptive_cap,
    int stc_cap,
    int queue_cap,
    double* __restrict__ corr_values_scratch,
    int* __restrict__ queue_idx_scratch,
    double* __restrict__ queue_val_scratch,
    double* __restrict__ out_stc,
    double* __restrict__ out_histogram
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int adaptive_length = adaptive_lengths[row];
    const int stc_length = stc_lengths[row];
    const double smoothing_factor = smoothing_factors[row];
    const int fast_length = fast_lengths[row];
    const int slow_length = slow_lengths[row];

    double* row_stc = out_stc + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_histogram = out_histogram + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_stc[i] = NAN;
        row_histogram[i] = NAN;
    }

    if (adaptive_length <= 0 || adaptive_length > adaptive_cap || stc_length <= 0 ||
        stc_length > stc_cap || !isfinite(smoothing_factor) || smoothing_factor <= 0.0 ||
        smoothing_factor > 1.0 || fast_length <= 0 || slow_length <= 0) {
        return;
    }

    double* corr_values = corr_values_scratch + static_cast<size_t>(row) * static_cast<size_t>(adaptive_cap);
    int* row_queue_idx = queue_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(queue_cap * 4);
    double* row_queue_val = queue_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(queue_cap * 4);

    int* macd_min_idx = row_queue_idx;
    int* macd_max_idx = row_queue_idx + queue_cap;
    int* smooth_min_idx = row_queue_idx + queue_cap * 2;
    int* smooth_max_idx = row_queue_idx + queue_cap * 3;

    double* macd_min_val = row_queue_val;
    double* macd_max_val = row_queue_val + queue_cap;
    double* smooth_min_val = row_queue_val + queue_cap * 2;
    double* smooth_max_val = row_queue_val + queue_cap * 3;

    RollingCorrelationTimeDevice correlation;
    correlation.init(adaptive_length, corr_values);

    EmaStateDevice range_ema;
    range_ema.init(slow_length);
    EmaStateDevice histogram_ema;
    histogram_ema.init(HISTOGRAM_EMA_PERIOD);

    double fast_alpha = 2.0 / (static_cast<double>(fast_length) + 1.0);
    double slow_alpha = 2.0 / (static_cast<double>(slow_length) + 1.0);
    double prev_close = NAN;
    double macd_prev1 = 0.0;
    double macd_prev2 = 0.0;
    double normalized_prev = 0.0;
    double smoothed_macd_prev = 0.0;
    bool smoothed_macd_initialized = false;
    double smoothed_normalized_prev = 0.0;
    double stc_prev = 0.0;
    bool stc_initialized = false;

    int macd_next_index = 0;
    int macd_min_count = 0;
    int macd_max_count = 0;
    int smooth_next_index = 0;
    int smooth_min_count = 0;
    int smooth_max_count = 0;

    for (int i = 0; i < len; ++i) {
        if (!valid_bar(high[i], low[i], close[i])) {
            correlation.reset();
            range_ema.reset();
            histogram_ema.reset();
            prev_close = NAN;
            macd_prev1 = 0.0;
            macd_prev2 = 0.0;
            normalized_prev = 0.0;
            smoothed_macd_prev = 0.0;
            smoothed_macd_initialized = false;
            smoothed_normalized_prev = 0.0;
            stc_prev = 0.0;
            stc_initialized = false;
            macd_next_index = 0;
            macd_min_count = 0;
            macd_max_count = 0;
            smooth_next_index = 0;
            smooth_min_count = 0;
            smooth_max_count = 0;
            continue;
        }

        const double range_ema_value = range_ema.update(high[i] - low[i]);
        double corr = NAN;
        const bool has_corr = correlation.update(close[i], &corr);
        const double prev_close_value = prev_close;
        prev_close = close[i];

        if (!has_corr) {
            continue;
        }

        const double delta = isfinite(prev_close_value) ? (close[i] - prev_close_value) : 0.0;
        const double r2 = 0.5 * corr * corr + 0.5;
        const double k = r2 * ((1.0 - fast_alpha) * (1.0 - slow_alpha)) +
            (1.0 - r2) * ((1.0 - fast_alpha) / (1.0 - slow_alpha));
        const double macd =
            delta * (fast_alpha - slow_alpha) + (2.0 - fast_alpha - slow_alpha) * macd_prev1 -
            k * macd_prev2;
        macd_prev2 = macd_prev1;
        macd_prev1 = macd;

        double histogram = NAN;
        if (fabs(range_ema_value) > EPS) {
            const double normalized_macd = macd / range_ema_value * SCALE_100;
            const double histogram_ema_value = histogram_ema.update(normalized_macd);
            histogram = (normalized_macd - histogram_ema_value) * 0.5;
        }
        row_histogram[i] = histogram;

        double macd_min = NAN;
        double macd_max = NAN;
        if (!rolling_minmax_update(
                stc_length,
                &macd_next_index,
                macd_min_idx,
                macd_min_val,
                &macd_min_count,
                macd_max_idx,
                macd_max_val,
                &macd_max_count,
                macd,
                &macd_min,
                &macd_max)) {
            continue;
        }

        const double macd_span = macd_max - macd_min;
        const double normalized =
            macd_span > EPS ? ((macd - macd_min) / macd_span * SCALE_100) : normalized_prev;
        normalized_prev = normalized;

        const double smoothed_macd = !smoothed_macd_initialized
            ? normalized
            : (smoothed_macd_prev + smoothing_factor * (normalized - smoothed_macd_prev));
        smoothed_macd_prev = smoothed_macd;
        smoothed_macd_initialized = true;

        double smoothed_min = NAN;
        double smoothed_max = NAN;
        if (!rolling_minmax_update(
                stc_length,
                &smooth_next_index,
                smooth_min_idx,
                smooth_min_val,
                &smooth_min_count,
                smooth_max_idx,
                smooth_max_val,
                &smooth_max_count,
                smoothed_macd,
                &smoothed_min,
                &smoothed_max)) {
            continue;
        }

        const double smoothed_span = smoothed_max - smoothed_min;
        const double smoothed_normalized = smoothed_span > EPS
            ? ((smoothed_macd - smoothed_min) / smoothed_span * SCALE_100)
            : smoothed_normalized_prev;
        smoothed_normalized_prev = smoothed_normalized;

        const double stc_raw = !stc_initialized
            ? smoothed_normalized
            : (stc_prev + smoothing_factor * (smoothed_normalized - stc_prev));
        stc_prev = stc_raw;
        stc_initialized = true;

        row_stc[i] = stc_raw - CENTER;
    }
}
