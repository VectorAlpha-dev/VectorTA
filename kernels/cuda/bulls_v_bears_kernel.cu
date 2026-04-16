#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_WMA = 2;
constexpr int METHOD_NORMALIZED = 0;
constexpr int METHOD_RAW = 1;

__device__ inline bool finite3(double a, double b, double c) {
    return isfinite(a) && isfinite(b) && isfinite(c);
}
}

extern "C" __global__ void bulls_v_bears_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* periods,
    const int* normalized_bars_backs,
    const int* raw_rolling_periods,
    const double* raw_threshold_percentiles,
    const double* threshold_levels,
    int ma_type,
    int calculation_method,
    int rows,
    double* out_value,
    double* out_bull,
    double* out_bear,
    double* out_ma,
    double* out_upper,
    double* out_lower,
    double* out_bullish_signal,
    double* out_bearish_signal,
    double* out_zero_cross_up,
    double* out_zero_cross_down
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int period = periods[row];
    const int normalized_bars_back = normalized_bars_backs[row];
    const int raw_rolling_period = raw_rolling_periods[row];
    const double raw_threshold_percentile = raw_threshold_percentiles[row];
    const double threshold_level = threshold_levels[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bull = out_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear = out_bear + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_signal =
        out_bullish_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_signal =
        out_bearish_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_zero_cross_up =
        out_zero_cross_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_zero_cross_down =
        out_zero_cross_down + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_bull[i] = NAN;
        row_bear[i] = NAN;
        row_ma[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_bullish_signal[i] = NAN;
        row_bearish_signal[i] = NAN;
        row_zero_cross_up[i] = NAN;
        row_zero_cross_down[i] = NAN;
    }

    if (period <= 0 || normalized_bars_back <= 0 || raw_rolling_period <= 0
        || !isfinite(raw_threshold_percentile) || raw_threshold_percentile < 80.0
        || raw_threshold_percentile > 99.0 || !isfinite(threshold_level)
        || threshold_level < 0.0 || threshold_level > 100.0
        || (ma_type != MA_EMA && ma_type != MA_SMA && ma_type != MA_WMA)
        || (calculation_method != METHOD_NORMALIZED && calculation_method != METHOD_RAW)) {
        return;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double ema_prev = NAN;
    double prev_total = NAN;

    for (int i = 0; i < len; ++i) {
        const double c = close[i];

        if (ma_type == MA_EMA) {
            if (!isfinite(c)) {
                ema_prev = NAN;
                row_ma[i] = NAN;
            } else {
                ema_prev = isfinite(ema_prev) ? (ema_prev + ema_alpha * (c - ema_prev)) : c;
                row_ma[i] = ema_prev;
            }
        } else if (period <= i + 1) {
            bool full_valid = true;
            double sum = 0.0;
            double weighted = 0.0;
            const int start = i + 1 - period;
            for (int j = start; j <= i; ++j) {
                const double value = close[j];
                if (!isfinite(value)) {
                    full_valid = false;
                    break;
                }
                sum += value;
                if (ma_type == MA_WMA) {
                    weighted += value * static_cast<double>(j - start + 1);
                }
            }
            if (full_valid) {
                row_ma[i] = ma_type == MA_SMA
                    ? (sum / static_cast<double>(period))
                    : (weighted / static_cast<double>(period * (period + 1) / 2));
            }
        }

        if (finite3(high[i], low[i], row_ma[i])) {
            row_bull[i] = high[i] - row_ma[i];
            row_bear[i] = row_ma[i] - low[i];
        }

        if (calculation_method == METHOD_NORMALIZED) {
            row_upper[i] = threshold_level;
            row_lower[i] = -threshold_level;

            if (isfinite(row_bull[i]) && isfinite(row_bear[i])) {
                const int start = (i + 1 > normalized_bars_back) ? (i + 1 - normalized_bars_back) : 0;
                double bull_min = NAN;
                double bull_max = NAN;
                double bear_min = NAN;
                double bear_max = NAN;
                for (int j = start; j <= i; ++j) {
                    const double bull = row_bull[j];
                    const double bear = row_bear[j];
                    if (isfinite(bull)) {
                        bull_min = isfinite(bull_min) ? fmin(bull_min, bull) : bull;
                        bull_max = isfinite(bull_max) ? fmax(bull_max, bull) : bull;
                    }
                    if (isfinite(bear)) {
                        bear_min = isfinite(bear_min) ? fmin(bear_min, bear) : bear;
                        bear_max = isfinite(bear_max) ? fmax(bear_max, bear) : bear;
                    }
                }
                const double bull_range = bull_max - bull_min;
                const double bear_range = bear_max - bear_min;
                if (bull_range > 0.0 && bear_range > 0.0) {
                    const double norm_bull = ((row_bull[i] - bull_min) / bull_range - 0.5) * 100.0;
                    const double norm_bear = ((row_bear[i] - bear_min) / bear_range - 0.5) * 100.0;
                    row_value[i] = norm_bull - norm_bear;
                }
            }
        } else {
            if (isfinite(row_bull[i]) && isfinite(row_bear[i])) {
                row_value[i] = row_bull[i] - row_bear[i];
            }

            const int start = (i + 1 > raw_rolling_period) ? (i + 1 - raw_rolling_period) : 0;
            double lowest = NAN;
            double highest = NAN;
            for (int j = start; j <= i; ++j) {
                const double value = row_value[j];
                if (isfinite(value)) {
                    lowest = isfinite(lowest) ? fmin(lowest, value) : value;
                    highest = isfinite(highest) ? fmax(highest, value) : value;
                }
            }
            if (isfinite(lowest) && isfinite(highest)) {
                const double range = highest - lowest;
                row_upper[i] = lowest + range * (raw_threshold_percentile / 100.0);
                row_lower[i] = lowest + range * ((100.0 - raw_threshold_percentile) / 100.0);
            }
        }

        if (isfinite(row_value[i]) && isfinite(row_upper[i]) && isfinite(row_lower[i])) {
            row_bullish_signal[i] = row_value[i] > row_upper[i] ? 1.0 : 0.0;
            row_bearish_signal[i] = row_value[i] < row_lower[i] ? 1.0 : 0.0;
            row_zero_cross_up[i] =
                isfinite(prev_total) && row_value[i] > 0.0 && prev_total <= 0.0 ? 1.0 : 0.0;
            row_zero_cross_down[i] =
                isfinite(prev_total) && row_value[i] < 0.0 && prev_total >= 0.0 ? 1.0 : 0.0;
            prev_total = row_value[i];
        }
    }
}
