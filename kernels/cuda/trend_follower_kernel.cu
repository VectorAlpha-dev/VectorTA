#include <cmath>
#include <cstddef>

namespace {
constexpr int CHANNEL_WINDOW = 280;
constexpr int MATYPE_EMA = 0;
constexpr int MATYPE_SMA = 1;
constexpr int MATYPE_RMA = 2;
constexpr int MATYPE_WMA = 3;
constexpr int MATYPE_VWMA = 4;

struct EmaState {
    int period;
    int valid_count;
    bool has_value;
    double value;
    double alpha;
    double beta;

    __device__ inline void init(int len) {
        period = len;
        valid_count = 0;
        has_value = false;
        value = NAN;
        alpha = 2.0 / (static_cast<double>(len) + 1.0);
        beta = 1.0 - alpha;
    }

    __device__ inline double update(double input) {
        if (!has_value) {
            valid_count = 1;
            has_value = true;
            value = input;
            return value;
        }
        if (valid_count < period) {
            valid_count += 1;
            const double vc = static_cast<double>(valid_count);
            value = ((vc - 1.0) * value + input) / vc;
            return value;
        }
        value = beta * value + alpha * input;
        return value;
    }
};

struct RmaState {
    int period;
    int seed_count;
    bool has_value;
    double seed_sum;
    double value;

    __device__ inline void init(int len) {
        period = len;
        seed_count = 0;
        has_value = false;
        seed_sum = 0.0;
        value = NAN;
    }

    __device__ inline double update(double input) {
        if (has_value) {
            value = value + (input - value) / static_cast<double>(period);
            return value;
        }
        seed_sum += input;
        seed_count += 1;
        if (seed_count == period) {
            value = seed_sum / static_cast<double>(period);
            has_value = true;
            return value;
        }
        return NAN;
    }
};
}

extern "C" __global__ void trend_follower_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    int len,
    const int* trend_periods,
    const int* ma_periods,
    const double* channel_rate_fractions,
    const int* linear_regression_periods,
    const int* ma_type_ids,
    int use_linear_regression,
    int rows,
    double* out_values,
    double* base_ma_history,
    double* ma_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int trend_period = trend_periods[row];
    const int ma_period = ma_periods[row];
    const double channel_rate_fraction = channel_rate_fractions[row];
    const int linear_regression_period = linear_regression_periods[row];
    const int ma_type = ma_type_ids[row];

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_base_ma_history =
        base_ma_history + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma_history = ma_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
        row_base_ma_history[i] = NAN;
        row_ma_history[i] = NAN;
    }

    if (trend_period < 1 || ma_period <= 0 || !isfinite(channel_rate_fraction)
        || channel_rate_fraction <= 0.0 || (use_linear_regression != 0 && linear_regression_period < 2)
        || ma_type < MATYPE_EMA || ma_type > MATYPE_VWMA) {
        return;
    }

    EmaState ema_state;
    RmaState rma_state;
    ema_state.init(ma_period);
    rma_state.init(ma_period);

    int segment_start = 0;
    for (int i = 0; i < len; ++i) {
        const bool needs_volume = ma_type == MATYPE_VWMA;
        if (!(isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i]))
            || (needs_volume && !isfinite(volume[i]))) {
            segment_start = i + 1;
            ema_state.init(ma_period);
            rma_state.init(ma_period);
            continue;
        }

        const int bars_in_segment = i - segment_start + 1;
        double base_ma = NAN;

        if (ma_type == MATYPE_EMA) {
            base_ma = ema_state.update(close[i]);
        } else if (ma_type == MATYPE_RMA) {
            base_ma = rma_state.update(close[i]);
        } else if (ma_type == MATYPE_SMA) {
            if (bars_in_segment >= ma_period) {
                double sum = 0.0;
                for (int j = i - ma_period + 1; j <= i; ++j) {
                    sum += close[j];
                }
                base_ma = sum / static_cast<double>(ma_period);
            }
        } else if (ma_type == MATYPE_WMA) {
            if (bars_in_segment >= ma_period) {
                double weighted_sum = 0.0;
                double weight_sum = 0.0;
                int weight = 1;
                for (int j = i - ma_period + 1; j <= i; ++j, ++weight) {
                    weighted_sum += close[j] * static_cast<double>(weight);
                    weight_sum += static_cast<double>(weight);
                }
                base_ma = weighted_sum / weight_sum;
            }
        } else if (ma_type == MATYPE_VWMA) {
            if (bars_in_segment >= ma_period) {
                double sum_pv = 0.0;
                double sum_v = 0.0;
                for (int j = i - ma_period + 1; j <= i; ++j) {
                    sum_pv += close[j] * volume[j];
                    sum_v += volume[j];
                }
                if (sum_v != 0.0) {
                    base_ma = sum_pv / sum_v;
                }
            }
        }

        row_base_ma_history[i] = base_ma;

        double ma_value = base_ma;
        if (use_linear_regression != 0) {
            ma_value = NAN;
            if (isfinite(base_ma) && bars_in_segment >= linear_regression_period) {
                const int start = i - linear_regression_period + 1;
                double y_sum = 0.0;
                double xy_sum = 0.0;
                bool all_finite = true;
                int x = 1;
                for (int j = start; j <= i; ++j, ++x) {
                    const double y = row_base_ma_history[j];
                    if (!isfinite(y)) {
                        all_finite = false;
                        break;
                    }
                    y_sum += y;
                    xy_sum += y * static_cast<double>(x);
                }
                if (all_finite) {
                    const double pf = static_cast<double>(linear_regression_period);
                    const double x_sum = pf * (pf + 1.0) * 0.5;
                    const double x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
                    const double denom = pf * x2_sum - x_sum * x_sum;
                    if (denom != 0.0) {
                        const double b = (pf * xy_sum - x_sum * y_sum) / denom;
                        const double a = (y_sum - b * x_sum) / pf;
                        ma_value = a + b * pf;
                    }
                }
            }
        }

        row_ma_history[i] = ma_value;
        if (!isfinite(ma_value)) {
            continue;
        }

        const int channel_start =
            (i - CHANNEL_WINDOW + 1 > segment_start) ? (i - CHANNEL_WINDOW + 1) : segment_start;
        double channel_high = high[channel_start];
        double channel_low = low[channel_start];
        for (int j = channel_start + 1; j <= i; ++j) {
            channel_high = fmax(channel_high, high[j]);
            channel_low = fmin(channel_low, low[j]);
        }

        const int ma_start = (i - trend_period + 1 > segment_start) ? (i - trend_period + 1)
                                                                     : segment_start;
        bool have_ma = false;
        double hh = NAN;
        double ll = NAN;
        for (int j = ma_start; j <= i; ++j) {
            const double value = row_ma_history[j];
            if (!isfinite(value)) {
                continue;
            }
            if (!have_ma) {
                hh = value;
                ll = value;
                have_ma = true;
            } else {
                hh = fmax(hh, value);
                ll = fmin(ll, value);
            }
        }
        if (!have_ma) {
            continue;
        }

        const double chan = (channel_high - channel_low) * channel_rate_fraction;
        if (!isfinite(chan) || chan == 0.0) {
            continue;
        }

        const double diff = fabs(hh - ll);
        double trend = 0.0;
        if (diff > chan) {
            if (ma_value > ll + chan) {
                trend = 1.0;
            } else if (ma_value < hh - chan) {
                trend = -1.0;
            }
        }

        row_out[i] = trend * diff / chan;
    }
}
