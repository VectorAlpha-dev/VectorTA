#include <cmath>
#include <cstddef>

namespace {

constexpr int SOURCE_OPEN = 0;
constexpr int SOURCE_HIGH = 1;
constexpr int SOURCE_LOW = 2;
constexpr int SOURCE_CLOSE = 3;
constexpr int SOURCE_HL2 = 4;
constexpr int SOURCE_HLC3 = 5;
constexpr int SOURCE_OHLC4 = 6;
constexpr int SOURCE_HLCC4 = 7;

constexpr int TP_LOW = 0;
constexpr int TP_MEDIUM = 1;
constexpr int TP_HIGH = 2;

constexpr double MULT1 = 0.618;
constexpr double MULT2 = 1.0;
constexpr double MULT3 = 1.618;
constexpr double MULT4 = 2.618;
constexpr double FLOAT_TOL = 1e-12;

__device__ inline bool source_needs_open(int source_mode) {
    return source_mode == SOURCE_OPEN || source_mode == SOURCE_OHLC4;
}

__device__ inline bool valid_bar(int source_mode, double open, double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close) &&
        (!source_needs_open(source_mode) || isfinite(open));
}

__device__ inline double source_value(
    int source_mode,
    double open,
    double high,
    double low,
    double close
) {
    switch (source_mode) {
        case SOURCE_OPEN:
            return open;
        case SOURCE_HIGH:
            return high;
        case SOURCE_LOW:
            return low;
        case SOURCE_CLOSE:
            return close;
        case SOURCE_HL2:
            return 0.5 * (high + low);
        case SOURCE_HLC3:
            return (high + low + close) / 3.0;
        case SOURCE_OHLC4:
            return 0.25 * (open + high + low + close);
        case SOURCE_HLCC4:
            return 0.25 * (high + low + close + close);
        default:
            return NAN;
    }
}

__device__ inline bool matches_true(double value) {
    return isfinite(value) && value > 0.5;
}

__device__ inline bool crossover(
    double current_a,
    double current_b,
    bool has_prev_a,
    double prev_a,
    bool has_prev_b,
    double prev_b
) {
    return isfinite(current_a) && isfinite(current_b) && current_a > current_b + FLOAT_TOL &&
        has_prev_a && has_prev_b && prev_a <= prev_b + FLOAT_TOL;
}

__device__ inline bool crossunder(
    double current_a,
    double current_b,
    bool has_prev_a,
    double prev_a,
    bool has_prev_b,
    double prev_b
) {
    return isfinite(current_a) && isfinite(current_b) && current_a < current_b - FLOAT_TOL &&
        has_prev_a && has_prev_b && prev_a >= prev_b - FLOAT_TOL;
}

struct RollingStdevState {
    double* buffer;
    int period;
    int head;
    int count;
    double sum;
    double sum_sq;

    __device__ void init(double* buffer_ptr, int period_value) {
        buffer = buffer_ptr;
        period = period_value;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ bool update(double value, double* out) {
        if (count == period) {
            const double old = buffer[head];
            sum -= old;
            sum_sq -= old * old;
        } else {
            count += 1;
        }
        buffer[head] = value;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += value;
        sum_sq += value * value;
        if (count < period) {
            return false;
        }
        const double mean = sum / static_cast<double>(period);
        *out = sqrt(fmax(sum_sq / static_cast<double>(period) - mean * mean, 0.0));
        return true;
    }
};

}

extern "C" __global__ void fibonacci_entry_bands_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ atr_lengths,
    int source_mode,
    int use_atr,
    int tp_mode,
    int rows,
    int stdev_cap,
    double* __restrict__ stdev_scratch,
    double* __restrict__ out_basis,
    double* __restrict__ out_trend,
    double* __restrict__ out_upper_0618,
    double* __restrict__ out_upper_1000,
    double* __restrict__ out_upper_1618,
    double* __restrict__ out_upper_2618,
    double* __restrict__ out_lower_0618,
    double* __restrict__ out_lower_1000,
    double* __restrict__ out_lower_1618,
    double* __restrict__ out_lower_2618,
    double* __restrict__ out_tp_long_band,
    double* __restrict__ out_tp_short_band,
    double* __restrict__ out_long_entry,
    double* __restrict__ out_short_entry,
    double* __restrict__ out_rejection_long,
    double* __restrict__ out_rejection_short,
    double* __restrict__ out_long_bounce,
    double* __restrict__ out_short_bounce
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_basis = out_basis + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_0618 = out_upper_0618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_1000 = out_upper_1000 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_1618 = out_upper_1618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_2618 = out_upper_2618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_0618 = out_lower_0618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_1000 = out_lower_1000 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_1618 = out_lower_1618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_2618 = out_lower_2618 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_tp_long_band =
        out_tp_long_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_tp_short_band =
        out_tp_short_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_entry =
        out_long_entry + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_entry =
        out_short_entry + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_rejection_long =
        out_rejection_long + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_rejection_short =
        out_rejection_short + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_bounce =
        out_long_bounce + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_bounce =
        out_short_bounce + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_basis[i] = NAN;
        row_trend[i] = NAN;
        row_upper_0618[i] = NAN;
        row_upper_1000[i] = NAN;
        row_upper_1618[i] = NAN;
        row_upper_2618[i] = NAN;
        row_lower_0618[i] = NAN;
        row_lower_1000[i] = NAN;
        row_lower_1618[i] = NAN;
        row_lower_2618[i] = NAN;
        row_tp_long_band[i] = NAN;
        row_tp_short_band[i] = NAN;
        row_long_entry[i] = NAN;
        row_short_entry[i] = NAN;
        row_rejection_long[i] = NAN;
        row_rejection_short[i] = NAN;
        row_long_bounce[i] = NAN;
        row_short_bounce[i] = NAN;
    }

    const int length = lengths[row];
    const int atr_length = atr_lengths[row];
    if (length <= 0 || atr_length <= 0 || stdev_cap < length || source_mode < 0 || source_mode > 7 ||
        tp_mode < 0 || tp_mode > 2) {
        return;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(length) + 1.0);
    RollingStdevState stdev_state;
    stdev_state.init(
        stdev_scratch + static_cast<size_t>(row) * static_cast<size_t>(stdev_cap),
        length
    );

    bool has_ema1 = false;
    double ema1 = NAN;
    bool has_ema2 = false;
    double ema2 = NAN;
    bool has_prev_basis = false;
    double prev_basis = NAN;
    bool has_prev_prev_basis = false;
    double prev_prev_basis = NAN;
    double trend = 0.0;
    bool has_prev_close = false;
    double prev_close = NAN;
    double atr_sum = 0.0;
    int atr_count = 0;
    bool atr_seeded = false;
    double atr_value = NAN;
    bool has_prev_tp_long_band = false;
    double prev_tp_long_band = NAN;
    bool has_prev_tp_short_band = false;
    double prev_tp_short_band = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!valid_bar(source_mode, o, h, l, c)) {
            has_ema1 = false;
            ema1 = NAN;
            has_ema2 = false;
            ema2 = NAN;
            has_prev_basis = false;
            prev_basis = NAN;
            has_prev_prev_basis = false;
            prev_prev_basis = NAN;
            trend = 0.0;
            has_prev_close = false;
            prev_close = NAN;
            atr_sum = 0.0;
            atr_count = 0;
            atr_seeded = false;
            atr_value = NAN;
            has_prev_tp_long_band = false;
            prev_tp_long_band = NAN;
            has_prev_tp_short_band = false;
            prev_tp_short_band = NAN;
            stdev_state.reset();
            continue;
        }

        const double source = source_value(source_mode, o, h, l, c);
        ema1 = has_ema1 ? (ema1 + ema_alpha * (source - ema1)) : source;
        has_ema1 = true;
        const double basis = has_ema2 ? (ema2 + ema_alpha * (ema1 - ema2)) : ema1;
        ema2 = basis;
        has_ema2 = true;

        double long_entry = NAN;
        double short_entry = NAN;
        if (has_prev_basis && has_prev_prev_basis) {
            const double curr_delta = basis - prev_basis;
            const double prev_delta = prev_basis - prev_prev_basis;
            long_entry = (curr_delta > FLOAT_TOL && prev_delta <= FLOAT_TOL) ? 1.0 : 0.0;
            short_entry = (curr_delta < -FLOAT_TOL && prev_delta >= -FLOAT_TOL) ? 1.0 : 0.0;
        }

        if (has_prev_basis) {
            if (basis > prev_basis + FLOAT_TOL) {
                trend = 1.0;
            } else if (basis < prev_basis - FLOAT_TOL) {
                trend = -1.0;
            }
        }

        double volatility = NAN;
        bool vol_ready = false;
        if (use_atr != 0) {
            const double tr = has_prev_close
                ? fmax(h - l, fmax(fabs(h - prev_close), fabs(l - prev_close)))
                : (h - l);
            if (!atr_seeded) {
                atr_count += 1;
                atr_sum += tr;
                if (atr_count == atr_length) {
                    atr_value = atr_sum / static_cast<double>(atr_length);
                    atr_seeded = true;
                    volatility = atr_value;
                    vol_ready = true;
                }
            } else {
                atr_value =
                    ((static_cast<double>(atr_length - 1) * atr_value) + tr) / static_cast<double>(atr_length);
                volatility = atr_value;
                vol_ready = true;
            }
        } else {
            vol_ready = stdev_state.update(source, &volatility);
        }

        row_basis[i] = basis;
        row_trend[i] = trend;
        row_long_entry[i] = long_entry;
        row_short_entry[i] = short_entry;
        row_long_bounce[i] = trend > 0.0 && l < basis - FLOAT_TOL && c > basis + FLOAT_TOL &&
                !matches_true(long_entry)
            ? 1.0
            : 0.0;
        row_short_bounce[i] = trend < 0.0 && h > basis + FLOAT_TOL && c < basis - FLOAT_TOL &&
                !matches_true(short_entry)
            ? 1.0
            : 0.0;

        double tp_long_band = NAN;
        double tp_short_band = NAN;
        if (vol_ready && isfinite(volatility)) {
            const double upper_0618 = basis + volatility * MULT1;
            const double upper_1000 = basis + volatility * MULT2;
            const double upper_1618 = basis + volatility * MULT3;
            const double upper_2618 = basis + volatility * MULT4;
            const double lower_0618 = basis - volatility * MULT1;
            const double lower_1000 = basis - volatility * MULT2;
            const double lower_1618 = basis - volatility * MULT3;
            const double lower_2618 = basis - volatility * MULT4;

            if (tp_mode == TP_LOW) {
                tp_long_band = lower_2618;
                tp_short_band = upper_2618;
            } else if (tp_mode == TP_MEDIUM) {
                tp_long_band = lower_1000;
                tp_short_band = upper_1000;
            } else {
                tp_long_band = lower_0618;
                tp_short_band = upper_0618;
            }

            row_upper_0618[i] = upper_0618;
            row_upper_1000[i] = upper_1000;
            row_upper_1618[i] = upper_1618;
            row_upper_2618[i] = upper_2618;
            row_lower_0618[i] = lower_0618;
            row_lower_1000[i] = lower_1000;
            row_lower_1618[i] = lower_1618;
            row_lower_2618[i] = lower_2618;
            row_tp_long_band[i] = tp_long_band;
            row_tp_short_band[i] = tp_short_band;
            row_rejection_long[i] = trend < 0.0 &&
                    crossunder(c, tp_long_band, has_prev_close, prev_close, has_prev_tp_long_band, prev_tp_long_band)
                ? 1.0
                : 0.0;
            row_rejection_short[i] = trend > 0.0 &&
                    crossover(c, tp_short_band, has_prev_close, prev_close, has_prev_tp_short_band, prev_tp_short_band)
                ? 1.0
                : 0.0;
        }

        prev_prev_basis = prev_basis;
        has_prev_prev_basis = has_prev_basis;
        prev_basis = basis;
        has_prev_basis = true;
        prev_close = c;
        has_prev_close = true;
        prev_tp_long_band = tp_long_band;
        has_prev_tp_long_band = isfinite(tp_long_band);
        prev_tp_short_band = tp_short_band;
        has_prev_tp_short_band = isfinite(tp_short_band);
    }
}
