#include <cmath>
#include <cstddef>

namespace {
constexpr double FLOAT_TOL = 1e-12;
constexpr int SOURCE_OPEN = 0;
constexpr int SOURCE_HIGH = 1;
constexpr int SOURCE_LOW = 2;
constexpr int SOURCE_CLOSE = 3;
constexpr int SOURCE_HL2 = 4;
constexpr int SOURCE_HLC3 = 5;
constexpr int SOURCE_OHLC4 = 6;
constexpr int SOURCE_HLCC4 = 7;

__device__ inline bool source_needs_open(int source_kind) {
    return source_kind == SOURCE_OPEN || source_kind == SOURCE_OHLC4;
}

__device__ inline double source_value(
    int source_kind,
    double open,
    double high,
    double low,
    double close
) {
    switch (source_kind) {
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
            return (open + high + low + close) * 0.25;
        case SOURCE_HLCC4:
            return (high + low + close + close) * 0.25;
        default:
            return NAN;
    }
}

__device__ inline bool valid_bar(
    int source_kind,
    double open,
    double high,
    double low,
    double close
) {
    if (!(isfinite(high) && isfinite(low) && isfinite(close))) {
        return false;
    }
    if (source_needs_open(source_kind) && !isfinite(open)) {
        return false;
    }
    return isfinite(source_value(source_kind, open, high, low, close));
}

__device__ inline double signum_with_tol(double value) {
    if (!isfinite(value) || fabs(value) <= FLOAT_TOL) {
        return 0.0;
    }
    return value > 0.0 ? 1.0 : -1.0;
}

struct RsiStreamState {
    int period;
    double inv_p;
    double beta;
    bool has_prev;
    double prev;
    int seed_count;
    double sum_gain;
    double sum_loss;
    bool poisoned;
    double avg_gain;
    double avg_loss;
    bool seeded;

    __device__ void init(int value) {
        period = value;
        inv_p = 1.0 / static_cast<double>(value);
        beta = 1.0 - inv_p;
        reset();
    }

    __device__ void reset() {
        has_prev = false;
        prev = NAN;
        seed_count = 0;
        sum_gain = 0.0;
        sum_loss = 0.0;
        poisoned = false;
        avg_gain = 0.0;
        avg_loss = 0.0;
        seeded = false;
    }

    __device__ double update(double value, bool* ready) {
        if (!has_prev) {
            prev = value;
            has_prev = true;
            *ready = false;
            return NAN;
        }

        const double delta = value - prev;
        prev = value;

        if (!seeded) {
            if (!isfinite(delta)) {
                poisoned = true;
            }
            const double gain = fmax(delta, 0.0);
            const double loss = fmax(-delta, 0.0);
            sum_gain += gain;
            sum_loss += loss;
            seed_count += 1;
            if (seed_count == period) {
                seeded = true;
                *ready = true;
                if (poisoned) {
                    avg_gain = NAN;
                    avg_loss = NAN;
                    return NAN;
                }
                avg_gain = sum_gain * inv_p;
                avg_loss = sum_loss * inv_p;
                const double denom = avg_gain + avg_loss;
                return denom == 0.0 ? 50.0 : (100.0 * avg_gain / denom);
            }
            *ready = false;
            return NAN;
        }

        const double gain = fmax(delta, 0.0);
        const double loss = fmax(-delta, 0.0);
        avg_gain = fma(beta, avg_gain, inv_p * gain);
        avg_loss = fma(beta, avg_loss, inv_p * loss);
        const double denom = avg_gain + avg_loss;
        *ready = true;
        return denom == 0.0 ? 50.0 : (100.0 * avg_gain / denom);
    }
};
}

extern "C" __global__ void grover_llorens_cycle_oscillator_batch_f64(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* lengths,
    const double* mults,
    int source_kind,
    int smooth,
    const int* rsi_periods,
    int rows,
    double* out_values
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];
    const int rsi_period = rsi_periods[row];

    double* row_values = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_values[i] = NAN;
    }

    if (length <= 0 || !isfinite(mult) || rsi_period <= 0
        || source_kind < SOURCE_OPEN || source_kind > SOURCE_HLCC4) {
        return;
    }

    const double ema_alpha = 2.0 / (static_cast<double>(rsi_period) + 1.0);

    RsiStreamState rsi;
    rsi.init(rsi_period);

    int segment_start = 0;
    int valid_src_count = 0;
    bool have_prev_src = false;
    double prev_src = NAN;
    bool have_prev_close = false;
    double prev_close = NAN;
    int atr_seed_count = 0;
    double atr_seed_sum = 0.0;
    bool have_atr = false;
    double atr_value = NAN;
    int os = 0;
    bool have_prev_diff = false;
    double prev_diff = NAN;
    bool have_prev_ts = false;
    double prev_ts = NAN;
    bool have_last_event_step = false;
    double last_event_step = NAN;
    int bars_since_event = 0;
    bool have_ema_value = false;
    double ema_value = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!valid_bar(source_kind, o, h, l, c)) {
            rsi.reset();
            segment_start = i + 1;
            valid_src_count = 0;
            have_prev_src = false;
            prev_src = NAN;
            have_prev_close = false;
            prev_close = NAN;
            atr_seed_count = 0;
            atr_seed_sum = 0.0;
            have_atr = false;
            atr_value = NAN;
            os = 0;
            have_prev_diff = false;
            prev_diff = NAN;
            have_prev_ts = false;
            prev_ts = NAN;
            have_last_event_step = false;
            last_event_step = NAN;
            bars_since_event = 0;
            have_ema_value = false;
            ema_value = NAN;
            continue;
        }

        const double src = source_value(source_kind, o, h, l, c);
        const double diff = have_prev_src ? (src - (have_prev_ts ? prev_ts : prev_src)) : NAN;

        const double tr = have_prev_close
            ? fmax(h - l, fmax(fabs(h - prev_close), fabs(l - prev_close)))
            : (h - l);
        prev_close = c;
        have_prev_close = true;
        if (atr_seed_count < length) {
            atr_seed_count += 1;
            atr_seed_sum += tr;
            if (atr_seed_count == length) {
                atr_value = atr_seed_sum / static_cast<double>(length);
                have_atr = true;
            }
        } else if (have_atr) {
            atr_value = ((static_cast<double>(length - 1) * atr_value) + tr) / static_cast<double>(length);
        }

        bool rising = false;
        bool falling = false;
        if (valid_src_count >= length) {
            double max_prev = open[0];
            double min_prev = open[0];
            bool first_prev = true;
            const int start = i - length;
            for (int j = start; j < i; ++j) {
                const double prior_src = source_value(source_kind, open[j], high[j], low[j], close[j]);
                if (first_prev) {
                    max_prev = prior_src;
                    min_prev = prior_src;
                    first_prev = false;
                } else {
                    max_prev = fmax(max_prev, prior_src);
                    min_prev = fmin(min_prev, prior_src);
                }
            }
            if (!first_prev) {
                rising = src > max_prev + FLOAT_TOL;
                falling = src < min_prev - FLOAT_TOL;
            }
        }

        const int prev_os = os;
        const int new_os = rising ? 1 : (falling ? -1 : prev_os);
        const double prev_diff_value = have_prev_diff ? prev_diff : NAN;
        const bool rise = (new_os - prev_os == 2) && isfinite(prev_diff_value) && prev_diff_value < 0.0;
        const bool fall = (new_os - prev_os == -2) && isfinite(prev_diff_value) && prev_diff_value > 0.0;
        const bool up =
            isfinite(prev_diff_value) && prev_diff_value <= 0.0 && isfinite(diff) && diff > 0.0;
        const bool dn =
            isfinite(prev_diff_value) && prev_diff_value >= 0.0 && isfinite(diff) && diff < 0.0;
        const bool event = up || dn || rise || fall;

        double ts = NAN;
        if (have_atr) {
            const double step = atr_value / static_cast<double>(length);
            if (event) {
                last_event_step = step;
                have_last_event_step = true;
                bars_since_event = 0;
            } else if (have_last_event_step) {
                bars_since_event += 1;
            }

            const double prev_ts_or_src = have_prev_ts ? prev_ts : src;
            if (up) {
                ts = prev_ts_or_src - atr_value * mult;
            } else if (dn) {
                ts = prev_ts_or_src + atr_value * mult;
            } else if (rise) {
                ts = src - atr_value * mult;
            } else if (fall) {
                ts = src + atr_value * mult;
            } else if (have_last_event_step) {
                ts = prev_ts_or_src
                    + signum_with_tol(diff) * last_event_step * static_cast<double>(bars_since_event);
            }
        }

        if (isfinite(ts)) {
            const double osc = src - ts;
            const double smoothed = smooth != 0
                ? (have_ema_value ? fma(ema_alpha, osc, (1.0 - ema_alpha) * ema_value) : osc)
                : osc;
            ema_value = smoothed;
            have_ema_value = true;

            bool rsi_ready = false;
            row_values[i] = rsi.update(smoothed, &rsi_ready);
            if (!rsi_ready) {
                row_values[i] = NAN;
            }
        }

        prev_src = src;
        have_prev_src = true;
        prev_diff = diff;
        have_prev_diff = isfinite(diff);
        prev_ts = ts;
        have_prev_ts = isfinite(ts);
        os = new_os;
        valid_src_count = i - segment_start + 1;
    }
}
