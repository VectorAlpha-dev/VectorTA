#include <cmath>
#include <cstddef>

namespace {

constexpr int MODE_BOLLINGER = 0;
constexpr int MODE_DONCHIAN = 1;
constexpr double BOLLINGER_STD_MULTIPLIER = 2.0;

__device__ inline bool finite_quad(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct AtrState {
    int length;
    double alpha;
    double prev_close;
    double rma;
    double warm_sum;
    int warm_count;
    bool seeded;

    __device__ void init(int period) {
        length = period;
        alpha = period > 0 ? (1.0 / static_cast<double>(period)) : 0.0;
        reset();
    }

    __device__ void reset() {
        prev_close = NAN;
        rma = NAN;
        warm_sum = 0.0;
        warm_count = 0;
        seeded = false;
    }

    __device__ bool update(double high, double low, double close, double* out) {
        if (!isfinite(high) || !isfinite(low) || !isfinite(close) || length <= 0) {
            reset();
            return false;
        }

        const double tr = isnan(prev_close)
            ? (high - low)
            : (fmax(high, prev_close) - fmin(low, prev_close));
        prev_close = close;

        if (!seeded) {
            warm_sum += tr;
            warm_count += 1;
            if (warm_count == length) {
                rma = warm_sum * alpha;
                seeded = true;
                *out = rma;
                return true;
            }
            return false;
        }

        rma = alpha * (tr - rma) + rma;
        *out = rma;
        return true;
    }
};

struct RingWmaState {
    double* buffer;
    int cap;
    int period;
    int head;
    int len;
    double sum;
    double weighted_sum;
    double denom;

    __device__ void init(double* buffer_ptr, int cap_value, int period_value) {
        buffer = buffer_ptr;
        cap = period_value > 0 ? period_value : cap_value;
        period = period_value;
        head = 0;
        len = 0;
        sum = 0.0;
        weighted_sum = 0.0;
        denom = static_cast<double>(period * (period + 1) / 2);
    }

    __device__ void reset() {
        head = 0;
        len = 0;
        sum = 0.0;
        weighted_sum = 0.0;
    }

    __device__ bool update(double value, double* out) {
        if (!isfinite(value)) {
            reset();
            return false;
        }
        if (period <= 1) {
            *out = value;
            return true;
        }
        if (len < period) {
            const int pos = (head + len) % cap;
            buffer[pos] = value;
            len += 1;
            sum += value;
            weighted_sum += value * static_cast<double>(len);
            if (len == period) {
                *out = weighted_sum / denom;
                return true;
            }
            return false;
        }

        const double oldest = buffer[head];
        const double old_sum = sum;
        buffer[head] = value;
        head = (head + 1) % cap;
        sum = old_sum - oldest + value;
        weighted_sum = weighted_sum - old_sum + static_cast<double>(period) * value;
        *out = weighted_sum / denom;
        return true;
    }
};

struct BollingerState {
    double* buffer;
    int cap;
    int period;
    int head;
    int len;
    double sum;
    double sum_sq;
    double inv_n;

    __device__ void init(double* buffer_ptr, int cap_value, int period_value) {
        buffer = buffer_ptr;
        cap = period_value > 0 ? period_value : cap_value;
        period = period_value;
        head = 0;
        len = 0;
        sum = 0.0;
        sum_sq = 0.0;
        inv_n = period > 0 ? (1.0 / static_cast<double>(period)) : 0.0;
    }

    __device__ void reset() {
        head = 0;
        len = 0;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ bool update(double value, double* highs, double* lows, double* mid) {
        if (!isfinite(value)) {
            reset();
            return false;
        }
        if (period <= 1) {
            *highs = value;
            *lows = value;
            *mid = value;
            return true;
        }
        if (len < period) {
            const int pos = (head + len) % cap;
            buffer[pos] = value;
            len += 1;
            sum += value;
            sum_sq += value * value;
            if (len < period) {
                return false;
            }
        } else {
            const double old = buffer[head];
            buffer[head] = value;
            head = (head + 1) % cap;
            sum += value - old;
            sum_sq += value * value - old * old;
        }
        const double mean = sum * inv_n;
        const double var = fmax(sum_sq * inv_n - mean * mean, 0.0);
        const double stddev = var > 0.0 ? sqrt(var) : 0.0;
        *highs = mean + BOLLINGER_STD_MULTIPLIER * stddev;
        *lows = mean - BOLLINGER_STD_MULTIPLIER * stddev;
        *mid = mean;
        return true;
    }
};

struct DonchianState {
    double* buffer;
    int cap;
    int period;
    int head;
    int len;

    __device__ void init(double* buffer_ptr, int cap_value, int period_value) {
        buffer = buffer_ptr;
        cap = period_value > 0 ? period_value : cap_value;
        period = period_value;
        head = 0;
        len = 0;
    }

    __device__ void reset() {
        head = 0;
        len = 0;
    }

    __device__ bool update(double value, double* highs, double* lows, double* mid) {
        if (!isfinite(value)) {
            reset();
            return false;
        }
        if (period <= 1) {
            *highs = value;
            *lows = value;
            *mid = value;
            return true;
        }
        if (len < period) {
            const int pos = (head + len) % cap;
            buffer[pos] = value;
            len += 1;
            if (len < period) {
                return false;
            }
        } else {
            buffer[head] = value;
            head = (head + 1) % cap;
        }

        double hi = -INFINITY;
        double lo = INFINITY;
        for (int i = 0; i < period; ++i) {
            const double sample = buffer[(head + i) % cap];
            hi = fmax(hi, sample);
            lo = fmin(lo, sample);
        }
        *highs = hi;
        *lows = lo;
        *mid = 0.5 * (hi + lo);
        return true;
    }
};

}

extern "C" __global__ void candle_strength_oscillator_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ periods,
    const int* __restrict__ atr_lengths,
    int atr_enabled,
    int mode,
    int rows,
    int full_cap,
    int half_cap,
    int sqrt_cap,
    int level_cap,
    double* __restrict__ full_scratch,
    double* __restrict__ half_scratch,
    double* __restrict__ sqrt_scratch,
    double* __restrict__ level_scratch,
    double* __restrict__ out_strength,
    double* __restrict__ out_highs,
    double* __restrict__ out_lows,
    double* __restrict__ out_mid,
    double* __restrict__ out_long_signal,
    double* __restrict__ out_short_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int period = periods[row];
    const int atr_length = atr_lengths[row];

    double* row_strength = out_strength + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_highs = out_highs + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lows = out_lows + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_mid = out_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_signal =
        out_long_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_signal =
        out_short_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_strength[i] = NAN;
        row_highs[i] = NAN;
        row_lows[i] = NAN;
        row_mid[i] = NAN;
        row_long_signal[i] = 0.0;
        row_short_signal[i] = 0.0;
    }

    if (period <= 0 || (atr_enabled != 0 && atr_length <= 0)) {
        return;
    }
    if (mode != MODE_BOLLINGER && mode != MODE_DONCHIAN) {
        return;
    }

    const int half_period = period / 2 > 0 ? period / 2 : 1;
    const int sqrt_period = static_cast<int>(floor(sqrt(static_cast<double>(period)))) > 0
        ? static_cast<int>(floor(sqrt(static_cast<double>(period))))
        : 1;

    RingWmaState full_state;
    RingWmaState half_state;
    RingWmaState sqrt_state;
    full_state.init(
        full_scratch + static_cast<size_t>(row) * static_cast<size_t>(full_cap),
        full_cap,
        period
    );
    half_state.init(
        half_scratch + static_cast<size_t>(row) * static_cast<size_t>(half_cap),
        half_cap,
        half_period
    );
    sqrt_state.init(
        sqrt_scratch + static_cast<size_t>(row) * static_cast<size_t>(sqrt_cap),
        sqrt_cap,
        sqrt_period
    );

    BollingerState bollinger_state;
    DonchianState donchian_state;
    if (mode == MODE_BOLLINGER) {
        bollinger_state.init(
            level_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
            level_cap,
            period
        );
    } else {
        donchian_state.init(
            level_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
            level_cap,
            period
        );
    }

    AtrState atr_state;
    atr_state.init(atr_length);

    double prev_strength = NAN;
    double prev_mid = NAN;
    bool has_prev_levels = false;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!finite_quad(o, h, l, c)) {
            atr_state.reset();
            full_state.reset();
            half_state.reset();
            sqrt_state.reset();
            if (mode == MODE_BOLLINGER) {
                bollinger_state.reset();
            } else {
                donchian_state.reset();
            }
            prev_strength = NAN;
            prev_mid = NAN;
            has_prev_levels = false;
            continue;
        }

        double atr_factor = 1.0;
        if (atr_enabled != 0) {
            if (!atr_state.update(h, l, c, &atr_factor)) {
                continue;
            }
        }

        const double range = h - l;
        if (!isfinite(range) || fabs(range) <= DBL_EPSILON) {
            atr_state.reset();
            full_state.reset();
            half_state.reset();
            sqrt_state.reset();
            if (mode == MODE_BOLLINGER) {
                bollinger_state.reset();
            } else {
                donchian_state.reset();
            }
            prev_strength = NAN;
            prev_mid = NAN;
            has_prev_levels = false;
            continue;
        }

        const double body = fabs(c - o);
        const double sign = c > o ? 1.0 : -1.0;
        const double signed_score = sign * body / range * atr_factor * 100.0;

        double full_value = NAN;
        double half_value = NAN;
        double strength = NAN;
        const bool full_ready = full_state.update(signed_score, &full_value);
        const bool half_ready = half_state.update(signed_score, &half_value);
        if (!(full_ready && half_ready)) {
            continue;
        }
        if (!sqrt_state.update(2.0 * half_value - full_value, &strength)) {
            continue;
        }

        row_strength[i] = strength;

        double highs = NAN;
        double lows = NAN;
        double mid = NAN;
        const bool levels_ready = mode == MODE_BOLLINGER
            ? bollinger_state.update(strength, &highs, &lows, &mid)
            : donchian_state.update(strength, &highs, &lows, &mid);
        if (!levels_ready) {
            continue;
        }

        row_highs[i] = highs;
        row_lows[i] = lows;
        row_mid[i] = mid;

        if (has_prev_levels) {
            if (prev_strength <= prev_mid && strength > mid) {
                row_long_signal[i] = 1.0;
            }
            if (prev_strength >= prev_mid && strength < mid) {
                row_short_signal[i] = 1.0;
            }
        }

        prev_strength = strength;
        prev_mid = mid;
        has_prev_levels = true;
    }
}
