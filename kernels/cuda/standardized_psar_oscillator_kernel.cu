#include <cmath>
#include <cstddef>

namespace {
constexpr double REVERSAL_LEVEL = 600.0;
constexpr double REVERSAL_MARKER = 900.0;
constexpr int MAX_PIVOT_BARS = 80;

struct EmaState {
    int period;
    double alpha;
    double beta;
    int count;
    double mean;

    __device__ void init(int value) {
        period = value;
        alpha = 2.0 / (static_cast<double>(value) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
    }

    __device__ double update(double value, bool* ready) {
        count += 1;
        if (count == 1) {
            mean = value;
        } else if (count <= period) {
            const double inv = 1.0 / static_cast<double>(count);
            mean = fma(inv, value - mean, mean);
        } else {
            mean = fma(alpha, value, beta * mean);
        }
        *ready = count >= period;
        return *ready ? mean : NAN;
    }
};

struct WmaState {
    double* buffer;
    int period;
    int head;
    int count;
    double sum;
    double weighted_sum;
    double denominator;

    __device__ void init(double* scratch, int value) {
        buffer = scratch;
        period = value;
        denominator = static_cast<double>(value * (value + 1) / 2);
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        weighted_sum = 0.0;
    }

    __device__ double update(double value, bool* ready) {
        if (count < period) {
            count += 1;
            buffer[head] = value;
            head += 1;
            if (head == period) {
                head = 0;
            }
            sum += value;
            weighted_sum += value * static_cast<double>(count);
            *ready = count == period;
            return *ready ? (weighted_sum / denominator) : NAN;
        }

        const double oldest = buffer[head];
        const double old_sum = sum;
        weighted_sum = weighted_sum - old_sum + value * static_cast<double>(period);
        sum = old_sum - oldest + value;
        buffer[head] = value;
        head += 1;
        if (head == period) {
            head = 0;
        }
        *ready = true;
        return weighted_sum / denominator;
    }
};

struct PsarTrendState {
    bool trend_up;
    double sar;
    double ep;
    double acc;
    double prev_high;
    double prev_high2;
    double prev_low;
    double prev_low2;
};

struct PsarState {
    double start;
    double increment;
    double maximum;
    bool initialized;
    int idx;
    PsarTrendState state;

    __device__ void init(double start_value, double increment_value, double maximum_value) {
        start = start_value;
        increment = increment_value;
        maximum = maximum_value;
        reset();
    }

    __device__ void reset() {
        initialized = false;
        idx = 0;
    }

    __device__ double update(double high, double low, bool* ready) {
        if (!initialized) {
            state.trend_up = false;
            state.sar = NAN;
            state.ep = NAN;
            state.acc = start;
            state.prev_high = high;
            state.prev_high2 = high;
            state.prev_low = low;
            state.prev_low2 = low;
            initialized = true;
            idx = 1;
            *ready = false;
            return NAN;
        }

        if (idx == 1) {
            const bool trend_up = high > state.prev_high;
            const double sar = trend_up ? state.prev_low : state.prev_high;
            const double ep = trend_up ? high : low;

            state.prev_high2 = state.prev_high;
            state.prev_low2 = state.prev_low;
            state.prev_high = high;
            state.prev_low = low;
            state.trend_up = trend_up;
            state.sar = sar;
            state.ep = ep;
            state.acc = start;
            idx = 2;
            *ready = true;
            return sar;
        }

        double next_sar = fma(state.acc, state.ep - state.sar, state.sar);
        if (state.trend_up) {
            if (low < next_sar) {
                state.trend_up = false;
                next_sar = state.ep;
                state.ep = low;
                state.acc = start;
            } else {
                if (high > state.ep) {
                    state.ep = high;
                    state.acc = fmin(state.acc + increment, maximum);
                }
                next_sar = fmin(next_sar, fmin(state.prev_low, state.prev_low2));
            }
        } else if (high > next_sar) {
            state.trend_up = true;
            next_sar = state.ep;
            state.ep = high;
            state.acc = start;
        } else {
            if (low < state.ep) {
                state.ep = low;
                state.acc = fmin(state.acc + increment, maximum);
            }
            next_sar = fmax(next_sar, fmax(state.prev_high, state.prev_high2));
        }

        state.prev_high2 = state.prev_high;
        state.prev_low2 = state.prev_low;
        state.prev_high = high;
        state.prev_low = low;
        state.sar = next_sar;
        idx += 1;
        *ready = true;
        return next_sar;
    }
};
}

extern "C" __global__ void standardized_psar_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const double* starts,
    const double* increments,
    const double* maximums,
    const int* standardization_lengths,
    const int* wma_lengths,
    const int* wma_lags,
    const int* pivot_lefts,
    const int* pivot_rights,
    int plot_bullish,
    int plot_bearish,
    int rows,
    int max_wma_length,
    double* out_oscillator,
    double* out_ma,
    double* out_bullish_reversal,
    double* out_bearish_reversal,
    double* out_regular_bullish,
    double* out_regular_bearish,
    double* out_bullish_weakening,
    double* out_bearish_weakening,
    double* wma_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const double start = starts[row];
    const double increment = increments[row];
    const double maximum = maximums[row];
    const int standardization_length = standardization_lengths[row];
    const int wma_length = wma_lengths[row];
    const int wma_lag = wma_lags[row];
    const int pivot_left = pivot_lefts[row];
    const int pivot_right = pivot_rights[row];

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_reversal =
        out_bullish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_reversal =
        out_bearish_reversal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_regular_bullish =
        out_regular_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_regular_bearish =
        out_regular_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_weakening =
        out_bullish_weakening + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_weakening =
        out_bearish_weakening + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_wma_buffer =
        wma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_wma_length);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = NAN;
        row_ma[i] = NAN;
        row_bullish_reversal[i] = NAN;
        row_bearish_reversal[i] = NAN;
        row_regular_bullish[i] = NAN;
        row_regular_bearish[i] = NAN;
        row_bullish_weakening[i] = NAN;
        row_bearish_weakening[i] = NAN;
    }

    if (!isfinite(start) || start <= 0.0 || !isfinite(increment) || increment <= 0.0
        || !isfinite(maximum) || maximum <= 0.0 || maximum < start || standardization_length <= 0
        || standardization_length > len || wma_length <= 0 || wma_length > len
        || wma_length > max_wma_length || wma_lag < 0 || wma_lag > len || pivot_left <= 0
        || pivot_left > len || pivot_right < 0 || pivot_right > len) {
        return;
    }

    PsarState psar;
    psar.init(start, increment, maximum);

    EmaState range_ema;
    range_ema.init(standardization_length);

    WmaState wma;
    wma.init(row_wma_buffer, wma_length);

    bool have_previous_low_pivot = false;
    bool have_previous_high_pivot = false;
    int previous_low_confirm_index = 0;
    int previous_high_confirm_index = 0;
    double previous_low_oscillator = NAN;
    double previous_high_oscillator = NAN;
    double previous_low_price = NAN;
    double previous_high_price = NAN;
    double previous_oscillator = NAN;
    int segment_count = 0;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            psar.reset();
            range_ema.reset();
            wma.reset();
            have_previous_low_pivot = false;
            have_previous_high_pivot = false;
            previous_oscillator = NAN;
            segment_count = 0;
            continue;
        }

        bool psar_ready = false;
        const double psar_value = psar.update(h, l, &psar_ready);

        bool range_ready = false;
        const double ema_range = range_ema.update(h - l, &range_ready);

        double oscillator = NAN;
        if (psar_ready && range_ready && isfinite(ema_range) && ema_range != 0.0) {
            oscillator = (c - psar_value) / ema_range * 100.0;
        }

        double ma = NAN;
        if (isfinite(oscillator)) {
            bool wma_ready = false;
            ma = wma.update(oscillator, &wma_ready);
            if (!wma_ready) {
                ma = NAN;
            }
        }

        double bearish_reversal = NAN;
        if (isfinite(previous_oscillator) && isfinite(oscillator) && previous_oscillator >= REVERSAL_LEVEL
            && oscillator < REVERSAL_LEVEL) {
            bearish_reversal = REVERSAL_MARKER;
        }

        double bullish_reversal = NAN;
        if (isfinite(previous_oscillator) && isfinite(oscillator) && previous_oscillator <= -REVERSAL_LEVEL
            && oscillator > -REVERSAL_LEVEL) {
            bullish_reversal = -REVERSAL_MARKER;
        }

        double lag_ma = NAN;
        if (wma_lag != 0 && segment_count >= wma_lag) {
            lag_ma = row_ma[i - wma_lag];
        }

        double bullish_weakening = NAN;
        if (isfinite(ma) && isfinite(lag_ma)) {
            bullish_weakening = (oscillator > 0.0 && ma < lag_ma) ? 1.0 : 0.0;
        }

        double bearish_weakening = NAN;
        if (isfinite(ma) && isfinite(lag_ma)) {
            bearish_weakening = (oscillator < 0.0 && ma > lag_ma) ? 1.0 : 0.0;
        }

        row_oscillator[i] = oscillator;
        row_ma[i] = ma;
        previous_oscillator = oscillator;

        double regular_bullish = NAN;
        double regular_bearish = NAN;

        const int current_segment_index = segment_count;
        const int needed = pivot_left + pivot_right + 1;
        if (current_segment_index + 1 >= needed) {
            const int center_actual = i - pivot_right;
            const int start_actual = center_actual - pivot_left;
            const int end_actual = center_actual + pivot_right;
            const double center_oscillator = row_oscillator[center_actual];

            if (isfinite(center_oscillator)) {
                bool pivot_low = true;
                bool pivot_high = true;
                for (int idx = start_actual; idx <= end_actual; ++idx) {
                    const double value = row_oscillator[idx];
                    if (!isfinite(value)) {
                        pivot_low = false;
                        pivot_high = false;
                        break;
                    }
                    if (idx != center_actual) {
                        if (value < center_oscillator) {
                            pivot_low = false;
                        }
                        if (value > center_oscillator) {
                            pivot_high = false;
                        }
                    }
                    if (!pivot_low && !pivot_high) {
                        break;
                    }
                }

                const int confirm_index = current_segment_index;

                if (pivot_low) {
                    const double price = low[center_actual];
                    if (plot_bullish != 0 && have_previous_low_pivot) {
                        const int bars = confirm_index - previous_low_confirm_index;
                        if (bars >= 1 && bars <= MAX_PIVOT_BARS
                            && center_oscillator > previous_low_oscillator
                            && price < previous_low_price) {
                            regular_bullish = center_oscillator;
                        }
                    }
                    have_previous_low_pivot = true;
                    previous_low_confirm_index = confirm_index;
                    previous_low_oscillator = center_oscillator;
                    previous_low_price = price;
                }

                if (pivot_high) {
                    const double price = high[center_actual];
                    if (plot_bearish != 0 && have_previous_high_pivot) {
                        const int bars = confirm_index - previous_high_confirm_index;
                        if (bars >= 1 && bars <= MAX_PIVOT_BARS
                            && center_oscillator < previous_high_oscillator
                            && price > previous_high_price) {
                            regular_bearish = center_oscillator;
                        }
                    }
                    have_previous_high_pivot = true;
                    previous_high_confirm_index = confirm_index;
                    previous_high_oscillator = center_oscillator;
                    previous_high_price = price;
                }
            }
        }

        if (isfinite(oscillator)) {
            row_bullish_reversal[i] = bullish_reversal;
            row_bearish_reversal[i] = bearish_reversal;
            row_regular_bullish[i] = regular_bullish;
            row_regular_bearish[i] = regular_bearish;
            row_bullish_weakening[i] = bullish_weakening;
            row_bearish_weakening[i] = bearish_weakening;
        }

        segment_count += 1;
    }
}
