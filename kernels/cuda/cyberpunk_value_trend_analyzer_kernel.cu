#include <cmath>
#include <cstddef>

namespace {

constexpr int SMA13_WINDOW = 13;
constexpr int RANGE75_WINDOW = 75;
constexpr int RANGE75_QUEUE_CAP = RANGE75_WINDOW + 1;

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct RollingSumDevice {
    double buf[SMA13_WINDOW];
    int pos;
    int count;
    double sum;

    __device__ void reset() {
        pos = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ double push(double value, bool* ready) {
        if (count < SMA13_WINDOW) {
            buf[pos] = value;
            pos = (pos + 1) % SMA13_WINDOW;
            count += 1;
            sum += value;
            if (count == SMA13_WINDOW) {
                *ready = true;
                return sum / static_cast<double>(SMA13_WINDOW);
            }
            *ready = false;
            return NAN;
        }

        const double old = buf[pos];
        buf[pos] = value;
        pos = (pos + 1) % SMA13_WINDOW;
        sum += value - old;
        *ready = true;
        return sum / static_cast<double>(SMA13_WINDOW);
    }
};

struct MonotonicQueueDevice {
    int idx[RANGE75_QUEUE_CAP];
    double val[RANGE75_QUEUE_CAP];
    int head;
    int tail;

    __device__ void reset() {
        head = 0;
        tail = 0;
    }

    __device__ void compact() {
        if (head <= 0) {
            return;
        }
        const int size = tail - head;
        for (int i = 0; i < size; ++i) {
            idx[i] = idx[head + i];
            val[i] = val[head + i];
        }
        head = 0;
        tail = size;
    }

    __device__ void prepare_for_push() {
        if (tail >= RANGE75_QUEUE_CAP && head > 0) {
            compact();
        }
    }

    __device__ void push_min(int index, double value) {
        while (tail > head) {
            if (val[tail - 1] <= value) {
                break;
            }
            tail -= 1;
        }
        prepare_for_push();
        if (tail < RANGE75_QUEUE_CAP) {
            idx[tail] = index;
            val[tail] = value;
            tail += 1;
        }
    }

    __device__ void push_max(int index, double value) {
        while (tail > head) {
            if (val[tail - 1] >= value) {
                break;
            }
            tail -= 1;
        }
        prepare_for_push();
        if (tail < RANGE75_QUEUE_CAP) {
            idx[tail] = index;
            val[tail] = value;
            tail += 1;
        }
    }

    __device__ void prune(int min_index) {
        while (tail > head && idx[head] < min_index) {
            head += 1;
        }
        if (head == tail) {
            head = 0;
            tail = 0;
        }
    }

    __device__ double current() const {
        return tail > head ? val[head] : NAN;
    }
};

struct WeightedSmaDevice {
    double alpha;
    double value;
    bool has_value;

    __device__ void init(double next_alpha) {
        alpha = next_alpha;
        reset();
    }

    __device__ void reset() {
        value = NAN;
        has_value = false;
    }

    __device__ double update(double source) {
        if (!isfinite(source)) {
            reset();
            return NAN;
        }
        const double next = has_value ? (alpha * source + (1.0 - alpha) * value) : source;
        value = next;
        has_value = true;
        return next;
    }
};

}

extern "C" __global__ void cyberpunk_value_trend_analyzer_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ entry_levels,
    const int* __restrict__ exit_levels,
    int rows,
    double* __restrict__ out_value_trend,
    double* __restrict__ out_value_trend_lag,
    double* __restrict__ out_deviation_index,
    double* __restrict__ out_overbought_signal,
    double* __restrict__ out_buy_signal,
    double* __restrict__ out_sell_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int entry_level = entry_levels[row];
    const int exit_level = exit_levels[row];

    double* row_value_trend = out_value_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_value_trend_lag =
        out_value_trend_lag + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_deviation_index =
        out_deviation_index + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_overbought_signal =
        out_overbought_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_buy_signal = out_buy_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_sell_signal = out_sell_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value_trend[i] = NAN;
        row_value_trend_lag[i] = NAN;
        row_deviation_index[i] = NAN;
        row_overbought_signal[i] = NAN;
        row_buy_signal[i] = NAN;
        row_sell_signal[i] = NAN;
    }

    if (entry_level < 1 || entry_level > 100 || exit_level < 1 || exit_level > 100) {
        return;
    }

    RollingSumDevice sma13;
    MonotonicQueueDevice lowest75;
    MonotonicQueueDevice highest75;
    WeightedSmaDevice close_norm_sma;
    WeightedSmaDevice smooth5;
    sma13.reset();
    lowest75.reset();
    highest75.reset();
    close_norm_sma.init(1.0 / 20.0);
    smooth5.init(1.0 / 5.0);

    int valid_run = 0;
    double prev_value_trend = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!is_valid_ohlc(o, h, l, c)) {
            sma13.reset();
            lowest75.reset();
            highest75.reset();
            close_norm_sma.reset();
            smooth5.reset();
            valid_run = 0;
            prev_value_trend = NAN;
            continue;
        }

        valid_run += 1;

        bool sma_ready = false;
        const double avg13 = sma13.push(c, &sma_ready);
        lowest75.push_min(i, l);
        highest75.push_max(i, h);
        const int min_index = i >= RANGE75_WINDOW - 1 ? (i - (RANGE75_WINDOW - 1)) : 0;
        lowest75.prune(min_index);
        highest75.prune(min_index);

        if (isfinite(prev_value_trend)) {
            row_value_trend_lag[i] = prev_value_trend;
        }

        double current_value_trend = NAN;
        if (valid_run >= RANGE75_WINDOW) {
            const double range_low = lowest75.current();
            const double range_high = highest75.current();
            const double range = range_high - range_low;
            if (isfinite(range) && range > 0.0) {
                const double close_norm = (c - range_low) * 100.0 / range;
                const double close_norm_avg = close_norm_sma.update(close_norm);
                const double smooth = smooth5.update(close_norm_avg);
                if (isfinite(close_norm_avg) && isfinite(smooth)) {
                    current_value_trend = 3.0 * close_norm_avg - 2.0 * smooth;
                    row_value_trend[i] = current_value_trend;
                    row_buy_signal[i] = 0.0;
                    row_sell_signal[i] = 0.0;
                }
            } else {
                close_norm_sma.reset();
                smooth5.reset();
            }
        }

        if (sma_ready && isfinite(avg13) && avg13 != 0.0) {
            const double deviation_index = 100.0 - fabs(((c - avg13) / avg13) * 100.0);
            row_deviation_index[i] = deviation_index;
            if (isfinite(current_value_trend) && current_value_trend > deviation_index) {
                row_overbought_signal[i] = deviation_index;
            }
        }

        if (isfinite(current_value_trend) && isfinite(prev_value_trend)) {
            if (prev_value_trend <= static_cast<double>(entry_level) &&
                current_value_trend > static_cast<double>(entry_level)) {
                row_buy_signal[i] = 1.0;
            }
            if (prev_value_trend >= static_cast<double>(exit_level) &&
                current_value_trend < static_cast<double>(exit_level)) {
                row_sell_signal[i] = 1.0;
            }
        }

        prev_value_trend = current_value_trend;
    }
}
