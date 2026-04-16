#include <cmath>
#include <cstddef>

namespace {

constexpr int ATR_PERIOD = 200;
constexpr int LOOKBACK_MODE_BAR_COUNT = 0;
constexpr int LOOKBACK_MODE_FVG_COUNT = 1;

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct LevelDeque {
    int* left;
    double* value;
    int head;
    int tail;
    int cap;
    double sum;

    __device__ void init(int* left_ptr, double* value_ptr, int cap_value) {
        left = left_ptr;
        value = value_ptr;
        cap = cap_value;
        reset();
    }

    __device__ void reset() {
        head = 0;
        tail = 0;
        sum = 0.0;
    }

    __device__ int len() const {
        return tail - head;
    }

    __device__ void compact() {
        if (head <= 0) {
            return;
        }
        const int size = tail - head;
        for (int i = 0; i < size; ++i) {
            left[i] = left[head + i];
            value[i] = value[head + i];
        }
        head = 0;
        tail = size;
    }

    __device__ void ensure_capacity() {
        if (tail >= cap && head > 0) {
            compact();
        }
    }

    __device__ void push_back(int left_index, double level_value) {
        ensure_capacity();
        if (tail < cap) {
            left[tail] = left_index;
            value[tail] = level_value;
            tail += 1;
            sum += level_value;
        }
    }

    __device__ void pop_front() {
        if (tail <= head) {
            return;
        }
        sum -= value[head];
        head += 1;
        if (head == tail) {
            head = 0;
            tail = 0;
        }
    }

    __device__ void push_count_mode(int left_index, double level_value, int lookback) {
        push_back(left_index, level_value);
        while (len() > lookback) {
            pop_front();
        }
    }

    __device__ void prune_bar_count(int current_idx, int lookback) {
        const int cutoff = current_idx > lookback ? (current_idx - lookback) : 0;
        while (tail > head && left[head] < cutoff) {
            pop_front();
        }
    }

    __device__ double average() const {
        const int size = tail - head;
        return size <= 0 ? NAN : (sum / static_cast<double>(size));
    }
};

}

extern "C" __global__ void fvg_positioning_average_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lookbacks,
    const double* __restrict__ atr_multipliers,
    int lookback_mode,
    int rows,
    int level_cap,
    int* __restrict__ bull_left_scratch,
    double* __restrict__ bull_value_scratch,
    int* __restrict__ bear_left_scratch,
    double* __restrict__ bear_value_scratch,
    double* __restrict__ out_bull_average,
    double* __restrict__ out_bear_average,
    double* __restrict__ out_bull_mid,
    double* __restrict__ out_bear_mid
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int lookback = lookbacks[row];
    const double atr_multiplier = atr_multipliers[row];

    double* row_bull_average =
        out_bull_average + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear_average =
        out_bear_average + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bull_mid = out_bull_mid + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear_mid = out_bear_mid + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_bull_average[i] = NAN;
        row_bear_average[i] = NAN;
        row_bull_mid[i] = NAN;
        row_bear_mid[i] = NAN;
    }

    if (lookback <= 0 || !isfinite(atr_multiplier) || atr_multiplier < 0.0) {
        return;
    }
    if (lookback_mode != LOOKBACK_MODE_BAR_COUNT && lookback_mode != LOOKBACK_MODE_FVG_COUNT) {
        return;
    }

    LevelDeque bull_levels;
    LevelDeque bear_levels;
    bull_levels.init(
        bull_left_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
        bull_value_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
        level_cap
    );
    bear_levels.init(
        bear_left_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
        bear_value_scratch + static_cast<size_t>(row) * static_cast<size_t>(level_cap),
        level_cap
    );

    int valid_count = 0;
    double cumulative_range = 0.0;
    double tr_sum = 0.0;
    double atr = NAN;
    double prev_close = 0.0;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!is_valid_ohlc(o, h, l, c)) {
            bull_levels.reset();
            bear_levels.reset();
            valid_count = 0;
            cumulative_range = 0.0;
            tr_sum = 0.0;
            atr = NAN;
            continue;
        }

        valid_count += 1;
        const double high_low = h - l;
        cumulative_range += high_low;
        const double tr = valid_count == 1
            ? high_low
            : fmax(high_low, fmax(fabs(h - prev_close), fabs(l - prev_close)));

        double threshold = NAN;
        if (valid_count < ATR_PERIOD) {
            tr_sum += tr;
            threshold = cumulative_range / static_cast<double>(valid_count);
        } else if (valid_count == ATR_PERIOD) {
            tr_sum += tr;
            const double seed = tr_sum / static_cast<double>(ATR_PERIOD);
            atr = seed;
            threshold = seed * atr_multiplier;
        } else {
            const double next = ((isnan(atr) ? tr : atr) * static_cast<double>(ATR_PERIOD - 1) + tr)
                / static_cast<double>(ATR_PERIOD);
            atr = next;
            threshold = next * atr_multiplier;
        }

        if (valid_count >= 3) {
            const int idx1 = i - 1;
            const int idx2 = i - 2;

            if (l > high[idx2] && close[idx1] > high[idx2] && (l - high[idx2]) > threshold) {
                if (lookback_mode == LOOKBACK_MODE_BAR_COUNT) {
                    bull_levels.push_back(idx2, high[idx2]);
                } else {
                    bull_levels.push_count_mode(idx2, high[idx2], lookback);
                }
            }

            if (h < low[idx2] && close[idx1] < low[idx2] && (low[idx2] - h) > threshold) {
                if (lookback_mode == LOOKBACK_MODE_BAR_COUNT) {
                    bear_levels.push_back(idx2, low[idx2]);
                } else {
                    bear_levels.push_count_mode(idx2, low[idx2], lookback);
                }
            }
        }

        if (lookback_mode == LOOKBACK_MODE_BAR_COUNT) {
            bull_levels.prune_bar_count(i, lookback);
            bear_levels.prune_bar_count(i, lookback);
        }

        const double bull_average = bull_levels.average();
        const double bear_average = bear_levels.average();
        const double body_mid = 0.5 * (o + c);
        row_bull_average[i] = bull_average;
        row_bear_average[i] = bear_average;
        row_bull_mid[i] = isnan(bull_average) ? NAN : fmax(body_mid, bull_average);
        row_bear_mid[i] = isnan(bear_average) ? NAN : fmin(body_mid, bear_average);
        prev_close = c;
    }
}
