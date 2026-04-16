#include <cmath>
#include <cstddef>

namespace {

constexpr int DIFF_FAST_PERIOD = 12;
constexpr int DIFF_SLOW_PERIOD = 26;
constexpr int DEA_PERIOD = 9;
constexpr int LINE_LONG_PERIOD = 40;
constexpr int LINE_SHORT_START = 6;
constexpr int LINE_SHORT_END = 20;
constexpr int LINE_SHORT_COUNT = LINE_SHORT_END - LINE_SHORT_START + 1;
constexpr double LINE_SHORT_AVG_INV = 1.0 / static_cast<double>(LINE_SHORT_COUNT);
constexpr double MID_CLOSE_WEIGHT = 7.0;
constexpr double MID_DIVISOR = 10.0;
constexpr double MACD_SCALE = 2.0;
constexpr int LINE_SHORT_TOTAL = 195;

struct SeededEma {
    int period;
    double alpha;
    double beta;
    int count;
    double sum;
    double value;
    bool ready;

    __device__ void init(int period_value) {
        period = period_value;
        alpha = 2.0 / (static_cast<double>(period_value) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        ready = false;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            count += 1;
            sum += input;
            if (count == period) {
                value = sum / static_cast<double>(period);
                ready = true;
                *out = value;
                return true;
            }
            *out = NAN;
            return false;
        }

        value = input * alpha + beta * value;
        *out = value;
        return true;
    }
};

struct RollingSma {
    double* values;
    int period;
    int index;
    int count;
    double sum;

    __device__ void init(double* storage, int period_value) {
        values = storage;
        period = period_value;
        reset();
    }

    __device__ void reset() {
        index = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            values[index] = input;
            sum += input;
            index += 1;
            if (index == period) {
                index = 0;
            }
            count += 1;
            if (count == period) {
                *out = sum / static_cast<double>(period);
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = values[index];
        values[index] = input;
        sum += input - old;
        index += 1;
        if (index == period) {
            index = 0;
        }
        *out = sum / static_cast<double>(period);
        return true;
    }
};

struct CoreState {
    SeededEma ema_fast;
    SeededEma ema_slow;
    SeededEma ema_dea;
    RollingSma line_short[LINE_SHORT_COUNT];
    RollingSma line_long;
    double prev_diff;
    double prev_dea;

    __device__ void init(double* short_storage, double* long_storage) {
        ema_fast.init(DIFF_FAST_PERIOD);
        ema_slow.init(DIFF_SLOW_PERIOD);
        ema_dea.init(DEA_PERIOD);

        int offset = 0;
        for (int period = LINE_SHORT_START; period <= LINE_SHORT_END; ++period) {
            line_short[period - LINE_SHORT_START].init(short_storage + offset, period);
            offset += period;
        }
        line_long.init(long_storage, LINE_LONG_PERIOD);
        prev_diff = NAN;
        prev_dea = NAN;
    }

    __device__ void reset() {
        ema_fast.reset();
        ema_slow.reset();
        ema_dea.reset();
        for (int i = 0; i < LINE_SHORT_COUNT; ++i) {
            line_short[i].reset();
        }
        line_long.reset();
        prev_diff = NAN;
        prev_dea = NAN;
    }
};

__device__ inline bool valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

}

extern "C" __global__ void macd_wave_signal_pro_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int rows,
    double* __restrict__ out_diff,
    double* __restrict__ out_dea,
    double* __restrict__ out_macd_histogram,
    double* __restrict__ out_line_convergence,
    double* __restrict__ out_buy_signal,
    double* __restrict__ out_sell_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_diff = out_diff + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_dea = out_dea + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_macd = out_macd_histogram + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_line =
        out_line_convergence + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_buy = out_buy_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_sell = out_sell_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_diff[i] = NAN;
        row_dea[i] = NAN;
        row_macd[i] = NAN;
        row_line[i] = NAN;
        row_buy[i] = NAN;
        row_sell[i] = NAN;
    }

    double short_storage[LINE_SHORT_TOTAL];
    double long_storage[LINE_LONG_PERIOD];
    CoreState state;
    state.init(short_storage, long_storage);

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!valid_ohlc(o, h, l, c)) {
            state.reset();
            continue;
        }

        double diff_value = NAN;
        double dea_value = NAN;
        double macd_value = NAN;
        double line_value = NAN;
        double buy_value = NAN;
        double sell_value = NAN;

        double fast = NAN;
        double slow = NAN;
        const bool has_fast = state.ema_fast.update(c, &fast);
        const bool has_slow = state.ema_slow.update(c, &slow);
        if (has_fast && has_slow) {
            const double diff = fast - slow;
            diff_value = diff;

            double dea = NAN;
            if (state.ema_dea.update(diff, &dea)) {
                dea_value = dea;
                macd_value = MACD_SCALE * (diff - dea);
                if (isfinite(state.prev_diff) && isfinite(state.prev_dea)) {
                    buy_value = (diff > dea && state.prev_diff <= state.prev_dea) ? 1.0 : 0.0;
                    sell_value = (diff < dea && state.prev_diff >= state.prev_dea) ? 1.0 : 0.0;
                } else {
                    buy_value = 0.0;
                    sell_value = 0.0;
                }
            }

            state.prev_diff = diff;
            state.prev_dea = dea_value;
        }

        const double mid = (MID_CLOSE_WEIGHT * c + (o + h + l)) / MID_DIVISOR;
        double short_sum = 0.0;
        int short_ready = 0;
        for (int idx = 0; idx < LINE_SHORT_COUNT; ++idx) {
            double sma_value = NAN;
            if (state.line_short[idx].update(mid, &sma_value)) {
                short_sum += sma_value;
                short_ready += 1;
            }
        }
        double long_value = NAN;
        if (state.line_long.update(mid, &long_value)) {
            if (short_ready == LINE_SHORT_COUNT) {
                line_value = short_sum * LINE_SHORT_AVG_INV - long_value;
            }
        }

        row_diff[i] = diff_value;
        row_dea[i] = dea_value;
        row_macd[i] = macd_value;
        row_line[i] = line_value;
        row_buy[i] = buy_value;
        row_sell[i] = sell_value;
    }
}
