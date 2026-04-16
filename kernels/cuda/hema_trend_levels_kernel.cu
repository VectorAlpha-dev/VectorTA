#include <cmath>
#include <cstddef>

namespace {

constexpr int ATR_PERIOD = 14;
constexpr double ATR_SCALE = 0.5;

struct EmaState {
    double alpha;
    double value;
    bool initialized;

    __device__ void init(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        reset();
    }

    __device__ void reset() {
        value = 0.0;
        initialized = false;
    }

    __device__ double update(double input) {
        if (!initialized) {
            value = input;
            initialized = true;
        } else {
            value += alpha * (input - value);
        }
        return value;
    }
};

struct HemaState {
    EmaState ema_half;
    EmaState ema_full;
    EmaState ema_diff;

    __device__ void init(int length) {
        const int half_length =
            max(static_cast<int>(llround(static_cast<double>(length) / 2.0)), 1);
        const int sqrt_length =
            max(static_cast<int>(llround(sqrt(static_cast<double>(length)))), 1);
        ema_half.init(half_length);
        ema_full.init(length);
        ema_diff.init(sqrt_length);
    }

    __device__ void reset() {
        ema_half.reset();
        ema_full.reset();
        ema_diff.reset();
    }

    __device__ double update(double input) {
        const double half = ema_half.update(input);
        const double full = ema_full.update(input);
        return ema_diff.update(2.0 * half - full);
    }
};

struct AtrState {
    double prev_close;
    double sum;
    double value;
    int count;
    bool initialized;

    __device__ void init() {
        reset();
    }

    __device__ void reset() {
        prev_close = NAN;
        sum = 0.0;
        value = 0.0;
        count = 0;
        initialized = false;
    }

    __device__ bool update(double high, double low, double close, double* out) {
        const double tr = isfinite(prev_close)
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;

        if (!initialized) {
            sum += tr;
            count += 1;
            if (count >= ATR_PERIOD) {
                value = sum / static_cast<double>(ATR_PERIOD);
                initialized = true;
                *out = value;
                return true;
            }
            return false;
        }

        value = ((static_cast<double>(ATR_PERIOD - 1) * value) + tr) / static_cast<double>(ATR_PERIOD);
        *out = value;
        return true;
    }
};

struct BoxState {
    double top;
    double bottom;
    bool valid;

    __device__ void reset() {
        top = NAN;
        bottom = NAN;
        valid = false;
    }
};

__device__ inline bool finite_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

}

extern "C" __global__ void hema_trend_levels_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ fast_lengths,
    const int* __restrict__ slow_lengths,
    int rows,
    double* __restrict__ out_fast_hema,
    double* __restrict__ out_slow_hema,
    double* __restrict__ out_trend_direction,
    double* __restrict__ out_bar_state,
    double* __restrict__ out_bullish_crossover,
    double* __restrict__ out_bearish_crossunder,
    double* __restrict__ out_box_offset,
    double* __restrict__ out_bull_box_top,
    double* __restrict__ out_bull_box_bottom,
    double* __restrict__ out_bear_box_top,
    double* __restrict__ out_bear_box_bottom,
    double* __restrict__ out_bullish_test,
    double* __restrict__ out_bearish_test,
    double* __restrict__ out_bullish_test_level,
    double* __restrict__ out_bearish_test_level
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int fast_length = fast_lengths[row];
    const int slow_length = slow_lengths[row];

    double* row_fast_hema =
        out_fast_hema + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_slow_hema =
        out_slow_hema + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend_direction =
        out_trend_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bar_state =
        out_bar_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_crossover =
        out_bullish_crossover + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_crossunder =
        out_bearish_crossunder + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_box_offset =
        out_box_offset + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bull_box_top =
        out_bull_box_top + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bull_box_bottom =
        out_bull_box_bottom + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear_box_top =
        out_bear_box_top + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bear_box_bottom =
        out_bear_box_bottom + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_test =
        out_bullish_test + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_test =
        out_bearish_test + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_test_level =
        out_bullish_test_level + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_test_level =
        out_bearish_test_level + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_fast_hema[i] = NAN;
        row_slow_hema[i] = NAN;
        row_trend_direction[i] = NAN;
        row_bar_state[i] = NAN;
        row_bullish_crossover[i] = NAN;
        row_bearish_crossunder[i] = NAN;
        row_box_offset[i] = NAN;
        row_bull_box_top[i] = NAN;
        row_bull_box_bottom[i] = NAN;
        row_bear_box_top[i] = NAN;
        row_bear_box_bottom[i] = NAN;
        row_bullish_test[i] = NAN;
        row_bearish_test[i] = NAN;
        row_bullish_test_level[i] = NAN;
        row_bearish_test_level[i] = NAN;
    }

    if (fast_length <= 0 || slow_length <= 0) {
        return;
    }

    HemaState fast_state;
    HemaState slow_state;
    AtrState atr_state;
    BoxState bull_box;
    BoxState bear_box;
    fast_state.init(fast_length);
    slow_state.init(slow_length);
    atr_state.init();
    bull_box.reset();
    bear_box.reset();

    double prev_fast = NAN;
    double prev_slow = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!finite_ohlc(o, h, l, c)) {
            fast_state.reset();
            slow_state.reset();
            atr_state.reset();
            bull_box.reset();
            bear_box.reset();
            prev_fast = NAN;
            prev_slow = NAN;
            continue;
        }

        const double fast_hema = fast_state.update(c);
        const double slow_hema = slow_state.update(c);
        double atr = NAN;
        const bool atr_ready = atr_state.update(h, l, c, &atr);
        const double box_offset = atr_ready ? atr * ATR_SCALE : NAN;

        const bool bullish_crossover =
            isfinite(prev_fast) && isfinite(prev_slow) && prev_fast <= prev_slow && fast_hema > slow_hema;
        const bool bearish_crossunder =
            isfinite(prev_fast) && isfinite(prev_slow) && prev_fast >= prev_slow && fast_hema < slow_hema;

        if (bullish_crossover && isfinite(box_offset)) {
            bull_box.top = l + box_offset;
            bull_box.bottom = l;
            bull_box.valid = true;
        } else if (bearish_crossunder && isfinite(box_offset)) {
            bear_box.top = h - box_offset;
            bear_box.bottom = h;
            bear_box.valid = true;
        }

        const double trend_direction =
            fast_hema > slow_hema ? 1.0 : (fast_hema < slow_hema ? -1.0 : 0.0);
        const bool bullish_condition = c > fast_hema && fast_hema > slow_hema;
        const bool bearish_condition = c < fast_hema && fast_hema < slow_hema;
        const double bar_state = bullish_condition ? 1.0 : (bearish_condition ? -1.0 : 0.0);

        double bullish_test = 0.0;
        double bearish_test = 0.0;
        double bullish_test_level = NAN;
        double bearish_test_level = NAN;
        if (bull_box.valid &&
            l < bull_box.top &&
            h > bull_box.top &&
            o > bull_box.top &&
            c > bull_box.top) {
            bullish_test = 1.0;
            bullish_test_level = bull_box.bottom;
        }
        if (bear_box.valid &&
            h > bear_box.top &&
            l < bear_box.top &&
            o < bear_box.top &&
            c < bear_box.top) {
            bearish_test = 1.0;
            bearish_test_level = bear_box.bottom;
        }

        row_fast_hema[i] = fast_hema;
        row_slow_hema[i] = slow_hema;
        row_trend_direction[i] = trend_direction;
        row_bar_state[i] = bar_state;
        row_bullish_crossover[i] = bullish_crossover ? 1.0 : 0.0;
        row_bearish_crossunder[i] = bearish_crossunder ? 1.0 : 0.0;
        row_box_offset[i] = box_offset;
        row_bull_box_top[i] = bull_box.valid ? bull_box.top : NAN;
        row_bull_box_bottom[i] = bull_box.valid ? bull_box.bottom : NAN;
        row_bear_box_top[i] = bear_box.valid ? bear_box.top : NAN;
        row_bear_box_bottom[i] = bear_box.valid ? bear_box.bottom : NAN;
        row_bullish_test[i] = bullish_test;
        row_bearish_test[i] = bearish_test;
        row_bullish_test_level[i] = bullish_test_level;
        row_bearish_test_level[i] = bearish_test_level;

        prev_fast = fast_hema;
        prev_slow = slow_hema;
    }
}
