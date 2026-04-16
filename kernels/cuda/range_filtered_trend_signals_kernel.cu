#include <cmath>
#include <cstddef>

namespace {

constexpr int WMA_PERIOD = 200;

struct KalmanState {
    double alpha_mul_period;
    double beta_div_period;
    double value;
    double covariance;
    bool has_value;

    __device__ void init(double alpha, int period, double beta) {
        alpha_mul_period = alpha * static_cast<double>(period);
        beta_div_period = beta / static_cast<double>(period);
        reset();
    }

    __device__ void reset() {
        value = NAN;
        covariance = 1.0;
        has_value = false;
    }

    __device__ bool update(double input, bool has_prev_input, double prev_input, double* out) {
        const double gain = covariance / (covariance + alpha_mul_period);
        if (!has_value && has_prev_input) {
            value = prev_input;
            has_value = true;
        }
        bool ready = false;
        if (has_value) {
            const double next = value + gain * (input - value);
            value = next;
            *out = next;
            ready = true;
        }
        covariance = (1.0 - gain) * covariance + beta_div_period;
        return ready;
    }
};

struct AtrState {
    int period;
    int count;
    double sum;
    double value;
    double prev_close;
    bool seeded;
    bool has_prev_close;

    __device__ void init(int period_value) {
        period = period_value;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        prev_close = NAN;
        seeded = false;
        has_prev_close = false;
    }

    __device__ bool update(double high, double low, double close, double* out) {
        const double tr = has_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        has_prev_close = true;

        if (seeded) {
            value = ((value * static_cast<double>(period - 1)) + tr) / static_cast<double>(period);
            *out = value;
            return true;
        }

        count += 1;
        sum += tr;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *out = value;
            return true;
        }
        return false;
    }
};

struct WmaState {
    double* buffer;
    int head;
    int len;
    double sum;
    double weighted_sum;
    double divisor;

    __device__ void init(double* buffer_ptr) {
        buffer = buffer_ptr;
        divisor = static_cast<double>(WMA_PERIOD * (WMA_PERIOD + 1) / 2);
        reset();
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
        if (len < WMA_PERIOD) {
            const int pos = (head + len) % WMA_PERIOD;
            buffer[pos] = value;
            len += 1;
            sum += value;
            weighted_sum += static_cast<double>(len) * value;
            if (len == WMA_PERIOD) {
                *out = weighted_sum / divisor;
                return true;
            }
            return false;
        }

        const double oldest = buffer[head];
        const double old_sum = sum;
        buffer[head] = value;
        head = (head + 1) % WMA_PERIOD;
        weighted_sum = weighted_sum - old_sum + static_cast<double>(WMA_PERIOD) * value;
        sum = old_sum - oldest + value;
        *out = weighted_sum / divisor;
        return true;
    }
};

struct SuperTrendState {
    double factor;
    double prev_lower_band;
    double prev_upper_band;
    double prev_k;
    bool has_prev_lower_band;
    bool has_prev_upper_band;
    bool has_prev_k;
    bool prev_atr_ready;
    int prev_direction;

    __device__ void init(double factor_value) {
        factor = factor_value;
        reset();
    }

    __device__ void reset() {
        prev_lower_band = NAN;
        prev_upper_band = NAN;
        prev_k = NAN;
        has_prev_lower_band = false;
        has_prev_upper_band = false;
        has_prev_k = false;
        prev_atr_ready = false;
        prev_direction = 1;
    }

    __device__ void update(double k, double atr, double* supertrend, int* direction) {
        double upper_band = k + factor * atr;
        double lower_band = k - factor * atr;
        const double prev_lower = has_prev_lower_band ? prev_lower_band : lower_band;
        const double prev_upper = has_prev_upper_band ? prev_upper_band : upper_band;
        const double prev_k_value = has_prev_k ? prev_k : k;

        if (!(lower_band > prev_lower || prev_k_value < prev_lower)) {
            lower_band = prev_lower;
        }
        if (!(upper_band < prev_upper || prev_k_value > prev_upper)) {
            upper_band = prev_upper;
        }

        if (!prev_atr_ready) {
            *direction = 1;
        } else if (prev_direction == 1) {
            *direction = k > upper_band ? -1 : 1;
        } else if (k < lower_band) {
            *direction = 1;
        } else {
            *direction = -1;
        }

        *supertrend = *direction == -1 ? lower_band : upper_band;
        prev_lower_band = lower_band;
        prev_upper_band = upper_band;
        prev_k = k;
        has_prev_lower_band = true;
        has_prev_upper_band = true;
        has_prev_k = true;
        prev_atr_ready = true;
        prev_direction = *direction;
    }
};

}

extern "C" __global__ void range_filtered_trend_signals_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ kalman_alphas,
    const double* __restrict__ kalman_betas,
    const int* __restrict__ kalman_periods,
    const double* __restrict__ devs,
    const double* __restrict__ supertrend_factors,
    const int* __restrict__ supertrend_atr_periods,
    int rows,
    double* __restrict__ wma_scratch,
    double* __restrict__ out_kalman,
    double* __restrict__ out_supertrend,
    double* __restrict__ out_upper_band,
    double* __restrict__ out_lower_band,
    double* __restrict__ out_trend,
    double* __restrict__ out_kalman_trend,
    double* __restrict__ out_state,
    double* __restrict__ out_market_trending,
    double* __restrict__ out_market_ranging,
    double* __restrict__ out_short_term_bullish,
    double* __restrict__ out_short_term_bearish,
    double* __restrict__ out_long_term_bullish,
    double* __restrict__ out_long_term_bearish
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_kalman = out_kalman + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_supertrend =
        out_supertrend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_band =
        out_upper_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_band =
        out_lower_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kalman_trend =
        out_kalman_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_market_trending =
        out_market_trending + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_market_ranging =
        out_market_ranging + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_term_bullish =
        out_short_term_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_term_bearish =
        out_short_term_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_term_bullish =
        out_long_term_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_long_term_bearish =
        out_long_term_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_kalman[i] = NAN;
        row_supertrend[i] = NAN;
        row_upper_band[i] = NAN;
        row_lower_band[i] = NAN;
        row_trend[i] = NAN;
        row_kalman_trend[i] = NAN;
        row_state[i] = NAN;
        row_market_trending[i] = NAN;
        row_market_ranging[i] = NAN;
        row_short_term_bullish[i] = NAN;
        row_short_term_bearish[i] = NAN;
        row_long_term_bullish[i] = NAN;
        row_long_term_bearish[i] = NAN;
    }

    const double kalman_alpha = kalman_alphas[row];
    const double kalman_beta = kalman_betas[row];
    const int kalman_period = kalman_periods[row];
    const double dev = devs[row];
    const double supertrend_factor = supertrend_factors[row];
    const int supertrend_atr_period = supertrend_atr_periods[row];
    if (!isfinite(kalman_alpha) || kalman_alpha <= 0.0 || !isfinite(kalman_beta) ||
        kalman_beta < 0.0 || kalman_period <= 0 || !isfinite(dev) || dev < 0.0 ||
        !isfinite(supertrend_factor) || supertrend_factor < 0.0 ||
        supertrend_atr_period <= 0) {
        return;
    }

    KalmanState kalman_state;
    kalman_state.init(kalman_alpha, kalman_period, kalman_beta);
    AtrState atr_state;
    atr_state.init(supertrend_atr_period);
    WmaState wma_state;
    wma_state.init(wma_scratch + static_cast<size_t>(row) * static_cast<size_t>(WMA_PERIOD));
    SuperTrendState supertrend_state;
    supertrend_state.init(supertrend_factor);

    bool has_prev_close = false;
    double prev_close = NAN;
    double trend_state = 0.0;
    bool has_prev_trend = false;
    double prev_trend = NAN;
    bool has_prev_kalman_trend = false;
    double prev_kalman_trend = NAN;
    bool has_prev_state = false;
    double prev_state = NAN;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            kalman_state.reset();
            atr_state.reset();
            wma_state.reset();
            supertrend_state.reset();
            has_prev_close = false;
            prev_close = NAN;
            trend_state = 0.0;
            has_prev_trend = false;
            prev_trend = NAN;
            has_prev_kalman_trend = false;
            prev_kalman_trend = NAN;
            has_prev_state = false;
            prev_state = NAN;
            continue;
        }

        double kalman_value = NAN;
        const bool kalman_ready =
            kalman_state.update(c, has_prev_close, prev_close, &kalman_value);
        prev_close = c;
        has_prev_close = true;

        double atr_value = NAN;
        const bool atr_ready = atr_state.update(h, l, c, &atr_value);

        double vola = NAN;
        const bool vola_ready = wma_state.update(h - l, &vola);

        double supertrend = NAN;
        int direction = 0;
        bool supertrend_ready = false;
        if (kalman_ready && atr_ready) {
            supertrend_state.update(kalman_value, atr_value, &supertrend, &direction);
            supertrend_ready = true;
        }

        if (!(kalman_ready && atr_ready && vola_ready && supertrend_ready)) {
            continue;
        }

        const double upper_band = kalman_value + vola * dev;
        const double lower_band = kalman_value - vola * dev;
        if (c > upper_band) {
            trend_state = 1.0;
        } else if (c < lower_band) {
            trend_state = -1.0;
        }

        const double kalman_trend = direction < 0 ? 1.0 : -1.0;
        const double state = kalman_trend * trend_state;
        const double market_trending =
            has_prev_state && state > 0.0 && prev_state <= 0.0 ? 1.0 : 0.0;
        const double market_ranging =
            has_prev_state && state < 0.0 && prev_state >= 0.0 ? 1.0 : 0.0;
        const double short_term_bullish =
            has_prev_trend && trend_state > 0.0 && prev_trend <= 0.0 ? 1.0 : 0.0;
        const double short_term_bearish =
            has_prev_trend && trend_state < 0.0 && prev_trend >= 0.0 ? 1.0 : 0.0;
        const double long_term_bullish =
            has_prev_kalman_trend && kalman_trend > 0.0 && prev_kalman_trend <= 0.0 ? 1.0 : 0.0;
        const double long_term_bearish =
            has_prev_kalman_trend && kalman_trend < 0.0 && prev_kalman_trend >= 0.0 ? 1.0 : 0.0;

        row_kalman[i] = kalman_value;
        row_supertrend[i] = supertrend;
        row_upper_band[i] = upper_band;
        row_lower_band[i] = lower_band;
        row_trend[i] = trend_state;
        row_kalman_trend[i] = kalman_trend;
        row_state[i] = state;
        row_market_trending[i] = market_trending;
        row_market_ranging[i] = market_ranging;
        row_short_term_bullish[i] = short_term_bullish;
        row_short_term_bearish[i] = short_term_bearish;
        row_long_term_bullish[i] = long_term_bullish;
        row_long_term_bearish[i] = long_term_bearish;

        prev_trend = trend_state;
        has_prev_trend = true;
        prev_kalman_trend = kalman_trend;
        has_prev_kalman_trend = true;
        prev_state = state;
        has_prev_state = true;
    }
}
