#include <cmath>
#include <cstddef>

namespace {
constexpr double DEFAULT_TREND = 1.0;
constexpr double MIN_MULTIPLIER = 0.1;
constexpr double MIN_ALPHA_PERCENT = 0.1;
constexpr double MAX_ALPHA_PERCENT = 100.0;

struct AtrState {
    int length;
    int count;
    double sum;
    double value;

    __device__ void init(int period) {
        length = period;
        reset();
    }

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
    }

    __device__ double update(double tr, bool* ready) {
        if (count < length) {
            count += 1;
            sum += tr;
            if (count == length) {
                value = sum / static_cast<double>(length);
                *ready = true;
                return value;
            }
            *ready = false;
            return NAN;
        }

        value = ((value * static_cast<double>(length - 1)) + tr) / static_cast<double>(length);
        *ready = true;
        return value;
    }
};

__device__ inline double true_range(double high, double low, double prev_close) {
    return fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)));
}
}

extern "C" __global__ void supertrend_recovery_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* atr_lengths,
    const double* multipliers,
    const double* alpha_percents,
    const double* threshold_atrs,
    int rows,
    double* out_band,
    double* out_switch_price,
    double* out_trend,
    double* out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int atr_length = atr_lengths[row];
    const double multiplier = multipliers[row];
    const double alpha_percent = alpha_percents[row];
    const double threshold_atr = threshold_atrs[row];

    double* row_band = out_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_switch_price =
        out_switch_price + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_band[i] = NAN;
        row_switch_price[i] = NAN;
        row_trend[i] = NAN;
        row_changed[i] = NAN;
    }

    if (atr_length <= 0 || atr_length > len || !isfinite(multiplier) || multiplier < MIN_MULTIPLIER
        || !isfinite(alpha_percent) || alpha_percent < MIN_ALPHA_PERCENT
        || alpha_percent > MAX_ALPHA_PERCENT || !isfinite(threshold_atr)
        || threshold_atr < 0.0) {
        return;
    }

    const double alpha = alpha_percent * 0.01;

    AtrState atr;
    atr.init(atr_length);

    double prev_close = NAN;
    double band = NAN;
    double switch_price = NAN;
    double trend = DEFAULT_TREND;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            atr.reset();
            prev_close = NAN;
            band = NAN;
            switch_price = NAN;
            trend = DEFAULT_TREND;
            continue;
        }

        if (!isfinite(switch_price)) {
            switch_price = c;
        }

        const double tr = isfinite(prev_close) ? true_range(h, l, prev_close) : (h - l);
        prev_close = c;

        bool atr_ready = false;
        const double atr_value = atr.update(tr, &atr_ready);
        if (!atr_ready) {
            continue;
        }

        const double src = 0.5 * (h + l);
        const double upper_base = src + multiplier * atr_value;
        const double lower_base = src - multiplier * atr_value;
        const double deviation = threshold_atr * atr_value;
        const bool is_at_loss =
            (trend == 1.0 && (switch_price - c) > deviation)
            || (trend == -1.0 && (c - switch_price) > deviation);
        const double prev_band = isfinite(band) ? band : (trend == 1.0 ? lower_base : upper_base);

        double changed = 0.0;

        if (trend == 1.0) {
            const double target_band =
                is_at_loss ? fma(alpha, c, (1.0 - alpha) * prev_band) : lower_base;
            band = fmax(target_band, prev_band);
            if (c < band) {
                trend = -1.0;
                band = upper_base;
                switch_price = c;
                changed = 1.0;
            }
        } else {
            const double target_band =
                is_at_loss ? fma(alpha, c, (1.0 - alpha) * prev_band) : upper_base;
            band = fmin(target_band, prev_band);
            if (c > band) {
                trend = 1.0;
                band = lower_base;
                switch_price = c;
                changed = 1.0;
            }
        }

        row_band[i] = band;
        row_switch_price[i] = switch_price;
        row_trend[i] = trend;
        row_changed[i] = changed;
    }
}
