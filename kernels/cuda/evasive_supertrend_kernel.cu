#include <cmath>
#include <cstddef>

namespace {

__device__ inline bool is_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close);
}

struct AtrTrackerDevice {
    int period;
    int count;
    double tr_sum;
    double prev_close;
    bool has_prev_close;
    double atr;

    __device__ void init(int period_value) {
        period = period_value;
        reset();
    }

    __device__ void reset() {
        count = 0;
        tr_sum = 0.0;
        prev_close = 0.0;
        has_prev_close = false;
        atr = NAN;
    }

    __device__ bool update(double high, double low, double close, double* out_atr) {
        const double tr = has_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        has_prev_close = true;

        if (count < period) {
            count += 1;
            tr_sum += tr;
            if (count == period) {
                atr = tr_sum / static_cast<double>(period);
                *out_atr = atr;
                return true;
            }
            return false;
        }

        atr = ((atr * static_cast<double>(period - 1)) + tr) / static_cast<double>(period);
        *out_atr = atr;
        return true;
    }
};

}

extern "C" __global__ void evasive_supertrend_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ atr_lengths,
    const double* __restrict__ base_multipliers,
    const double* __restrict__ noise_thresholds,
    const double* __restrict__ expansion_alphas,
    int rows,
    double* __restrict__ out_band,
    double* __restrict__ out_state,
    double* __restrict__ out_noisy,
    double* __restrict__ out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int atr_length = atr_lengths[row];
    const double base_multiplier = base_multipliers[row];
    const double noise_threshold = noise_thresholds[row];
    const double expansion_alpha = expansion_alphas[row];

    double* row_band = out_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_state = out_state + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_noisy = out_noisy + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_band[i] = NAN;
        row_state[i] = NAN;
        row_noisy[i] = NAN;
        row_changed[i] = NAN;
    }

    if (atr_length <= 0 || !isfinite(base_multiplier) || base_multiplier < 0.1 ||
        !isfinite(noise_threshold) || noise_threshold < 0.1 || !isfinite(expansion_alpha) ||
        expansion_alpha < 0.0) {
        return;
    }

    AtrTrackerDevice tracker;
    tracker.init(atr_length);
    int trend = 1;
    double band = NAN;

    for (int i = 0; i < len; ++i) {
        if (!is_valid_ohlc(open[i], high[i], low[i], close[i])) {
            tracker.reset();
            trend = 1;
            band = NAN;
            continue;
        }

        double atr = NAN;
        if (!tracker.update(high[i], low[i], close[i], &atr)) {
            continue;
        }

        const double src = (high[i] + low[i]) * 0.5;
        const double upper_base = src + base_multiplier * atr;
        const double lower_base = src - base_multiplier * atr;
        const double prev_band = isnan(band) ? (trend == 1 ? lower_base : upper_base) : band;
        const bool is_noisy = fabs(close[i] - prev_band) < atr * noise_threshold;
        const int prev_trend = trend;
        double next_band = NAN;

        if (prev_trend == 1) {
            next_band = is_noisy ? (prev_band - atr * expansion_alpha) : fmax(lower_base, prev_band);
            if (close[i] < next_band) {
                trend = -1;
                next_band = upper_base;
            }
        } else {
            next_band = is_noisy ? (prev_band + atr * expansion_alpha) : fmin(upper_base, prev_band);
            if (close[i] > next_band) {
                trend = 1;
                next_band = lower_base;
            }
        }

        band = next_band;
        row_band[i] = next_band;
        row_state[i] = static_cast<double>(trend);
        row_noisy[i] = is_noisy ? 1.0 : 0.0;
        row_changed[i] = trend != prev_trend ? 1.0 : 0.0;
    }
}
