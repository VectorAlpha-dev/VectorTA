#include <cmath>
#include <cstddef>

static __device__ inline void exponential_trend_reset_state(
    int* atr_count,
    double* atr_sum,
    double* atr_value,
    double* atr_prev_close,
    bool* atr_have_prev_close,
    double* prev_upper_band,
    double* prev_lower_band,
    double* prev_close,
    bool* prev_atr_ready,
    double* initial_line,
    double* prev_initial_line,
    int* trend,
    int* bars_since_change,
    int* segment_index
) {
    *atr_count = 0;
    *atr_sum = 0.0;
    *atr_value = NAN;
    *atr_prev_close = NAN;
    *atr_have_prev_close = false;
    *prev_upper_band = NAN;
    *prev_lower_band = NAN;
    *prev_close = NAN;
    *prev_atr_ready = false;
    *initial_line = 0.0;
    *prev_initial_line = 0.0;
    *trend = 0;
    *bars_since_change = 0;
    *segment_index = 0;
}

extern "C" __global__ void exponential_trend_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ exp_rates,
    const double* __restrict__ initial_distances,
    const double* __restrict__ width_multipliers,
    int rows,
    double* __restrict__ out_uptrend_base,
    double* __restrict__ out_downtrend_base,
    double* __restrict__ out_uptrend_extension,
    double* __restrict__ out_downtrend_extension,
    double* __restrict__ out_bullish_change,
    double* __restrict__ out_bearish_change
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double exp_rate = exp_rates[row];
    double initial_distance = initial_distances[row];
    double width_multiplier = width_multipliers[row];

    double* row_out_uptrend_base =
        out_uptrend_base + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_downtrend_base =
        out_downtrend_base + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_uptrend_extension =
        out_uptrend_extension + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_downtrend_extension =
        out_downtrend_extension + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bullish_change =
        out_bullish_change + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_bearish_change =
        out_bearish_change + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_uptrend_base[i] = NAN;
        row_out_downtrend_base[i] = NAN;
        row_out_uptrend_extension[i] = NAN;
        row_out_downtrend_extension[i] = NAN;
        row_out_bullish_change[i] = NAN;
        row_out_bearish_change[i] = NAN;
    }

    if (!isfinite(exp_rate) || exp_rate < 0.0 || exp_rate > 0.5 ||
        !isfinite(initial_distance) || initial_distance < 0.1 ||
        !isfinite(width_multiplier) || width_multiplier < 0.1) {
        return;
    }

    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double atr_prev_close = NAN;
    bool atr_have_prev_close = false;
    double prev_upper_band = NAN;
    double prev_lower_band = NAN;
    double prev_close = NAN;
    bool prev_atr_ready = false;
    double initial_line = 0.0;
    double prev_initial_line = 0.0;
    int trend = 0;
    int bars_since_change = 0;
    int segment_index = 0;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            exponential_trend_reset_state(
                &atr_count,
                &atr_sum,
                &atr_value,
                &atr_prev_close,
                &atr_have_prev_close,
                &prev_upper_band,
                &prev_lower_band,
                &prev_close,
                &prev_atr_ready,
                &initial_line,
                &prev_initial_line,
                &trend,
                &bars_since_change,
                &segment_index
            );
            continue;
        }

        double tr_prev_close = atr_have_prev_close ? atr_prev_close : c;
        double tr = fmax(h - l, fmax(fabs(h - tr_prev_close), fabs(l - tr_prev_close)));
        atr_prev_close = c;
        atr_have_prev_close = true;

        bool atr_ready = false;
        if (atr_count < 14) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == 14) {
                atr_value = atr_sum / 14.0;
                atr_ready = true;
            }
        } else {
            atr_value = ((atr_value * 13.0) + tr) / 14.0;
            atr_ready = true;
        }

        double upper = NAN;
        double lower = NAN;
        int direction = 1;

        if (atr_ready) {
            double src = (h + l) * 0.5;
            double raw_upper = src + initial_distance * atr_value;
            double raw_lower = src - initial_distance * atr_value;
            double prev_lower = isfinite(prev_lower_band) ? prev_lower_band : 0.0;
            double prev_upper = isfinite(prev_upper_band) ? prev_upper_band : 0.0;
            double prev_close_value = isfinite(prev_close) ? prev_close : c;

            lower = (raw_lower > prev_lower || prev_close_value < prev_lower) ? raw_lower : prev_lower;
            upper = (raw_upper < prev_upper || prev_close_value > prev_upper) ? raw_upper : prev_upper;
            direction = !prev_atr_ready ? 1 : ((c < lower) ? 1 : -1);
        }

        int prev_trend = trend;
        double saved_prev_initial = prev_initial_line;
        double saved_prev_close = prev_close;

        if (segment_index == 100 && isfinite(upper) && isfinite(lower)) {
            if (direction < 0) {
                initial_line = lower;
                trend = 1;
            } else {
                initial_line = upper;
                trend = -1;
            }
        }

        bool crossover = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c > initial_line && saved_prev_close <= saved_prev_initial;
        bool crossunder = isfinite(initial_line) && isfinite(saved_prev_close) &&
            isfinite(saved_prev_initial) && c < initial_line && saved_prev_close >= saved_prev_initial;

        if (crossover && isfinite(lower)) {
            initial_line = lower;
            trend = 1;
        } else if (crossunder && isfinite(upper)) {
            initial_line = upper;
            trend = -1;
        }

        if (trend != prev_trend) {
            bars_since_change = 0;
        } else if (trend != 0) {
            bars_since_change += 1;
        }

        if (trend != 0) {
            double exp_multiplier =
                1.0 + static_cast<double>(trend) *
                (1.0 - exp(-exp_rate * static_cast<double>(bars_since_change)));
            if (exp_multiplier > 900.0) {
                exp_multiplier = 900.0;
            }
            initial_line *= exp_multiplier;
        }

        if (atr_ready) {
            double extension = initial_line +
                ((trend > 0) ? atr_value * width_multiplier : -atr_value * width_multiplier);

            if (trend == 1) {
                row_out_uptrend_base[i] = initial_line;
                row_out_uptrend_extension[i] = extension;
            } else if (trend == -1) {
                row_out_downtrend_base[i] = initial_line;
                row_out_downtrend_extension[i] = extension;
            }

            if (crossover) {
                row_out_bullish_change[i] = initial_line - atr_value;
            }
            if (crossunder) {
                row_out_bearish_change[i] = initial_line + atr_value;
            }
        }

        prev_upper_band = upper;
        prev_lower_band = lower;
        prev_close = c;
        prev_initial_line = initial_line;
        prev_atr_ready = atr_ready;
        segment_index += 1;
    }
}
