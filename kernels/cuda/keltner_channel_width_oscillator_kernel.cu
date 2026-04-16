#include <cmath>
#include <cstddef>

namespace {
constexpr int BANDS_STYLE_ATR = 0;
constexpr int BANDS_STYLE_TR = 1;
constexpr int BANDS_STYLE_RANGE = 2;

__device__ inline bool is_valid_bar(double high, double low, double close, double source) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(source) && high >= low;
}

__device__ inline void reset_rolling_sma(
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    *count = 0;
    *head = 0;
    *sum = 0.0;
    for (int i = 0; i < period; ++i) {
        buffer[i] = 0.0;
    }
}

__device__ inline double update_rolling_sma(
    double value,
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    if (isfinite(value)) {
        if (*count < period) {
            buffer[*count] = value;
            *sum += value;
            *count += 1;
        } else {
            const double old = buffer[*head];
            buffer[*head] = value;
            *sum += value - old;
            *head += 1;
            if (*head == period) {
                *head = 0;
            }
        }
    }

    return *count == period ? (*sum / static_cast<double>(period)) : NAN;
}

__device__ inline void reset_seeded_avg(
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    *count = 0;
    *sum = 0.0;
    *value = NAN;
    *seeded = false;
}

__device__ inline double update_seeded_ema(
    double input,
    int period,
    double alpha,
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    if (!*seeded) {
        *sum += input;
        *count += 1;
        if (*count == period) {
            *value = *sum / static_cast<double>(period);
            *seeded = true;
            return *value;
        }
        return NAN;
    }

    *value = fma(alpha, input - *value, *value);
    return *value;
}

__device__ inline double update_seeded_rma(
    double input,
    int period,
    double alpha,
    int* count,
    double* sum,
    double* value,
    bool* seeded
) {
    if (!*seeded) {
        *sum += input;
        *count += 1;
        if (*count == period) {
            *value = *sum / static_cast<double>(period);
            *seeded = true;
            return *value;
        }
        return NAN;
    }

    *value = fma(alpha, input - *value, *value);
    return *value;
}
}

extern "C" __global__ void keltner_channel_width_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    const double* source,
    int len,
    const int* lengths,
    const double* multipliers,
    const int* use_exponentials,
    const int* bands_styles,
    const int* atr_lengths,
    int rows,
    int max_length,
    double* out_kbw,
    double* out_kbw_sma,
    double* center_sma_buffers,
    double* width_sma_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double multiplier = multipliers[row];
    const int use_exponential = use_exponentials[row];
    const int bands_style = bands_styles[row];
    const int atr_length = atr_lengths[row];

    double* row_kbw = out_kbw + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kbw_sma = out_kbw_sma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_center_sma =
        center_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_width_sma =
        width_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_kbw[i] = NAN;
        row_kbw_sma[i] = NAN;
    }

    if (length <= 0 || length > max_length || atr_length <= 0 || !isfinite(multiplier)
        || multiplier < 0.0 || (use_exponential != 0 && use_exponential != 1)
        || bands_style < BANDS_STYLE_ATR || bands_style > BANDS_STYLE_RANGE) {
        return;
    }

    int center_sma_count = 0;
    int center_sma_head = 0;
    double center_sma_sum = 0.0;
    int center_ema_count = 0;
    double center_ema_sum = 0.0;
    double center_ema_value = NAN;
    bool center_ema_seeded = false;

    int width_sma_count = 0;
    int width_sma_head = 0;
    double width_sma_sum = 0.0;

    int atr_rma_count = 0;
    double atr_rma_sum = 0.0;
    double atr_rma_value = NAN;
    bool atr_rma_seeded = false;

    int range_rma_count = 0;
    double range_rma_sum = 0.0;
    double range_rma_value = NAN;
    bool range_rma_seeded = false;

    double prev_close = NAN;
    const double center_ema_alpha = 2.0 / (static_cast<double>(length) + 1.0);
    const double atr_rma_alpha = 1.0 / static_cast<double>(atr_length);
    const double range_rma_alpha = 1.0 / static_cast<double>(length);

    reset_rolling_sma(
        &center_sma_count,
        &center_sma_head,
        &center_sma_sum,
        row_center_sma,
        length
    );
    reset_rolling_sma(
        &width_sma_count,
        &width_sma_head,
        &width_sma_sum,
        row_width_sma,
        length
    );

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double s = source[i];

        if (!is_valid_bar(h, l, c, s)) {
            reset_rolling_sma(
                &center_sma_count,
                &center_sma_head,
                &center_sma_sum,
                row_center_sma,
                length
            );
            reset_rolling_sma(
                &width_sma_count,
                &width_sma_head,
                &width_sma_sum,
                row_width_sma,
                length
            );
            reset_seeded_avg(
                &center_ema_count,
                &center_ema_sum,
                &center_ema_value,
                &center_ema_seeded
            );
            reset_seeded_avg(
                &atr_rma_count,
                &atr_rma_sum,
                &atr_rma_value,
                &atr_rma_seeded
            );
            reset_seeded_avg(
                &range_rma_count,
                &range_rma_sum,
                &range_rma_value,
                &range_rma_seeded
            );
            prev_close = NAN;
            continue;
        }

        const double middle =
            use_exponential != 0
                ? update_seeded_ema(
                      s,
                      length,
                      center_ema_alpha,
                      &center_ema_count,
                      &center_ema_sum,
                      &center_ema_value,
                      &center_ema_seeded
                  )
                : update_rolling_sma(
                      s,
                      &center_sma_count,
                      &center_sma_head,
                      &center_sma_sum,
                      row_center_sma,
                      length
                  );

        const double tr = isfinite(prev_close)
                              ? fmax(h - l, fmax(fabs(h - prev_close), fabs(l - prev_close)))
                              : (h - l);
        prev_close = c;

        double range = NAN;
        if (bands_style == BANDS_STYLE_ATR) {
            range = update_seeded_rma(
                tr,
                atr_length,
                atr_rma_alpha,
                &atr_rma_count,
                &atr_rma_sum,
                &atr_rma_value,
                &atr_rma_seeded
            );
        } else if (bands_style == BANDS_STYLE_TR) {
            range = tr;
        } else {
            range = update_seeded_rma(
                h - l,
                length,
                range_rma_alpha,
                &range_rma_count,
                &range_rma_sum,
                &range_rma_value,
                &range_rma_seeded
            );
        }

        if (!isfinite(middle) || !isfinite(range)) {
            continue;
        }

        if (middle == 0.0) {
            row_kbw[i] = NAN;
            row_kbw_sma[i] = NAN;
            continue;
        }

        const double kbw = (2.0 * multiplier * range) / middle;
        const double kbw_sma = update_rolling_sma(
            kbw,
            &width_sma_count,
            &width_sma_head,
            &width_sma_sum,
            row_width_sma,
            length
        );
        row_kbw[i] = kbw;
        row_kbw_sma[i] = kbw_sma;
    }
}
