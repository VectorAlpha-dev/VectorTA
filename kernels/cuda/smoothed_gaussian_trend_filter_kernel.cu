#include <cmath>
#include <cstddef>

static __device__ inline bool sg_valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close) && high >= low;
}

static __device__ inline double sg_gaussian_alpha(int length, int poles) {
    double freq = (2.0 * 3.14159265358979323846) / static_cast<double>(length);
    double factor_b =
        (1.0 - cos(freq)) / (pow(1.414, 2.0 / static_cast<double>(poles)) - 1.0);
    return -factor_b + sqrt(factor_b * factor_b + 2.0 * factor_b);
}

static __device__ inline double sg_gaussian_update(
    int poles,
    double alpha,
    double input,
    double* history
) {
    double oma = 1.0 - alpha;
    double out = 0.0;
    if (poles == 1) {
        out = alpha * input + oma * history[0];
    } else if (poles == 2) {
        out = alpha * alpha * input + 2.0 * oma * history[0] - oma * oma * history[1];
    } else if (poles == 3) {
        double oma2 = oma * oma;
        double oma3 = oma2 * oma;
        out = alpha * alpha * alpha * input + 3.0 * oma * history[0] - 3.0 * oma2 * history[1] +
            oma3 * history[2];
    } else {
        double oma2 = oma * oma;
        double oma3 = oma2 * oma;
        double oma4 = oma3 * oma;
        double alpha4 = alpha * alpha * alpha * alpha;
        out = alpha4 * input + 4.0 * oma * history[0] - 6.0 * oma2 * history[1] +
            4.0 * oma3 * history[2] - oma4 * history[3];
    }
    history[3] = history[2];
    history[2] = history[1];
    history[1] = history[0];
    history[0] = out;
    return out;
}

static __device__ inline bool sg_linreg_update(
    double value,
    int period,
    int offset,
    double* buffer,
    int* count,
    int* head,
    double* out
) {
    if (*count < period) {
        buffer[*count] = value;
        *count += 1;
    } else {
        buffer[*head] = value;
        *head += 1;
        if (*head == period) {
            *head = 0;
        }
    }

    if (*count < period) {
        return false;
    }

    double x_sum = 0.0;
    double x2_sum = 0.0;
    for (int i = 1; i <= period; ++i) {
        double x = static_cast<double>(i);
        x_sum += x;
        x2_sum += x * x;
    }
    double period_f = static_cast<double>(period);
    double denom = period_f * x2_sum - x_sum * x_sum;
    double y_sum = 0.0;
    double xy_sum = 0.0;

    for (int i = 0; i < period; ++i) {
        int idx = *count < period ? i : ((*head + i) % period);
        double x = static_cast<double>(i + 1);
        double y = buffer[idx];
        y_sum += y;
        xy_sum += x * y;
    }

    double b = (period_f * xy_sum - x_sum * y_sum) / denom;
    double a = (y_sum - b * x_sum) / period_f;
    *out = a + b * (period_f - static_cast<double>(offset));
    return true;
}

static __device__ inline bool sg_atr_update(
    double high,
    double low,
    double close,
    int period,
    double* prev_close,
    double* rma,
    double* warm_sum,
    int* warm_count,
    bool* seeded,
    double* out
) {
    double tr = 0.0;
    if (isnan(*prev_close)) {
        tr = high - low;
    } else {
        double up = high > *prev_close ? high : *prev_close;
        double dn = low < *prev_close ? low : *prev_close;
        tr = up - dn;
    }

    *prev_close = close;

    if (!*seeded) {
        *warm_sum += tr;
        *warm_count += 1;
        if (*warm_count == period) {
            *rma = *warm_sum / static_cast<double>(period);
            *seeded = true;
            *out = *rma;
            return true;
        }
        return false;
    }

    double alpha = 1.0 / static_cast<double>(period);
    *rma = alpha * (tr - *rma) + *rma;
    *out = *rma;
    return true;
}

static __device__ inline double sg_supertrend_update(
    double src,
    double atr,
    double factor,
    double* prev_src,
    double* prev_upper,
    double* prev_lower,
    double* prev_supertrend,
    bool* prev_atr_valid,
    bool* initialized
) {
    double upper = src + factor * atr;
    double lower = src - factor * atr;

    double effective_prev_upper = *initialized ? *prev_upper : upper;
    double effective_prev_lower = *initialized ? *prev_lower : lower;
    double effective_prev_src = *initialized ? *prev_src : src;

    if (!(lower > effective_prev_lower || effective_prev_src < effective_prev_lower)) {
        lower = effective_prev_lower;
    }
    if (!(upper < effective_prev_upper || effective_prev_src > effective_prev_upper)) {
        upper = effective_prev_upper;
    }

    double direction = 0.0;
    if (!*prev_atr_valid) {
        direction = 1.0;
    } else if (*prev_supertrend == *prev_upper) {
        direction = src > upper ? -1.0 : 1.0;
    } else {
        direction = src < lower ? 1.0 : -1.0;
    }

    double supertrend = direction == -1.0 ? lower : upper;
    *prev_src = src;
    *prev_upper = upper;
    *prev_lower = lower;
    *prev_supertrend = supertrend;
    *prev_atr_valid = true;
    *initialized = true;
    return supertrend;
}

extern "C" __global__ void smoothed_gaussian_trend_filter_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ gaussian_lengths,
    const int* __restrict__ poles_values,
    const int* __restrict__ smoothing_lengths,
    const int* __restrict__ linreg_offsets,
    int rows,
    double* __restrict__ out_filter,
    double* __restrict__ out_supertrend,
    double* __restrict__ out_trend,
    double* __restrict__ out_ranging
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int gaussian_length = gaussian_lengths[row];
    int poles = poles_values[row];
    int smoothing_length = smoothing_lengths[row];
    int linreg_offset = linreg_offsets[row];

    double* row_filter = out_filter + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_supertrend = out_supertrend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ranging = out_ranging + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_filter[i] = NAN;
        row_supertrend[i] = NAN;
        row_trend[i] = NAN;
        row_ranging[i] = NAN;
    }

    if (gaussian_length <= 0 || gaussian_length > len || smoothing_length <= 0 ||
        smoothing_length > len || poles < 1 || poles > 4) {
        return;
    }

    double alpha = sg_gaussian_alpha(gaussian_length, poles);
    double gaussian_history[4] = {0.0, 0.0, 0.0, 0.0};

    double* linreg_buffer = new double[smoothing_length];
    if (linreg_buffer == nullptr) {
        return;
    }
    int linreg_count = 0;
    int linreg_head = 0;

    double atr_prev_close = NAN;
    double atr_rma = NAN;
    double atr_warm_sum = 0.0;
    int atr_warm_count = 0;
    bool atr_seeded = false;

    double prev_src = NAN;
    double prev_upper = NAN;
    double prev_lower = NAN;
    double prev_supertrend = NAN;
    bool prev_atr_valid = false;
    bool supertrend_initialized = false;

    double prev_final = NAN;
    bool has_prev_final = false;

    for (int i = 0; i < len; ++i) {
        if (!sg_valid_bar(high[i], low[i], close[i])) {
            gaussian_history[0] = 0.0;
            gaussian_history[1] = 0.0;
            gaussian_history[2] = 0.0;
            gaussian_history[3] = 0.0;
            linreg_count = 0;
            linreg_head = 0;
            atr_prev_close = NAN;
            atr_rma = NAN;
            atr_warm_sum = 0.0;
            atr_warm_count = 0;
            atr_seeded = false;
            prev_src = NAN;
            prev_upper = NAN;
            prev_lower = NAN;
            prev_supertrend = NAN;
            prev_atr_valid = false;
            supertrend_initialized = false;
            prev_final = NAN;
            has_prev_final = false;
            continue;
        }

        double atr_value = NAN;
        bool atr_ready = sg_atr_update(
            high[i],
            low[i],
            close[i],
            21,
            &atr_prev_close,
            &atr_rma,
            &atr_warm_sum,
            &atr_warm_count,
            &atr_seeded,
            &atr_value
        );

        double gaussian_value = sg_gaussian_update(poles, alpha, close[i], gaussian_history);

        double final_value = NAN;
        if (!sg_linreg_update(
                gaussian_value,
                smoothing_length,
                linreg_offset,
                linreg_buffer,
                &linreg_count,
                &linreg_head,
                &final_value
            )) {
            continue;
        }
        row_filter[i] = final_value;

        if (!atr_ready) {
            continue;
        }

        double supertrend_value = sg_supertrend_update(
            final_value,
            atr_value,
            0.15,
            &prev_src,
            &prev_upper,
            &prev_lower,
            &prev_supertrend,
            &prev_atr_valid,
            &supertrend_initialized
        );
        row_supertrend[i] = supertrend_value;

        double trend = final_value > supertrend_value ? 1.0 : -1.0;
        double slope_trend = has_prev_final && final_value > prev_final ? 1.0 : -1.0;
        row_trend[i] = trend;
        row_ranging[i] = slope_trend * trend < 0.0 ? 1.0 : 0.0;
        prev_final = final_value;
        has_prev_final = true;
    }

    delete[] linreg_buffer;
}
