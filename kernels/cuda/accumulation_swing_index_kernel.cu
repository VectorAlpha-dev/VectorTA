#include <cmath>
#include <cstddef>

namespace {
__device__ double compute_increment(
    double prev_open,
    double prev_close,
    double open,
    double high,
    double low,
    double close,
    double daily_limit
) {
    const double abs_high_close = fabs(high - prev_close);
    const double abs_low_close = fabs(low - prev_close);
    const double abs_close_open = fabs(prev_close - prev_open);
    const double k = abs_high_close >= abs_low_close ? abs_high_close : abs_low_close;
    const double range = high - low;
    double r = 0.0;
    if (abs_high_close >= abs_low_close) {
        if (abs_high_close >= range) {
            r = abs_high_close - 0.5 * abs_low_close + 0.25 * abs_close_open;
        } else {
            r = range + 0.25 * abs_close_open;
        }
    } else if (abs_low_close >= range) {
        r = abs_low_close - 0.5 * abs_high_close + 0.25 * abs_close_open;
    } else {
        r = range + 0.25 * abs_close_open;
    }

    if (r != 0.0) {
        return 50.0 *
            (((close - prev_close) + 0.5 * (close - open) + 0.25 * (prev_close - prev_open)) / r) *
            k / daily_limit;
    }
    return 0.0;
}
}

extern "C" __global__ void accumulation_swing_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ daily_limits,
    int rows,
    double* __restrict__ out_values
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    const double daily_limit = daily_limits[row];
    if (!isfinite(daily_limit) || daily_limit <= 0.0) {
        return;
    }

    int first = -1;
    for (int i = 0; i < len; ++i) {
        if (isfinite(open[i]) && isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return;
    }

    double accum = 0.0;
    row_out[first] = 0.0;
    double prev_open = open[first];
    double prev_close = close[first];

    for (int i = first + 1; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
            isfinite(prev_open) && isfinite(prev_close)) {
            const double delta = compute_increment(prev_open, prev_close, o, h, l, c, daily_limit);
            if (isfinite(delta)) {
                accum += delta;
            }
        }
        row_out[i] = accum;
        prev_open = o;
        prev_close = c;
    }
}
