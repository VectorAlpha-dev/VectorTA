#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void daily_factor_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const double* __restrict__ threshold_levels,
    int n_combos,
    double* __restrict__ out_value,
    double* __restrict__ out_ema,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double threshold_level = threshold_levels[combo_idx];
    double* row_value = out_value + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_ema = out_ema + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = CUDART_NAN;
        row_ema[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
    }

    if (!isfinite(threshold_level) || threshold_level < 0.0 || threshold_level > 1.0) {
        return;
    }

    double alpha = 2.0 / 15.0;
    double prev_open = CUDART_NAN;
    double prev_high = CUDART_NAN;
    double prev_low = CUDART_NAN;
    double prev_close = CUDART_NAN;
    double prev_ema = CUDART_NAN;
    bool has_prev = false;

    for (int i = 0; i < len; ++i) {
        double o = open[i];
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!(isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;
        }

        double ema = isfinite(prev_ema) ? prev_ema + alpha * (c - prev_ema) : c;
        double value = 0.0;
        if (has_prev) {
            double range = prev_high - prev_low;
            if (isfinite(range) && range != 0.0) {
                value = fabs(prev_open - prev_close) / range;
            }
        }

        double signal = 0.0;
        if (value > threshold_level && c > ema) {
            signal = 2.0;
        } else if (value > threshold_level && c < ema) {
            signal = -2.0;
        } else if (c > ema) {
            signal = 1.0;
        } else if (c < ema) {
            signal = -1.0;
        }

        row_value[i] = value;
        row_ema[i] = ema;
        row_signal[i] = signal;

        prev_open = o;
        prev_high = h;
        prev_low = l;
        prev_close = c;
        prev_ema = ema;
        has_prev = true;
    }
}
