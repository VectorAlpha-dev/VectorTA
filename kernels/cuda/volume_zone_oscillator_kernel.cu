#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void volume_zone_oscillator_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ noise_filters,
    const int* __restrict__ intraday_flags,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int noise_filter = noise_filters[combo_idx];
    bool intraday_smoothing = intraday_flags[combo_idx] != 0;
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || noise_filter < 2) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(length) + 1.0);
    double beta = 1.0 - alpha;
    double smooth_alpha = 2.0 / (static_cast<double>(noise_filter) + 1.0);
    double smooth_beta = 1.0 - smooth_alpha;

    double prev_close = CUDART_NAN;
    double ema_direction = 0.0;
    double ema_total = 0.0;
    double smooth = 0.0;
    bool smooth_valid = false;
    bool started = false;

    for (int i = 0; i < len; ++i) {
        double vol = volume[i];
        if (!started) {
            if (!isfinite(vol)) {
                continue;
            }
            started = true;
        }

        double raw = CUDART_NAN;
        bool raw_valid = false;
        if (!isfinite(vol)) {
            if (ema_total != 0.0) {
                raw = 100.0 * ema_direction / ema_total;
                raw_valid = true;
            }
        } else {
            double current_close = close[i];
            double directed =
                (isfinite(current_close) && isfinite(prev_close) && current_close > prev_close)
                    ? vol
                    : -vol;
            ema_direction = beta * ema_direction + alpha * directed;
            ema_total = beta * ema_total + alpha * vol;
            if (ema_total != 0.0) {
                raw = 100.0 * ema_direction / ema_total;
                raw_valid = true;
            }
        }

        if (isfinite(close[i])) {
            prev_close = close[i];
        }

        if (intraday_smoothing) {
            if (raw_valid) {
                smooth = smooth_beta * smooth + smooth_alpha * raw;
                smooth_valid = true;
                row[i] = smooth;
            } else if (smooth_valid) {
                row[i] = smooth;
            }
        } else if (raw_valid) {
            row[i] = raw;
        }
    }
}
