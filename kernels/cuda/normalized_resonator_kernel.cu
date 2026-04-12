#include <cmath>
#include <cstdint>

extern "C" __global__ void normalized_resonator_batch_f64(
    const double* data,
    int len,
    const int* periods,
    const double* deltas,
    const double* lookback_mults,
    const int* signal_lengths,
    int rows,
    double* out_oscillator,
    double* out_signal,
    double* bp_history
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int period = periods[row];
    double delta = deltas[row];
    double lookback_mult = lookback_mults[row];
    int signal_length = signal_lengths[row];
    if (period < 2 || !isfinite(delta) || delta <= 0.0 || delta > 1.0 || !isfinite(lookback_mult)
        || lookback_mult <= 0.0 || signal_length <= 0) {
        return;
    }

    const double nan = NAN;
    const double pi = 3.14159265358979323846;

    double alpha = tan(pi * delta / static_cast<double>(period));
    if (!isfinite(alpha)) {
        return;
    }
    double beta = cos(2.0 * pi / static_cast<double>(period));
    double r = 1.0 / (1.0 + alpha);
    double c1 = 2.0 * r * beta;
    double c2 = -(2.0 * r - 1.0);
    double gain = alpha * r;
    double peak_lookback_raw = floor(static_cast<double>(period) * lookback_mult);
    int peak_lookback = static_cast<int>(peak_lookback_raw < 1.0 ? 1.0 : peak_lookback_raw);
    double ema_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bp = bp_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    double src_prev1 = 0.0;
    double src_prev2 = 0.0;
    int src_count = 0;
    double bp_prev1 = 0.0;
    double bp_prev2 = 0.0;
    double ema_value = 0.0;
    bool ema_seeded = false;
    int run_start = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        row_oscillator[i] = nan;
        row_signal[i] = nan;
        row_bp[i] = nan;

        if (!isfinite(value)) {
            src_prev1 = 0.0;
            src_prev2 = 0.0;
            src_count = 0;
            bp_prev1 = 0.0;
            bp_prev2 = 0.0;
            ema_value = 0.0;
            ema_seeded = false;
            run_start = i + 1;
            continue;
        }

        if (src_count >= 2) {
            double bp = gain * (value - src_prev2) + c1 * bp_prev1 + c2 * bp_prev2;
            row_bp[i] = bp;

            int peak_start = i - peak_lookback + 1;
            if (peak_start < run_start) {
                peak_start = run_start;
            }
            double peak = 0.0;
            for (int j = peak_start; j <= i; ++j) {
                double hist = row_bp[j];
                if (isfinite(hist)) {
                    double abs_hist = fabs(hist);
                    if (abs_hist > peak) {
                        peak = abs_hist;
                    }
                }
            }

            double oscillator = peak > 0.0 ? bp / peak : 0.0;
            double signal = oscillator;
            if (ema_seeded) {
                ema_value += ema_alpha * (oscillator - ema_value);
                signal = ema_value;
            } else {
                ema_value = oscillator;
                ema_seeded = true;
            }

            bp_prev2 = bp_prev1;
            bp_prev1 = bp;
            row_oscillator[i] = oscillator;
            row_signal[i] = signal;
        }

        if (src_count == 0) {
            src_prev1 = value;
            src_count = 1;
        } else if (src_count == 1) {
            src_prev2 = src_prev1;
            src_prev1 = value;
            src_count = 2;
        } else {
            src_prev2 = src_prev1;
            src_prev1 = value;
        }
    }
}
