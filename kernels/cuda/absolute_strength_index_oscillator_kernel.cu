#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void absolute_strength_index_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ signal_lengths,
    int n_combos,
    double* __restrict__ out_oscillator,
    double* __restrict__ out_signal,
    double* __restrict__ out_histogram
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int ema_length = ema_lengths[combo_idx];
    int signal_length = signal_lengths[combo_idx];
    double* row_oscillator =
        out_oscillator + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal =
        out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_histogram =
        out_histogram + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
        row_histogram[i] = CUDART_NAN;
    }

    if (ema_length <= 0 || signal_length <= 1) {
        return;
    }

    double ema_alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    double signal_alpha = 2.0 / (static_cast<double>(signal_length) + 1.0);
    double signal_beta = 1.0 - signal_alpha;

    bool have_prev_close = false;
    bool have_ema_abssi = false;
    double prev_close = CUDART_NAN;
    double a = 0.0;
    double m = 0.0;
    double d = 0.0;
    double ema_abssi = CUDART_NAN;
    double mt = 0.0;
    double ut = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_prev_close = false;
            have_ema_abssi = false;
            prev_close = CUDART_NAN;
            a = 0.0;
            m = 0.0;
            d = 0.0;
            ema_abssi = CUDART_NAN;
            mt = 0.0;
            ut = 0.0;
            continue;
        }

        double abssi = 1.0;
        if (have_prev_close) {
            if (value > prev_close) {
                if (prev_close != 0.0) {
                    a += value / prev_close - 1.0;
                }
            } else if (value < prev_close) {
                if (value != 0.0) {
                    d += prev_close / value - 1.0;
                }
            } else {
                m += 0.1;
            }

            double denom = d + m * 0.5;
            if (denom == 0.0) {
                abssi = 1.0;
            } else {
                abssi = 1.0 - 1.0 / (1.0 + (a + m * 0.5) / denom);
            }
        }

        prev_close = value;
        have_prev_close = true;

        if (have_ema_abssi) {
            ema_abssi = ema_alpha * abssi + (1.0 - ema_alpha) * ema_abssi;
        } else {
            ema_abssi = abssi;
            have_ema_abssi = true;
        }

        double oscillator = abssi - ema_abssi;
        mt = signal_alpha * oscillator + signal_beta * mt;
        ut = signal_alpha * mt + signal_beta * ut;

        double signal = ((2.0 - signal_alpha) * mt - ut) / signal_beta;
        double histogram = oscillator - signal;

        row_oscillator[i] = oscillator;
        row_signal[i] = signal;
        row_histogram[i] = histogram;
    }
}
