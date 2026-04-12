#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void stochastic_distance_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookback_lengths,
    const int* __restrict__ length1s,
    const int* __restrict__ length2s,
    const int* __restrict__ ob_levels,
    const int* __restrict__ os_levels,
    int n_combos,
    int max_lookback,
    int max_length1,
    double* __restrict__ close_buffer,
    double* __restrict__ distance_buffer,
    double* __restrict__ out_oscillator,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback_length = lookback_lengths[combo_idx];
    int length1 = length1s[combo_idx];
    int length2 = length2s[combo_idx];
    double ob_level = static_cast<double>(ob_levels[combo_idx]);
    double os_level = static_cast<double>(os_levels[combo_idx]);
    double* close_ring =
        close_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length1);
    double* distance_ring =
        distance_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_lookback);
    double* row_oscillator =
        out_oscillator + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal =
        out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = CUDART_NAN;
        row_signal[i] = CUDART_NAN;
    }

    if (lookback_length <= 0 ||
        length1 <= 0 ||
        length2 <= 0 ||
        os_level >= ob_level ||
        lookback_length > max_lookback ||
        length1 > max_length1) {
        return;
    }

    int close_head = 0;
    int close_count = 0;
    int distance_head = 0;
    int distance_count = 0;
    double ema = 0.0;
    bool have_ema = false;
    double prev_sdo = 0.0;
    double alpha = 2.0 / (static_cast<double>(length2) + 1.0);
    const double tol = 1e-12;

    for (int i = 0; i < len; ++i) {
        double close = data[i];
        if (!isfinite(close)) {
            close_head = 0;
            close_count = 0;
            distance_head = 0;
            distance_count = 0;
            ema = 0.0;
            have_ema = false;
            prev_sdo = 0.0;
            continue;
        }

        bool have_lag = close_count >= length1;
        double lag_close = have_lag ? close_ring[close_head] : CUDART_NAN;

        close_ring[close_head] = close;
        close_head += 1;
        if (close_head == length1) {
            close_head = 0;
        }
        if (close_count < length1) {
            close_count += 1;
        }
        if (!have_lag) {
            continue;
        }

        double distance = fabs(close - lag_close);
        if (distance_count < lookback_length) {
            distance_ring[distance_count] = distance;
            distance_count += 1;
        } else {
            distance_ring[distance_head] = distance;
            distance_head += 1;
            if (distance_head == lookback_length) {
                distance_head = 0;
            }
        }
        if (distance_count < lookback_length) {
            continue;
        }

        double hh = -CUDART_INF;
        double ll = CUDART_INF;
        for (int j = 0; j < lookback_length; ++j) {
            double v = distance_ring[j];
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        double spread = hh - ll;
        double distance_sto = fabs(spread) > tol ? (distance - ll) / spread * 100.0 : 0.0;
        double distance_d = 0.0;
        if (close > lag_close + tol) {
            distance_d = distance_sto;
        } else if (close + tol < lag_close) {
            distance_d = -distance_sto;
        }

        if (have_ema) {
            ema = alpha * distance_d + (1.0 - alpha) * ema;
        } else {
            ema = distance_d;
            have_ema = true;
        }

        double signal = 0.0;
        if (distance_d > ema || (prev_sdo < os_level && ema > os_level)) {
            signal = 1.0;
        } else if (distance_d < ema || (prev_sdo > ob_level && ema < ob_level)) {
            signal = -1.0;
        }
        prev_sdo = ema;
        row_oscillator[i] = ema;
        row_signal[i] = signal;
    }
}
