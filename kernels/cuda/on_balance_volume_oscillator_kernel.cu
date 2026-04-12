#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool obvo_valid_bar(double source, double volume) {
    return isfinite(source) && isfinite(volume);
}

extern "C" __global__ void on_balance_volume_oscillator_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ obv_lengths,
    const int* __restrict__ ema_lengths,
    int n_combos,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int obv_length = obv_lengths[combo_idx];
    int ema_length = ema_lengths[combo_idx];
    double* row_line = out_line + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_line[t] = CUDART_NAN;
        row_signal[t] = CUDART_NAN;
    }

    if (obv_length <= 0 || ema_length <= 0) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    bool ema_initialized = false;
    double ema_value = CUDART_NAN;
    int run_start = 0;

    for (int t = 0; t < len; ++t) {
        double src = source[t];
        double vol = volume[t];
        if (!obvo_valid_bar(src, vol)) {
            run_start = t + 1;
            ema_initialized = false;
            ema_value = CUDART_NAN;
            continue;
        }

        int run_len = t - run_start + 1;
        if (run_len < obv_length) {
            continue;
        }

        double signed_sum = 0.0;
        double volume_sum = 0.0;
        int count = 0;

        for (int j = t; j >= run_start && count < obv_length; --j) {
            double signed_volume;
            if (j == run_start) {
                signed_volume = 0.0;
            } else {
                double prev_source = source[j - 1];
                double sign =
                    source[j] > prev_source ? 1.0 : (source[j] < prev_source ? -1.0 : 0.0);
                signed_volume = volume[j] * sign;
            }
            signed_sum += signed_volume;
            volume_sum += volume[j];
            count += 1;
        }

        if (count < obv_length) {
            continue;
        }

        double line = volume_sum == 0.0 ? CUDART_NAN : signed_sum / volume_sum;
        row_line[t] = line;

        if (isfinite(line)) {
            if (ema_initialized) {
                ema_value += alpha * (line - ema_value);
            } else {
                ema_value = line;
                ema_initialized = true;
            }
            row_signal[t] = ema_value;
        }
    }
}
