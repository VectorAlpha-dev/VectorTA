#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool atr_percentile_valid_bar(double high, double low, double close) {
    return isfinite(high) && isfinite(low) && isfinite(close);
}

__device__ inline bool atr_percentile_compute_atr(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int t,
    int atr_length,
    double* out_atr
) {
    if (atr_length <= 0 || t < atr_length - 1) {
        return false;
    }

    int start = t + 1 - atr_length;
    double sum = 0.0;

    for (int i = start; i <= t; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!atr_percentile_valid_bar(h, l, c)) {
            return false;
        }

        double tr = h - l;
        if (i > 0) {
            double prev_h = high[i - 1];
            double prev_l = low[i - 1];
            double prev_c = close[i - 1];
            if (atr_percentile_valid_bar(prev_h, prev_l, prev_c)) {
                double hc = fabs(h - prev_c);
                double lc = fabs(l - prev_c);
                if (hc > tr) {
                    tr = hc;
                }
                if (lc > tr) {
                    tr = lc;
                }
            }
        }

        sum += tr;
    }

    *out_atr = sum / static_cast<double>(atr_length);
    return true;
}

extern "C" __global__ void atr_percentile_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ percentile_lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int atr_length = atr_lengths[combo_idx];
    int percentile_length = percentile_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (atr_length <= 0 || percentile_length <= 0) {
        return;
    }

    int first_output = atr_length + percentile_length - 1;
    if (first_output >= len) {
        return;
    }

    for (int t = first_output; t < len; ++t) {
        double current_atr = 0.0;
        if (!atr_percentile_compute_atr(high, low, close, t, atr_length, &current_atr)) {
            continue;
        }

        bool valid = true;
        int below = 0;
        for (int offset = 1; offset <= percentile_length; ++offset) {
            double prev_atr = 0.0;
            if (!atr_percentile_compute_atr(high, low, close, t - offset, atr_length, &prev_atr)) {
                valid = false;
                break;
            }
            below += static_cast<int>(current_atr > prev_atr);
        }

        if (valid) {
            row[t] = 100.0 * static_cast<double>(below) / static_cast<double>(percentile_length);
        }
    }
}
