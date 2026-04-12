#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void rolling_z_score_trend_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lookbacks,
    int n_combos,
    double* __restrict__ out_zscore,
    double* __restrict__ out_momentum
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback = lookbacks[combo_idx];
    double* row_zscore = out_zscore + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_momentum =
        out_momentum + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row_zscore[t] = CUDART_NAN;
        row_momentum[t] = CUDART_NAN;
    }

    if (lookback <= 0) {
        return;
    }

    bool has_smoothed = false;
    double smoothed = CUDART_NAN;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!isfinite(value)) {
            has_smoothed = false;
            smoothed = CUDART_NAN;
            continue;
        }

        double sum = 0.0;
        double sumsq = 0.0;
        int count = 0;

        for (int i = t; i >= 0 && count < lookback; --i) {
            double v = data[i];
            if (!isfinite(v)) {
                break;
            }
            sum += v;
            sumsq += v * v;
            count += 1;
        }

        if (count < lookback) {
            continue;
        }

        double n = static_cast<double>(lookback);
        double mean = sum / n;
        double variance = sumsq / n - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        double stddev = sqrt(variance);
        double raw_zscore = stddev > DBL_EPSILON ? (value - mean) / stddev : 0.0;

        if (!has_smoothed) {
            smoothed = raw_zscore;
            has_smoothed = true;
            row_zscore[t] = smoothed;
            row_momentum[t] = CUDART_NAN;
            continue;
        }

        double prev_smoothed = smoothed;
        smoothed = 0.5 * raw_zscore + 0.5 * prev_smoothed;
        row_zscore[t] = smoothed;
        row_momentum[t] = smoothed - prev_smoothed;
    }
}
