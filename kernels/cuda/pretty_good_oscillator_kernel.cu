#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool pgo_valid_bar(double high, double low, double close, double source) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(source) && high >= low;
}

extern "C" __global__ void pretty_good_oscillator_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ source,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length <= 0) {
        return;
    }

    double alpha = 1.0 / static_cast<double>(length);
    double prev_close = CUDART_NAN;
    double atr = CUDART_NAN;
    double warm_sum_tr = 0.0;
    int valid_seen = 0;
    bool atr_seeded = false;

    for (int t = 0; t < len; ++t) {
        if (!pgo_valid_bar(high[t], low[t], close[t], source[t])) {
            continue;
        }

        double tr;
        if (isnan(prev_close)) {
            tr = high[t] - low[t];
        } else {
            double up = high[t] > prev_close ? high[t] : prev_close;
            double dn = low[t] < prev_close ? low[t] : prev_close;
            tr = up - dn;
        }
        prev_close = close[t];
        valid_seen += 1;

        if (!atr_seeded) {
            warm_sum_tr += tr;
            if (valid_seen < length) {
                continue;
            }
            atr = warm_sum_tr * alpha;
            atr_seeded = true;
        } else {
            atr = atr + alpha * (tr - atr);
        }

        double sma_sum = 0.0;
        int count = 0;
        for (int j = t; j >= 0 && count < length; --j) {
            if (pgo_valid_bar(high[j], low[j], close[j], source[j])) {
                sma_sum += source[j];
                count += 1;
            }
        }

        if (count < length) {
            continue;
        }

        double sma = sma_sum * alpha;
        row[t] = atr != 0.0 ? (source[t] - sma) / atr : CUDART_NAN;
    }
}
