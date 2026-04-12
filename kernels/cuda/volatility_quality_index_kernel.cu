#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void volatility_quality_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ fast_lengths,
    const int* __restrict__ slow_lengths,
    int n_combos,
    double* __restrict__ out_vqi_sum,
    double* __restrict__ out_fast_sma,
    double* __restrict__ out_slow_sma
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int fast_length = fast_lengths[combo_idx];
    int slow_length = slow_lengths[combo_idx];
    double* row_vqi = out_vqi_sum + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_fast = out_fast_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_slow = out_slow_sma + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    double prev_close = CUDART_NAN;
    double prev_vqi_t = 0.0;
    double cumulative = 0.0;

    for (int i = 0; i < len; ++i) {
        double o = open[i];
        double h = high[i];
        double l = low[i];
        double c = close[i];
        double range = h - l;

        double tr;
        if (isfinite(h) && isfinite(l)) {
            if (isfinite(prev_close)) {
                tr = range;
                double hc = fabs(h - prev_close);
                double lc = fabs(l - prev_close);
                if (hc > tr) {
                    tr = hc;
                }
                if (lc > tr) {
                    tr = lc;
                }
            } else {
                tr = range;
            }
        } else {
            tr = CUDART_NAN;
        }

        double vqi_t;
        if (isfinite(prev_close) &&
            isfinite(o) &&
            isfinite(h) &&
            isfinite(l) &&
            isfinite(c) &&
            isfinite(tr) &&
            tr != 0.0 &&
            isfinite(range) &&
            range != 0.0) {
            vqi_t = 0.5 * (((c - prev_close) / tr) + ((c - o) / range));
        } else {
            vqi_t = prev_vqi_t;
        }

        double raw;
        if (isfinite(prev_close) && isfinite(o) && isfinite(c)) {
            raw = fabs(vqi_t) * 0.5 * ((c - prev_close) + (c - o));
        } else {
            raw = 0.0;
        }

        prev_vqi_t = vqi_t;
        prev_close = c;
        cumulative += raw;
        row_vqi[i] = cumulative;
        row_fast[i] = CUDART_NAN;
        row_slow[i] = CUDART_NAN;
    }

    if (fast_length > 0 && fast_length <= len) {
        double sum = 0.0;
        for (int i = 0; i < fast_length; ++i) {
            sum += row_vqi[i];
        }
        row_fast[fast_length - 1] = sum / static_cast<double>(fast_length);
        for (int i = fast_length; i < len; ++i) {
            sum += row_vqi[i] - row_vqi[i - fast_length];
            row_fast[i] = sum / static_cast<double>(fast_length);
        }
    }

    if (slow_length > 0 && slow_length <= len) {
        double sum = 0.0;
        for (int i = 0; i < slow_length; ++i) {
            sum += row_vqi[i];
        }
        row_slow[slow_length - 1] = sum / static_cast<double>(slow_length);
        for (int i = slow_length; i < len; ++i) {
            sum += row_vqi[i] - row_vqi[i - slow_length];
            row_slow[i] = sum / static_cast<double>(slow_length);
        }
    }
}
