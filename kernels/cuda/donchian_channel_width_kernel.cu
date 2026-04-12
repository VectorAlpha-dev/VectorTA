#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void donchian_channel_width_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int len,
    const int* __restrict__ periods,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int period = periods[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    for (int t = 0; t < len; ++t) {
        double h = high[t];
        double l = low[t];
        if (!isfinite(h) || !isfinite(l)) {
            continue;
        }

        double max_h = -CUDART_INF;
        double min_l = CUDART_INF;
        int count = 0;

        for (int i = t; i >= 0 && count < period; --i) {
            double hh = high[i];
            double ll = low[i];
            if (!isfinite(hh) || !isfinite(ll)) {
                break;
            }
            if (hh > max_h) {
                max_h = hh;
            }
            if (ll < min_l) {
                min_l = ll;
            }
            count += 1;
        }

        if (count == period) {
            row[t] = max_h - min_l;
        }
    }
}
