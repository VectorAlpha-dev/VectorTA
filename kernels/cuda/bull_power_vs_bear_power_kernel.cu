#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool bbpower_valid_ohlc(double open, double high, double low, double close) {
    return isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close) && close != 0.0;
}

extern "C" __global__ void bull_power_vs_bear_power_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
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

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (period <= 0) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    double beta = 1.0 - alpha;
    int count = 0;
    double mean = CUDART_NAN;

    for (int i = 0; i < len; ++i) {
        if (!bbpower_valid_ohlc(open[i], high[i], low[i], close[i])) {
            count = 0;
            mean = CUDART_NAN;
            continue;
        }

        double value = ((high[i] + low[i]) - (2.0 * open[i])) * (100.0 / close[i]);
        count += 1;
        if (count == 1) {
            mean = value;
        } else if (count <= period) {
            double c = static_cast<double>(count);
            mean = ((c - 1.0) * mean + value) / c;
        } else {
            mean = beta * mean + alpha * value;
        }

        if (count >= period) {
            row[i] = mean;
        }
    }
}
