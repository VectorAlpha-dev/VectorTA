#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
struct Coefficients {
    double c1;
    double c2;
    double c3;
};

__device__ inline Coefficients coefficients(int period) {
    double period_f = static_cast<double>(period);
    double a1 = exp(-1.414 * CUDART_PI / period_f);
    double b1 = 2.0 * a1 * cos(1.414 * CUDART_PI / period_f);
    Coefficients out;
    out.c2 = b1;
    out.c3 = -(a1 * a1);
    out.c1 = 1.0 - out.c2 - out.c3;
    return out;
}
}

extern "C" __global__ void ehlers_fm_demodulator_batch_f64(
    const double* __restrict__ open,
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

    Coefficients coeffs = coefficients(period);
    int warmup_bars = period > 3 ? period - 3 : 0;
    int valid_count = 0;
    double prev_hl = 0.0;
    double ss1 = 0.0;
    double ss2 = 0.0;

    for (int i = 0; i < len; ++i) {
        double open_value = open[i];
        double close_value = close[i];
        if (isnan(open_value) || isnan(close_value)) {
            valid_count = 0;
            prev_hl = 0.0;
            ss1 = 0.0;
            ss2 = 0.0;
            continue;
        }

        double derivative = close_value - open_value;
        double hl = fmin(fmax(10.0 * derivative, -1.0), 1.0);
        double value = valid_count < 3
            ? derivative
            : coeffs.c1 * (hl + prev_hl) * 0.5 + coeffs.c2 * ss1 + coeffs.c3 * ss2;

        prev_hl = hl;
        ss2 = ss1;
        ss1 = value;
        valid_count += 1;

        if (valid_count > warmup_bars) {
            row[i] = value;
        }
    }
}
