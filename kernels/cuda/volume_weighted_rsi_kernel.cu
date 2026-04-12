#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool vw_rsi_valid_pair(double close, double volume) {
    return isfinite(close) && isfinite(volume);
}

__device__ inline double vw_rsi_from_components(double avg_up, double avg_down) {
    double denom = avg_up + avg_down;
    if (denom == 0.0) {
        return 50.0;
    }
    return 100.0 * avg_up / denom;
}

extern "C" __global__ void volume_weighted_rsi_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
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

    double inv_period = 1.0 / static_cast<double>(period);
    double beta = 1.0 - inv_period;
    double prev_close = CUDART_NAN;
    bool has_prev = false;
    int seeded = 0;
    double sum_up = 0.0;
    double sum_down = 0.0;
    double avg_up = 0.0;
    double avg_down = 0.0;

    for (int i = 0; i < len; ++i) {
        if (!vw_rsi_valid_pair(close[i], volume[i])) {
            prev_close = CUDART_NAN;
            has_prev = false;
            seeded = 0;
            sum_up = 0.0;
            sum_down = 0.0;
            avg_up = 0.0;
            avg_down = 0.0;
            continue;
        }

        double up = 0.0;
        double down = 0.0;
        if (has_prev) {
            if (close[i] > prev_close) {
                up = volume[i];
            } else if (close[i] < prev_close) {
                down = volume[i];
            }
        }

        prev_close = close[i];
        has_prev = true;

        if (seeded < period) {
            sum_up += up;
            sum_down += down;
            seeded += 1;
            if (seeded < period) {
                continue;
            }
            avg_up = sum_up * inv_period;
            avg_down = sum_down * inv_period;
            row[i] = vw_rsi_from_components(avg_up, avg_down);
            continue;
        }

        avg_up = beta * avg_up + inv_period * up;
        avg_down = beta * avg_down + inv_period * down;
        row[i] = vw_rsi_from_components(avg_up, avg_down);
    }
}
