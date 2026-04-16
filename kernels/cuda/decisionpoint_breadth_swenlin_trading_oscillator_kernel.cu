#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
constexpr int SMA_LENGTH = 5;
constexpr double EMA_ALPHA = 2.0 / 5.0;
constexpr double MULTIPLIER = 1000.0;
constexpr double EPSILON = 1e-12;

__device__ inline bool valid_breadth_pair(double advancing, double declining) {
    if (!isfinite(advancing) || !isfinite(declining)) {
        return false;
    }
    double total = advancing + declining;
    return isfinite(total) && fabs(total) > EPSILON;
}
}

extern "C" __global__ void decisionpoint_breadth_swenlin_trading_oscillator_batch_f64(
    const double* __restrict__ advancing,
    const double* __restrict__ declining,
    int len,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    bool ema_started = false;
    double ema_value = CUDART_NAN;
    double sma_values[SMA_LENGTH] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int sma_idx = 0;
    int sma_count = 0;
    double sma_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double adv = advancing[i];
        double dec = declining[i];
        if (!valid_breadth_pair(adv, dec)) {
            row[i] = CUDART_NAN;
            ema_started = false;
            ema_value = CUDART_NAN;
            sma_idx = 0;
            sma_count = 0;
            sma_sum = 0.0;
            for (int j = 0; j < SMA_LENGTH; ++j) {
                sma_values[j] = 0.0;
            }
            continue;
        }

        double breadth = ((adv - dec) / (adv + dec)) * MULTIPLIER;
        if (!ema_started) {
            ema_started = true;
            ema_value = breadth;
        } else {
            ema_value += EMA_ALPHA * (breadth - ema_value);
        }

        if (sma_count == SMA_LENGTH) {
            sma_sum -= sma_values[sma_idx];
        } else {
            sma_count += 1;
        }
        sma_values[sma_idx] = ema_value;
        sma_sum += ema_value;
        sma_idx += 1;
        if (sma_idx == SMA_LENGTH) {
            sma_idx = 0;
        }

        row[i] = sma_count < SMA_LENGTH ? CUDART_NAN : (sma_sum / static_cast<double>(SMA_LENGTH));
    }
}
