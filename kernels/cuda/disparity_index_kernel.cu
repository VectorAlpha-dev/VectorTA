#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void disparity_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ ema_periods,
    const int* __restrict__ lookback_periods,
    const int* __restrict__ smoothing_periods,
    const int* __restrict__ smoothing_flags,
    int n_combos,
    int max_lookback,
    int max_smoothing,
    double* __restrict__ disparity_buffer,
    double* __restrict__ sma_buffer,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_lookback <= 0 || max_smoothing <= 0) {
        return;
    }

    int ema_period = ema_periods[combo_idx];
    int lookback_period = lookback_periods[combo_idx];
    int smoothing_period = smoothing_periods[combo_idx];
    int smoothing_flag = smoothing_flags[combo_idx];
    double* disparity_ring =
        disparity_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_lookback);
    double* sma_ring =
        sma_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_smoothing);
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (ema_period <= 0 ||
        lookback_period <= 0 ||
        smoothing_period <= 0 ||
        lookback_period > max_lookback ||
        smoothing_period > max_smoothing ||
        (smoothing_flag != 0 && smoothing_flag != 1)) {
        return;
    }

    double ema_alpha = 2.0 / (static_cast<double>(ema_period) + 1.0);
    double ema_beta = 1.0 - ema_alpha;
    double smoothing_alpha = 2.0 / (static_cast<double>(smoothing_period) + 1.0);
    double smoothing_beta = 1.0 - smoothing_alpha;

    int ema_seed_count = 0;
    double ema_seed_sum = 0.0;
    double ema = CUDART_NAN;
    bool ema_ready = false;

    int disparity_count = 0;
    int disparity_index = 0;

    int smoothing_seed_count = 0;
    double smoothing_seed_sum = 0.0;
    double smoothed = CUDART_NAN;
    bool smoothed_ready = false;

    int sma_count = 0;
    int sma_index = 0;
    double sma_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            ema_seed_count = 0;
            ema_seed_sum = 0.0;
            ema = CUDART_NAN;
            ema_ready = false;
            disparity_count = 0;
            disparity_index = 0;
            smoothing_seed_count = 0;
            smoothing_seed_sum = 0.0;
            smoothed = CUDART_NAN;
            smoothed_ready = false;
            sma_count = 0;
            sma_index = 0;
            sma_sum = 0.0;
            continue;
        }

        if (!ema_ready) {
            ema_seed_sum += value;
            ema_seed_count += 1;
            if (ema_seed_count < ema_period) {
                continue;
            }
            ema = ema_seed_sum / static_cast<double>(ema_period);
            ema_ready = true;
        } else {
            ema = ema_beta * ema + ema_alpha * value;
        }

        double disparity = CUDART_NAN;
        if (fabs(ema) <= 2.2204460492503131e-16) {
            if (fabs(value) <= 2.2204460492503131e-16) {
                disparity = 0.0;
            } else {
                continue;
            }
        } else {
            disparity = (value - ema) / ema * 100.0;
        }

        disparity_ring[disparity_index] = disparity;
        disparity_index += 1;
        if (disparity_index == lookback_period) {
            disparity_index = 0;
        }
        if (disparity_count < lookback_period) {
            disparity_count += 1;
        }
        if (disparity_count < lookback_period) {
            continue;
        }

        double high = -CUDART_INF;
        double low = CUDART_INF;
        for (int j = 0; j < lookback_period; ++j) {
            double window_value = disparity_ring[j];
            if (window_value > high) {
                high = window_value;
            }
            if (window_value < low) {
                low = window_value;
            }
        }

        double scaled = !(high > low) ? 50.0 : (disparity - low) / (high - low) * 100.0;

        if (smoothing_flag == 0) {
            if (!smoothed_ready) {
                smoothing_seed_sum += scaled;
                smoothing_seed_count += 1;
                if (smoothing_seed_count < smoothing_period) {
                    continue;
                }
                smoothed = smoothing_seed_sum / static_cast<double>(smoothing_period);
                smoothed_ready = true;
                row[i] = smoothed;
            } else {
                smoothed = smoothing_beta * smoothed + smoothing_alpha * scaled;
                row[i] = smoothed;
            }
        } else {
            if (sma_count < smoothing_period) {
                sma_ring[sma_count] = scaled;
                sma_sum += scaled;
                sma_count += 1;
                if (sma_count < smoothing_period) {
                    continue;
                }
                row[i] = sma_sum / static_cast<double>(smoothing_period);
            } else {
                double old = sma_ring[sma_index];
                sma_ring[sma_index] = scaled;
                sma_sum += scaled - old;
                sma_index += 1;
                if (sma_index == smoothing_period) {
                    sma_index = 0;
                }
                row[i] = sma_sum / static_cast<double>(smoothing_period);
            }
        }
    }
}
