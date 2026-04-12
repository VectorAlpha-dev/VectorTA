#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double dynamic_period(
    int rsi_period,
    double std_value,
    double avg_std,
    int lower_limit,
    int upper_limit
) {
    if (!isfinite(std_value) || !isfinite(avg_std) || std_value <= 0.0 || avg_std <= 0.0) {
        return static_cast<double>(upper_limit);
    }
    double raw = floor(static_cast<double>(rsi_period) * avg_std / std_value);
    int period = (isfinite(raw) && raw > 0.0) ? static_cast<int>(raw) : upper_limit;
    if (period < lower_limit) {
        period = lower_limit;
    }
    if (period > upper_limit) {
        period = upper_limit;
    }
    return static_cast<double>(period);
}
}

extern "C" __global__ void dynamic_momentum_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ rsi_periods,
    const int* __restrict__ volatility_periods,
    const int* __restrict__ volatility_sma_periods,
    const int* __restrict__ upper_limits,
    const int* __restrict__ lower_limits,
    int n_combos,
    int max_volatility_period,
    int max_volatility_sma_period,
    int max_upper_limit,
    double* __restrict__ close_buffer,
    double* __restrict__ std_buffer,
    double* __restrict__ gain_buffer,
    double* __restrict__ loss_buffer,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int rsi_period = rsi_periods[combo_idx];
    int volatility_period = volatility_periods[combo_idx];
    int volatility_sma_period = volatility_sma_periods[combo_idx];
    int upper_limit = upper_limits[combo_idx];
    int lower_limit = lower_limits[combo_idx];
    double* close_ring =
        close_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_volatility_period);
    double* std_ring =
        std_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_volatility_sma_period);
    double* gain_ring =
        gain_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_upper_limit);
    double* loss_ring =
        loss_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_upper_limit);
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (rsi_period <= 0 ||
        volatility_period <= 0 ||
        volatility_sma_period <= 0 ||
        upper_limit <= 0 ||
        lower_limit <= 0 ||
        lower_limit > upper_limit ||
        volatility_period > max_volatility_period ||
        volatility_sma_period > max_volatility_sma_period ||
        upper_limit > max_upper_limit) {
        return;
    }

    double prev_close = CUDART_NAN;
    bool has_prev = false;
    double close_sum = 0.0;
    double close_sumsq = 0.0;
    int close_idx = 0;
    int close_count = 0;
    double std_sum = 0.0;
    int std_idx = 0;
    int std_count = 0;
    int gl_idx = 0;
    int gl_count = 0;

    for (int i = 0; i < len; ++i) {
        double close = data[i];
        if (!isfinite(close)) {
            prev_close = CUDART_NAN;
            has_prev = false;
            close_sum = 0.0;
            close_sumsq = 0.0;
            close_idx = 0;
            close_count = 0;
            std_sum = 0.0;
            std_idx = 0;
            std_count = 0;
            gl_idx = 0;
            gl_count = 0;
            continue;
        }

        bool have_std = false;
        double std_value = CUDART_NAN;

        if (close_count == volatility_period) {
            double old = close_ring[close_idx];
            close_sum -= old;
            close_sumsq -= old * old;
        } else {
            close_count += 1;
        }
        close_ring[close_idx] = close;
        close_sum += close;
        close_sumsq += close * close;
        close_idx += 1;
        if (close_idx == volatility_period) {
            close_idx = 0;
        }
        if (close_count == volatility_period) {
            double n = static_cast<double>(volatility_period);
            double mean = close_sum / n;
            double var = close_sumsq / n - mean * mean;
            if (var < 0.0) {
                var = 0.0;
            }
            std_value = sqrt(var);
            have_std = true;
        }

        if (has_prev) {
            double delta = close - prev_close;
            gain_ring[gl_idx] = delta > 0.0 ? delta : 0.0;
            loss_ring[gl_idx] = delta < 0.0 ? -delta : 0.0;
            gl_idx += 1;
            if (gl_idx == upper_limit) {
                gl_idx = 0;
            }
            if (gl_count < upper_limit) {
                gl_count += 1;
            }
        }
        prev_close = close;
        has_prev = true;

        if (!have_std) {
            continue;
        }

        bool have_avg_std = false;
        double avg_std = CUDART_NAN;
        if (std_count == volatility_sma_period) {
            std_sum -= std_ring[std_idx];
        } else {
            std_count += 1;
        }
        std_ring[std_idx] = std_value;
        std_sum += std_value;
        std_idx += 1;
        if (std_idx == volatility_sma_period) {
            std_idx = 0;
        }
        if (std_count == volatility_sma_period) {
            avg_std = std_sum / static_cast<double>(volatility_sma_period);
            have_avg_std = true;
        }
        if (!have_avg_std) {
            continue;
        }

        int period = static_cast<int>(dynamic_period(
            rsi_period,
            std_value,
            avg_std,
            lower_limit,
            upper_limit
        ));
        if (gl_count < period) {
            continue;
        }

        double sum_gain = 0.0;
        double sum_loss = 0.0;
        int idx = gl_idx;
        for (int j = 0; j < period; ++j) {
            idx = (idx == 0) ? upper_limit - 1 : idx - 1;
            sum_gain += gain_ring[idx];
            sum_loss += loss_ring[idx];
        }
        double denom = sum_gain + sum_loss;
        row[i] = denom == 0.0 ? 50.0 : 100.0 * sum_gain / denom;
    }
}
