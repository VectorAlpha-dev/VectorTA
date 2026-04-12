#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double ema_seeded_update(
    int period,
    double alpha,
    double beta,
    int* count,
    double* mean,
    bool* filled,
    double value,
    bool* produced
) {
    *count += 1;
    int current = *count;
    if (current == 1) {
        *mean = value;
    } else if (current <= period) {
        double inv = 1.0 / static_cast<double>(current);
        *mean = (value - *mean) * inv + *mean;
    } else {
        *mean = beta * (*mean) + alpha * value;
    }
    if (!*filled && current >= period) {
        *filled = true;
    }
    *produced = *filled;
    return *mean;
}

extern "C" __global__ void trend_direction_force_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    int max_norm_window,
    int* __restrict__ deque_indices,
    double* __restrict__ deque_values,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_norm_window <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    int* dq_idx = deque_indices + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_norm_window);
    double* dq_val = deque_values + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_norm_window);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length <= 0) {
        return;
    }

    int half = length / 2;
    if (half < 1) {
        half = 1;
    }
    int norm_window = length * 3;
    if (norm_window < 1 || norm_window > max_norm_window) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(half) + 1.0);
    double beta = 1.0 - alpha;
    int ema1_count = 0;
    int ema2_count = 0;
    double ema1_mean = CUDART_NAN;
    double ema2_mean = CUDART_NAN;
    bool ema1_filled = false;
    bool ema2_filled = false;
    double prev_ema1 = CUDART_NAN;
    double prev_ema2 = CUDART_NAN;
    bool have_prev_emas = false;
    int next_index = 0;
    int dq_head = 0;
    int dq_size = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            ema1_count = 0;
            ema2_count = 0;
            ema1_mean = CUDART_NAN;
            ema2_mean = CUDART_NAN;
            ema1_filled = false;
            ema2_filled = false;
            prev_ema1 = CUDART_NAN;
            prev_ema2 = CUDART_NAN;
            have_prev_emas = false;
            next_index = 0;
            dq_head = 0;
            dq_size = 0;
            continue;
        }

        int idx = next_index;
        next_index += 1;

        bool ema1_ready = false;
        double ema1 = ema_seeded_update(
            half,
            alpha,
            beta,
            &ema1_count,
            &ema1_mean,
            &ema1_filled,
            value * 1000.0,
            &ema1_ready
        );
        if (!ema1_ready) {
            continue;
        }

        bool ema2_ready = false;
        double ema2 = ema_seeded_update(
            half,
            alpha,
            beta,
            &ema2_count,
            &ema2_mean,
            &ema2_filled,
            ema1,
            &ema2_ready
        );
        if (!ema2_ready) {
            continue;
        }

        if (!have_prev_emas) {
            prev_ema1 = ema1;
            prev_ema2 = ema2;
            have_prev_emas = true;
            continue;
        }

        double ema_diff_avg = ((ema1 - prev_ema1) + (ema2 - prev_ema2)) * 0.5;
        double tdf = fabs(ema1 - ema2) * ema_diff_avg * ema_diff_avg * ema_diff_avg;
        prev_ema1 = ema1;
        prev_ema2 = ema2;

        double abs_tdf = fabs(tdf);
        int window_start = idx + 1 - norm_window;
        if (window_start < 0) {
            window_start = 0;
        }
        while (dq_size > 0 && dq_idx[dq_head] < window_start) {
            dq_head += 1;
            if (dq_head == norm_window) {
                dq_head = 0;
            }
            dq_size -= 1;
        }

        while (dq_size > 0) {
            int back_pos = dq_head + dq_size - 1;
            if (back_pos >= norm_window) {
                back_pos -= norm_window;
            }
            if (dq_val[back_pos] <= abs_tdf) {
                dq_size -= 1;
            } else {
                break;
            }
        }
        int insert_pos = dq_head + dq_size;
        if (insert_pos >= norm_window) {
            insert_pos -= norm_window;
        }
        dq_idx[insert_pos] = idx;
        dq_val[insert_pos] = abs_tdf;
        dq_size += 1;

        double max_abs = dq_size > 0 ? dq_val[dq_head] : 0.0;
        row[i] = max_abs == 0.0 ? 0.0 : tdf / max_abs;
    }
}
