#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline void copy_ordered_window(
    double* dst,
    const double* ring,
    int len,
    int head
) {
    if (head == 0) {
        for (int i = 0; i < len; ++i) {
            dst[i] = ring[i];
        }
        return;
    }

    int tail = len - head;
    for (int i = 0; i < tail; ++i) {
        dst[i] = ring[head + i];
    }
    for (int i = 0; i < head; ++i) {
        dst[tail + i] = ring[i];
    }
}

__device__ inline void pava_fit(
    const double* data,
    int len,
    bool non_decreasing,
    double* pool_vals,
    int* pool_weights,
    double* mse,
    int* pools,
    double* start_value,
    double* end_value
) {
    int pool_count = 0;
    for (int i = 0; i < len; ++i) {
        double current_pool = data[i];
        int current_weight = 1;
        while (pool_count > 0) {
            double prev_pool = pool_vals[pool_count - 1];
            bool violation = non_decreasing ? (prev_pool > current_pool) : (prev_pool < current_pool);
            if (!violation) {
                break;
            }
            int prev_weight = pool_weights[pool_count - 1];
            double last_pool = pool_vals[pool_count - 1];
            pool_count -= 1;
            int combined_weight = prev_weight + current_weight;
            current_pool =
                (last_pool * static_cast<double>(prev_weight) +
                 current_pool * static_cast<double>(current_weight)) /
                static_cast<double>(combined_weight);
            current_weight = combined_weight;
        }
        pool_vals[pool_count] = current_pool;
        pool_weights[pool_count] = current_weight;
        pool_count += 1;
    }

    double total_error = 0.0;
    int idx = 0;
    for (int pool = 0; pool < pool_count; ++pool) {
        double pool_value = pool_vals[pool];
        int pool_weight = pool_weights[pool];
        for (int j = 0; j < pool_weight; ++j) {
            double delta = data[idx] - pool_value;
            total_error += delta * delta;
            idx += 1;
        }
    }

    *mse = total_error / static_cast<double>(len);
    *pools = pool_count;
    *start_value = pool_count > 0 ? pool_vals[0] : 0.0;
    *end_value = pool_count > 0 ? pool_vals[pool_count - 1] : 0.0;
}

extern "C" __global__ void monotonicity_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ index_smooths,
    const int* __restrict__ mode_flags,
    int n_combos,
    int max_length,
    int max_index_smooth,
    double* __restrict__ window_ring,
    double* __restrict__ window_copy,
    double* __restrict__ inc_pool_vals,
    int* __restrict__ inc_pool_weights,
    double* __restrict__ dec_pool_vals,
    int* __restrict__ dec_pool_weights,
    double* __restrict__ sma_buf,
    double* __restrict__ out_index,
    double* __restrict__ out_cumulative_mean,
    double* __restrict__ out_upper_bound
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0 || max_index_smooth <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int index_smooth = index_smooths[combo_idx];
    int mode_flag = mode_flags[combo_idx];

    double* row_index = out_index + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_cumulative =
        out_cumulative_mean + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_upper =
        out_upper_bound + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* ring =
        window_ring + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* ordered =
        window_copy + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* inc_vals =
        inc_pool_vals + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* inc_weights =
        inc_pool_weights + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* dec_vals =
        dec_pool_vals + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    int* dec_weights =
        dec_pool_weights + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* sma =
        sma_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_index_smooth);

    for (int i = 0; i < len; ++i) {
        row_index[i] = CUDART_NAN;
        row_cumulative[i] = CUDART_NAN;
        row_upper[i] = CUDART_NAN;
    }

    if (length < 2 || length > max_length || index_smooth <= 0 || index_smooth > max_index_smooth) {
        return;
    }

    int window_next = 0;
    int window_len = 0;
    int sma_next = 0;
    int sma_len = 0;
    double sma_sum = 0.0;
    double cumulative_sum = 0.0;
    int cumulative_count = 0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            window_next = 0;
            window_len = 0;
            sma_next = 0;
            sma_len = 0;
            sma_sum = 0.0;
            cumulative_sum = 0.0;
            cumulative_count = 0;
            continue;
        }

        ring[window_next] = value;
        window_next += 1;
        if (window_next == length) {
            window_next = 0;
        }
        if (window_len < length) {
            window_len += 1;
        }
        if (window_len < length) {
            continue;
        }

        copy_ordered_window(ordered, ring, length, window_next);

        double inc_mse = 0.0;
        double dec_mse = 0.0;
        int inc_pools = 0;
        int dec_pools = 0;
        double inc_start = 0.0;
        double inc_end = 0.0;
        double dec_start = 0.0;
        double dec_end = 0.0;
        pava_fit(
            ordered,
            length,
            true,
            inc_vals,
            inc_weights,
            &inc_mse,
            &inc_pools,
            &inc_start,
            &inc_end
        );
        pava_fit(
            ordered,
            length,
            false,
            dec_vals,
            dec_weights,
            &dec_mse,
            &dec_pools,
            &dec_start,
            &dec_end
        );

        bool use_inc = inc_mse < dec_mse;
        double raw_index = 0.0;
        if (mode_flag == 0) {
            double start_value = use_inc ? inc_start : dec_start;
            double end_value = use_inc ? inc_end : dec_end;
            double price_path = 0.0;
            for (int j = 1; j < length; ++j) {
                price_path += fabs(ordered[j] - ordered[j - 1]);
            }
            if (price_path > 0.0) {
                raw_index = fabs(end_value - start_value) / price_path * 100.0;
            }
        } else {
            int pools = use_inc ? inc_pools : dec_pools;
            raw_index =
                (static_cast<double>(pools > 0 ? pools - 1 : 0) / static_cast<double>(length - 1)) *
                100.0;
        }

        if (sma_len == index_smooth) {
            sma_sum -= sma[sma_next];
        } else {
            sma_len += 1;
        }
        sma[sma_next] = raw_index;
        sma_sum += raw_index;
        sma_next += 1;
        if (sma_next == index_smooth) {
            sma_next = 0;
        }
        if (sma_len < index_smooth) {
            continue;
        }

        double smoothed = sma_sum / static_cast<double>(index_smooth);
        cumulative_sum += smoothed;
        cumulative_count += 1;
        double cumulative_mean = cumulative_sum / static_cast<double>(cumulative_count);
        row_index[i] = smoothed;
        row_cumulative[i] = cumulative_mean;
        row_upper[i] = cumulative_mean * 2.0;
    }
}
