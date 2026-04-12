#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline bool market_meanness_valid_bar(double open, double close, int mode_flag) {
    if (mode_flag == 0) {
        return isfinite(close);
    }
    return isfinite(open) && isfinite(close);
}

__device__ inline double market_meanness_source_value(double open, double close, int mode_flag) {
    return mode_flag == 0 ? close : close - open;
}

__device__ inline void ordered_window_from_ring_device(
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

__device__ inline void insertion_sort_device(double* data, int len) {
    for (int i = 1; i < len; ++i) {
        double value = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > value) {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = value;
    }
}

__device__ inline double median_from_sorted_device(const double* data, int len) {
    int mid = len / 2;
    if ((len & 1) == 1) {
        return data[mid];
    }
    return 0.5 * (data[mid - 1] + data[mid]);
}

extern "C" __global__ void market_meanness_index_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ mode_flags,
    int n_combos,
    int max_length,
    double* __restrict__ source_ring,
    double* __restrict__ window_buf,
    double* __restrict__ median_buf,
    double* __restrict__ smoothing_buf,
    double* __restrict__ out_mmi,
    double* __restrict__ out_mmi_smoothed
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int mode_flag = mode_flags[combo_idx];
    double* row_mmi = out_mmi + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_smoothed =
        out_mmi_smoothed + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* source =
        source_ring + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* window =
        window_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* median =
        median_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* smooth =
        smoothing_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_mmi[i] = CUDART_NAN;
        row_smoothed[i] = CUDART_NAN;
    }

    if (length < 6 || length > max_length) {
        return;
    }

    int source_count = 0;
    int source_head = 0;
    int smooth_count = 0;
    int smooth_head = 0;
    double smooth_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double open_value = open[i];
        double close_value = close[i];
        if (!market_meanness_valid_bar(open_value, close_value, mode_flag)) {
            source_count = 0;
            source_head = 0;
            smooth_count = 0;
            smooth_head = 0;
            smooth_sum = 0.0;
            continue;
        }

        double value = market_meanness_source_value(open_value, close_value, mode_flag);
        if (source_count < length) {
            source[source_count] = value;
            source_count += 1;
            if (source_count < length) {
                continue;
            }
        } else {
            source[source_head] = value;
            source_head += 1;
            if (source_head == length) {
                source_head = 0;
            }
        }

        ordered_window_from_ring_device(window, source, length, source_head);
        for (int j = 0; j < length; ++j) {
            median[j] = window[j];
        }
        insertion_sort_device(median, length);
        double median_value = median_from_sorted_device(median, length);

        int count = 0;
        for (int j = 1; j < length; ++j) {
            double prev = window[j - 1];
            double curr = window[j];
            if ((curr > median_value && curr > prev) || (curr < median_value && curr < prev)) {
                count += 1;
            }
        }

        double mmi = static_cast<double>(count) * (100.0 / static_cast<double>(length - 1));
        row_mmi[i] = mmi;

        if (smooth_count < length) {
            smooth[smooth_count] = mmi;
            smooth_sum += mmi;
            smooth_count += 1;
            if (smooth_count == length) {
                row_smoothed[i] = smooth_sum / static_cast<double>(length);
            }
            continue;
        }

        double old = smooth[smooth_head];
        smooth[smooth_head] = mmi;
        smooth_sum += mmi - old;
        smooth_head += 1;
        if (smooth_head == length) {
            smooth_head = 0;
        }
        row_smoothed[i] = smooth_sum / static_cast<double>(length);
    }
}
