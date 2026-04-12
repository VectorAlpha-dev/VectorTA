#include <cmath>
#include <cstdint>

static __device__ inline double vacd_ring_get(
    const double* ring,
    int head,
    int count,
    int capacity,
    int lookback
) {
    if (lookback <= 0 || lookback > count) {
        return 0.0;
    }
    int idx = head - lookback;
    while (idx < 0) {
        idx += capacity;
    }
    return ring[idx];
}

static __device__ inline void vacd_ring_push(
    double* ring,
    int capacity,
    int* head,
    int* count,
    double value
) {
    ring[*head] = value;
    *head += 1;
    if (*head >= capacity) {
        *head = 0;
    }
    if (*count < capacity) {
        *count += 1;
    }
}

static __device__ inline double vacd_compute_velocity_current(
    const double* history,
    int head,
    int count,
    int capacity,
    double current,
    int length
) {
    double sum = 0.0;
    for (int i = 1; i <= length; ++i) {
        double prev =
            i <= count ? vacd_ring_get(history, head, count, capacity, i) : 0.0;
        sum += (current - prev) / static_cast<double>(i);
    }
    return sum / static_cast<double>(length);
}

static __device__ inline double vacd_compute_wma_tail(
    const double* history,
    int head,
    int capacity,
    int period
) {
    double numerator = 0.0;
    double denominator = 0.0;
    int start = head - period;
    while (start < 0) {
        start += capacity;
    }
    for (int offset = 0; offset < period; ++offset) {
        int idx = start + offset;
        if (idx >= capacity) {
            idx -= capacity;
        }
        double weight = static_cast<double>(offset + 1);
        numerator += history[idx] * weight;
        denominator += weight;
    }
    return numerator / denominator;
}

static __device__ inline double vacd_classify_signal(double vacd, double prev_vacd_nz) {
    if (vacd > 0.0) {
        return vacd > prev_vacd_nz ? 2.0 : 1.0;
    }
    if (vacd < 0.0) {
        return vacd < prev_vacd_nz ? -2.0 : -1.0;
    }
    return 0.0;
}

extern "C" __global__ void velocity_acceleration_convergence_divergence_indicator_batch_f64(
    const double* data,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* source_history,
    double* raw_velocity_history,
    double* velocity_avg_history,
    double* out_vacd,
    double* out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length < 2 || smooth_length <= 0 || max_length <= 0 || max_smooth_length <= 0) {
        return;
    }

    const double nan = NAN;

    double* row_source = source_history + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_raw = raw_velocity_history
        + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_velocity_avg =
        velocity_avg_history + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_vacd = out_vacd + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    int source_head = 0;
    int source_count = 0;
    int raw_head = 0;
    int raw_count = 0;
    int velocity_avg_head = 0;
    int velocity_avg_count = 0;
    double prev_vacd = nan;
    bool has_prev_vacd = false;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_head = 0;
            source_count = 0;
            raw_head = 0;
            raw_count = 0;
            velocity_avg_head = 0;
            velocity_avg_count = 0;
            prev_vacd = nan;
            has_prev_vacd = false;
            row_vacd[i] = nan;
            row_signal[i] = nan;
            continue;
        }

        double raw_velocity = vacd_compute_velocity_current(
            row_source,
            source_head,
            source_count,
            max_length,
            value,
            length);
        vacd_ring_push(row_source, max_length, &source_head, &source_count, value);
        vacd_ring_push(row_raw, max_smooth_length, &raw_head, &raw_count, raw_velocity);

        if (raw_count < smooth_length) {
            row_vacd[i] = nan;
            row_signal[i] = nan;
            continue;
        }

        double velocity_avg =
            vacd_compute_wma_tail(row_raw, raw_head, max_smooth_length, smooth_length);
        double acceleration = vacd_compute_velocity_current(
            row_velocity_avg,
            velocity_avg_head,
            velocity_avg_count,
            max_length,
            velocity_avg,
            length);
        double vacd = velocity_avg - acceleration;
        double signal = vacd_classify_signal(vacd, has_prev_vacd ? prev_vacd : 0.0);

        vacd_ring_push(
            row_velocity_avg, max_length, &velocity_avg_head, &velocity_avg_count, velocity_avg);
        prev_vacd = vacd;
        has_prev_vacd = true;

        row_vacd[i] = vacd;
        row_signal[i] = signal;
    }
}
