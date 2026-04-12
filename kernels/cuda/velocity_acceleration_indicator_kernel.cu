#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double vai_weighted_past_sum(
    const double* values,
    int length,
    int next,
    int count
) {
    int upto = count < length ? count : length;
    double sum = 0.0;
    for (int lag = 1; lag <= upto; ++lag) {
        int idx = next >= lag ? next - lag : length + next - lag;
        sum += values[idx] / static_cast<double>(lag);
    }
    return sum;
}

__device__ inline void vai_push(double* values, int length, int* next, int* count, double value) {
    if (length <= 0) {
        return;
    }
    values[*next] = value;
    *next += 1;
    if (*next == length) {
        *next = 0;
    }
    if (*count < length) {
        *count += 1;
    }
}

extern "C" __global__ void velocity_acceleration_indicator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooth_lengths,
    int n_combos,
    int max_length,
    int max_smooth_length,
    double* __restrict__ source_histories,
    double* __restrict__ acceleration_histories,
    double* __restrict__ wma_values,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0 || max_smooth_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth_length = smooth_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* source_history =
        source_histories + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* acceleration_history =
        acceleration_histories + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* wma_history =
        wma_values + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_smooth_length);

    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || length > max_length || smooth_length <= 0 ||
        smooth_length > max_smooth_length) {
        return;
    }

    double harmonic_sum = 0.0;
    for (int lag = 1; lag <= length; ++lag) {
        harmonic_sum += 1.0 / static_cast<double>(lag);
    }
    double inv_length = 1.0 / static_cast<double>(length);
    double wma_denominator =
        static_cast<double>(smooth_length * (smooth_length + 1) / 2);

    int source_next = 0;
    int source_count = 0;
    int acceleration_next = 0;
    int acceleration_count = 0;
    int wma_next = 0;
    int wma_count = 0;
    double wma_sum = 0.0;
    double wma_weighted_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_next = 0;
            source_count = 0;
            acceleration_next = 0;
            acceleration_count = 0;
            wma_next = 0;
            wma_count = 0;
            wma_sum = 0.0;
            wma_weighted_sum = 0.0;
            continue;
        }

        double velocity =
            (value * harmonic_sum -
             vai_weighted_past_sum(source_history, length, source_next, source_count)) *
            inv_length;
        vai_push(source_history, length, &source_next, &source_count, value);

        double velocity_avg = CUDART_NAN;
        bool have_velocity_avg = false;
        if (smooth_length == 1) {
            wma_history[0] = velocity;
            wma_next = 0;
            wma_count = 1;
            wma_sum = velocity;
            wma_weighted_sum = velocity;
            velocity_avg = velocity;
            have_velocity_avg = true;
        } else if (wma_count < smooth_length) {
            wma_history[wma_next] = velocity;
            wma_count += 1;
            wma_next += 1;
            if (wma_next == smooth_length) {
                wma_next = 0;
            }
            wma_sum += velocity;
            wma_weighted_sum += static_cast<double>(wma_count) * velocity;
            if (wma_count == smooth_length) {
                velocity_avg = wma_weighted_sum / wma_denominator;
                have_velocity_avg = true;
            }
        } else {
            double old = wma_history[wma_next];
            double previous_sum = wma_sum;
            wma_history[wma_next] = velocity;
            wma_next += 1;
            if (wma_next == smooth_length) {
                wma_next = 0;
            }
            wma_sum = previous_sum - old + velocity;
            wma_weighted_sum =
                wma_weighted_sum - previous_sum + static_cast<double>(smooth_length) * velocity;
            velocity_avg = wma_weighted_sum / wma_denominator;
            have_velocity_avg = true;
        }

        if (!have_velocity_avg) {
            continue;
        }

        double acceleration =
            (velocity_avg * harmonic_sum -
             vai_weighted_past_sum(
                 acceleration_history,
                 length,
                 acceleration_next,
                 acceleration_count
             )) *
            inv_length;
        vai_push(
            acceleration_history,
            length,
            &acceleration_next,
            &acceleration_count,
            velocity_avg
        );
        row[i] = acceleration;
    }
}
