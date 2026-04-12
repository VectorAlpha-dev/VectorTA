#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

constexpr int VELOCITY_MAX_LENGTH = 60;
constexpr int VELOCITY_MAX_SMOOTH_LENGTH = 9;

extern "C" __global__ void velocity_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooth_lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth_length = smooth_lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length < 2 || length > VELOCITY_MAX_LENGTH || smooth_length < 1 ||
        smooth_length > VELOCITY_MAX_SMOOTH_LENGTH) {
        return;
    }

    double harmonic = 0.0;
    for (int lag = 1; lag <= length; ++lag) {
        harmonic += 1.0 / static_cast<double>(lag);
    }
    double harmonic_over_length = harmonic / static_cast<double>(length);
    double smooth_denom = static_cast<double>(smooth_length * (smooth_length + 1) / 2);

    double history[VELOCITY_MAX_LENGTH];
    double raw_ring[VELOCITY_MAX_SMOOTH_LENGTH];
    for (int i = 0; i < VELOCITY_MAX_LENGTH; ++i) {
        history[i] = CUDART_NAN;
    }
    for (int i = 0; i < VELOCITY_MAX_SMOOTH_LENGTH; ++i) {
        raw_ring[i] = CUDART_NAN;
    }

    int history_head = 0;
    int history_count = 0;
    int raw_head = 0;
    int raw_count = 0;
    bool started = false;

    for (int t = 0; t < len; ++t) {
        double value = data[t];
        if (!started) {
            if (isnan(value)) {
                continue;
            }
            started = true;
        }

        double raw;
        if (isfinite(value)) {
            double weighted_past = 0.0;
            for (int lag = 1; lag <= length; ++lag) {
                double past = 0.0;
                if (lag <= history_count) {
                    int idx = (history_head + length - lag) % length;
                    double hist_value = history[idx];
                    if (isfinite(hist_value)) {
                        past = hist_value;
                    }
                }
                weighted_past += past / static_cast<double>(lag);
            }
            raw = value * harmonic_over_length - weighted_past / static_cast<double>(length);
        } else {
            raw = CUDART_NAN;
        }

        history[history_head] = value;
        history_head += 1;
        if (history_head == length) {
            history_head = 0;
        }
        if (history_count < length) {
            history_count += 1;
        }

        raw_ring[raw_head] = raw;
        raw_head += 1;
        if (raw_head == smooth_length) {
            raw_head = 0;
        }
        if (raw_count < smooth_length) {
            raw_count += 1;
        }
        if (raw_count < smooth_length) {
            continue;
        }

        double weighted = 0.0;
        bool valid = true;
        for (int offset = 0; offset < smooth_length; ++offset) {
            int idx = (raw_head + offset) % smooth_length;
            double raw_value = raw_ring[idx];
            if (!isfinite(raw_value)) {
                valid = false;
                break;
            }
            weighted += static_cast<double>(offset + 1) * raw_value;
        }

        if (valid) {
            row[t] = weighted / smooth_denom;
        }
    }
}
