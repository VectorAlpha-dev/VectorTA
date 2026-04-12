#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void trend_continuation_factor_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    int max_length,
    double* __restrict__ plus_buffer,
    double* __restrict__ minus_buffer,
    double* __restrict__ out_plus,
    double* __restrict__ out_minus
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row_plus = out_plus + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_minus = out_minus + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* plus_ring =
        plus_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* minus_ring =
        minus_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);

    for (int i = 0; i < len; ++i) {
        row_plus[i] = CUDART_NAN;
        row_minus[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length) {
        return;
    }

    bool have_prev = false;
    bool have_plus_cf = false;
    bool have_minus_cf = false;
    double prev = CUDART_NAN;
    double plus_cf = 0.0;
    double minus_cf = 0.0;
    int comparisons_seen = 0;
    int head = 0;
    double sum_plus = 0.0;
    double sum_minus = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_prev = false;
            have_plus_cf = false;
            have_minus_cf = false;
            prev = CUDART_NAN;
            plus_cf = 0.0;
            minus_cf = 0.0;
            comparisons_seen = 0;
            head = 0;
            sum_plus = 0.0;
            sum_minus = 0.0;
            continue;
        }

        if (!have_prev) {
            prev = value;
            have_prev = true;
            continue;
        }

        double change = value - prev;
        double plus_change = change > 0.0 ? change : 0.0;
        double minus_change = change < 0.0 ? -change : 0.0;

        double next_plus_cf = plus_change == 0.0
            ? 0.0
            : plus_change + (have_plus_cf ? plus_cf : 1.0);
        double next_minus_cf = minus_change == 0.0
            ? 0.0
            : minus_change + (have_minus_cf ? minus_cf : 1.0);

        have_plus_cf = true;
        have_minus_cf = true;
        plus_cf = next_plus_cf;
        minus_cf = next_minus_cf;
        prev = value;

        double plus = plus_change - next_minus_cf;
        double minus = minus_change - next_plus_cf;

        if (comparisons_seen < length) {
            plus_ring[comparisons_seen] = plus;
            minus_ring[comparisons_seen] = minus;
            sum_plus += plus;
            sum_minus += minus;
            comparisons_seen += 1;
            if (comparisons_seen == length) {
                row_plus[i] = sum_plus;
                row_minus[i] = sum_minus;
            }
            continue;
        }

        double old_plus = plus_ring[head];
        double old_minus = minus_ring[head];
        plus_ring[head] = plus;
        minus_ring[head] = minus;
        sum_plus += plus - old_plus;
        sum_minus += minus - old_minus;
        head += 1;
        if (head == length) {
            head = 0;
        }

        row_plus[i] = sum_plus;
        row_minus[i] = sum_minus;
    }
}
