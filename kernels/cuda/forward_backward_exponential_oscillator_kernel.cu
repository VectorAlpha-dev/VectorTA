#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline double ring_at(
    const double* ring,
    int head,
    int period,
    int idx
) {
    int pos = head + idx;
    if (pos >= period) {
        pos -= period;
    }
    return ring[pos];
}

__device__ inline double compute_forward_backward_value(
    const double* ema1_ring,
    int ema1_head,
    int length,
    double alpha
) {
    double current = ring_at(ema1_ring, ema1_head, length, length - 1);
    double ema2 = current;
    double prev = ema2;
    double num = 0.0;
    double den = 0.0;

    for (int idx = length - 2; idx >= 0; --idx) {
        double value = ring_at(ema1_ring, ema1_head, length, idx);
        ema2 += alpha * (value - ema2);
        double dt = prev - ema2;
        num += dt;
        den += fabs(dt);
        prev = ema2;
    }

    if (den == 0.0) {
        return CUDART_NAN;
    }
    return num / den * 50.0 + 50.0;
}
}

extern "C" __global__ void forward_backward_exponential_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooths,
    int n_combos,
    int max_length,
    double* __restrict__ ema1_buffer,
    double* __restrict__ diff_buffer,
    double* __restrict__ out_forward_backward,
    double* __restrict__ out_backward,
    double* __restrict__ out_histogram
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    int smooth = smooths[combo_idx];
    double* ema1_ring =
        ema1_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* diff_ring =
        diff_buffer + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_length);
    double* row_forward_backward =
        out_forward_backward + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_backward =
        out_backward + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_histogram =
        out_histogram + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_forward_backward[i] = CUDART_NAN;
        row_backward[i] = CUDART_NAN;
        row_histogram[i] = CUDART_NAN;
    }

    if (length <= 0 || length > max_length || smooth <= 0) {
        return;
    }

    double alpha = 2.0 / (static_cast<double>(smooth) + 1.0);
    double beta = 1.0 - alpha;

    bool have_ema1_state = false;
    bool have_ema2_state = false;
    bool have_prev_ema2 = false;
    double ema1_state = CUDART_NAN;
    double ema2_state = CUDART_NAN;
    double prev_ema2 = CUDART_NAN;

    int ema1_count = 0;
    int ema1_head = 0;

    int diff_count = 0;
    int diff_head = 0;
    double diff_sum = 0.0;
    double diff_abs_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            have_ema1_state = false;
            have_ema2_state = false;
            have_prev_ema2 = false;
            ema1_state = CUDART_NAN;
            ema2_state = CUDART_NAN;
            prev_ema2 = CUDART_NAN;
            ema1_count = 0;
            ema1_head = 0;
            diff_count = 0;
            diff_head = 0;
            diff_sum = 0.0;
            diff_abs_sum = 0.0;
            continue;
        }

        if (have_ema1_state) {
            ema1_state = alpha * value + beta * ema1_state;
        } else {
            ema1_state = value;
            have_ema1_state = true;
        }

        if (ema1_count < length) {
            ema1_ring[ema1_count] = ema1_state;
            ema1_count += 1;
        } else {
            ema1_ring[ema1_head] = ema1_state;
            ema1_head += 1;
            if (ema1_head == length) {
                ema1_head = 0;
            }
        }

        if (ema1_count == length) {
            row_forward_backward[i] =
                compute_forward_backward_value(ema1_ring, ema1_head, length, alpha);
        }

        if (have_ema2_state) {
            ema2_state = alpha * ema1_state + beta * ema2_state;
        } else {
            ema2_state = ema1_state;
            have_ema2_state = true;
        }

        if (have_prev_ema2) {
            double diff = ema2_state - prev_ema2;
            if (diff_count < length) {
                diff_ring[diff_count] = diff;
                diff_count += 1;
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
            } else {
                double removed = diff_ring[diff_head];
                diff_sum -= removed;
                diff_abs_sum -= fabs(removed);
                diff_ring[diff_head] = diff;
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
                diff_head += 1;
                if (diff_head == length) {
                    diff_head = 0;
                }
            }

            if (diff_count == length && diff_abs_sum != 0.0) {
                double backward = diff_sum / diff_abs_sum * 50.0 + 50.0;
                row_backward[i] = backward;
                double forward_backward = row_forward_backward[i];
                if (isfinite(forward_backward)) {
                    row_histogram[i] = (forward_backward - backward) * 0.25 + 50.0;
                }
            }
        }

        prev_ema2 = ema2_state;
        have_prev_ema2 = true;
    }
}
