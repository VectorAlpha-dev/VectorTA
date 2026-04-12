#include <cmath>
#include <cstddef>

static __device__ inline double amo_linreg_from_ring(
    const double* ring,
    int head,
    int period
) {
    double x_sum = 0.0;
    double x2_sum = 0.0;
    for (int i = 1; i <= period; ++i) {
        double x = static_cast<double>(i);
        x_sum += x;
        x2_sum += x * x;
    }

    double period_f = static_cast<double>(period);
    double denom = period_f * x2_sum - x_sum * x_sum;
    if (denom == 0.0 || !isfinite(denom)) {
        return NAN;
    }

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int i = 0; i < period; ++i) {
        double y = ring[(head + i) % period];
        y_sum += y;
        xy_sum += y * static_cast<double>(i + 1);
    }

    double b = (period_f * xy_sum - x_sum * y_sum) / denom;
    double a = (y_sum - b * x_sum) / period_f;
    return a + b * period_f;
}

extern "C" __global__ void adaptive_momentum_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smoothing_lengths,
    int rows,
    int max_length,
    int max_smoothing_length,
    double* __restrict__ raw_ring_buf,
    double* __restrict__ change_ring_buf,
    double* __restrict__ linreg_ring_buf,
    double* __restrict__ out_amo,
    double* __restrict__ out_ama
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    int smoothing_length = smoothing_lengths[row];
    double* raw_ring =
        raw_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* change_ring =
        change_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* linreg_ring =
        linreg_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smoothing_length);
    double* row_out_amo = out_amo + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_ama = out_ama + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_amo[i] = NAN;
        row_out_ama[i] = NAN;
    }

    if (length <= 0 || smoothing_length <= 0 || length > max_length ||
        smoothing_length > max_smoothing_length) {
        return;
    }

    int raw_head = 0;
    int raw_count = 0;
    int linreg_head = 0;
    bool linreg_filled = false;
    int change_head = 0;
    int change_count = 0;
    double change_sum = 0.0;
    bool avg_have_prev = false;
    double avg_prev = NAN;
    double ama_value = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];

        double raw = NAN;
        if (isfinite(value) && raw_count >= length) {
            bool valid = true;
            double best_abs = -1.0;
            double best_delta = NAN;
            for (int lag = 1; lag <= length; ++lag) {
                int hist_idx = (raw_head + length - lag) % length;
                double past = raw_ring[hist_idx];
                if (!isfinite(past)) {
                    valid = false;
                    break;
                }
                double delta = value - past;
                double abs_delta = fabs(delta);
                if (abs_delta >= best_abs) {
                    best_abs = abs_delta;
                    best_delta = delta;
                }
            }
            if (valid) {
                raw = best_delta;
            }
        }

        raw_ring[raw_head] = value;
        raw_head += 1;
        if (raw_head == length) {
            raw_head = 0;
        }
        if (raw_count < length) {
            raw_count += 1;
        }

        linreg_ring[linreg_head] = raw;
        linreg_head += 1;
        if (linreg_head == smoothing_length) {
            linreg_head = 0;
            linreg_filled = true;
        }

        double amo = NAN;
        if (linreg_filled) {
            amo = amo_linreg_from_ring(linreg_ring, linreg_head, smoothing_length);
        }

        double change = 0.0;
        if (avg_have_prev && isfinite(amo) && isfinite(avg_prev)) {
            change = fabs(amo - avg_prev);
        }
        double normalized_change = isfinite(change) ? change : 0.0;
        if (change_count < length) {
            change_ring[change_head] = normalized_change;
            change_sum += normalized_change;
            change_count += 1;
        } else {
            double old = change_ring[change_head];
            change_ring[change_head] = normalized_change;
            change_sum += normalized_change - old;
        }
        change_head += 1;
        if (change_head == length) {
            change_head = 0;
        }

        if (isfinite(amo) && change_sum > 0.0) {
            double efficiency_ratio = fabs(amo) / change_sum;
            double delta = efficiency_ratio * (amo - ama_value);
            if (isfinite(delta)) {
                ama_value += delta;
            }
        }

        avg_prev = amo;
        avg_have_prev = true;

        if (isfinite(amo)) {
            row_out_amo[i] = amo;
            row_out_ama[i] = ama_value;
        }
    }
}
