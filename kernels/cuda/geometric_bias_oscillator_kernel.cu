#include <cmath>
#include <cstddef>

static __device__ inline double geometric_bias_true_range(
    double high,
    double low,
    double prev_close
) {
    double value = high - low;
    double high_close = fabs(high - prev_close);
    double low_close = fabs(low - prev_close);
    if (high_close > value) {
        value = high_close;
    }
    if (low_close > value) {
        value = low_close;
    }
    return value;
}

static __device__ inline double point_line_distance(
    double x1,
    double y1,
    double x2,
    double y2,
    double x0,
    double y0
) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double denominator = sqrt(dx * dx + dy * dy);
    if (denominator == 0.0) {
        return 0.0;
    }
    return fabs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denominator;
}

static __device__ inline double compute_raw_geometric_bias(
    const double* ordered,
    int length,
    double atr,
    double threshold,
    int* keep,
    int* stack_start,
    int* stack_end
) {
    if (!isfinite(atr) || atr <= 0.0) {
        return 0.0;
    }

    for (int i = 0; i < length; ++i) {
        keep[i] = 0;
    }
    keep[0] = 1;
    keep[length - 1] = 1;

    int stack_len = 0;
    stack_start[stack_len] = 0;
    stack_end[stack_len] = length - 1;
    stack_len += 1;

    double atr_inv = 1.0 / atr;
    while (stack_len > 0) {
        stack_len -= 1;
        int first_idx = stack_start[stack_len];
        int last_idx = stack_end[stack_len];
        if (last_idx <= first_idx + 1) {
            continue;
        }

        double x1 = static_cast<double>(first_idx);
        double y1 = ordered[first_idx] * atr_inv;
        double x2 = static_cast<double>(last_idx);
        double y2 = ordered[last_idx] * atr_inv;

        double max_dist = 0.0;
        int split_idx = first_idx;
        for (int i = first_idx + 1; i < last_idx; ++i) {
            double distance =
                point_line_distance(x1, y1, x2, y2, static_cast<double>(i), ordered[i] * atr_inv);
            if (distance > max_dist) {
                max_dist = distance;
                split_idx = i;
            }
        }

        if (max_dist > threshold) {
            keep[split_idx] = 1;
            stack_start[stack_len] = first_idx;
            stack_end[stack_len] = split_idx;
            stack_len += 1;
            stack_start[stack_len] = split_idx;
            stack_end[stack_len] = last_idx;
            stack_len += 1;
        }
    }

    double bull_sum = 0.0;
    double bear_sum = 0.0;
    int last_kept = 0;
    for (int i = 1; i < length; ++i) {
        if (keep[i] != 0) {
            double diff = ordered[i] - ordered[last_kept];
            if (diff > 0.0) {
                bull_sum += diff;
            } else if (diff < 0.0) {
                bear_sum += -diff;
            }
            last_kept = i;
        }
    }

    double total = bull_sum + bear_sum;
    if (total > 0.0) {
        return ((bull_sum - bear_sum) / total) * 100.0;
    }
    return 0.0;
}

extern "C" __global__ void geometric_bias_oscillator_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ multipliers,
    const int* __restrict__ atr_lengths,
    const int* __restrict__ smooths,
    int rows,
    int max_length,
    int max_smooth,
    double* __restrict__ price_ring_buf,
    double* __restrict__ ordered_buf,
    int* __restrict__ keep_buf,
    int* __restrict__ stack_start_buf,
    int* __restrict__ stack_end_buf,
    double* __restrict__ smooth_ring_buf,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    double multiplier = multipliers[row];
    int atr_length = atr_lengths[row];
    int smooth = smooths[row];

    double* price_ring = price_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* ordered = ordered_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* keep = keep_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* stack_start =
        stack_start_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* stack_end = stack_end_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* smooth_ring =
        smooth_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_smooth);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (length <= 0 ||
        length > max_length ||
        atr_length <= 0 ||
        smooth <= 0 ||
        smooth > max_smooth ||
        !isfinite(multiplier) ||
        multiplier < 0.1) {
        return;
    }

    int price_head = 0;
    int price_count = 0;
    int smooth_head = 0;
    int smooth_count = 0;
    double smooth_sum = 0.0;
    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double prev_close = NAN;

    for (int i = 0; i < len; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            price_head = 0;
            price_count = 0;
            smooth_head = 0;
            smooth_count = 0;
            smooth_sum = 0.0;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = NAN;
            prev_close = NAN;
            row_out[i] = NAN;
            continue;
        }

        double tr = isfinite(prev_close) ? geometric_bias_true_range(h, l, prev_close) : (h - l);
        prev_close = c;

        price_ring[price_head] = c;
        price_head += 1;
        if (price_head == length) {
            price_head = 0;
        }
        if (price_count < length) {
            price_count += 1;
        }

        if (atr_count < atr_length) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == atr_length) {
                atr_value = atr_sum / static_cast<double>(atr_length);
            }
        } else {
            atr_value = ((atr_value * static_cast<double>(atr_length - 1)) + tr) /
                        static_cast<double>(atr_length);
        }

        if (atr_count < atr_length || price_count < length) {
            row_out[i] = NAN;
            continue;
        }

        for (int j = 0; j < length; ++j) {
            ordered[j] = price_ring[(price_head + j) % length];
        }
        double raw = compute_raw_geometric_bias(
            ordered,
            length,
            atr_value,
            multiplier,
            keep,
            stack_start,
            stack_end
        );

        if (smooth_count == smooth) {
            smooth_sum -= smooth_ring[smooth_head];
        } else {
            smooth_count += 1;
        }
        smooth_ring[smooth_head] = raw;
        smooth_sum += raw;
        smooth_head += 1;
        if (smooth_head == smooth) {
            smooth_head = 0;
        }

        row_out[i] = smooth_count == smooth ? smooth_sum / static_cast<double>(smooth) : NAN;
    }
}
