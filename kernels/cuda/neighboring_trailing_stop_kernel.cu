#include <cmath>
#include <cstddef>

namespace {
__device__ inline int lower_bound_sorted(const double* sorted, int size, double value) {
    int left = 0;
    int right = size;
    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        if (sorted[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ inline void insert_sorted(double* sorted, int* size, double value, int capacity) {
    if (*size >= capacity) {
        return;
    }
    const int idx = lower_bound_sorted(sorted, *size, value);
    for (int i = *size; i > idx; --i) {
        sorted[i] = sorted[i - 1];
    }
    sorted[idx] = value;
    *size += 1;
}

__device__ inline void remove_sorted_once(double* sorted, int* size, double value) {
    const int idx = lower_bound_sorted(sorted, *size, value);
    if (idx < *size && sorted[idx] == value) {
        for (int i = idx; i + 1 < *size; ++i) {
            sorted[i] = sorted[i + 1];
        }
        *size -= 1;
    }
}

__device__ inline double percentile_sorted_slice(
    const double* sorted,
    int size,
    double percentile
) {
    if (size <= 0) {
        return NAN;
    }
    if (size == 1) {
        return sorted[0];
    }

    const double idx = static_cast<double>(size - 1) * percentile / 100.0;
    const int i1 = static_cast<int>(floor(idx));
    const int i2 = static_cast<int>(ceil(idx));
    if (i1 == i2) {
        return sorted[i1];
    }
    const double v1 = sorted[i1];
    const double v2 = sorted[i2];
    return v1 + (v2 - v1) * (idx - static_cast<double>(i1));
}

__device__ inline void sma_reset(
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    *count = 0;
    *head = 0;
    *sum = 0.0;
    for (int i = 0; i < period; ++i) {
        buffer[i] = 0.0;
    }
}

__device__ inline double sma_update_ignore_nan(
    double value,
    int* count,
    int* head,
    double* sum,
    double* buffer,
    int period
) {
    if (isfinite(value)) {
        if (*count < period) {
            buffer[*count] = value;
            *sum += value;
            *count += 1;
        } else {
            const double old = buffer[*head];
            buffer[*head] = value;
            *sum += value - old;
            *head += 1;
            if (*head == period) {
                *head = 0;
            }
        }
    }

    return *count == period ? (*sum / static_cast<double>(period)) : NAN;
}
}

extern "C" __global__ void neighboring_trailing_stop_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* buffer_sizes,
    const int* ks,
    const double* percentiles,
    const int* smooths,
    int rows,
    int max_buffer_size,
    int max_smooth,
    double* out_trailing_stop,
    double* out_bullish_band,
    double* out_bearish_band,
    double* out_direction,
    double* out_discovery_bull,
    double* out_discovery_bear,
    double* price_buffers,
    double* sorted_buffers,
    double* bull_sma_buffers,
    double* bear_sma_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int buffer_size = buffer_sizes[row];
    const int k = ks[row];
    const double percentile = percentiles[row];
    const int smooth = smooths[row];

    double* row_trailing_stop =
        out_trailing_stop + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish_band =
        out_bullish_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish_band =
        out_bearish_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction = out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_discovery_bull =
        out_discovery_bull + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_discovery_bear =
        out_discovery_bear + static_cast<size_t>(row) * static_cast<size_t>(len);

    double* row_price_buffer =
        price_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_buffer_size);
    double* row_sorted_buffer =
        sorted_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_buffer_size);
    double* row_bull_sma =
        bull_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_smooth);
    double* row_bear_sma =
        bear_sma_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_smooth);

    for (int i = 0; i < len; ++i) {
        row_trailing_stop[i] = NAN;
        row_bullish_band[i] = NAN;
        row_bearish_band[i] = NAN;
        row_direction[i] = NAN;
        row_discovery_bull[i] = NAN;
        row_discovery_bear[i] = NAN;
    }

    if (buffer_size < 100 || buffer_size > max_buffer_size || k < 5 || !isfinite(percentile)
        || percentile < 1.0 || percentile > 99.0 || smooth <= 0 || smooth > max_smooth) {
        return;
    }

    int price_count = 0;
    int price_head = 0;
    int sorted_count = 0;
    int bull_sma_count = 0;
    int bull_sma_head = 0;
    double bull_sma_sum = 0.0;
    int bear_sma_count = 0;
    int bear_sma_head = 0;
    double bear_sma_sum = 0.0;
    int direction = 0;
    double trailing_stop = NAN;

    sma_reset(&bull_sma_count, &bull_sma_head, &bull_sma_sum, row_bull_sma, smooth);
    sma_reset(&bear_sma_count, &bear_sma_head, &bear_sma_sum, row_bear_sma, smooth);

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            price_count = 0;
            price_head = 0;
            sorted_count = 0;
            direction = 0;
            trailing_stop = NAN;
            sma_reset(&bull_sma_count, &bull_sma_head, &bull_sma_sum, row_bull_sma, smooth);
            sma_reset(&bear_sma_count, &bear_sma_head, &bear_sma_sum, row_bear_sma, smooth);
            continue;
        }

        double bear_val = NAN;
        double bull_val = NAN;
        const int size = sorted_count;

        if (size > 5) {
            const int idx = lower_bound_sorted(row_sorted_buffer, size, c);
            const int bear_start = idx > k ? (idx - k) : 0;
            if (idx > bear_start) {
                bear_val = percentile_sorted_slice(
                    row_sorted_buffer + bear_start,
                    idx - bear_start,
                    100.0 - percentile
                );
            }

            if (size > 0) {
                const int bull_end = min(idx + k, size - 1);
                if (bull_end > idx) {
                    bull_val = percentile_sorted_slice(
                        row_sorted_buffer + idx,
                        bull_end - idx + 1,
                        percentile
                    );
                }
            }
        }

        if (price_count < buffer_size) {
            const int insert_idx = (price_head + price_count) % buffer_size;
            row_price_buffer[insert_idx] = c;
            price_count += 1;
        } else {
            const double old = row_price_buffer[price_head];
            remove_sorted_once(row_sorted_buffer, &sorted_count, old);
            row_price_buffer[price_head] = c;
            price_head += 1;
            if (price_head == buffer_size) {
                price_head = 0;
            }
        }
        insert_sorted(row_sorted_buffer, &sorted_count, c, max_buffer_size);

        const double final_bull = sma_update_ignore_nan(
            bull_val,
            &bull_sma_count,
            &bull_sma_head,
            &bull_sma_sum,
            row_bull_sma,
            smooth
        );
        const double final_bear = sma_update_ignore_nan(
            bear_val,
            &bear_sma_count,
            &bear_sma_head,
            &bear_sma_sum,
            row_bear_sma,
            smooth
        );

        const bool discovery_bull = !isfinite(bull_val) && isfinite(bear_val);
        const bool discovery_bear = !isfinite(bear_val) && isfinite(bull_val);

        const int prev_direction = direction;
        if (discovery_bull) {
            direction = 1;
        } else if (discovery_bear) {
            direction = -1;
        }

        if (direction > prev_direction) {
            trailing_stop = isfinite(final_bear) ? final_bear : l;
        } else if (direction < prev_direction) {
            trailing_stop = isfinite(final_bull) ? final_bull : h;
        }

        if (direction == 1) {
            const double candidate = isfinite(final_bear) ? final_bear : trailing_stop;
            trailing_stop = isfinite(trailing_stop) ? fmax(trailing_stop, candidate) : candidate;
        } else if (direction == -1) {
            const double candidate = isfinite(final_bull) ? final_bull : trailing_stop;
            trailing_stop = isfinite(trailing_stop) ? fmin(trailing_stop, candidate) : candidate;
        }

        row_trailing_stop[i] = trailing_stop;
        row_bullish_band[i] = final_bull;
        row_bearish_band[i] = final_bear;
        row_direction[i] = static_cast<double>(direction);
        row_discovery_bull[i] = discovery_bull ? 1.0 : 0.0;
        row_discovery_bear[i] = discovery_bear ? 1.0 : 0.0;
    }
}
