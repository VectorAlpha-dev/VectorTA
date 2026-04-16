#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {
constexpr int ATR_LENGTH = 14;
constexpr double ATR_MULTIPLIER = 1.2;
constexpr double VOLATILITY_THRESHOLD = 1.2;
constexpr double BULLISH_LOCATION_WEIGHT = 0.15;
constexpr double BEARISH_LOCATION_WEIGHT = 0.85;

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

__device__ inline unsigned char bool_window_get_ago(
    const unsigned char* ring,
    int count,
    int head,
    int len,
    int ago
) {
    if (ago >= count) {
        return 0;
    }
    if (count < len) {
        return ring[count - 1 - ago];
    }
    const int latest = head == 0 ? (len - 1) : (head - 1);
    return ring[(latest + len - ago) % len];
}
}

extern "C" __global__ void range_breakout_signals_batch_f64(
    const double* open,
    const double* high,
    const double* low,
    const double* close,
    const double* volume,
    int len,
    const int* range_lengths,
    const int* confirmation_lengths,
    int rows,
    int max_range_length,
    int max_confirmation_window,
    double* out_range_top,
    double* out_range_bottom,
    double* out_bullish,
    double* out_extra_bullish,
    double* out_bearish,
    double* out_extra_bearish,
    double* dist_ring_buffers,
    double* dist_sorted_buffers,
    double* up_volume_buffers,
    double* down_volume_buffers,
    unsigned char* under_buffers
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int range_length = range_lengths[row];
    const int confirmation_length = confirmation_lengths[row];
    const int confirmation_window = confirmation_length + 1;

    double* row_range_top = out_range_top + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_bottom =
        out_range_bottom + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bullish = out_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extra_bullish =
        out_extra_bullish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bearish = out_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_extra_bearish =
        out_extra_bearish + static_cast<size_t>(row) * static_cast<size_t>(len);

    double* row_dist_ring =
        dist_ring_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_range_length);
    double* row_dist_sorted =
        dist_sorted_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_range_length);
    double* row_up_volume =
        up_volume_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);
    double* row_down_volume = down_volume_buffers
        + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);
    unsigned char* row_under =
        under_buffers + static_cast<size_t>(row) * static_cast<size_t>(max_confirmation_window);

    for (int i = 0; i < len; ++i) {
        row_range_top[i] = NAN;
        row_range_bottom[i] = NAN;
        row_bullish[i] = NAN;
        row_extra_bullish[i] = NAN;
        row_bearish[i] = NAN;
        row_extra_bearish[i] = NAN;
    }

    if (range_length <= 0 || range_length > max_range_length || confirmation_length <= 0
        || confirmation_window > max_confirmation_window) {
        return;
    }

    int dist_head = 0;
    int dist_count = 0;
    int dist_sorted_count = 0;
    double dist_sum = 0.0;

    int atr_count = 0;
    double atr_sum = 0.0;
    double atr_value = NAN;
    double prev_close = NAN;
    bool have_prev_close = false;

    int volume_head = 0;
    int volume_count = 0;
    double up_volume_sum = 0.0;
    double down_volume_sum = 0.0;

    int under_head = 0;
    int under_count = 0;

    double prev_volatility = NAN;
    bool active_range = false;
    double active_top = NAN;
    double active_bottom = NAN;

    for (int i = 0; i < len; ++i) {
        const double o = open[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!isfinite(o) || !isfinite(h) || !isfinite(l) || !isfinite(c) || !isfinite(v)) {
            dist_head = 0;
            dist_count = 0;
            dist_sorted_count = 0;
            dist_sum = 0.0;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = NAN;
            prev_close = NAN;
            have_prev_close = false;
            volume_head = 0;
            volume_count = 0;
            up_volume_sum = 0.0;
            down_volume_sum = 0.0;
            under_head = 0;
            under_count = 0;
            prev_volatility = NAN;
            active_range = false;
            active_top = NAN;
            active_bottom = NAN;
            continue;
        }

        const double tr_prev_close = have_prev_close ? prev_close : c;
        const double tr = fmax(h - l, fmax(fabs(h - tr_prev_close), fabs(l - tr_prev_close)));
        prev_close = c;
        have_prev_close = true;

        bool atr_ready = false;
        if (atr_count < ATR_LENGTH) {
            atr_count += 1;
            atr_sum += tr;
            if (atr_count == ATR_LENGTH) {
                atr_value = atr_sum / static_cast<double>(ATR_LENGTH);
                atr_ready = true;
            }
        } else {
            atr_value =
                ((atr_value * static_cast<double>(ATR_LENGTH - 1)) + tr) / static_cast<double>(ATR_LENGTH);
            atr_ready = true;
        }

        const double dist_value = fabs(c - o);
        if (dist_count == range_length) {
            const double old = row_dist_ring[dist_head];
            dist_sum -= old;
            remove_sorted_once(row_dist_sorted, &dist_sorted_count, old);
            row_dist_ring[dist_head] = dist_value;
            dist_head += 1;
            if (dist_head == range_length) {
                dist_head = 0;
            }
        } else {
            row_dist_ring[dist_count] = dist_value;
            dist_count += 1;
            if (dist_count == range_length) {
                dist_head = 0;
            }
        }
        dist_sum += dist_value;
        insert_sorted(row_dist_sorted, &dist_sorted_count, dist_value, max_range_length);

        double volatility = NAN;
        if (dist_count == range_length) {
            const double median =
                (range_length & 1) == 1
                    ? row_dist_sorted[range_length >> 1]
                    : (row_dist_sorted[(range_length >> 1) - 1]
                       + row_dist_sorted[range_length >> 1])
                        * 0.5;
            if (median > 0.0) {
                volatility = (dist_sum / static_cast<double>(range_length)) / median;
            }
        }

        const bool current_isunder = isfinite(volatility) && volatility < VOLATILITY_THRESHOLD;
        double up_volume = 0.0;
        double down_volume = 0.0;
        if (c > o) {
            up_volume = v;
        } else if (c < o) {
            down_volume = v;
        } else {
            up_volume = v * 0.5;
            down_volume = v * 0.5;
        }

        if (volume_count == confirmation_window) {
            up_volume_sum -= row_up_volume[volume_head];
            down_volume_sum -= row_down_volume[volume_head];
            row_up_volume[volume_head] = up_volume;
            row_down_volume[volume_head] = down_volume;
            volume_head += 1;
            if (volume_head == confirmation_window) {
                volume_head = 0;
            }
        } else {
            row_up_volume[volume_count] = up_volume;
            row_down_volume[volume_count] = down_volume;
            volume_count += 1;
            if (volume_count == confirmation_window) {
                volume_head = 0;
            }
        }
        up_volume_sum += up_volume;
        down_volume_sum += down_volume;

        if (under_count == confirmation_window) {
            row_under[under_head] = current_isunder ? 1 : 0;
            under_head += 1;
            if (under_head == confirmation_window) {
                under_head = 0;
            }
        } else {
            row_under[under_count] = current_isunder ? 1 : 0;
            under_count += 1;
            if (under_count == confirmation_window) {
                under_head = 0;
            }
        }

        const bool ready = isfinite(volatility) && atr_ready && isfinite(prev_volatility)
            && volume_count == confirmation_window && under_count == confirmation_window;

        double range_top = NAN;
        double range_bottom = NAN;
        double bullish = NAN;
        double extra_bullish = NAN;
        double bearish = NAN;
        double extra_bearish = NAN;

        if (ready) {
            const bool under_ago =
                bool_window_get_ago(
                    row_under,
                    under_count,
                    under_head,
                    confirmation_window,
                    confirmation_length
                )
                != 0;
            const bool crossed_under =
                prev_volatility >= VOLATILITY_THRESHOLD && volatility < VOLATILITY_THRESHOLD;
            if (!active_range && crossed_under && current_isunder && under_ago) {
                const double offset = atr_value * ATR_MULTIPLIER;
                active_top = c + offset;
                active_bottom = c - offset;
                active_range = true;
            }

            if (active_range) {
                range_top = active_top;
                range_bottom = active_bottom;
                if (c > active_top || c < active_bottom) {
                    const bool bullish_break = c > active_top;
                    const double location = active_bottom
                        + (active_top - active_bottom)
                            * (bullish_break ? BULLISH_LOCATION_WEIGHT : BEARISH_LOCATION_WEIGHT);
                    const bool bullish_volume_bias = up_volume_sum > down_volume_sum;
                    if (bullish_break) {
                        bullish = location;
                        if (bullish_volume_bias) {
                            extra_bullish = location;
                        }
                    } else {
                        bearish = location;
                        if (!bullish_volume_bias) {
                            extra_bearish = location;
                        }
                    }
                    active_range = false;
                    active_top = NAN;
                    active_bottom = NAN;
                }
            }

            row_range_top[i] = range_top;
            row_range_bottom[i] = range_bottom;
            row_bullish[i] = bullish;
            row_extra_bullish[i] = extra_bullish;
            row_bearish[i] = bearish;
            row_extra_bearish[i] = extra_bearish;
        }

        prev_volatility = isfinite(volatility) ? volatility : NAN;
    }
}
