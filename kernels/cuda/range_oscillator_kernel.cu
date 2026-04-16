#include <cmath>
#include <cstddef>

namespace {
constexpr int ATR_FALLBACK_PERIOD = 200;
constexpr int ATR_PRIMARY_PERIOD = 2000;
constexpr double ZERO_EPS = 1e-12;

struct AtrState {
    int count;
    double sum;
    double value;
    bool seeded;
    double prev_close;
    bool have_prev;

    __device__ void reset() {
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
        prev_close = NAN;
        have_prev = false;
    }

    __device__ double update(int period, double high, double low, double close, bool* ready) {
        const double tr = have_prev
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        have_prev = true;

        if (seeded) {
            value = (value * static_cast<double>(period - 1) + tr) / static_cast<double>(period);
            *ready = true;
            return value;
        }

        count += 1;
        sum += tr;
        if (count == period) {
            value = sum / static_cast<double>(period);
            seeded = true;
            *ready = true;
            return value;
        }

        *ready = false;
        return NAN;
    }
};

__device__ inline void push_close(double* ring, int cap, int* head, int* count, double value) {
    if (*count < cap) {
        ring[(*head + *count) % cap] = value;
        *count += 1;
        return;
    }
    ring[*head] = value;
    *head += 1;
    if (*head == cap) {
        *head = 0;
    }
}

__device__ inline int back_index(int cap, int head, int count, int offset) {
    int idx = head + count - 1 - offset;
    idx %= cap;
    if (idx < 0) {
        idx += cap;
    }
    return idx;
}
}

extern "C" __global__ void range_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* lengths,
    const double* mults,
    int rows,
    int storage_cols,
    double* out_oscillator,
    double* out_ma,
    double* out_upper_band,
    double* out_lower_band,
    double* out_range_width,
    double* out_in_range,
    double* out_trend,
    double* out_break_up,
    double* out_break_down,
    double* close_storage
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_ma = out_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper_band =
        out_upper_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower_band =
        out_lower_band + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_width =
        out_range_width + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_in_range = out_in_range + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_break_up = out_break_up + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_break_down =
        out_break_down + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = NAN;
        row_ma[i] = NAN;
        row_upper_band[i] = NAN;
        row_lower_band[i] = NAN;
        row_range_width[i] = NAN;
        row_in_range[i] = NAN;
        row_trend[i] = NAN;
        row_break_up[i] = NAN;
        row_break_down[i] = NAN;
    }

    if (length <= 0 || length + 1 > storage_cols || !isfinite(mult) || mult < 0.1) {
        return;
    }

    double* ring = close_storage + static_cast<size_t>(row) * static_cast<size_t>(storage_cols);
    int ring_head = 0;
    int ring_count = 0;
    double trend_state = 0.0;

    AtrState atr_fallback;
    AtrState atr_primary;
    atr_fallback.reset();
    atr_primary.reset();

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            atr_fallback.reset();
            atr_primary.reset();
            ring_head = 0;
            ring_count = 0;
            trend_state = 0.0;
            continue;
        }

        bool fallback_ready = false;
        bool primary_ready = false;
        const double atr200 = atr_fallback.update(ATR_FALLBACK_PERIOD, h, l, c, &fallback_ready);
        const double atr2000 = atr_primary.update(ATR_PRIMARY_PERIOD, h, l, c, &primary_ready);

        push_close(ring, storage_cols, &ring_head, &ring_count, c);

        const bool atr_ready = primary_ready || fallback_ready;
        if (!atr_ready || ring_count < length + 1) {
            continue;
        }

        const double atr_raw = primary_ready ? atr2000 : atr200;
        const double range_width = atr_raw * mult;

        double sum_weighted = 0.0;
        double sum_weights = 0.0;
        for (int j = 0; j < length; ++j) {
            const double curr = ring[back_index(storage_cols, ring_head, ring_count, j)];
            const double prev = ring[back_index(storage_cols, ring_head, ring_count, j + 1)];
            if (fabs(prev) <= ZERO_EPS) {
                continue;
            }
            const double weight = fabs(curr - prev) / prev;
            sum_weighted += curr * weight;
            sum_weights += weight;
        }
        if (fabs(sum_weights) <= ZERO_EPS) {
            continue;
        }

        const double ma = sum_weighted / sum_weights;
        double max_dist = 0.0;
        for (int j = 0; j < length; ++j) {
            const double value = ring[back_index(storage_cols, ring_head, ring_count, j)];
            const double dist = fabs(value - ma);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }

        if (c > ma) {
            trend_state = 1.0;
        } else if (c < ma) {
            trend_state = -1.0;
        }

        const double upper_band = ma + range_width;
        const double lower_band = ma - range_width;
        const double break_up = c > upper_band ? 1.0 : 0.0;
        const double break_down = c < lower_band ? 1.0 : 0.0;
        const double oscillator =
            fabs(range_width) <= ZERO_EPS ? NAN : (100.0 * (c - ma) / range_width);

        row_oscillator[i] = oscillator;
        row_ma[i] = ma;
        row_upper_band[i] = upper_band;
        row_lower_band[i] = lower_band;
        row_range_width[i] = range_width;
        row_in_range[i] = max_dist <= range_width ? 1.0 : 0.0;
        row_trend[i] = trend_state;
        row_break_up[i] = break_up;
        row_break_down[i] = break_down;
    }
}
