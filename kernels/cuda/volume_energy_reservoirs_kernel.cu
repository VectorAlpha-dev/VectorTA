#include <cmath>
#include <cstddef>

namespace {

constexpr int VOLUME_STDEV_LENGTH = 100;
constexpr double MOMENTUM_EMA_ALPHA = 2.0 / 6.0;
constexpr double RESERVOIR_CAP = 10.0;
constexpr double RESERVOIR_SQUEEZE_THRESHOLD = 5.0;
constexpr double STABILITY_THRESHOLD = 0.2;
constexpr double FLOAT_TOL = 1.0e-12;

__device__ inline bool finite_ohlcv(double high, double low, double close, double volume) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(volume);
}

struct ExtremumDeque {
    int* indices;
    double* values;
    int head;
    int tail;
    int cap;

    __device__ void init(int* indices_ptr, double* values_ptr, int cap_value) {
        indices = indices_ptr;
        values = values_ptr;
        cap = cap_value;
        reset();
    }

    __device__ void reset() {
        head = 0;
        tail = 0;
    }

    __device__ void compact() {
        if (head <= 0) {
            return;
        }
        const int size = tail - head;
        for (int i = 0; i < size; ++i) {
            indices[i] = indices[head + i];
            values[i] = values[head + i];
        }
        head = 0;
        tail = size;
    }

    __device__ void ensure_capacity() {
        if (tail >= cap && head > 0) {
            compact();
        }
    }

    __device__ void normalize_if_empty() {
        if (head == tail) {
            head = 0;
            tail = 0;
        }
    }

    __device__ double front_value(double fallback) const {
        return tail > head ? values[head] : fallback;
    }

    __device__ void push_high(int idx, double value, int length) {
        while (tail > head && values[tail - 1] <= value) {
            tail -= 1;
        }
        ensure_capacity();
        if (tail < cap) {
            indices[tail] = idx;
            values[tail] = value;
            tail += 1;
        }
        while (tail > head && indices[head] + length <= idx) {
            head += 1;
        }
        normalize_if_empty();
    }

    __device__ void push_low(int idx, double value, int length) {
        while (tail > head && values[tail - 1] >= value) {
            tail -= 1;
        }
        ensure_capacity();
        if (tail < cap) {
            indices[tail] = idx;
            values[tail] = value;
            tail += 1;
        }
        while (tail > head && indices[head] + length <= idx) {
            head += 1;
        }
        normalize_if_empty();
    }
};

}

extern "C" __global__ void volume_energy_reservoirs_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ lengths,
    const double* __restrict__ sensitivities,
    int rows,
    int window_cap,
    int* __restrict__ high_idx_scratch,
    double* __restrict__ high_val_scratch,
    int* __restrict__ low_idx_scratch,
    double* __restrict__ low_val_scratch,
    double* __restrict__ out_momentum,
    double* __restrict__ out_reservoir,
    double* __restrict__ out_squeeze_active,
    double* __restrict__ out_squeeze_start,
    double* __restrict__ out_range_high,
    double* __restrict__ out_range_low
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const double sensitivity = sensitivities[row];

    double* row_momentum = out_momentum + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_reservoir = out_reservoir + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_squeeze_active =
        out_squeeze_active + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_squeeze_start =
        out_squeeze_start + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_high =
        out_range_high + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_range_low = out_range_low + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_momentum[i] = NAN;
        row_reservoir[i] = NAN;
        row_squeeze_active[i] = NAN;
        row_squeeze_start[i] = NAN;
        row_range_high[i] = NAN;
        row_range_low[i] = NAN;
    }

    if (length < 5 || !isfinite(sensitivity) || sensitivity < 0.5) {
        return;
    }

    ExtremumDeque high_window;
    ExtremumDeque low_window;
    high_window.init(
        high_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        high_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        window_cap
    );
    low_window.init(
        low_idx_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        low_val_scratch + static_cast<size_t>(row) * static_cast<size_t>(window_cap),
        window_cap
    );

    double volume_ring[VOLUME_STDEV_LENGTH];
    for (int i = 0; i < VOLUME_STDEV_LENGTH; ++i) {
        volume_ring[i] = 0.0;
    }

    int segment_index = 0;
    int volume_head = 0;
    int volume_count = 0;
    double volume_sum = 0.0;
    double volume_sum_sq = 0.0;
    double reservoir = 0.0;
    double ema = 0.0;
    bool ema_ready = false;
    bool prev_squeeze_active = false;
    double current_high = NAN;
    double current_low = NAN;
    bool has_range = false;
    bool is_extending = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!finite_ohlcv(h, l, c, v)) {
            segment_index = 0;
            volume_head = 0;
            volume_count = 0;
            volume_sum = 0.0;
            volume_sum_sq = 0.0;
            high_window.reset();
            low_window.reset();
            reservoir = 0.0;
            ema = 0.0;
            ema_ready = false;
            prev_squeeze_active = false;
            current_high = NAN;
            current_low = NAN;
            has_range = false;
            is_extending = false;
            continue;
        }

        const int idx = segment_index;
        segment_index += 1;

        if (volume_count == VOLUME_STDEV_LENGTH) {
            const double old = volume_ring[volume_head];
            volume_sum -= old;
            volume_sum_sq -= old * old;
        } else {
            volume_count += 1;
        }
        volume_ring[volume_head] = v;
        volume_head = (volume_head + 1) % VOLUME_STDEV_LENGTH;
        volume_sum += v;
        volume_sum_sq += v * v;

        high_window.push_high(idx, h, length);
        low_window.push_low(idx, l, length);

        const double hi = high_window.front_value(h);
        const double lo = low_window.front_value(l);
        const double mid_price = 0.5 * (hi + lo);
        const double price_range = hi - lo;
        const double hl2 = 0.5 * (h + l);
        const double price_rel =
            fabs(price_range) <= FLOAT_TOL ? 0.0 : (hl2 - mid_price) / price_range;

        double norm_vol = 0.0;
        if (volume_count >= VOLUME_STDEV_LENGTH) {
            const double mean = volume_sum / static_cast<double>(VOLUME_STDEV_LENGTH);
            const double variance =
                fmax(volume_sum_sq / static_cast<double>(VOLUME_STDEV_LENGTH) - mean * mean, 0.0);
            const double stdev = sqrt(variance);
            norm_vol = fabs(stdev) <= FLOAT_TOL ? 1.0 : (v / stdev);
        }

        if (norm_vol < 1.0 && fabs(price_rel) < STABILITY_THRESHOLD) {
            reservoir += 0.5;
        } else if (norm_vol > sensitivity) {
            reservoir *= 0.7;
        } else {
            reservoir = fmax(reservoir - 0.1, 0.0);
        }
        reservoir = fmin(reservoir, RESERVOIR_CAP);

        const double momentum = price_rel * norm_vol * 20.0;
        if (!ema_ready) {
            ema = momentum;
            ema_ready = true;
        } else {
            ema += MOMENTUM_EMA_ALPHA * (momentum - ema);
        }

        const bool squeeze_active = reservoir > RESERVOIR_SQUEEZE_THRESHOLD;
        const bool squeeze_start = squeeze_active && !prev_squeeze_active;
        const bool squeeze_end = !squeeze_active && prev_squeeze_active;

        if (squeeze_start) {
            current_high = h;
            current_low = l;
            has_range = true;
            is_extending = false;
        }

        if (squeeze_active && has_range) {
            current_high = fmax(current_high, h);
            current_low = fmin(current_low, l);
        }

        bool range_visible = squeeze_active || is_extending;
        if (squeeze_end && has_range) {
            is_extending = true;
            range_visible = true;
        }
        if (is_extending && has_range) {
            range_visible = true;
            if (c > current_high || c < current_low) {
                is_extending = false;
            }
        }

        prev_squeeze_active = squeeze_active;

        row_momentum[i] = ema;
        row_reservoir[i] = reservoir;
        row_squeeze_active[i] = squeeze_active ? 1.0 : 0.0;
        row_squeeze_start[i] = squeeze_start ? 1.0 : 0.0;
        row_range_high[i] = range_visible && has_range ? current_high : NAN;
        row_range_low[i] = range_visible && has_range ? current_low : NAN;
    }
}
