#include <cmath>
#include <cstddef>

namespace {
constexpr int MIN_DATA_LENGTH = 1;
constexpr int MIN_NORMALIZATION_LENGTH = 10;
constexpr double MIN_POSITIVE = 2.2250738585072014e-308;
constexpr double BIAS_BEARISH = 0.0;
constexpr double BIAS_BULLISH = 1.0;

struct MonoDeque {
    int* idx;
    double* vals;
    int cap;
    int head;
    int tail;
    int count;
    bool descending;

    __device__ void init(int* idx_ptr, double* vals_ptr, int capacity, bool is_descending) {
        idx = idx_ptr;
        vals = vals_ptr;
        cap = capacity;
        descending = is_descending;
        clear();
    }

    __device__ void clear() {
        head = 0;
        tail = 0;
        count = 0;
    }

    __device__ void pop_back() {
        if (count == 0) {
            return;
        }
        tail = (tail + cap - 1) % cap;
        count -= 1;
    }

    __device__ void pop_front() {
        if (count == 0) {
            return;
        }
        head = (head + 1) % cap;
        count -= 1;
    }

    __device__ int back_slot() const {
        return (tail + cap - 1) % cap;
    }

    __device__ void push(int index, double value) {
        while (count > 0) {
            const int slot = back_slot();
            const double last = vals[slot];
            const bool remove = descending ? (last <= value) : (last >= value);
            if (!remove) {
                break;
            }
            pop_back();
        }
        if (count == cap) {
            pop_front();
        }
        idx[tail] = index;
        vals[tail] = value;
        tail = (tail + 1) % cap;
        count += 1;
    }

    __device__ void expire(int min_index) {
        while (count > 0 && idx[head] < min_index) {
            pop_front();
        }
    }

    __device__ double front_value() const {
        return vals[head];
    }
};

struct RingHistory {
    double* values;
    int cap;
    int head;
    int count;

    __device__ void init(double* ptr, int capacity) {
        values = ptr;
        cap = capacity;
        clear();
    }

    __device__ void clear() {
        head = 0;
        count = 0;
    }

    __device__ void push(double value) {
        values[head] = value;
        head += 1;
        if (head == cap) {
            head = 0;
        }
        if (count < cap) {
            count += 1;
        }
    }

    __device__ bool get_from_end(int offset, double* out) const {
        if (offset <= 0 || offset > count) {
            return false;
        }
        int idx = head + cap - offset;
        idx %= cap;
        *out = values[idx];
        return true;
    }
};

struct RollingStats {
    double* ring;
    int cap;
    int head;
    int count;
    double sum;
    double sum_sq;

    __device__ void init(double* ptr, int capacity) {
        ring = ptr;
        cap = capacity;
        clear();
    }

    __device__ void clear() {
        head = 0;
        count = 0;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ bool push(double value, double* mean_out, double* stdev_out) {
        if (count < cap) {
            ring[head] = value;
            head += 1;
            if (head == cap) {
                head = 0;
            }
            count += 1;
            sum += value;
            sum_sq += value * value;
        } else {
            const double old = ring[head];
            ring[head] = value;
            head += 1;
            if (head == cap) {
                head = 0;
            }
            sum += value - old;
            sum_sq += value * value - old * old;
        }

        if (count < cap) {
            return false;
        }

        const double n = static_cast<double>(cap);
        const double mean = sum / n;
        const double variance = fmax(sum_sq / n - mean * mean, 0.0);
        *mean_out = mean;
        *stdev_out = sqrt(variance);
        return true;
    }
};

__device__ inline double hlc3(double high, double low, double close) {
    return (high + low + close) / 3.0;
}

__device__ inline double floor_positive(double value) {
    return value > 0.0 ? value : MIN_POSITIVE;
}
}

extern "C" __global__ void statistical_trailing_stop_batch_f64(
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* data_lengths,
    const int* normalization_lengths,
    const int* base_level_indices,
    int rows,
    int deque_cap,
    int stats_cap,
    double* out_level,
    double* out_anchor,
    double* out_bias,
    double* out_changed,
    int* max_high_idx_storage,
    double* max_high_val_storage,
    int* min_low_idx_storage,
    double* min_low_val_storage,
    double* close_history_storage,
    double* stats_ring_storage
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int data_length = data_lengths[row];
    const int normalization_length = normalization_lengths[row];
    const int base_level_index = base_level_indices[row];
    const int row_deque_cap = data_length + 2;

    double* row_level = out_level + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_anchor = out_anchor + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_bias = out_bias + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_level[i] = NAN;
        row_anchor[i] = NAN;
        row_bias[i] = NAN;
        row_changed[i] = NAN;
    }

    if (data_length < MIN_DATA_LENGTH || normalization_length < MIN_NORMALIZATION_LENGTH
        || data_length + normalization_length + 1 > len || base_level_index < 0
        || base_level_index > 3 || deque_cap < row_deque_cap || stats_cap < normalization_length) {
        return;
    }

    const size_t deque_offset = static_cast<size_t>(row) * static_cast<size_t>(deque_cap);
    const size_t stats_offset = static_cast<size_t>(row) * static_cast<size_t>(stats_cap);

    MonoDeque max_high;
    MonoDeque min_low;
    RingHistory close_history;
    RollingStats stats;

    max_high.init(
        max_high_idx_storage + deque_offset,
        max_high_val_storage + deque_offset,
        row_deque_cap,
        true
    );
    min_low.init(
        min_low_idx_storage + deque_offset,
        min_low_val_storage + deque_offset,
        row_deque_cap,
        false
    );
    close_history.init(close_history_storage + deque_offset, row_deque_cap);
    stats.init(stats_ring_storage + stats_offset, normalization_length);

    int valid_run = 0;
    double bias = BIAS_BEARISH;
    double level = NAN;
    double anchor = NAN;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];

        if (!isfinite(h) || !isfinite(l) || !isfinite(c)) {
            valid_run = 0;
            max_high.clear();
            min_low.clear();
            close_history.clear();
            stats.clear();
            bias = BIAS_BEARISH;
            level = NAN;
            anchor = NAN;
            continue;
        }

        valid_run += 1;
        max_high.push(i, h);
        min_low.push(i, l);
        const int window_start = i + 1 - (valid_run < data_length ? valid_run : data_length);
        max_high.expire(window_start);
        min_low.expire(window_start);
        close_history.push(c);

        if (valid_run < data_length + 2) {
            continue;
        }

        double previous_close = NAN;
        if (!close_history.get_from_end(data_length + 2, &previous_close)) {
            continue;
        }

        const double highest = max_high.front_value();
        const double lowest = min_low.front_value();
        const double tr =
            fmax(highest - lowest, fmax(fabs(highest - previous_close), fabs(lowest - previous_close)));

        double mean = NAN;
        double stdev = NAN;
        if (!stats.push(log(floor_positive(tr)), &mean, &stdev)) {
            continue;
        }

        const double delta = exp(mean + static_cast<double>(base_level_index) * stdev);
        const double current_hlc3 = hlc3(h, l, c);

        if (!isfinite(level)) {
            level =
                bias == BIAS_BEARISH ? (current_hlc3 + delta) : fmax(current_hlc3 - delta, 0.0);
        }

        if (bias == BIAS_BEARISH) {
            level = fmin(level, current_hlc3 + delta);
        } else {
            level = fmax(level, fmax(current_hlc3 - delta, 0.0));
        }

        const bool triggered =
            (bias == BIAS_BEARISH && c >= level) || (bias == BIAS_BULLISH && c <= level);
        double changed = 0.0;

        if (triggered) {
            anchor = c;
            bias = bias == BIAS_BEARISH ? BIAS_BULLISH : BIAS_BEARISH;
            level =
                bias == BIAS_BEARISH ? (current_hlc3 + delta) : fmax(current_hlc3 - delta, 0.0);
            changed = 1.0;
        }

        row_level[i] = level;
        row_anchor[i] = anchor;
        row_bias[i] = bias;
        row_changed[i] = changed;
    }
}
