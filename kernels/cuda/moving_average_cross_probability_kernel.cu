#include <cmath>
#include <cstddef>

namespace {
constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr double NBDEV = 4.0;

__device__ inline int max_i(int a, int b) {
    return a > b ? a : b;
}

__device__ inline int min_i(int a, int b) {
    return a < b ? a : b;
}

__device__ inline int sqrt_period(int period) {
    const double root = floor(sqrt(static_cast<double>(period)));
    const int value = static_cast<int>(root);
    return value > 0 ? value : 1;
}

struct EmaState {
    int period;
    int count;
    double alpha;
    double beta;
    double mean;
    bool filled;

    __device__ void init(int period) {
        this->period = period;
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        beta = 1.0 - alpha;
        reset();
    }

    __device__ void reset() {
        count = 0;
        mean = NAN;
        filled = false;
    }

    __device__ bool update(double input, double* out) {
        if (!isfinite(input)) {
            *out = filled ? mean : NAN;
            return filled;
        }
        count += 1;
        if (count == 1) {
            mean = input;
        } else if (count <= period) {
            mean += (input - mean) / static_cast<double>(count);
        } else {
            mean = alpha * input + beta * mean;
        }
        if (!filled && count >= period) {
            filled = true;
        }
        *out = filled ? mean : NAN;
        return filled;
    }
};

struct SmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            if (count == period) {
                *out = sum / static_cast<double>(period);
                return true;
            }
            *out = NAN;
            return false;
        }
        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += input - old;
        *out = sum / static_cast<double>(period);
        return true;
    }
};

struct WmaState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;
    double wsum;
    double inv_norm;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        const double norm = static_cast<double>(period) * (static_cast<double>(period) + 1.0) * 0.5;
        inv_norm = 1.0 / norm;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        wsum = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            wsum += static_cast<double>(count) * input;
            if (count == period) {
                *out = wsum * inv_norm;
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        const double prev_sum = sum;
        sum = prev_sum + input - old;
        wsum = static_cast<double>(period) * input + wsum - prev_sum;
        *out = wsum * inv_norm;
        return true;
    }
};

struct HmaState {
    WmaState wma_half;
    WmaState wma_full;
    WmaState wma_sqrt;

    __device__ void init(int period, double* half_storage, double* full_storage, double* sqrt_storage) {
        const int half = max_i(period / 2, 1);
        const int sqrt_len = sqrt_period(period);
        wma_half.init(half, half_storage);
        wma_full.init(period, full_storage);
        wma_sqrt.init(sqrt_len, sqrt_storage);
    }

    __device__ void reset() {
        wma_half.reset();
        wma_full.reset();
        wma_sqrt.reset();
    }

    __device__ bool update(double input, double* out) {
        double half_value = NAN;
        double full_value = NAN;
        const bool half_ready = wma_half.update(input, &half_value);
        const bool full_ready = wma_full.update(input, &full_value);
        if (half_ready && full_ready) {
            return wma_sqrt.update(2.0 * half_value - full_value, out);
        }
        *out = NAN;
        return false;
    }
};

struct StddevState {
    double* ring;
    int period;
    int head;
    int count;
    double sum;
    double sumsq;

    __device__ void init(int p, double* storage) {
        period = p;
        ring = storage;
        reset();
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
        sumsq = 0.0;
    }

    __device__ bool update(double input, double* out) {
        if (count < period) {
            ring[count] = input;
            count += 1;
            sum += input;
            sumsq += input * input;
            if (count == period) {
                const double mean = sum / static_cast<double>(period);
                const double var = fmax(sumsq / static_cast<double>(period) - mean * mean, 0.0);
                *out = sqrt(var) * NBDEV;
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        ring[head] = input;
        head += 1;
        if (head == period) {
            head = 0;
        }
        sum += input - old;
        sumsq += input * input - old * old;
        const double mean = sum / static_cast<double>(period);
        const double var = fmax(sumsq / static_cast<double>(period) - mean * mean, 0.0);
        *out = sqrt(var) * NBDEV;
        return true;
    }
};

__device__ bool finite_window(
    const double* data,
    int start,
    int end
) {
    for (int i = start; i <= end; ++i) {
        if (!isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

__device__ bool truncated_ema_from_slice(
    const double* data,
    int end_idx,
    int history_len,
    double alpha,
    double beta,
    double* out
) {
    const int start_idx = end_idx - history_len + 1;
    if (start_idx < 0) {
        *out = NAN;
        return false;
    }
    double ema = data[start_idx];
    if (!isfinite(ema)) {
        *out = NAN;
        return false;
    }
    for (int idx = start_idx + 1; idx <= end_idx; ++idx) {
        const double value = data[idx];
        if (!isfinite(value)) {
            *out = NAN;
            return false;
        }
        ema = alpha * value + beta * ema;
    }
    *out = ema;
    return true;
}

__device__ bool probability_from_slice(
    const double* data,
    int end_idx,
    int ma_type,
    int slow_length,
    int fast_length,
    int resolution,
    int history_window_len,
    double lower,
    double upper,
    double direction,
    double* out
) {
    const int start_idx = end_idx - history_window_len + 1;
    if (start_idx < 0 || !finite_window(data, start_idx, end_idx)) {
        *out = NAN;
        return false;
    }

    const double step = (upper - lower) / static_cast<double>(resolution - 1);
    int hits = 0;

    if (ma_type == MA_EMA) {
        const double slow_alpha = 2.0 / (static_cast<double>(slow_length) + 1.0);
        const double slow_beta = 1.0 - slow_alpha;
        const double fast_alpha = 2.0 / (static_cast<double>(fast_length) + 1.0);
        const double fast_beta = 1.0 - fast_alpha;
        double slow_current = NAN;
        double fast_current = NAN;
        if (!truncated_ema_from_slice(data, end_idx, history_window_len, slow_alpha, slow_beta, &slow_current) ||
            !truncated_ema_from_slice(data, end_idx, history_window_len, fast_alpha, fast_beta, &fast_current)) {
            *out = NAN;
            return false;
        }
        for (int idx = 0; idx < resolution; ++idx) {
            const double price = lower + step * static_cast<double>(idx);
            const double slow_future = slow_alpha * price + slow_beta * slow_current;
            const double fast_future = fast_alpha * price + fast_beta * fast_current;
            const bool crossed = direction < 0.0 ? (slow_future > fast_future) : (slow_future <= fast_future);
            if (crossed) {
                hits += 1;
            }
        }
    } else {
        const int slow_needed = slow_length - 1;
        const int fast_needed = fast_length - 1;
        double slow_sum = 0.0;
        double fast_sum = 0.0;
        for (int idx = 0; idx < slow_needed; ++idx) {
            slow_sum += data[end_idx - idx];
        }
        for (int idx = 0; idx < fast_needed; ++idx) {
            fast_sum += data[end_idx - idx];
        }
        for (int idx = 0; idx < resolution; ++idx) {
            const double price = lower + step * static_cast<double>(idx);
            const double slow_future = (price + slow_sum) / static_cast<double>(slow_length);
            const double fast_future = (price + fast_sum) / static_cast<double>(fast_length);
            const bool crossed = direction < 0.0 ? (slow_future > fast_future) : (slow_future <= fast_future);
            if (crossed) {
                hits += 1;
            }
        }
    }

    *out = 100.0 * static_cast<double>(hits) / static_cast<double>(resolution);
    return true;
}
}

extern "C" __global__ void moving_average_cross_probability_batch_f64(
    const double* data,
    int len,
    const int* smoothing_windows,
    const int* slow_lengths,
    const int* fast_lengths,
    const int* resolutions,
    const int* ma_types,
    int rows,
    int max_smoothing_window,
    int max_slow_length,
    int max_fast_length,
    double* scratch,
    double* out_value,
    double* out_slow_ma,
    double* out_fast_ma,
    double* out_forecast,
    double* out_upper,
    double* out_lower,
    double* out_direction
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int smoothing_window = smoothing_windows[row];
    const int slow_length = slow_lengths[row];
    const int fast_length = fast_lengths[row];
    const int resolution = resolutions[row];
    const int ma_type = ma_types[row];

    double* row_value = out_value + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_slow_ma = out_slow_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_fast_ma = out_fast_ma + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_forecast = out_forecast + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction = out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_value[i] = NAN;
        row_slow_ma[i] = NAN;
        row_fast_ma[i] = NAN;
        row_forecast[i] = NAN;
        row_upper[i] = NAN;
        row_lower[i] = NAN;
        row_direction[i] = NAN;
    }

    if (smoothing_window < 2 || slow_length < 2 || fast_length <= 0 || slow_length <= fast_length ||
        resolution < 2 || ma_type < MA_EMA || ma_type > MA_SMA) {
        return;
    }

    const int history_window_len = 2 * slow_length + 1;
    const int row_stride = max_smoothing_window * 4 + max_slow_length + max_fast_length;
    double* row_scratch = scratch + static_cast<size_t>(row) * static_cast<size_t>(row_stride);
    double* hma_half = row_scratch;
    double* hma_full = hma_half + max_smoothing_window;
    double* hma_sqrt = hma_full + max_smoothing_window;
    double* stddev_ring = hma_sqrt + max_smoothing_window;
    double* slow_ring = stddev_ring + max_smoothing_window;
    double* fast_ring = slow_ring + max_slow_length;

    EmaState slow_ema;
    EmaState fast_ema;
    SmaState slow_sma;
    SmaState fast_sma;
    HmaState hma;
    StddevState stddev;

    slow_ema.init(slow_length);
    fast_ema.init(fast_length);
    slow_sma.init(slow_length, slow_ring);
    fast_sma.init(fast_length, fast_ring);
    hma.init(smoothing_window, hma_half, hma_full, hma_sqrt);
    stddev.init(smoothing_window, stddev_ring);

    double previous_hma = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            slow_ema.reset();
            fast_ema.reset();
            slow_sma.reset();
            fast_sma.reset();
            hma.reset();
            stddev.reset();
            previous_hma = NAN;
            continue;
        }

        double slow_ma = NAN;
        double fast_ma = NAN;
        double current_hma = NAN;
        double current_std = NAN;

        if (ma_type == MA_EMA) {
            slow_ema.update(value, &slow_ma);
            fast_ema.update(value, &fast_ma);
        } else {
            slow_sma.update(value, &slow_ma);
            fast_sma.update(value, &fast_ma);
        }
        hma.update(value, &current_hma);
        stddev.update(value, &current_std);

        row_slow_ma[i] = slow_ma;
        row_fast_ma[i] = fast_ma;

        double direction = NAN;
        if (isfinite(slow_ma) && isfinite(fast_ma)) {
            direction = fast_ma > slow_ma ? -1.0 : 1.0;
        }
        row_direction[i] = direction;

        if (isfinite(current_hma) && isfinite(previous_hma) && isfinite(current_std)) {
            const double forecast = current_hma + (current_hma - previous_hma);
            const double upper = forecast + current_std;
            const double lower = forecast - current_std;
            row_forecast[i] = forecast;
            row_upper[i] = upper;
            row_lower[i] = lower;

            if (isfinite(direction) && i + 1 >= history_window_len) {
                double probability = NAN;
                if (probability_from_slice(
                        data,
                        i,
                        ma_type,
                        slow_length,
                        fast_length,
                        resolution,
                        history_window_len,
                        lower,
                        upper,
                        direction,
                        &probability)) {
                    row_value[i] = probability;
                }
            }
        }

        previous_hma = current_hma;
    }
}
