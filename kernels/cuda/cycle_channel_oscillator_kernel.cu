#include <cmath>
#include <cstddef>

namespace {
struct RmaState {
    int length;
    int count;
    double sum;
    double value;
    bool seeded;

    __device__ inline void init(int len) {
        length = len;
        count = 0;
        sum = 0.0;
        value = NAN;
        seeded = false;
    }

    __device__ inline double update(double input) {
        if (count < length) {
            sum += input;
            count += 1;
            if (count == length) {
                value = sum / static_cast<double>(length);
                seeded = true;
            }
        } else {
            value = value + (input - value) / static_cast<double>(length);
            count += 1;
        }
        return value;
    }
};

struct AtrState {
    RmaState rma;
    bool have_prev_close;
    double prev_close;

    __device__ inline void init(int len) {
        rma.init(len);
        have_prev_close = false;
        prev_close = NAN;
    }

    __device__ inline double update(double high, double low, double close) {
        const double tr = have_prev_close
            ? fmax(high - low, fmax(fabs(high - prev_close), fabs(low - prev_close)))
            : (high - low);
        prev_close = close;
        have_prev_close = true;
        return rma.update(tr);
    }
};
}

extern "C" __global__ void cycle_channel_oscillator_batch_f64(
    const double* source,
    const double* high,
    const double* low,
    const double* close,
    int len,
    const int* short_cycle_lengths,
    const int* medium_cycle_lengths,
    const double* short_multipliers,
    const double* medium_multipliers,
    int rows,
    double* out_fast,
    double* out_slow,
    double* short_history,
    double* medium_history
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int short_cycle_length = short_cycle_lengths[row];
    const int medium_cycle_length = medium_cycle_lengths[row];
    const double short_multiplier = short_multipliers[row];
    const double medium_multiplier = medium_multipliers[row];

    double* row_fast = out_fast + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_slow = out_slow + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_short_history =
        short_history + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_medium_history =
        medium_history + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_fast[i] = NAN;
        row_slow[i] = NAN;
        row_short_history[i] = NAN;
        row_medium_history[i] = NAN;
    }

    if (short_cycle_length < 2 || medium_cycle_length < 2 || !isfinite(short_multiplier)
        || short_multiplier < 0.0 || !isfinite(medium_multiplier) || medium_multiplier < 0.0) {
        return;
    }

    const int short_period = short_cycle_length / 2;
    const int medium_period = medium_cycle_length / 2;
    const int short_delay = short_period / 2;
    const int medium_delay = medium_period / 2;

    RmaState short_rma;
    RmaState medium_rma;
    AtrState medium_atr;
    short_rma.init(short_period);
    medium_rma.init(medium_period);
    medium_atr.init(medium_period);

    int valid_count = 0;
    for (int i = 0; i < len; ++i) {
        const double src = source[i];
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (!(isfinite(src) && isfinite(h) && isfinite(l) && isfinite(c))) {
            continue;
        }

        const double short_ma = short_rma.update(src);
        const double medium_ma = medium_rma.update(src);
        const double medium_atr_value = medium_atr.update(h, l, c);

        row_short_history[valid_count] = short_ma;
        row_medium_history[valid_count] = medium_ma;

        double short_center = src;
        if (valid_count + 1 > short_delay) {
            const double delayed = row_short_history[valid_count - short_delay];
            if (isfinite(delayed)) {
                short_center = delayed;
            }
        }

        double medium_center = src;
        if (valid_count + 1 > medium_delay) {
            const double delayed = row_medium_history[valid_count - medium_delay];
            if (isfinite(delayed)) {
                medium_center = delayed;
            }
        }

        const double offset = medium_multiplier * medium_atr_value;
        const double denom = 2.0 * offset;
        if (isfinite(denom) && denom != 0.0) {
            const double medium_bottom = medium_center - offset;
            row_fast[i] = (src - medium_bottom) / denom;
            row_slow[i] = (short_center - medium_bottom) / denom;
        }

        valid_count += 1;
    }
}
