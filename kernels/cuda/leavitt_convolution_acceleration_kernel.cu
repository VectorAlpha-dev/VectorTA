#include <cmath>
#include <cstddef>

namespace {

__device__ inline int sqrt_length_device(int length) {
    int value = static_cast<int>(floor(sqrt(static_cast<double>(length))));
    return value > 1 ? value : 1;
}

struct RollingLinRegState {
    double* buffer;
    int period;
    int head;
    int count;
    bool filled;
    double n;
    double sum_x;
    double inv_n;
    double inv_denom;
    double mean_x;
    double forecast_x;
    double sum_y;
    double sum_xy;

    __device__ void init(double* storage, int len) {
        buffer = storage;
        period = len;
        head = 0;
        count = 0;
        filled = false;
        const double period_f = static_cast<double>(len);
        const double m = static_cast<double>(len - 1);
        const double sum_x2 = (m * period_f) * (2.0 * m + 1.0) / 6.0;
        n = period_f > 0.0 ? period_f : 1.0;
        sum_x = 0.5 * m * period_f;
        inv_n = period_f > 0.0 ? 1.0 / period_f : 0.0;
        const double denom = period_f * sum_x2 - sum_x * sum_x;
        inv_denom = fabs(denom) > 0.0 ? 1.0 / denom : 0.0;
        mean_x = period_f > 0.0 ? sum_x / period_f : 0.0;
        forecast_x = period_f;
        sum_y = 0.0;
        sum_xy = 0.0;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        filled = false;
        sum_y = 0.0;
        sum_xy = 0.0;
    }

    __device__ double slope() const {
        if (period <= 1) {
            return 0.0;
        }
        return (n * sum_xy - sum_x * sum_y) * inv_denom;
    }

    __device__ double forecast_next() const {
        if (period == 1) {
            return buffer[(head + period - 1) % period];
        }
        const double slope_value = slope();
        const double mean_y = sum_y * inv_n;
        return mean_y + slope_value * (forecast_x - mean_x);
    }

    __device__ bool update(double value, double* forecast_out, double* slope_out) {
        if (period == 1) {
            buffer[0] = value;
            count = 1;
            filled = true;
            *forecast_out = value;
            *slope_out = 0.0;
            return true;
        }

        if (!filled) {
            const double j = static_cast<double>(count);
            buffer[head] = value;
            head = (head + 1) % period;
            sum_y += value;
            sum_xy += j * value;
            count += 1;
            if (count < period) {
                *forecast_out = NAN;
                *slope_out = NAN;
                return false;
            }
            filled = true;
            *forecast_out = forecast_next();
            *slope_out = slope();
            return true;
        }

        const double old = buffer[head];
        buffer[head] = value;
        const double new_sum_y = sum_y + value - old;
        const double new_sum_xy = n * value + sum_xy - new_sum_y;
        sum_y = new_sum_y;
        sum_xy = new_sum_xy;
        head = (head + 1) % period;
        *forecast_out = forecast_next();
        *slope_out = slope();
        return true;
    }
};

struct RollingMeanStdState {
    double* buffer;
    int period;
    int head;
    int count;
    bool filled;
    double sum;
    double sum_sq;

    __device__ void init(double* storage, int len) {
        buffer = storage;
        period = len;
        head = 0;
        count = 0;
        filled = false;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        filled = false;
        sum = 0.0;
        sum_sq = 0.0;
    }

    __device__ bool update(double value, double* mean_out, double* std_out) {
        if (!filled) {
            buffer[head] = value;
            head = (head + 1) % period;
            count += 1;
            sum += value;
            sum_sq += value * value;
            if (count < period) {
                *mean_out = NAN;
                *std_out = NAN;
                return false;
            }
            filled = true;
        } else {
            const double old = buffer[head];
            buffer[head] = value;
            head = (head + 1) % period;
            sum += value - old;
            sum_sq += value * value - old * old;
        }

        const double n = static_cast<double>(period);
        const double mean = sum / n;
        double variance = sum_sq / n - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        *mean_out = mean;
        *std_out = sqrt(variance);
        return true;
    }
};

__device__ inline double logistic(double z) {
    return 1.0 / (1.0 + exp(-z));
}

__device__ inline double hyperbolic(double z) {
    const double e = exp(-z);
    return (1.0 - e) / (1.0 + e);
}

__device__ inline void bump_source_history(
    double value,
    double* prev_src1,
    double* prev_src2,
    bool* have_src1,
    bool* have_src2
) {
    if (*have_src1) {
        *prev_src2 = *prev_src1;
        *have_src2 = true;
    }
    *prev_src1 = value;
    *have_src1 = true;
}

}

extern "C" __global__ void leavitt_convolution_acceleration_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ norm_lengths,
    const int* __restrict__ use_norm_hyperbolic,
    int rows,
    int scratch_cap,
    double* __restrict__ scratch_buf,
    double* __restrict__ out_conv_acceleration,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int norm_length = norm_lengths[row];
    const bool norm_hyperbolic = use_norm_hyperbolic[row] != 0;

    double* row_conv =
        out_conv_acceleration + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_conv[i] = NAN;
        row_signal[i] = NAN;
    }

    if (length <= 0 || norm_length <= 0) {
        return;
    }

    const int slope_length = sqrt_length_device(length);
    const int needed = length + slope_length + norm_length;
    if (needed > scratch_cap) {
        return;
    }

    double* row_scratch =
        scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap);
    double* source_projection_storage = row_scratch;
    double* projection_slope_storage = source_projection_storage + length;
    double* norm_storage = projection_slope_storage + slope_length;

    RollingLinRegState source_projection;
    RollingLinRegState projection_slope;
    RollingMeanStdState norm;
    source_projection.init(source_projection_storage, length);
    projection_slope.init(projection_slope_storage, slope_length);
    norm.init(norm_storage, norm_length);

    double prev_scaled = 0.0;
    double prev_conv_acceleration = 0.0;
    double prev_slo = 0.0;
    double prev_src1 = 0.0;
    double prev_src2 = 0.0;
    bool have_src1 = false;
    bool have_src2 = false;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            source_projection.reset();
            projection_slope.reset();
            norm.reset();
            prev_scaled = 0.0;
            prev_conv_acceleration = 0.0;
            prev_slo = 0.0;
            prev_src1 = 0.0;
            prev_src2 = 0.0;
            have_src1 = false;
            have_src2 = false;
            continue;
        }

        const double src1 = have_src1 ? prev_src1 : 0.0;
        const double src2 = have_src2 ? prev_src2 : 0.0;
        const bool is_accelerated = (src2 - 2.0 * src1 + value) > 0.0;

        double projection = NAN;
        double projection_slope_value = NAN;
        if (!source_projection.update(value, &projection, &projection_slope_value) ||
            !isfinite(projection)) {
            projection_slope.reset();
            norm.reset();
            prev_scaled = 0.0;
            prev_conv_acceleration = 0.0;
            prev_slo = 0.0;
            bump_source_history(value, &prev_src1, &prev_src2, &have_src1, &have_src2);
            continue;
        }

        double projection_forecast = NAN;
        double conv_slope = NAN;
        if (!projection_slope.update(projection, &projection_forecast, &conv_slope) ||
            !isfinite(conv_slope)) {
            norm.reset();
            prev_scaled = 0.0;
            prev_conv_acceleration = 0.0;
            prev_slo = 0.0;
            bump_source_history(value, &prev_src1, &prev_src2, &have_src1, &have_src2);
            continue;
        }

        double mean = NAN;
        double dev = NAN;
        if (!norm.update(conv_slope, &mean, &dev)) {
            bump_source_history(value, &prev_src1, &prev_src2, &have_src1, &have_src2);
            continue;
        }

        const double z = dev != 0.0 ? (conv_slope - mean) / dev : 0.0;
        const double scaled = norm_hyperbolic ? hyperbolic(z) : logistic(z);
        const double conv_acceleration = scaled - prev_scaled;
        const double slo =
            norm_hyperbolic ? conv_acceleration : (conv_acceleration - prev_conv_acceleration);

        double signal = 0.0;
        if (slo > 0.0 && is_accelerated) {
            signal = slo > prev_slo ? 2.0 : 1.0;
        } else if (slo < 0.0 && !is_accelerated) {
            signal = slo < prev_slo ? -2.0 : -1.0;
        }

        row_conv[i] = conv_acceleration;
        row_signal[i] = signal;

        prev_scaled = scaled;
        prev_conv_acceleration = conv_acceleration;
        prev_slo = slo;
        bump_source_history(value, &prev_src1, &prev_src2, &have_src1, &have_src2);
    }
}
