#include <cmath>
#include <cstddef>

namespace {

constexpr int ATR_PERIOD = 200;

__device__ inline bool valid_bar(double high, double low, double source) {
    return isfinite(high) && isfinite(low) && isfinite(source) && high >= low;
}

__device__ inline double pine_sign(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

__device__ inline double true_range(double high, double low, double prev_close) {
    if (isfinite(prev_close)) {
        const double a = high - low;
        const double b = fabs(high - prev_close);
        const double c = fabs(low - prev_close);
        return fmax(a, fmax(b, c));
    }
    return high - low;
}

}

extern "C" __global__ void hypertrend_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ source,
    int len,
    const double* __restrict__ factors,
    const double* __restrict__ slopes,
    const double* __restrict__ width_ratios,
    int rows,
    double* __restrict__ out_upper,
    double* __restrict__ out_average,
    double* __restrict__ out_lower,
    double* __restrict__ out_trend,
    double* __restrict__ out_changed
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double factor = factors[row];
    const double slope = slopes[row];
    const double width_ratio = width_ratios[row];

    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_average = out_average + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_trend = out_trend + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_changed = out_changed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_upper[i] = NAN;
        row_average[i] = NAN;
        row_lower[i] = NAN;
        row_trend[i] = NAN;
        row_changed[i] = NAN;
    }

    if (!isfinite(factor) || factor <= 0.0 || !isfinite(slope) || slope <= 0.0 ||
        !isfinite(width_ratio) || width_ratio < 0.0 || width_ratio > 1.0) {
        return;
    }

    bool initialized = false;
    double avg = 0.0;
    double hold = 0.0;
    double os = 1.0;

    double prev_close = NAN;
    double seed_sum = 0.0;
    int seed_count = 0;
    double atr = NAN;

    for (int i = 0; i < len; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double src = source[i];

        if (!valid_bar(hi, lo, src)) {
            initialized = false;
            avg = 0.0;
            hold = 0.0;
            os = 1.0;
            prev_close = NAN;
            seed_sum = 0.0;
            seed_count = 0;
            atr = NAN;
            continue;
        }

        const double tr = true_range(hi, lo, prev_close);
        prev_close = src;

        double atr_value = 0.0;
        if (seed_count < ATR_PERIOD) {
            seed_sum += tr;
            seed_count += 1;
            if (seed_count == ATR_PERIOD) {
                atr = seed_sum / static_cast<double>(ATR_PERIOD);
                atr_value = atr;
            }
        } else {
            atr = ((atr * static_cast<double>(ATR_PERIOD - 1)) + tr) / static_cast<double>(ATR_PERIOD);
            atr_value = atr;
        }

        if (!initialized) {
            avg = src;
            hold = 0.0;
            os = 1.0;
            row_average[i] = avg;
            row_upper[i] = avg;
            row_lower[i] = avg;
            row_trend[i] = os;
            row_changed[i] = 0.0;
            initialized = true;
            continue;
        }

        const double atr_band = atr_value * factor;
        const double next_avg = fabs(src - avg) > atr_band
            ? 0.5 * (src + avg)
            : avg + os * (hold / factor / slope);
        const double next_os = pine_sign(next_avg - avg);
        const double changed = next_os != os ? 1.0 : 0.0;
        const double next_hold = changed != 0.0 ? atr_band : hold;
        const double upper = next_avg + width_ratio * next_hold;
        const double lower = next_avg - width_ratio * next_hold;

        row_upper[i] = upper;
        row_average[i] = next_avg;
        row_lower[i] = lower;
        row_trend[i] = next_os;
        row_changed[i] = changed;

        avg = next_avg;
        hold = next_hold;
        os = next_os;
    }
}
