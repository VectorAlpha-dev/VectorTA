#include <cmath>
#include <cstddef>

namespace {
constexpr double OUTPUT_SCALE = 100.0;

struct AtrStreamState {
    int length;
    double alpha;
    double prev_close;
    double rma;
    double warm_sum;
    int warm_count;
    bool seeded;

    __device__ void init(int value) {
        length = value;
        alpha = 1.0 / static_cast<double>(value);
        reset();
    }

    __device__ void reset() {
        prev_close = NAN;
        rma = NAN;
        warm_sum = 0.0;
        warm_count = 0;
        seeded = false;
    }

    __device__ double update(double high, double low, double close, bool* ready) {
        const double tr = isnan(prev_close) ? (high - low) : (fmax(high, prev_close) - fmin(low, prev_close));
        prev_close = close;

        if (!seeded) {
            warm_sum += tr;
            warm_count += 1;
            if (warm_count == length) {
                rma = warm_sum * alpha;
                seeded = true;
                *ready = true;
                return rma;
            }
            *ready = false;
            return NAN;
        }

        rma = fma(alpha, tr - rma, rma);
        *ready = true;
        return rma;
    }
};

__device__ inline bool valid_bar(double high, double low, double source) {
    return isfinite(high) && isfinite(low) && isfinite(source) && high >= low;
}

__device__ inline double clamp_unit(double value) {
    return value < -1.0 ? -1.0 : (value > 1.0 ? 1.0 : value);
}
}

extern "C" __global__ void supertrend_oscillator_batch_f64(
    const double* high,
    const double* low,
    const double* source,
    int len,
    const int* lengths,
    const double* mults,
    const int* smooths,
    int rows,
    double* out_oscillator,
    double* out_signal,
    double* out_histogram
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const int length = lengths[row];
    const double mult = mults[row];
    const int smooth = smooths[row];

    double* row_oscillator = out_oscillator + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_histogram = out_histogram + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_oscillator[i] = NAN;
        row_signal[i] = NAN;
        row_histogram[i] = NAN;
    }

    if (length <= 0 || smooth <= 0 || !isfinite(mult) || mult <= 0.0) {
        return;
    }

    const double hist_alpha = 2.0 / (static_cast<double>(smooth) + 1.0);
    const double length_f64 = static_cast<double>(length);

    AtrStreamState atr;
    atr.init(length);

    double prev_source = NAN;
    double prev_upper = NAN;
    double prev_lower = NAN;
    double prev_trend = 0.0;
    double ama_prev = NAN;
    bool have_ama = false;
    double hist_prev = NAN;
    bool have_hist = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double src = source[i];

        if (!valid_bar(h, l, src)) {
            atr.reset();
            prev_source = NAN;
            prev_upper = NAN;
            prev_lower = NAN;
            prev_trend = 0.0;
            ama_prev = NAN;
            have_ama = false;
            hist_prev = NAN;
            have_hist = false;
            continue;
        }

        bool atr_ready = false;
        const double atr_value = atr.update(h, l, src, &atr_ready);
        if (!atr_ready) {
            prev_source = src;
            continue;
        }

        const double mid = 0.5 * (h + l);
        const double band = atr_value * mult;
        const double up = mid + band;
        const double dn = mid - band;

        const double upper =
            (isfinite(prev_source) && isfinite(prev_upper) && prev_source < prev_upper)
            ? fmin(up, prev_upper)
            : up;
        const double lower =
            (isfinite(prev_source) && isfinite(prev_lower) && prev_source > prev_lower)
            ? fmax(dn, prev_lower)
            : dn;

        const double trend =
            (isfinite(prev_upper) && src > prev_upper) ? 1.0
            : ((isfinite(prev_lower) && src < prev_lower) ? 0.0 : prev_trend);
        const double supertrend = trend * lower + (1.0 - trend) * upper;
        const double width = upper - lower;
        const double osc =
            (isfinite(width) && width != 0.0) ? clamp_unit((src - supertrend) / width) : 0.0;
        const double alpha = (osc * osc) / length_f64;
        const double ama = have_ama ? (ama_prev + alpha * (osc - ama_prev)) : osc;
        const double diff = osc - ama;
        const double hist = have_hist ? (hist_prev + hist_alpha * (diff - hist_prev)) : diff;

        row_oscillator[i] = osc * OUTPUT_SCALE;
        row_signal[i] = ama * OUTPUT_SCALE;
        row_histogram[i] = hist * OUTPUT_SCALE;

        prev_source = src;
        prev_upper = upper;
        prev_lower = lower;
        prev_trend = trend;
        ama_prev = ama;
        have_ama = true;
        hist_prev = hist;
        have_hist = true;
    }
}
