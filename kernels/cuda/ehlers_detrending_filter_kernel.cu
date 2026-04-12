#include <cmath>
#include <cstddef>

extern "C" __global__ void ehlers_detrending_filter_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int rows,
    double* __restrict__ out_edf,
    double* __restrict__ out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    double* row_out_edf = out_edf + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_edf[i] = NAN;
        row_out_signal[i] = NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    double weight_sum = 0.0;
    double denom = static_cast<double>(length + 1);
    for (int i = 1; i <= length; ++i) {
        weight_sum += 1.0 - cos((2.0 * 3.14159265358979323846 * static_cast<double>(i)) / denom);
    }
    if (!(weight_sum > 0.0) || !isfinite(weight_sum)) {
        return;
    }

    bool initialized = false;
    double prev_src = 0.0;
    double prev_edf = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            row_out_edf[i] = NAN;
            initialized = false;
            prev_src = 0.0;
            prev_edf = 0.0;
            continue;
        }

        double prev = initialized ? prev_src : 0.0;
        double edf_raw = (0.95 * value) - (0.95 * prev) + (0.9 * prev_edf);
        row_out_edf[i] = edf_raw;
        prev_src = value;
        prev_edf = edf_raw;
        initialized = true;
    }

    int run = 0;
    for (int i = 0; i < len; ++i) {
        double raw = row_out_edf[i];
        if (!isfinite(raw)) {
            row_out_signal[i] = NAN;
            run = 0;
            continue;
        }

        run += 1;
        double filt = 0.0;
        for (int offset = 0; offset < run && offset < length; ++offset) {
            double hist = row_out_edf[i - offset];
            double weight = 1.0 -
                cos((2.0 * 3.14159265358979323846 * static_cast<double>(offset + 1)) / denom);
            filt += weight * hist;
        }
        row_out_signal[i] = filt / weight_sum;
    }

    run = 0;
    double prev_filt = 0.0;
    double prev_slo = 0.0;

    for (int i = 0; i < len; ++i) {
        double filt = row_out_signal[i];
        if (!isfinite(filt)) {
            row_out_edf[i] = NAN;
            row_out_signal[i] = NAN;
            run = 0;
            prev_filt = 0.0;
            prev_slo = 0.0;
            continue;
        }

        run += 1;
        double slo = filt - prev_filt;
        double signal = 0.0;
        if (slo > 0.0) {
            signal = (slo > prev_slo) ? 2.0 : 1.0;
        } else if (slo < 0.0) {
            signal = (slo < prev_slo) ? -2.0 : -1.0;
        }
        prev_filt = filt;
        prev_slo = slo;

        if (run >= length) {
            row_out_edf[i] = filt;
            row_out_signal[i] = signal;
        } else {
            row_out_edf[i] = NAN;
            row_out_signal[i] = NAN;
        }
    }
}
