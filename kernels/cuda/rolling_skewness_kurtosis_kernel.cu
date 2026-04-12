#include <cmath>
#include <cstdint>

extern "C" __global__ void rolling_skewness_kurtosis_batch_f64(
    const double* data,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* source_buffer,
    double* skew_buffer,
    double* kurt_buffer,
    double* out_skewness,
    double* out_kurtosis
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length <= 0 || smooth_length <= 0) {
        return;
    }

    const double nan = NAN;
    double* source_ring = source_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* skew_ring = skew_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* kurt_ring = kurt_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_skew = out_skewness + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kurt = out_kurtosis + static_cast<size_t>(row) * static_cast<size_t>(len);

    int source_head = 0;
    int source_count = 0;
    int skew_head = 0;
    int skew_count = 0;
    int kurt_head = 0;
    int kurt_count = 0;
    double skew_sum = 0.0;
    double kurt_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_head = 0;
            source_count = 0;
            skew_head = 0;
            skew_count = 0;
            kurt_head = 0;
            kurt_count = 0;
            skew_sum = 0.0;
            kurt_sum = 0.0;
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        source_ring[source_head] = value;
        source_head += 1;
        if (source_head == length) {
            source_head = 0;
        }
        if (source_count < length) {
            source_count += 1;
        }
        if (source_count < length) {
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        double n = static_cast<double>(length);
        double mean = 0.0;
        for (int j = 0; j < length; ++j) {
            mean += source_ring[j];
        }
        mean /= n;

        double m2 = 0.0;
        double m3 = 0.0;
        double m4 = 0.0;
        for (int j = 0; j < length; ++j) {
            double dev = source_ring[j] - mean;
            double dev2 = dev * dev;
            m2 += dev2;
            m3 += dev2 * dev;
            m4 += dev2 * dev2;
        }
        m2 /= n;
        if (!isfinite(m2) || m2 <= 2.2204460492503131e-16) {
            skew_head = 0;
            skew_count = 0;
            kurt_head = 0;
            kurt_count = 0;
            skew_sum = 0.0;
            kurt_sum = 0.0;
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        double sigma = sqrt(m2);
        double skew_raw = (m3 / n) / (sigma * sigma * sigma);
        double kurt_raw = (m4 / n) / (m2 * m2) - 3.0;

        if (skew_count == smooth_length) {
            skew_sum -= skew_ring[skew_head];
        } else {
            skew_count += 1;
        }
        skew_ring[skew_head] = skew_raw;
        skew_sum += skew_raw;
        skew_head += 1;
        if (skew_head == smooth_length) {
            skew_head = 0;
        }

        if (kurt_count == smooth_length) {
            kurt_sum -= kurt_ring[kurt_head];
        } else {
            kurt_count += 1;
        }
        kurt_ring[kurt_head] = kurt_raw;
        kurt_sum += kurt_raw;
        kurt_head += 1;
        if (kurt_head == smooth_length) {
            kurt_head = 0;
        }

        if (skew_count == smooth_length && kurt_count == smooth_length) {
            row_skew[i] = skew_sum / static_cast<double>(smooth_length);
            row_kurt[i] = kurt_sum / static_cast<double>(smooth_length);
        } else {
            row_skew[i] = nan;
            row_kurt[i] = nan;
        }
    }
}
