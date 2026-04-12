#include <cmath>
#include <cstdint>

extern "C" __global__ void vwap_zscore_with_signals_batch_f64(
    const double* close,
    const double* volume,
    int len,
    const int* lengths,
    const double* upper_bottoms,
    const double* lower_bottoms,
    int rows,
    int max_length,
    double* pv_values,
    double* vol_values,
    int* pv_valid,
    double* dev_values,
    int* dev_valid,
    double* out_zvwap,
    double* out_support,
    double* out_resistance
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    double upper_bottom = upper_bottoms[row];
    double lower_bottom = lower_bottoms[row];
    if (length <= 0 || max_length <= 0 || !isfinite(upper_bottom) || !isfinite(lower_bottom)) {
        return;
    }

    const double nan = NAN;
    double* row_pv_values =
        pv_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_vol_values =
        vol_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* row_pv_valid = pv_valid + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_dev_values =
        dev_values + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    int* row_dev_valid = dev_valid + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_zvwap = out_zvwap + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_support = out_support + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_resistance =
        out_resistance + static_cast<size_t>(row) * static_cast<size_t>(len);

    int idx = 0;
    int count = 0;
    int valid_count = 0;
    double pv_sum = 0.0;
    double vol_sum = 0.0;

    int dev_idx = 0;
    int dev_count = 0;
    int dev_valid_count = 0;
    double dev_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        row_zvwap[i] = nan;
        row_support[i] = nan;
        row_resistance[i] = nan;

        if (count >= length) {
            int old_idx = idx;
            if (row_pv_valid[old_idx] != 0) {
                valid_count -= 1;
                pv_sum -= row_pv_values[old_idx];
                vol_sum -= row_vol_values[old_idx];
            }
        } else {
            count += 1;
        }

        double c = close[i];
        double v = volume[i];
        if (isfinite(c) && isfinite(v) && v >= 0.0) {
            double pv = c * v;
            row_pv_values[idx] = pv;
            row_vol_values[idx] = v;
            row_pv_valid[idx] = 1;
            valid_count += 1;
            pv_sum += pv;
            vol_sum += v;
        } else {
            row_pv_values[idx] = 0.0;
            row_vol_values[idx] = 0.0;
            row_pv_valid[idx] = 0;
        }
        idx += 1;
        if (idx == length) {
            idx = 0;
        }

        if (dev_count >= length) {
            int old_idx = dev_idx;
            if (row_dev_valid[old_idx] != 0) {
                dev_valid_count -= 1;
                dev_sum -= row_dev_values[old_idx];
            }
        } else {
            dev_count += 1;
        }

        double mean = nan;
        if (count >= length && valid_count == length && vol_sum > 0.0) {
            mean = pv_sum / vol_sum;
            double dev = (c - mean) * (c - mean);
            row_dev_values[dev_idx] = dev;
            row_dev_valid[dev_idx] = 1;
            dev_valid_count += 1;
            dev_sum += dev;
        } else {
            row_dev_values[dev_idx] = 0.0;
            row_dev_valid[dev_idx] = 0;
        }
        dev_idx += 1;
        if (dev_idx == length) {
            dev_idx = 0;
        }

        if (dev_count < length || dev_valid_count != length || !isfinite(mean)) {
            continue;
        }

        double variance = dev_sum / static_cast<double>(length);
        if (variance < 0.0) {
            variance = 0.0;
        }
        double sd = sqrt(variance);
        if (!isfinite(sd) || sd <= 0.0) {
            continue;
        }

        double zvwap = (c - mean) / sd;
        row_zvwap[i] = zvwap;
        row_support[i] = zvwap < lower_bottom ? 1.0 : 0.0;
        row_resistance[i] = zvwap > upper_bottom ? 1.0 : 0.0;
    }
}
