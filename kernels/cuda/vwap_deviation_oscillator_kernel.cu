#include <cmath>
#include <cstddef>

extern "C" __global__ void vwap_deviation_oscillator_batch_f64(
    const double* source_values,
    int len,
    const int* modes,
    const int* windows,
    const double* guards,
    int rows,
    int max_window,
    double* scratch_values,
    double* out_osc,
    double* out_std1,
    double* out_std2,
    double* out_std3
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0 || max_window <= 0) {
        return;
    }

    const int mode = modes[row];
    const int window = windows[row];
    const double guard = guards[row];
    if (window <= 0 || window > max_window) {
        return;
    }

    const double* row_source = source_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_scratch =
        scratch_values + static_cast<size_t>(row) * static_cast<size_t>(max_window);
    double* row_osc = out_osc + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std1 = out_std1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std2 = out_std2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_std3 = out_std3 + static_cast<size_t>(row) * static_cast<size_t>(len);

    const double nan = NAN;
    int head = 0;
    int count = 0;
    double sum = 0.0;
    double sumsq = 0.0;

    for (int i = 0; i < len; ++i) {
        const double value = row_source[i];
        row_osc[i] = value;
        if (mode == 2) {
            row_std1[i] = 1.0;
            row_std2[i] = 2.0;
            row_std3[i] = 3.0;
        } else {
            row_std1[i] = nan;
            row_std2[i] = nan;
            row_std3[i] = nan;
        }

        if (isfinite(value)) {
            if (count < window) {
                row_scratch[count] = value;
                count += 1;
            } else {
                const double old = row_scratch[head];
                sum -= old;
                sumsq -= old * old;
                row_scratch[head] = value;
                head += 1;
                if (head == window) {
                    head = 0;
                }
            }
            sum += value;
            sumsq += value * value;
        }

        if (count < window) {
            if (mode == 2) {
                row_osc[i] = nan;
            }
            continue;
        }

        const double mean = sum / static_cast<double>(window);
        double variance = sumsq / static_cast<double>(window) - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        const double std = sqrt(variance);

        if (mode == 2) {
            if (!isfinite(value) || !isfinite(std) || std <= 0.0) {
                row_osc[i] = nan;
            } else {
                row_osc[i] = (value - mean) / std;
            }
            continue;
        }

        if (!isfinite(std)) {
            continue;
        }

        const double std1 = std > guard ? std : guard;
        row_std1[i] = std1;
        row_std2[i] = std1 * 2.0;
        row_std3[i] = std1 * 3.0;
    }
}
