#include <cmath>
#include <cstddef>

static __device__ inline bool finite_pair(
    const double* main,
    const double* compare,
    int idx
) {
    return isfinite(main[idx]) && isfinite(compare[idx]);
}

static __device__ inline bool finite_return_pair(
    const double* main,
    const double* compare,
    int idx
) {
    return idx > 0 && finite_pair(main, compare, idx - 1) && finite_pair(main, compare, idx);
}

static __device__ inline double return_value(const double* values, int idx) {
    return values[idx] - values[idx - 1];
}

extern "C" __global__ void spearman_correlation_batch_f64(
    const double* __restrict__ main,
    const double* __restrict__ compare,
    int len,
    const int* __restrict__ lookbacks,
    const int* __restrict__ smoothing_lengths,
    int rows,
    double* __restrict__ out_raw,
    double* __restrict__ out_smoothed
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int lookback = lookbacks[row];
    int smoothing_length = smoothing_lengths[row];

    double* row_out_raw = out_raw + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_smoothed = out_smoothed + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_raw[i] = NAN;
        row_out_smoothed[i] = NAN;
    }

    if (lookback <= 0 || smoothing_length <= 0 || lookback >= len) {
        return;
    }

    double mean_rank = (static_cast<double>(lookback) + 1.0) * 0.5;

    for (int i = 0; i < len; ++i) {
        int start = i + 1 - lookback;
        if (start < 1) {
            continue;
        }

        bool valid_window = true;
        for (int idx = start; idx <= i; ++idx) {
            if (!finite_return_pair(main, compare, idx)) {
                valid_window = false;
                break;
            }
        }
        if (!valid_window) {
            continue;
        }

        double cov = 0.0;
        double var_main = 0.0;
        double var_compare = 0.0;

        for (int a = start; a <= i; ++a) {
            double main_a = return_value(main, a);
            double compare_a = return_value(compare, a);

            int main_less = 0;
            int main_equal = 0;
            int compare_less = 0;
            int compare_equal = 0;

            for (int b = start; b <= i; ++b) {
                double main_b = return_value(main, b);
                double compare_b = return_value(compare, b);

                if (main_b < main_a) {
                    main_less += 1;
                } else if (main_b == main_a) {
                    main_equal += 1;
                }

                if (compare_b < compare_a) {
                    compare_less += 1;
                } else if (compare_b == compare_a) {
                    compare_equal += 1;
                }
            }

            double main_rank =
                1.0 + static_cast<double>(main_less) + 0.5 * static_cast<double>(main_equal - 1);
            double compare_rank = 1.0 +
                                  static_cast<double>(compare_less) +
                                  0.5 * static_cast<double>(compare_equal - 1);
            double dx = main_rank - mean_rank;
            double dy = compare_rank - mean_rank;
            cov += dx * dy;
            var_main += dx * dx;
            var_compare += dy * dy;
        }

        double denom = sqrt(var_main * var_compare);
        if (!isfinite(denom) || denom == 0.0) {
            continue;
        }

        double raw = cov / denom;
        row_out_raw[i] = raw;

        int smooth_start = i + 1 - smoothing_length;
        if (smooth_start < 0) {
            continue;
        }

        bool smooth_valid = true;
        double smooth_sum = 0.0;
        for (int j = smooth_start; j <= i; ++j) {
            double value = row_out_raw[j];
            if (!isfinite(value)) {
                smooth_valid = false;
                break;
            }
            smooth_sum += value;
        }

        if (smooth_valid) {
            row_out_smoothed[i] = smooth_sum / static_cast<double>(smoothing_length);
        }
    }
}
