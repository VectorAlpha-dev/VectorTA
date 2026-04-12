#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
constexpr double LN_2 = 0.69314718055994530942;
}

extern "C" __global__ void fractal_dimension_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    double* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row[i] = CUDART_NAN;
    }

    if (length < 2 || length > len) {
        return;
    }

    double log_den = log(static_cast<double>(2 * length));
    for (int end = length - 1; end < len; ++end) {
        int start = end + 1 - length;
        bool valid = true;
        double low = 0.0;
        double high = 0.0;

        for (int i = start; i <= end; ++i) {
            double value = data[i];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            if (i == start || value < low) {
                low = value;
            }
            if (i == start || value > high) {
                high = value;
            }
        }

        if (!valid) {
            continue;
        }

        double range = high - low;
        double length_sum;
        if (!isfinite(range) || range <= 0.0) {
            length_sum = static_cast<double>(length - 1) / static_cast<double>(length);
        } else {
            double inv_n_sq = 1.0 / static_cast<double>(length * length);
            double prev = (data[start] - low) / range;
            double acc = 0.0;
            for (int i = start + 1; i <= end; ++i) {
                double cur = (data[i] - low) / range;
                double delta = cur - prev;
                acc += sqrt(delta * delta + inv_n_sq);
                prev = cur;
            }
            length_sum = acc;
        }

        if (!isfinite(length_sum) || length_sum <= 0.0) {
            continue;
        }

        row[end] = 1.0 + (log(length_sum) + LN_2) / log_den;
    }
}
