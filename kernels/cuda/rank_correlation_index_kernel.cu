#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double rci_compute_window(const double* data, int start, int length) {
    double len_f = static_cast<double>(length);
    double denom = len_f * static_cast<double>(length * length - 1);
    double sum = 0.0;

    for (int c = 0; c < length; ++c) {
        double p = data[start + c];
        double o = 1.0;
        double s = 0.0;
        for (int j = 0; j < length; ++j) {
            double other = data[start + j];
            if (p < other) {
                o += 1.0;
            } else if (p == other) {
                s += 1.0;
            }
        }
        double ord = o + (s - 1.0) * 0.5;
        double time_rank = static_cast<double>(length - c);
        double diff = time_rank - ord;
        sum += diff * diff;
    }

    return (1.0 - 6.0 * sum / denom) * 100.0;
}

extern "C" __global__ void rank_correlation_index_batch_f64(
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

    for (int t = 0; t < len; ++t) {
        row[t] = CUDART_NAN;
    }

    if (length < 2) {
        return;
    }

    int run_start = 0;
    for (int t = 0; t < len; ++t) {
        if (!isfinite(data[t])) {
            run_start = t + 1;
            continue;
        }
        if (t - run_start + 1 < length) {
            continue;
        }
        row[t] = rci_compute_window(data, t + 1 - length, length);
    }
}
