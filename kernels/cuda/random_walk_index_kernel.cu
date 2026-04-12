#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

__device__ inline double nz_history_value(const double* src, int idx, int offset) {
    if (idx >= offset) {
        double value = src[idx - offset];
        if (isfinite(value)) {
            return value;
        }
    }
    return 0.0;
}

extern "C" __global__ void random_walk_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int first_valid,
    const int* __restrict__ lengths,
    int n_combos,
    double* __restrict__ out_high,
    double* __restrict__ out_low
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0 || first_valid < 0 || first_valid >= len) {
        return;
    }

    int length = lengths[combo_idx];
    double* row_high = out_high + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_low = out_low + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_high[i] = CUDART_NAN;
        row_low[i] = CUDART_NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    int warm = first_valid + length - 1;
    double sqrt_length = sqrt(static_cast<double>(length));
    double alpha = 1.0 / static_cast<double>(length);
    double prev_close = close[first_valid];
    double sum_tr = high[first_valid] - low[first_valid];
    double atr = CUDART_NAN;

    if (length == 1) {
        atr = sum_tr;
        double denom = atr * sqrt_length;
        if (isfinite(denom) && denom != 0.0) {
            row_high[first_valid] =
                (high[first_valid] - nz_history_value(low, first_valid, length)) / denom;
            row_low[first_valid] =
                (nz_history_value(high, first_valid, length) - low[first_valid]) / denom;
        }
    }

    for (int i = first_valid + 1; i < len; ++i) {
        double tr = fmax(
            high[i] - low[i],
            fmax(fabs(high[i] - prev_close), fabs(low[i] - prev_close))
        );

        if (i <= warm) {
            sum_tr += tr;
            if (i == warm) {
                atr = sum_tr / static_cast<double>(length);
            }
        } else {
            atr = alpha * (tr - atr) + atr;
        }

        if (i >= warm) {
            double denom = atr * sqrt_length;
            if (isfinite(denom) && denom != 0.0) {
                row_high[i] = (high[i] - nz_history_value(low, i, length)) / denom;
                row_low[i] = (nz_history_value(high, i, length) - low[i]) / denom;
            } else {
                row_high[i] = CUDART_NAN;
                row_low[i] = CUDART_NAN;
            }
        }

        prev_close = close[i];
    }
}
