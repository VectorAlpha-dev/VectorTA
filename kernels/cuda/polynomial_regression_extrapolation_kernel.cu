#include <cmath>
#include <cstddef>

extern "C" __global__ void polynomial_regression_extrapolation_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int rows,
    int max_length,
    const double* __restrict__ weights,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    const double* row_weights =
        weights + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (length <= 0 || length > len || length > max_length) {
        return;
    }

    int valid_run = 0;
    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            valid_run = 0;
            continue;
        }

        valid_run += 1;
        if (valid_run < length) {
            continue;
        }

        double acc = 0.0;
        for (int offset = 0; offset < length; ++offset) {
            acc += row_weights[offset] * data[i - offset];
        }
        row_out[i] = acc;
    }
}
