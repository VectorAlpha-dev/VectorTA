#include <cmath>
#include <cstddef>

extern "C" __global__ void emd_trend_batch_f64(
    const double* __restrict__ src,
    int len,
    const double* __restrict__ mults,
    const double* __restrict__ averages,
    const double* __restrict__ deviations,
    int rows,
    double* __restrict__ out_direction,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const double mult = mults[row];
    const double* row_avg = averages + static_cast<size_t>(row) * static_cast<size_t>(len);
    const double* row_dev = deviations + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_direction =
        out_direction + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_upper = out_upper + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_lower = out_lower + static_cast<size_t>(row) * static_cast<size_t>(len);

    double direction = 0.0;
    for (int i = 0; i < len; ++i) {
        const double avg = row_avg[i];
        const double dev = row_dev[i];
        if (isfinite(avg) && isfinite(dev)) {
            row_upper[i] = avg + dev * mult;
            row_lower[i] = avg - dev * mult;
        } else {
            row_upper[i] = NAN;
            row_lower[i] = NAN;
        }

        if (i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_upper[i]) &&
            isfinite(row_upper[i - 1]) && src[i] > row_upper[i] &&
            src[i - 1] <= row_upper[i - 1]) {
            direction = 1.0;
        } else if (
            i > 0 && isfinite(src[i]) && isfinite(src[i - 1]) && isfinite(row_lower[i]) &&
            isfinite(row_lower[i - 1]) && src[i] < row_lower[i] &&
            src[i - 1] >= row_lower[i - 1]
        ) {
            direction = -1.0;
        }
        row_direction[i] = direction;
    }
}
