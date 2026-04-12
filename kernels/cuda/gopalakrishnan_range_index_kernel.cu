#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void gopalakrishnan_range_index_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    const int* __restrict__ lengths,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int length = lengths[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (length <= 1) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    double log_length = log(static_cast<double>(length));

    for (int t = 0; t < len; ++t) {
        if (t + 1 < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        float highest = -CUDART_INF_F;
        float lowest = CUDART_INF_F;

        for (int i = start; i <= t; ++i) {
            float hi = high[i];
            float lo = low[i];
            if (!isfinite(hi) || !isfinite(lo)) {
                valid = false;
                break;
            }
            if (hi > highest) {
                highest = hi;
            }
            if (lo < lowest) {
                lowest = lo;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        double range = static_cast<double>(highest) - static_cast<double>(lowest);
        if (!(range > 0.0) || !isfinite(range)) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        row[t] = static_cast<float>(log(range) / log_length);
    }
}
