#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void vertical_horizontal_filter_batch_f32(
    const float* __restrict__ data,
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

    if (length <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        if (t + 1 < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - length;
        bool valid = true;
        float highest = -CUDART_INF_F;
        float lowest = CUDART_INF_F;
        float denom = 0.0f;

        for (int i = start; i <= t; ++i) {
            float value = data[i];
            if (!isfinite(value)) {
                valid = false;
                break;
            }
            if (value > highest) {
                highest = value;
            }
            if (value < lowest) {
                lowest = value;
            }
        }

        if (valid) {
            for (int i = start; i <= t; ++i) {
                if (i == 0) {
                    valid = false;
                    break;
                }
                float prev = data[i - 1];
                float curr = data[i];
                if (!isfinite(prev) || !isfinite(curr)) {
                    valid = false;
                    break;
                }
                denom += fabsf(curr - prev);
            }
        }

        if (!valid || !(denom > 0.0f) || !isfinite(highest) || !isfinite(lowest)) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        row[t] = fabsf(highest - lowest) / denom;
    }
}
