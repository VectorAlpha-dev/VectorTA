#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void psychological_line_batch_f32(
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

    float scale = 100.0f / static_cast<float>(length);

    for (int t = 0; t < len; ++t) {
        if (t < length) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t - length;
        bool valid = true;
        int rising = 0;

        for (int i = start; i <= t; ++i) {
            if (!isfinite(data[i])) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        for (int i = start + 1; i <= t; ++i) {
            rising += static_cast<int>(data[i] > data[i - 1]);
        }

        row[t] = static_cast<float>(rising) * scale;
    }
}
