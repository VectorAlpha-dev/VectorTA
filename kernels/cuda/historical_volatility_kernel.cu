#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void historical_volatility_batch_f32(
    const float* __restrict__ data,
    int len,
    const int* __restrict__ lookbacks,
    const float* __restrict__ annualization_scales,
    int n_combos,
    float* __restrict__ out
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int lookback = lookbacks[combo_idx];
    float annualization_scale = annualization_scales[combo_idx];
    float* row = out + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);

    if (lookback <= 0) {
        for (int t = 0; t < len; ++t) {
            row[t] = CUDART_NAN_F;
        }
        return;
    }

    for (int t = 0; t < len; ++t) {
        if (t < lookback) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        int start = t + 1 - lookback;
        bool valid = true;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = start; i <= t; ++i) {
            float prev = data[i - 1];
            float curr = data[i];
            if (!isfinite(prev) || !isfinite(curr) || prev == 0.0f) {
                valid = false;
                break;
            }
            double ret = ((static_cast<double>(curr) / static_cast<double>(prev)) - 1.0) * 100.0;
            sum += ret;
            sumsq += ret * ret;
        }

        if (!valid) {
            row[t] = CUDART_NAN_F;
            continue;
        }

        double inv_lb = 1.0 / static_cast<double>(lookback);
        double mean = sum * inv_lb;
        double variance = sumsq * inv_lb - mean * mean;
        if (variance < 0.0) {
            variance = 0.0;
        }
        row[t] = static_cast<float>(sqrt(variance) * static_cast<double>(annualization_scale));
    }
}
