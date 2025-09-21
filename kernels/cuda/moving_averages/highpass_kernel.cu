// CUDA kernels for the single-pole High-Pass filter.
//
// Each kernel mirrors the scalar recurrence used by the CPU implementation.
// Arithmetic is carried out in FP64 to keep the GPU output aligned with the
// reference f64 path before casting results back to FP32 for storage.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void highpass_batch_f32(const float* __restrict__ prices,
                        const int* __restrict__ periods,
                        int series_len,
                        int n_combos,
                        float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > series_len) {
        return;
    }

    const double theta = (2.0 * CUDART_PI) / static_cast<double>(period);
    const double cos_val = cos(theta);
    if (fabs(cos_val) < 1e-12) {
        return;
    }
    const double sin_val = sin(theta);
    const double alpha = 1.0 + ((sin_val - 1.0) / cos_val);
    const double c = 1.0 - 0.5 * alpha;  // (1 - α/2)
    const double oma = 1.0 - alpha;      // (1 - α)

    const int base = combo * series_len;

    double prev_x = static_cast<double>(prices[0]);
    double prev_y = prev_x;
    out[base] = static_cast<float>(prev_y);

    for (int t = 1; t < series_len; ++t) {
        const double x = static_cast<double>(prices[t]);
        const double diff = x - prev_x;
        const double y = oma * prev_y + c * diff;
        out[base + t] = static_cast<float>(y);
        prev_x = x;
        prev_y = y;
    }
}

extern "C" __global__
void highpass_many_series_one_param_time_major_f32(const float* __restrict__ prices_tm,
                                                   int period,
                                                   int num_series,
                                                   int series_len,
                                                   float* __restrict__ out_tm) {
    if (period <= 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const double theta = (2.0 * CUDART_PI) / static_cast<double>(period);
    const double cos_val = cos(theta);
    if (fabs(cos_val) < 1e-12) {
        return;
    }
    const double sin_val = sin(theta);
    const double alpha = 1.0 + ((sin_val - 1.0) / cos_val);
    const double c = 1.0 - 0.5 * alpha;  // (1 - α/2)
    const double oma = 1.0 - alpha;      // (1 - α)

    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }

    const int stride = num_series;
    double prev_x = static_cast<double>(prices_tm[series_idx]);
    double prev_y = prev_x;
    out_tm[series_idx] = static_cast<float>(prev_y);

    for (int t = 1; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const double x = static_cast<double>(prices_tm[idx]);
        const double diff = x - prev_x;
        const double y = oma * prev_y + c * diff;
        out_tm[idx] = static_cast<float>(y);
        prev_x = x;
        prev_y = y;
    }
}
