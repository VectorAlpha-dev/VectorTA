// CUDA kernels for Ehlers Distance Coefficient Filter (EDCF).
//
// These kernels operate entirely in FP32 and mirror the scalar CPU
// implementation: first compute squared-distance weights for each sample,
// then apply the weighted average to produce the filtered output.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void edcf_compute_dist_f32(const float* __restrict__ prices,
                           int len,
                           int period,
                           int first_valid,
                           float* __restrict__ dist) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int start = first_valid + period;

    for (int k = idx; k < len; k += stride) {
        if (k < start) {
            dist[k] = 0.0f;
            continue;
        }
        const float xk = prices[k];
        float sum_sq = 0.0f;
        for (int lb = 1; lb < period; ++lb) {
            const float diff = xk - prices[k - lb];
            sum_sq = fmaf(diff, diff, sum_sq);
        }
        dist[k] = sum_sq;
    }
}

extern "C" __global__
void edcf_apply_weights_f32(const float* __restrict__ prices,
                            const float* __restrict__ dist,
                            int len,
                            int period,
                            int first_valid,
                            float* __restrict__ out_row) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int warm = first_valid + 2 * period;
    if (warm > len) {
        warm = len;
    }

    for (int j = idx; j < len; j += stride) {
        if (j < warm) {
            out_row[j] = NAN;
            continue;
        }

        float num = 0.0f;
        float denom = 0.0f;
        for (int i = 0; i < period; ++i) {
            const int k = j - i;
            const float w = dist[k];
            const float v = prices[k];
            num = fmaf(w, v, num);
            denom += w;
        }

        out_row[j] = (denom != 0.0f) ? (num / denom) : NAN;
    }
}
