// CUDA kernels for the Normalized Moving Average (NMA).
//
// The batch kernel evaluates one price series against a sweep of period values,
// reusing shared-memory buffers for the square-root differentials to avoid
// redundant work per output row. A companion kernel handles the many-series ×
// one-parameter path using the same shared table of weights.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef NMA_NAN
#define NMA_NAN (__int_as_float(0x7fffffff))
#endif

extern "C" __global__ void nma_batch_f32(const float* __restrict__ prices,
                                          const float* __restrict__ abs_diffs,
                                          const int* __restrict__ periods,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int base = combo * series_len;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int t = idx; t < series_len; t += stride) {
        out[base + t] = NMA_NAN;
    }

    const int period = periods[combo];
    if (period <= 0 || period >= series_len) {
        return;
    }
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len <= period) {
        return;
    }

    extern __shared__ float sqrt_diffs[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float s0 = sqrtf(static_cast<float>(i));
        const float s1 = sqrtf(static_cast<float>(i + 1));
        sqrt_diffs[i] = s1 - s0;
    }
    __syncthreads();

    const int warm = first_valid + period;

    for (int t = idx; t < series_len; t += stride) {
        if (t < warm) {
            continue;
        }

        float num = 0.0f;
        float denom = 0.0f;
        int cur = t;
        for (int k = 0; k < period; ++k, --cur) {
            const float oi = abs_diffs[cur];
            const float weight = sqrt_diffs[k];
            num += oi * weight;
            denom += oi;
        }

        const float ratio = denom > 0.0f ? num / denom : 0.0f;
        const int anchor = t - period + 1;
        const float latest = prices[anchor];
        const float prev = prices[anchor - 1];
        out[base + t] = latest * ratio + prev * (1.0f - ratio);
    }
}

extern "C" __global__ void nma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ abs_diffs_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {
    extern __shared__ float sqrt_diffs[];
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const float s0 = sqrtf(static_cast<float>(i));
        const float s1 = sqrtf(static_cast<float>(i + 1));
        sqrt_diffs[i] = s1 - s0;
    }
    __syncthreads();

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    const int stride = num_series;
    for (int row = 0; row < series_len; ++row) {
        out_tm[row * stride + series] = NMA_NAN;
    }

    if (period <= 0 || period >= series_len) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len <= period) {
        return;
    }

    const int warm = first_valid + period;

    for (int row = warm; row < series_len; ++row) {
        float num = 0.0f;
        float denom = 0.0f;
        int cur = row;
        for (int k = 0; k < period; ++k, --cur) {
            const float oi = abs_diffs_tm[cur * stride + series];
            const float weight = sqrt_diffs[k];
            num += oi * weight;
            denom += oi;
        }

        const float ratio = denom > 0.0f ? num / denom : 0.0f;
        const int anchor = row - period + 1;
        const float latest = prices_tm[anchor * stride + series];
        const float prev = prices_tm[(anchor - 1) * stride + series];
        out_tm[row * stride + series] = latest * ratio + prev * (1.0f - ratio);
    }
}
