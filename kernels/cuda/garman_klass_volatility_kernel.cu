#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool valid_ohlc(float o, float h, float l, float c) {
    return isfinite(o) && isfinite(h) && isfinite(l) && isfinite(c) &&
           o > 0.0f && h > 0.0f && l > 0.0f && c > 0.0f;
}

static __forceinline__ __device__ float gk_term(float o, float h, float l, float c) {
    const float hl = logf(h / l);
    const float co = logf(c / o);
    const float coeff = 2.0f * logf(2.0f) - 1.0f;
    return 0.5f * hl * hl - coeff * co * co;
}

extern "C" __global__ void garman_klass_precompute_terms_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int* __restrict__ valid_flags,
    float* __restrict__ terms
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < series_len;
         j += blockDim.x * gridDim.x) {
        const float o = open[j];
        const float h = high[j];
        const float l = low[j];
        const float c = close[j];
        const bool valid = valid_ohlc(o, h, l, c);
        valid_flags[j] = valid ? 1 : 0;
        terms[j] = valid ? gk_term(o, h, l, c) : 0.0f;
    }
}

extern "C" __global__ void garman_klass_prefix_terms_f32(
    const int* __restrict__ valid_flags,
    const float* __restrict__ terms,
    int series_len,
    int* __restrict__ prefix_valid,
    float* __restrict__ prefix_sum
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    prefix_valid[0] = 0;
    prefix_sum[0] = 0.0f;

    int valid_acc = 0;
    float sum_acc = 0.0f;
    for (int j = 0; j < series_len; ++j) {
        valid_acc += valid_flags[j];
        sum_acc += terms[j];
        prefix_valid[j + 1] = valid_acc;
        prefix_sum[j + 1] = sum_acc;
    }
}

extern "C" __global__ void garman_klass_volatility_batch_prefix_f32(
    const int* __restrict__ lookbacks,
    int series_len,
    int first_valid,
    int n_combos,
    const int* __restrict__ prefix_valid,
    const float* __restrict__ prefix_sum,
    float* __restrict__ out
) {
    const int combo = (int)blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    __shared__ int lookback_s;
    __shared__ int warmup_s;
    __shared__ int combo_valid_s;
    __shared__ float inv_lb_s;

    if (threadIdx.x == 0) {
        const int lookback = lookbacks[combo];
        const int combo_valid = lookback > 0 && lookback <= series_len;
        lookback_s = lookback;
        warmup_s = first_valid + lookback - 1;
        combo_valid_s = combo_valid;
        inv_lb_s = combo_valid ? 1.0f / (float)lookback : 0.0f;
    }
    __syncthreads();

    const float nan_f = __int_as_float(0x7fffffff);
    const int base = combo * series_len;
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < series_len;
         t += (int)blockDim.x * (int)gridDim.x) {
        float out_v = nan_f;
        if (combo_valid_s != 0 && t >= warmup_s) {
            const int window_start = t + 1 - lookback_s;
            const int valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
            if (valid_count == lookback_s) {
                float variance = (prefix_sum[t + 1] - prefix_sum[window_start]) * inv_lb_s;
                if (variance < 0.0f) {
                    variance = 0.0f;
                }
                out_v = sqrtf(variance);
            }
        }
        out[base + t] = out_v;
    }
}

extern "C" __global__ void garman_klass_volatility_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int lookback,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) {
        return;
    }

    const float nan_f = __int_as_float(0x7fffffff);
    int first_valid = -1;
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + s;
        if (valid_ohlc(open_tm[idx], high_tm[idx], low_tm[idx], close_tm[idx])) {
            first_valid = t;
            break;
        }
    }

    if (first_valid < 0) {
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + s] = nan_f;
        }
        return;
    }

    const int warmup = first_valid + lookback - 1;
    for (int t = 0; t < series_len; ++t) {
        float out_v = nan_f;
        if (t >= warmup) {
            bool valid = true;
            float sum = 0.0f;
            for (int j = t + 1 - lookback; j <= t; ++j) {
                const int idx = j * num_series + s;
                const float o = open_tm[idx];
                const float h = high_tm[idx];
                const float l = low_tm[idx];
                const float c = close_tm[idx];
                if (!valid_ohlc(o, h, l, c)) {
                    valid = false;
                    break;
                }
                sum += gk_term(o, h, l, c);
            }
            if (valid) {
                float variance = sum / (float)lookback;
                if (variance < 0.0f) {
                    variance = 0.0f;
                }
                out_v = sqrtf(variance);
            }
        }
        out_tm[t * num_series + s] = out_v;
    }
}
