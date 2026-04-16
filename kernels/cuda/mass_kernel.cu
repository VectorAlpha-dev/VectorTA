#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mass_nan() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ float2 two_sum_f32(float a, float b) {
    float s = a + b;
    float z = s - a;
    float e = (a - (s - z)) + (b - z);
    return make_float2(s, e);
}


__device__ __forceinline__ float2 two_diff_f32(float a, float b) {
    float s = a - b;
    float z = s - a;
    float e = (a - (s - z)) - (b + z);
    return make_float2(s, e);
}


__device__ __forceinline__ float ds_diff_to_f32(const float2 A, const float2 B) {
    float2 d  = two_diff_f32(A.x, B.x);
    float2 s1 = two_sum_f32(d.x, A.y - B.y);
    float2 s2 = two_sum_f32(s1.x, d.y + s1.y);
    return s2.x + s2.y;
}

extern "C" __global__ void mass_build_prefix_one_series_ds_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    float2* __restrict__ prefix_ratio_ds,
    int* __restrict__ prefix_nan)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0 || first_valid < 0 || first_valid >= len) return;

    prefix_ratio_ds[0] = make_float2(0.0f, 0.0f);
    prefix_nan[0] = 0;

    const float alpha = 2.0f / 10.0f;
    const float inv_alpha = 1.0f - alpha;
    float ema1 = high[first_valid] - low[first_valid];
    float ema2 = ema1;
    const int start_ema2 = first_valid + 8;
    const int start_ratio = first_valid + 16;
    float acc_hi = 0.0f;
    float acc_lo = 0.0f;

    for (int i = 0; i < len; ++i) {
        if (i < first_valid) {
            prefix_ratio_ds[i + 1] = make_float2(acc_hi, acc_lo);
            prefix_nan[i + 1] = prefix_nan[i];
            continue;
        }

        const float hl = high[i] - low[i];
        ema1 = fmaf(alpha, hl, inv_alpha * ema1);
        if (i == start_ema2) {
            ema2 = ema1;
        }

        float ratio = mass_nan();
        if (i >= start_ema2) {
            ema2 = fmaf(alpha, ema1, inv_alpha * ema2);
            if (i >= start_ratio) {
                ratio = ema1 / ema2;
            }
        }

        const bool is_nan = !isfinite(ratio);
        if (!is_nan) {
            float2 s = two_sum_f32(acc_hi, ratio);
            float2 s2 = two_sum_f32(s.x, acc_lo);
            float2 s3 = two_sum_f32(s2.x, s.y + s2.y);
            acc_hi = s3.x;
            acc_lo = s3.y;
            prefix_nan[i + 1] = prefix_nan[i];
        } else {
            prefix_nan[i + 1] = prefix_nan[i] + 1;
        }
        prefix_ratio_ds[i + 1] = make_float2(acc_hi, acc_lo);
    }
}


extern "C" __global__ void mass_batch_f32(
    const float2* __restrict__ prefix_ratio_ds,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out
) {
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    const int period = periods[row];
    if (period <= 0) return;

    const int warm = first_valid + 16 + period - 1;
    const int row_off = row * len;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int t = t0;
    int start = t + 1 - period;
    while (t < len) {
        float out_val = mass_nan();
        if (t >= warm) {
            const int p1 = t + 1;
            const int bad = prefix_nan[p1] - prefix_nan[start];
            if (bad == 0) {
                const float2 a = prefix_ratio_ds[p1];
                const float2 b = prefix_ratio_ds[start];
                out_val = ds_diff_to_f32(a, b);
            }
        }
        out[row_off + t] = out_val;
        t     += stride;
        start += stride;
    }
}


extern "C" __global__ void mass_many_series_one_param_time_major_f32(
    const double* __restrict__ prefix_ratio_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float*        __restrict__ out_tm
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + 16 + period - 1;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int t   = t0;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = mass_nan();
        if (t >= warm) {
            const int start = (t + 1 - period) * num_series + series;
            const int bad = prefix_nan_tm[idx + 1] - prefix_nan_tm[start];
            if (bad == 0) {
                const double sum = prefix_ratio_tm[idx + 1] - prefix_ratio_tm[start];
                out_val = static_cast<float>(sum);
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}
