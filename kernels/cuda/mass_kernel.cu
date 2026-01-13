
















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

