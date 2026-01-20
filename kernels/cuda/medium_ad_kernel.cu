













#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef MEDIUM_AD_MAX_PERIOD
#define MEDIUM_AD_MAX_PERIOD 512
#endif



__device__ __forceinline__ float fabsf_fast(float x) {
    return fabsf(x);
}


__device__ __forceinline__ void two_sum_f32(float a, float b, float &s, float &err) {
    s = a + b;
    float z = s - a;
    err = (a - (s - z)) + (b - z);
}



__device__ __forceinline__ float avg2_compensated(float a, float b) {
    float s, e;
    two_sum_f32(a, b, s, e);
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(0.5f, e, 0.5f * s);
#else
    return 0.5f * (s + e);
#endif
}


__device__ __forceinline__ float median3f(float a, float b, float c) {
    float ab = fminf(a, b), AB = fmaxf(a, b);
    float bc = fminf(AB, c), BC = fmaxf(AB, c);
    (void)BC;
    return fmaxf(ab, bc);
}



__device__ __forceinline__ float nth_element_inplace(float* a, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        const int mid = (left + right) >> 1;
        const float pivot = median3f(a[left], a[mid], a[right]);

        int lt = left, i = left, gt = right;
        while (i <= gt) {
            const float v = a[i];
            if (v < pivot) {
                float tmp = a[lt]; a[lt] = a[i]; a[i] = tmp;
                ++lt; ++i;
            } else if (v > pivot) {
                float tmp = a[i]; a[i] = a[gt]; a[gt] = tmp;
                --gt;
            } else {
                ++i;
            }
        }
        if (k < lt) {
            right = lt - 1;
        } else if (k > gt) {
            left = gt + 1;
        } else {
            return a[k];
        }
    }
    return a[k];
}




__device__ __forceinline__ float median_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {

    for (int i = 0; i < n; ++i) scratch[i] = orig[i];

    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace(scratch, n, k);
    } else {
        const int k = n >> 1;
        const float upper = nth_element_inplace(scratch, n, k);

        float lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) {
            lower = fmaxf(lower, scratch[i]);
        }
        return avg2_compensated(lower, upper);
    }
}


__device__ __forceinline__ float mad_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {

    const float med = median_from_window(orig, n, scratch);


    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf_fast(orig[i] - med);
    }


    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace(scratch, n, k);
    } else {
        const int k = n >> 1;
        const float upper = nth_element_inplace(scratch, n, k);
        float lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) lower = fmaxf(lower, scratch[i]);
        return avg2_compensated(lower, upper);
    }
}


__device__ __forceinline__ float mad_period_2(float x0, float x1) {

    return 0.5f * fabsf_fast(x1 - x0);
}




extern "C" __global__ void medium_ad_batch_f32(
    const float* __restrict__ data,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = nanf("");


    float orig[MEDIUM_AD_MAX_PERIOD];
    float scratch[MEDIUM_AD_MAX_PERIOD];


    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            if (period == 1) {
                const float v = data[t];
                out_val = isfinite(v) ? 0.0f : nan_f;
            } else if (period == 2) {
                const float x0 = data[t - 1];
                const float x1 = data[t];
                out_val = (isfinite(x0) && isfinite(x1)) ? mad_period_2(x0, x1) : nan_f;
            } else {
                const int start = t + 1 - period;
                bool has_nan = false;


                #pragma unroll 1
                for (int k = 0; k < period; ++k) {
                    const float v = data[start + k];
                    if (!isfinite(v)) has_nan = true;
                    orig[k] = v;
                }

                if (!has_nan) {
                    out_val = mad_from_window(orig, period, scratch);
                }
            }
        }

        out[row_off + t] = out_val;
        t += stride;
    }
}



extern "C" __global__ void medium_ad_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) {
        const float nan_f = nanf("");
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    int first_valid = first_valids[s];
    if (first_valid < 0) first_valid = 0;
    const int warm = first_valid + period - 1;
    const float nan_f = nanf("");


    int prefill = warm < rows ? warm : rows;
    for (int t = 0; t < prefill; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (warm >= rows) return;

    float orig[MEDIUM_AD_MAX_PERIOD];
    float scratch[MEDIUM_AD_MAX_PERIOD];

    for (int t = warm; t < rows; ++t) {
        if (period == 1) {
            const float v = data_tm[t * cols + s];
            out_tm[t * cols + s] = isfinite(v) ? 0.0f : nan_f;
            continue;
        }
        if (period == 2) {
            const float x0 = data_tm[(t - 1) * cols + s];
            const float x1 = data_tm[t * cols + s];
            out_tm[t * cols + s] = (isfinite(x0) && isfinite(x1)) ? mad_period_2(x0, x1) : nan_f;
            continue;
        }

        const int start = t + 1 - period;
        bool has_nan = false;
        for (int k = 0; k < period; ++k) {
            const float v = data_tm[(start + k) * cols + s];
            if (!isfinite(v)) has_nan = true;
            orig[k] = v;
        }

        if (has_nan) {
            out_tm[t * cols + s] = nan_f;
        } else {
            out_tm[t * cols + s] = mad_from_window(orig, period, scratch);
        }
    }
}

