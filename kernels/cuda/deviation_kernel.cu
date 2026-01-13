














#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }





struct twof { float hi, lo; };


__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}

__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

__device__ __forceinline__ twof make_twof(float hi, float lo) { return {hi, lo}; }

__device__ __forceinline__ twof twof_add(twof x, twof y) {
    float s, e; two_sum(x.hi, y.hi, s, e);
    float t = x.lo + y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sub(twof x, twof y) {
    float s, e; two_sum(x.hi, -y.hi, s, e);
    float t = x.lo - y.lo;
    float sh, sl; quick_two_sum(s, e + t, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_scale(twof x, float k) {
    float p, e; two_prod(x.hi, k, p, e);
    e = fmaf(x.lo, k, e);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ twof twof_sqr(twof x) {

    float p, e; two_prod(x.hi, x.hi, p, e);
    e = fmaf(2.0f * x.hi, x.lo, e) + (x.lo * x.lo);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ float twof_to_f(twof x) { return x.hi + x.lo; }


__device__ __forceinline__ twof ld_twof(const float2* __restrict__ a, int idx) {
    float2 v = a[idx];
    return make_twof(v.x, v.y);
}



extern "C" __global__ void deviation_batch_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const size_t row_off = static_cast<size_t>(combo) * static_cast<size_t>(len);
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);
    const int nan_base = prefix_nan[first_valid];
    const bool any_nan_since_first = (prefix_nan[len] - nan_base) != 0;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = dev_nan();
        if (t >= warm) {
            const int start = t + 1 - period;
            bool ok = true;
            if (any_nan_since_first) {
                ok = (prefix_nan[t + 1] - prefix_nan[start]) == 0;
            }
            if (ok) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    const float2 ps_e  = prefix_sum[t + 1];
                    const float2 ps_s  = prefix_sum[start];
                    const float2 ps2_e = prefix_sum_sq[t + 1];
                    const float2 ps2_s = prefix_sum_sq[start];

                    const float sum  = (ps_e.x  - ps_s.x)  + (ps_e.y  - ps_s.y);
                    const float sum2 = (ps2_e.x - ps2_s.x) + (ps2_e.y - ps2_s.y);

                    const float mean = sum * inv_den;
                    const float ex2  = sum2 * inv_den;
                    float var_f = fmaf(-mean, mean, ex2);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}





extern "C" __global__ void deviation_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float*        __restrict__ out_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + period - 1;
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = dev_nan();
        if (t >= warm) {
            const int wr = idx + 1;
            const int wl = wr - period * num_series;
            const int bad = prefix_nan_tm[wr] - prefix_nan_tm[wl];
            if (bad == 0) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    twof s1  = twof_sub(ld_twof(prefix_sum_tm,    wr),
                                         ld_twof(prefix_sum_tm,    wl));
                    twof s2  = twof_sub(ld_twof(prefix_sum_sq_tm, wr),
                                         ld_twof(prefix_sum_sq_tm, wl));

                    twof mean  = twof_scale(s1, inv_den);
                    twof mean2 = twof_scale(s2, inv_den);
                    twof var_ds = twof_sub(mean2, twof_sqr(mean));

                    float var_f = twof_to_f(var_ds);
                    if (var_f < 0.0f) var_f = 0.0f;
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}
