// CUDA kernels for Deviation (rolling standard deviation, population) paths.
//
// This version removes FP64 from the hot path by converting the input prefix
// sums (stored as f64 by existing wrappers) into "double-single" two-float
// numbers on load and then performing all arithmetic in FP32 with
// error-compensated operations (TwoSum/QuickTwoSum/TwoProd + FMA). This retains
// CPU/NaN/warmup semantics while avoiding FP64 throughput bottlenecks on gaming
// GPUs.
//
// Notes:
// - Prefix arrays are passed as float2 (two-float double-single) for Σx and Σx²,
//   enabling aligned vectorized loads with no FP64 on device.
// - Population standard deviation (divide by period). Negative variances from
//   rounding are clamped to zero. period==1 fast-path returns 0.0 when valid.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }

// ----------------------------------------------------------------------------
// Double-single helpers (two-float precision)
// ----------------------------------------------------------------------------

struct twof { float hi, lo; }; // value ~= hi + lo, |lo| <= ~0.5 ulp(hi)

// Error-free transforms using IEEE754 RN
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
    e = fmaf(a, b, -p); // capture rounding error via FMA
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
    // (hi+lo)^2 = hi^2 + 2*hi*lo + lo^2
    float p, e; two_prod(x.hi, x.hi, p, e);
    e = fmaf(2.0f * x.hi, x.lo, e) + (x.lo * x.lo);
    float sh, sl; quick_two_sum(p, e, sh, sl);
    return make_twof(sh, sl);
}

__device__ __forceinline__ float twof_to_f(twof x) { return x.hi + x.lo; }

// Load a twof from float2 (native two-float prefix entry)
__device__ __forceinline__ twof ld_twof(const float2* __restrict__ a, int idx) {
    float2 v = a[idx];
    return make_twof(v.x, v.y);
}

// ----------------------- Batch: one series × many params -----------------------

extern "C" __global__ void deviation_batch_f32(
    const float2* __restrict__ prefix_sum,     // len+1 (two-float)
    const float2* __restrict__ prefix_sum_sq,  // len+1 (two-float)
    const int*    __restrict__ prefix_nan,     // len+1
    int len,
    int first_valid,
    const int*    __restrict__ periods,        // n_combos
    int n_combos,
    float*        __restrict__ out)            // [n_combos, len]
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const size_t row_off = static_cast<size_t>(combo) * static_cast<size_t>(len);
    const float inv_den = 1.0f / static_cast<float>(period);
    const bool is_one = (period == 1);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = dev_nan();
        if (t >= warm) {
            const int start = t + 1 - period; // using len+1 prefixes
            const int bad = prefix_nan[t + 1] - prefix_nan[start];
            if (bad == 0) {
                if (is_one) {
                    out_val = 0.0f;
                } else {
                    // Load window sums as double-single and compute variance in DS
                    twof s1  = twof_sub(ld_twof(prefix_sum,    t + 1),
                                         ld_twof(prefix_sum,    start));
                    twof s2  = twof_sub(ld_twof(prefix_sum_sq, t + 1),
                                         ld_twof(prefix_sum_sq, start));
                    twof mean  = twof_scale(s1, inv_den);
                    twof mean2 = twof_scale(s2, inv_den);
                    twof var_ds = twof_sub(mean2, twof_sqr(mean));

                    float var_f = twof_to_f(var_ds);
                    if (var_f < 0.0f) var_f = 0.0f; // clamp tiny negatives
                    out_val = (var_f > 0.0f) ? sqrtf(var_f) : 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// -------- Many-series × one param (time-major) --------
// Prefix arrays are time-major and sized rows*cols + 1, with prefix at (t,s)
// stored at index (t*cols + s) + 1.

extern "C" __global__ void deviation_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,   // cols
    int series_len,   // rows
    const int*    __restrict__ first_valids,   // per series
    float*        __restrict__ out_tm)         // time-major
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
        const int idx = t * num_series + series; // time-major index
        float out_val = dev_nan();
        if (t >= warm) {
            const int wr = idx + 1;                      // end prefix index
            const int wl = wr - period * num_series;     // start prefix index
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
