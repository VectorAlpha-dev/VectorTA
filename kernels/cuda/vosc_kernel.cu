// CUDA kernels for VOSC (Volume Oscillator)
//
// Category: Prefix-sum/rational.
// Given prefix sums P where P[0]=0 and P[t+1]=P[t]+x[t], each output is:
//   long_avg = (P[t+1] - P[t+1-L]) / L
//   short_avg = (P[t+1] - P[t+1-S]) / S
//   VOSC = 100 * (short_avg - long_avg) / long_avg
// Warmup: indices < first_valid + L - 1 are NaN.
//
// Optimization: Replace sustained FP64 math with compensated FP32 (double-single)
// using two-float arithmetic (hi+lo). We keep wrapper signatures unchanged
// (double* prefix inputs) and convert the few prefix entries we touch per output
// to float2 on-the-fly. This removes FP64 multiply/divide from the hot path and
// uses FP32 FMA and fast divide, while preserving accuracy against cancellation
// when subtracting adjacent prefix sums.

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef VOSC_NAN
#define VOSC_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// ---------------- Float-Float (double-single) utilities ----------------
// Represent a number as hi+lo (two floats). Based on TwoSum/Dekker family.
// Keeping intrinsics in FP32 gives high throughput on Ada+ while providing
// enough precision (~44-48 bits) for prefix-sum differences.
struct ds {
    float hi;
    float lo;
};

__device__ __forceinline__ ds ds_make(float hi, float lo) { ds r{hi, lo}; return r; }

// Convert a double to ds (hi,lo) with one FP64 subtraction. Used for on-the-fly
// packing when wrappers provide double prefix arrays.
__device__ __forceinline__ ds ds_from_double(double d) {
    float hi = (float)d;
    float lo = (float)(d - (double)hi); // one FP64 subtract during conversion
    return ds_make(hi, lo);
}

__device__ __forceinline__ ds ds_add(ds a, ds b) {
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float e  = (a.hi - (s - bb)) + (b.hi - bb);
    float t  = e + a.lo + b.lo;
    float hi = s + t;
    float lo = t - (hi - s);
    return ds_make(hi, lo);
}
__device__ __forceinline__ ds ds_neg(ds a) { return ds_make(-a.hi, -a.lo); }
__device__ __forceinline__ ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }

__device__ __forceinline__ ds ds_mul_f(ds a, float k) {
    // hi*k plus error term; use FMA to capture FP32 rounding error
    float p  = a.hi * k;
    float e  = fmaf(a.hi, k, -p) + a.lo * k;
    float hi = p + e;
    float lo = e - (hi - p);
    return ds_make(hi, lo);
}

__device__ __forceinline__ float ds_to_float(ds a) { return a.hi + a.lo; }

// One-series × many-params (plain), grid.y enumerates parameter combos
extern "C" __global__ void vosc_batch_prefix_f32(
    const double* __restrict__ prefix_sum, // length = len + 1
    int len,
    int first_valid,
    const int* __restrict__ short_periods, // [n_combos]
    const int* __restrict__ long_periods,  // [n_combos]
    int n_combos,
    float* __restrict__ out                // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;

    const int warm = first_valid + L - 1;
    const int row_off = combo * len;
    // Compute reciprocals in FP32 (fast path)
    const float inv_S = __fdividef(1.0f, (float)S);
    const float inv_L = __fdividef(1.0f, (float)L);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;
            // Convert the three needed prefix entries to ds (one FP64 subtract each)
            ds PT = ds_from_double(prefix_sum[t1]);
            ds PS = ds_from_double(prefix_sum[sS]);
            ds PL = ds_from_double(prefix_sum[sL]);

            // Window sums in DS, then average; compute numerator in DS to reduce cancellation
            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, inv_S);
            ds lavg_ds = ds_mul_f(long_sum,  inv_L);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series × one-param (time-major):
// prefix_tm: (rows+1) x cols, time-major; out_tm: rows x cols, time-major
extern "C" __global__ void vosc_many_series_one_param_f32(
    const double* __restrict__ prefix_tm, // (rows+1) x cols in time-major order
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, // [num_series]
    float* __restrict__ out_tm            // rows x cols, time-major
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;

    const int warm = first_valids[series] + long_period - 1;
    const int stride = num_series;
    const double inv_S = 1.0 / static_cast<double>(short_period);
    const double inv_L = 1.0 / static_cast<double>(long_period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    while (t < series_len) {
        const int out_idx = t * stride + series;
        float out_val = VOSC_NAN;
        if (t >= warm) {
            const int t1 = t + 1;
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_idx_t  = t1 * stride + series;
            const int p_idx_sS = sS * stride + series;
            const int p_idx_sL = sL * stride + series;
            const double short_sum = prefix_tm[p_idx_t] - prefix_tm[p_idx_sS];
            const double long_sum  = prefix_tm[p_idx_t] - prefix_tm[p_idx_sL];
            const double lavg = long_sum * inv_L;
            const double savg = short_sum * inv_S;
            const double v = 100.0 * (savg - lavg) / lavg;
            out_val = static_cast<float>(v);
        }
        out_tm[out_idx] = out_val;
        t += step;
    }
}

// ---------------- Optional DS variants and pack kernel ----------------
// These are not wired by the current Rust wrappers, but are provided for
// future use when switching to packed float2 prefix arrays.
extern "C" __global__ void pack_double_to_float2(
    const double* __restrict__ in, float2* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (i < n) {
        double d = in[i];
        float hi = (float)d;
        float lo = (float)(d - (double)hi);
        out[i] = make_float2(hi, lo);
        i += stride;
    }
}

extern "C" __global__ void vosc_batch_prefix_f32_ds(
    const float2* __restrict__ prefix_f2,
    int len,
    int first_valid,
    const int* __restrict__ short_periods,
    const int* __restrict__ long_periods,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int S = short_periods[combo];
    const int L = long_periods[combo];
    if (UNLIKELY(S <= 0 || L <= 0)) return;
    const int warm = first_valid + L - 1;
    const int row_off = combo * len;
    const float invS = __fdividef(1.0f, (float)S);
    const float invL = __fdividef(1.0f, (float)L);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        float out_val = VOSC_NAN;
        if (LIKELY(t >= warm)) {
            const int t1 = t + 1;
            int sS = t1 - S; if (sS < 0) sS = 0;
            int sL = t1 - L; if (sL < 0) sL = 0;
            float2 pt = prefix_f2[t1];
            float2 pS = prefix_f2[sS];
            float2 pL = prefix_f2[sL];
            ds PT = ds_make(pt.x, pt.y);
            ds PS = ds_make(pS.x, pS.y);
            ds PL = ds_make(pL.x, pL.y);
            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, invS);
            ds lavg_ds = ds_mul_f(long_sum,  invL);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

extern "C" __global__ void vosc_many_series_one_param_f32_ds_tm_coalesced(
    const float2* __restrict__ prefix_tm,
    int short_period,
    int long_period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm,
    int row_base)
{
    if (UNLIKELY(short_period <= 0 || long_period <= 0)) return;
    const int t_global = row_base + blockIdx.y;
    if (t_global >= series_len) return;
    const float invS = __fdividef(1.0f, (float)short_period);
    const float invL = __fdividef(1.0f, (float)long_period);
    const int stride = num_series;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    const int t1 = t_global + 1;
    const int p_t1 = t1 * stride;
    while (s < num_series) {
        const int warm = first_valids[s] + long_period - 1;
        float out_val = VOSC_NAN;
        if (t_global >= warm) {
            int sS = t1 - short_period; if (sS < 0) sS = 0;
            int sL = t1 - long_period;  if (sL < 0) sL = 0;
            const int p_sS = sS * stride;
            const int p_sL = sL * stride;
            float2 pt = prefix_tm[p_t1 + s];
            float2 ps = prefix_tm[p_sS + s];
            float2 pl = prefix_tm[p_sL + s];
            ds PT = ds_make(pt.x, pt.y);
            ds PS = ds_make(ps.x, ps.y);
            ds PL = ds_make(pl.x, pl.y);
            ds short_sum = ds_sub(PT, PS);
            ds long_sum  = ds_sub(PT, PL);
            ds savg_ds = ds_mul_f(short_sum, invS);
            ds lavg_ds = ds_mul_f(long_sum,  invL);
            float lavg = ds_to_float(lavg_ds);
            float num  = ds_to_float(ds_sub(savg_ds, lavg_ds));
            float v = 100.0f * num * __fdividef(1.0f, lavg);
            out_val = v;
        }
        out_tm[t_global * stride + s] = out_val;
        s += step;
    }
}

