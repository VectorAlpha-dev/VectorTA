// CUDA kernels for Kurtosis (excess kurtosis) indicator.
//
// DS rewrite: remove FP64 from device code and use double‑single (2×FP32)
// accumulators for improved precision on GeForce/Ada GPUs. Preserves the
// original semantics: NaN handling, warmup indexing, and output format.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// =================== DS (double-single) arithmetic on FP32 ===================
// hi+lo with ~48-50 bits of precision. Implemented via TwoSum/TwoProd + FMA.
// Based on Shewchuk (TwoSum) and Thall (double-float). No special flags needed.
// References: Shewchuk 1997; Thall 2006.

struct __align__(8) dsf { float hi, lo; };

__device__ __forceinline__ dsf ds_from_float(float a) {
    return {a, 0.0f};
}

// General TwoSum for floats
__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// a.hi + b.hi plus low parts
__device__ __forceinline__ dsf ds_add(dsf a, dsf b) {
    float s, e;
    two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_neg(dsf a) { return {-a.hi, -a.lo}; }
__device__ __forceinline__ dsf ds_sub(dsf a, dsf b) { return ds_add(a, ds_neg(b)); }

// TwoProd using FMA for exact error of hi*hi
__device__ __forceinline__ dsf ds_mul(dsf a, dsf b) {
    float p  = a.hi * b.hi;
    float e  = fmaf(a.hi, b.hi, -p);                 // error of hi*hi
    e += a.hi * b.lo + a.lo * b.hi;                  // cross terms
    e += a.lo * b.lo;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_scale(dsf a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p);
    e += a.lo * s;
    float hi = p + e;
    float lo = e - (hi - p);
    return {hi, lo};
}

__device__ __forceinline__ dsf ds_square(dsf a) { return ds_mul(a, a); }
__device__ __forceinline__ float ds_to_f32(dsf a) { return a.hi + a.lo; }

// Helpers to load DS from float2 prefix arrays
__device__ __forceinline__ dsf ld_ds(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return {v.x, v.y};
}

// Canonical quiet NaN
__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fffffff); }

// ================= One-series × many-params (prefix-sum based, DS) =================
// Inputs: DS prefix sums for x, x^2, x^3, x^4 and int prefix of NaN counts.
// Output: excess kurtosis (float). Semantics identical to scalar path.
// NOTE: signature changed: double* ps_* -> float2* ps_* (DS).
extern "C" __global__ void kurtosis_batch_f32(
    const float2* __restrict__ ps_x,    // [len+1] DS
    const float2* __restrict__ ps_x2,   // [len+1] DS
    const float2* __restrict__ ps_x3,   // [len+1] DS
    const float2* __restrict__ ps_x4,   // [len+1] DS
    const int*    __restrict__ ps_nan,  // [len+1]
    int len,
    int first_valid,
    const int* __restrict__ periods,    // [n_combos]
    int n_combos,
    float* __restrict__ out             // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float inv_n = 1.0f / (float)period;

    while (t < len) {
        float out_val = qnan_f32();
        if (t >= warm) {
            const int end   = t + 1;
            int start = end - period;
            if (start < 0) start = 0;

            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {
                // Window raw sums via DS prefix differences (fast path: hi/lo diffs -> float)
                const float2 px_e  = ps_x[end];
                const float2 px_s  = ps_x[start];
                const float2 px2_e = ps_x2[end];
                const float2 px2_s = ps_x2[start];
                const float2 px3_e = ps_x3[end];
                const float2 px3_s = ps_x3[start];
                const float2 px4_e = ps_x4[end];
                const float2 px4_s = ps_x4[start];

                const float sum1 = (px_e.x  - px_s.x)  + (px_e.y  - px_s.y);
                const float sum2 = (px2_e.x - px2_s.x) + (px2_e.y - px2_s.y);
                const float sum3 = (px3_e.x - px3_s.x) + (px3_e.y - px3_s.y);
                const float sum4 = (px4_e.x - px4_s.x) + (px4_e.y - px4_s.y);

                const float mean = sum1 * inv_n;
                const float Ex2  = sum2 * inv_n;
                const float Ex3  = sum3 * inv_n;
                const float Ex4  = sum4 * inv_n;

                const float mean2 = mean * mean;
                const float m2 = Ex2 - mean2;

                if (m2 > 0.0f) {
                    // m4 = E[x^4] - 4*mean*E[x^3] + 6*mean^2*E[x^2] - 3*mean^4
                    const float term1 = fmaf(-4.0f * mean, Ex3, Ex4);
                    const float term2 = fmaf(6.0f * mean2, Ex2, term1);
                    const float mean4 = mean2 * mean2;
                    const float m4 = fmaf(-3.0f, mean4, term2);

                    const float denom = m2 * m2;
                    if (denom > 0.0f && !isnan(denom)) {
                        out_val = (m4 / denom) - 3.0f;
                    }
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// ============= Many-series × one-param (time-major), DS accumulators =============
// Time-major [t][series]. One block per series; thread 0 runs the sliding scan.
// Rebuild-on-NaN path retained; accumulators are DS with FMA-based products.
extern "C" __global__ void kurtosis_many_series_one_param_f32(
    const float* __restrict__ data_tm,    // [series_len * num_series], time-major
    const int*   __restrict__ first_valids,// [num_series]
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm            // [series_len * num_series], time-major
) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;
    const int stride = num_series;

    // Fill column with NaNs cooperatively
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = qnan_f32();
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int warm = first_valid + period - 1;
    const float inv_n = 1.0f / (float)period;

    // Bootstrap raw sums over initial window [first_valid .. warm]
    dsf s1 = ds_from_float(0.0f), s2 = ds_from_float(0.0f),
        s3 = ds_from_float(0.0f), s4 = ds_from_float(0.0f);
    int nan_in_win = 0;

    const int init_end = (warm + 1 < series_len) ? (warm + 1) : series_len;
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else {
            const float d  = v;
            const float d2 = fmaf(d, d, 0.0f);
            const float d3 = d2 * d;
            const float d4 = d2 * d2;
            s1 = ds_add(s1, ds_from_float(d));
            s2 = ds_add(s2, ds_from_float(d2));
            s3 = ds_add(s3, ds_from_float(d3));
            s4 = ds_add(s4, ds_from_float(d4));
        }
    }

    if (warm < series_len && nan_in_win == 0) {
        const dsf mean = ds_scale(s1, inv_n);
        const dsf Ex2  = ds_scale(s2, inv_n);
        const dsf Ex3  = ds_scale(s3, inv_n);
        const dsf Ex4  = ds_scale(s4, inv_n);

        const dsf mean2 = ds_square(mean);
        const dsf m2_ds = ds_sub(Ex2, mean2);
        const float m2  = ds_to_f32(m2_ds);

        float out0 = qnan_f32();
        if (m2 > 0.0f) {
            const dsf term1 = ds_sub(Ex4, ds_scale(ds_mul(mean, Ex3), 4.0f));
            const dsf term2 = ds_add(term1, ds_scale(ds_mul(mean2, Ex2), 6.0f));
            const dsf m4_ds = ds_sub(term2, ds_scale(ds_square(mean2), 3.0f));
            const float m4  = ds_to_f32(m4_ds);
            const float denom = m2 * m2;
            if (denom > 0.0f && !isnan(denom)) {
                out0 = (m4 / denom) - 3.0f;
            }
        }
        out_tm[warm * stride + series] = out0;
    }

    // Slide window forward
    for (int t = warm + 1; t < series_len; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {
            // Rebuild raw sums over the window
            s1 = ds_from_float(0.0f); s2 = ds_from_float(0.0f);
            s3 = ds_from_float(0.0f); s4 = ds_from_float(0.0f);
            nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else {
                    const float d  = vv;
                    const float d2 = fmaf(d, d, 0.0f);
                    const float d3 = d2 * d;
                    const float d4 = d2 * d2;
                    s1 = ds_add(s1, ds_from_float(d));
                    s2 = ds_add(s2, ds_from_float(d2));
                    s3 = ds_add(s3, ds_from_float(d3));
                    s4 = ds_add(s4, ds_from_float(d4));
                }
            }
        } else {
            // O(1) updates of raw sums with DS
            const float od  = old_v;
            const float nd  = new_v;
            const float od2 = fmaf(od, od, 0.0f);
            const float nd2 = fmaf(nd, nd, 0.0f);

            s1 = ds_add(s1, ds_from_float(nd - od));
            s2 = ds_add(s2, ds_from_float(nd2 - od2));
            s3 = ds_add(s3, ds_from_float(nd2 * nd - od2 * od));
            s4 = ds_add(s4, ds_from_float(nd2 * nd2 - od2 * od2));
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = qnan_f32();
        } else {
            const dsf mean = ds_scale(s1, inv_n);
            const dsf Ex2  = ds_scale(s2, inv_n);
            const dsf Ex3  = ds_scale(s3, inv_n);
            const dsf Ex4  = ds_scale(s4, inv_n);
            const dsf mean2 = ds_square(mean);
            const dsf m2_ds = ds_sub(Ex2, mean2);
            const float m2  = ds_to_f32(m2_ds);

            float outv = qnan_f32();
            if (m2 > 0.0f) {
                const dsf term1 = ds_sub(Ex4, ds_scale(ds_mul(mean, Ex3), 4.0f));
                const dsf term2 = ds_add(term1, ds_scale(ds_mul(mean2, Ex2), 6.0f));
                const dsf m4_ds = ds_sub(term2, ds_scale(ds_square(mean2), 3.0f));
                const float m4  = ds_to_f32(m4_ds);
                const float denom = m2 * m2;
                if (denom > 0.0f && !isnan(denom)) {
                    outv = (m4 / denom) - 3.0f;
                }
            }
            out_tm[t * stride + series] = outv;
        }
    }
}

// Optional: DS prefix builder kept out per guide to leave host/Rust generation.

