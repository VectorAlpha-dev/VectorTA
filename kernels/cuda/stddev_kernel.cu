// CUDA kernels for rolling Standard Deviation (population) indicator, FP64-light.
//
// Semantics mirror src/indicators/stddev.rs scalar path:
// - Warmup: output is NaN before warm = first_valid + period - 1
// - NaN handling: if any NaN exists in a window, output is NaN for that index
// - Variance <= 0 -> output 0.0 (scaled by nbdev still yields 0.0)
// - Accumulations use double-single (float-float) math; outputs in float32

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ===== Double-Single (float-float) helpers =====
// Representation: value ~= hi + lo, with |lo| << |hi|.
// Ops follow Dekker/Kahan-style algorithms; fmaf used to capture product error.

struct ds {
    float hi, lo;
    __device__ __forceinline__ ds() {}
    __device__ __forceinline__ ds(float h, float l=0.f): hi(h), lo(l) {}
};

__device__ __forceinline__ ds ds_from_f(float x) { return ds(x, 0.f); }
__device__ __forceinline__ float ds_to_f(ds a)  { return a.hi + a.lo; }
__device__ __forceinline__ ds ds_neg(ds a)      { return ds(-a.hi, -a.lo); }

// Convert from double to DS: hi = (float)x; lo captures residual as float
// Uses minimal FP64 (one subtraction) per conversion to avoid hot-loop FP64.
__device__ __forceinline__ ds ds_from_d(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return ds(hi, lo);
}

// addition
__device__ __forceinline__ ds ds_add(ds a, ds b) {
    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float e  = (a.hi - (s - bb)) + (b.hi - bb);
    e += a.lo + b.lo;
    float hi = s + e;
    float lo = e - (hi - s);
    return ds(hi, lo);
}
__device__ __forceinline__ ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }

// multiplication (ignore a.lo*b.lo term; DS target ~48-bit mantissa)
__device__ __forceinline__ ds ds_mul(ds a, ds b) {
    float p   = a.hi * b.hi;
    float err = fmaf(a.hi, b.hi, -p); // capture product rounding error
    err += a.hi * b.lo + a.lo * b.hi;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_scale(ds a, float s) {
    float p   = a.hi * s;
    float err = fmaf(a.hi, s, -p) + a.lo * s;
    float hi  = p + err;
    float lo  = err - (hi - p);
    return ds(hi, lo);
}

__device__ __forceinline__ ds ds_square(ds a) { return ds_mul(a, a); }

// ----------------- One-series × many-params (prefix-sum based) -----------------
// Uses prefix sums for x and x^2, and a prefix of NaN counts. Each block-y is
// a parameter row. Threads in x sweep time indices.
extern "C" __global__ void stddev_batch_f32(
    const float2* __restrict__ ps_x,    // [len+1] (DS prefix sums of x)
    const float2* __restrict__ ps_x2,   // [len+1] (DS prefix sums of x^2)
    const int*    __restrict__ ps_nan,  // [len+1]
    int len,
    int first_valid,
    const int* __restrict__ periods,    // [n_combos]
    const float* __restrict__ nbdevs,   // [n_combos]
    int n_combos,
    float* __restrict__ out             // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const float nb = nbdevs[combo];

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_n = 1.0 / (double)period;

    // Fast fill when nbdev==0: keep NaN until warm, then 0
    if (nb == 0.0f) {
        while (t < len) {
            out[row_off + t] = (t >= warm) ? 0.0f : nan_f;
            t += stride;
        }
        return;
    }

    while (t < len) {
        float outv = nan_f;
        if (t >= warm) {
            const int end = t + 1;
            int start = end - period;
            if (start < 0) start = 0;
            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {
                // Reconstruct double from DS prefixes: (hi + lo) as f64
                float2 ex = ps_x[end];    float2 sx = ps_x[start];
                float2 ex2 = ps_x2[end];  float2 sx2 = ps_x2[start];
                const double s1 = ((double)ex.x + (double)ex.y) - ((double)sx.x + (double)sx.y);
                const double s2 = ((double)ex2.x + (double)ex2.y) - ((double)sx2.x + (double)sx2.y);
                const double mean = s1 * inv_n;
                const double var  = (s2 * inv_n) - (mean * mean);
                outv = (var > 0.0) ? (float)(sqrt(var) * (double)nb) : 0.0f;
            }
        }
        out[row_off + t] = outv;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------
// Time-major layout [t][series]. Each block handles one series (column).
// Thread 0 performs the sequential sliding-window scan; other threads help
// initialize the output with NaNs.
extern "C" __global__ void stddev_many_series_one_param_f32(
    const float* __restrict__ data_tm,    // [rows * cols], time-major
    const int*  __restrict__ first_valids,// [cols]
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm            // [rows * cols], time-major
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;

    // fill column with NaN cooperatively
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm     = first_valid + period - 1;
    const double inv_n = 1.0 / (double)period;

    // Bootstrap over initial window [first_valid .. warm]
    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else { double d = (double)v; s1 += d; s2 += d * d; }
    }

    if (warm < rows) {
        if (nan_in_win == 0) {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[warm * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        } else {
            out_tm[warm * stride + series] = __int_as_float(0x7fffffff);
        }
    }

    // Slide window forward
    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        // Maintain O(1) raw sums and NaN count
        if (!isnan(old_v)) { double od = (double)old_v; s1 -= od; s2 -= od * od; }
        else { nan_in_win--; }
        if (!isnan(new_v)) { double nd = (double)new_v; s1 += nd; s2 += nd * nd; }
        else { nan_in_win++; }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            double mean = s1 * inv_n;
            double var  = (s2 * inv_n) - (mean * mean);
            out_tm[t * stride + series] = (var > 0.0 && nbdev != 0.0f)
                ? (float)(sqrt(var) * (double)nbdev) : 0.0f;
        }
    }
}
