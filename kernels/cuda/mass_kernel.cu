// CUDA kernels for Mass Index (MASS) – FP64-free, double-single (float2) prefixes
//
// Math per scalar path (src/indicators/mass.rs):
// - ema1 = EMA_9(high - low)
// - ema2 = EMA_9(ema1)
// - ratio = ema1 / ema2, valid starting at first_valid + 16
// - MASS(period) = rolling sum over `period` of ratio
//
// For GPU batch (one series × many params), we precompute the ratio on host
// and provide double-single (float2) prefix sums of ratio along with a prefix
// count of NaNs so that any window containing a NaN yields a NaN output (matching
// scalar semantics with a ring buffer where NaN poisons the sum until it
// leaves the window). We avoid FP64 in device code completely.
//
// For many-series × one-param (time-major), the wrapper provides time-major
// prefix arrays of ratio and ratio-NaN counts.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mass_nan() { return __int_as_float(0x7fffffff); }

// ---------------------- error-free f32 helpers ----------------------

// Error-free transformations (Knuth/Dekker): all-f32
// Returns (s,e) such that a + b == s + e and s == fl(a+b)
__device__ __forceinline__ float2 two_sum_f32(float a, float b) {
    float s = a + b;
    float z = s - a;
    float e = (a - (s - z)) + (b - z);
    return make_float2(s, e);
}

// Returns (s,e) such that a - b == s + e and s == fl(a-b)
__device__ __forceinline__ float2 two_diff_f32(float a, float b) {
    float s = a - b;
    float z = s - a;
    float e = (a - (s - z)) - (b + z);
    return make_float2(s, e);
}

// Compute (A.hi+A.lo) - (B.hi+B.lo) in fp32 with ~48-bit effective precision
__device__ __forceinline__ float ds_diff_to_f32(const float2 A, const float2 B) {
    float2 d  = two_diff_f32(A.x, B.x);
    float2 s1 = two_sum_f32(d.x, A.y - B.y);
    float2 s2 = two_sum_f32(s1.x, d.y + s1.y);
    return s2.x + s2.y;
}

// ----------------------- Batch: one series × many params -----------------------

extern "C" __global__ void mass_batch_f32(
    const float2* __restrict__ prefix_ratio_ds, // len+1 (DS prefix)
    const int*    __restrict__ prefix_nan,   // len+1 (count of NaNs in ratio)
    int len,
    int first_valid,
    const int*    __restrict__ periods,      // n_combos
    int n_combos,
    float*        __restrict__ out           // [n_combos, len]
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
    int start = t + 1 - period; // maintain incrementally
    while (t < len) {
        float out_val = mass_nan();
        if (t >= warm) {
            const int p1 = t + 1; // using len+1 prefixes
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

// -------- Many-series × one param (time-major) --------
// Prefix arrays are time-major and sized rows*cols + 1, with prefix at (t,s)
// stored at index (t*cols + s) + 1.

extern "C" __global__ void mass_many_series_one_param_time_major_f32(
    const double* __restrict__ prefix_ratio_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,   // cols
    int series_len,   // rows
    const int*    __restrict__ first_valids,   // per series
    float*        __restrict__ out_tm          // time-major [rows * cols]
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + 16 + period - 1;

    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int t   = t0;

    while (t < series_len) {
        const int idx = t * num_series + series;               // (t,s)
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

