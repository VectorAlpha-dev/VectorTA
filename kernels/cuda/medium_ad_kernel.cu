// CUDA kernels for Median Absolute Deviation (MEDIUM_AD)
// Optimized selection + FP32 math for Ada+ (SM_89 semantics)
//
// Semantics preserved:
// - Warmup: write NaN until index warm = first_valid + period - 1
// - If any NaN in the window, output is NaN
// - Period == 1: output 0.0 for finite input, else NaN
// - Even-length median = average of the two middle elements
//
// Numeric/Perf notes:
// - Window computations are FP32 to avoid Ada FP64 1/64 throughput.
// - Even-length averages use a compensated FP32 average (TwoSum + half).
// - Selection uses in-place Quickselect (3-way partition) for expected O(n).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef MEDIUM_AD_MAX_PERIOD
#define MEDIUM_AD_MAX_PERIOD 512
#endif

// ---- FP32 helpers -----------------------------------------------------------

__device__ __forceinline__ float fabsf_fast(float x) {
    return fabsf(x);
}

// Error-free TwoSum for FP32 (Knuth/Shewchuk). Returns s = a+b and err so that s+err is exact.
__device__ __forceinline__ void two_sum_f32(float a, float b, float &s, float &err) {
    s = a + b;
    float z = s - a;
    err = (a - (s - z)) + (b - z);
}

// Averaging two FP32 numbers with reduced rounding + overflow risk.
// Uses TwoSum then halves the exact sum (s + err).
__device__ __forceinline__ float avg2_compensated(float a, float b) {
    float s, e;
    two_sum_f32(a, b, s, e);
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(0.5f, e, 0.5f * s);
#else
    return 0.5f * (s + e);
#endif
}

// Median-of-3 pivot (value) to stabilize partitioning.
__device__ __forceinline__ float median3f(float a, float b, float c) {
    float ab = fminf(a, b), AB = fmaxf(a, b);
    float bc = fminf(AB, c), BC = fmaxf(AB, c);
    (void)BC; // silence unused warning in some nvcc modes
    return fmaxf(ab, bc);
}

// In-place Quickselect with Dijkstra 3-way partitioning.
// Guarantees nth-element property on return: a[0..k-1] <= a[k] <= a[k+1..n-1]
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

// Compute the scalar median from a window in orig[], using scratch[] as a mutable copy.
// - odd n:  k = n/2; return nth_element(orig_copy, k)
// - even n: k = n/2; upper = nth_element(..., k), lower = max(scratch[0..k-1]); return avg(lower, upper)
__device__ __forceinline__ float median_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {
    // Copy orig into scratch for in-place selection.
    for (int i = 0; i < n; ++i) scratch[i] = orig[i];

    if (n & 1) {
        const int k = n >> 1;
        return nth_element_inplace(scratch, n, k);
    } else {
        const int k = n >> 1;
        const float upper = nth_element_inplace(scratch, n, k);
        // Lower median = max of left partition [0..k-1]
        float lower = scratch[0];
        #pragma unroll 1
        for (int i = 1; i < k; ++i) {
            lower = fmaxf(lower, scratch[i]);
        }
        return avg2_compensated(lower, upper);
    }
}

// Compute MAD from orig[], using scratch[] as working buffer (overwritten).
__device__ __forceinline__ float mad_from_window(const float* __restrict__ orig, int n, float* __restrict__ scratch) {
    // First median
    const float med = median_from_window(orig, n, scratch);

    // Transform to absolute deviations into scratch
    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf_fast(orig[i] - med);
    }

    // Median of absolute deviations
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

// Small, hot path special cases.
__device__ __forceinline__ float mad_period_2(float x0, float x1) {
    // median = (x0+x1)/2; deviations are equal to |x1-x0|/2; MAD = that value.
    return 0.5f * fabsf_fast(x1 - x0);
}

// ---- Kernels ---------------------------------------------------------------

// One series × many params (periods), time-major stride in X, combos in Y.
extern "C" __global__ void medium_ad_batch_f32(
    const float* __restrict__ data,     // [len]
    int len,
    int first_valid,
    const int* __restrict__ periods,    // [n_combos]
    int n_combos,
    float* __restrict__ out)            // [n_combos * len], row-major per combo
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > MEDIUM_AD_MAX_PERIOD) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = nanf("");

    // Per-thread local buffers (FP32). orig[] remains unmodified; scratch[] is mutated.
    float orig[MEDIUM_AD_MAX_PERIOD];
    float scratch[MEDIUM_AD_MAX_PERIOD];

    // Time index striding
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

                // Load window (FP32) into orig[], check NaNs/Inf.
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

// Many-series × one-param (time-major input/output)
// data_tm[t * cols + s]
extern "C" __global__ void medium_ad_many_series_one_param_f32(
    const float* __restrict__ data_tm, // [rows * cols], time-major
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids, // [cols]
    float* __restrict__ out_tm)           // [rows * cols], time-major
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
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

    // Only prefill [0..warm-1] with NaN (avoid writing the whole column).
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

