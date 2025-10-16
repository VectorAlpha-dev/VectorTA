// CUDA kernels for rolling Variance (VAR) with optional nbdev scaling.
// Math pattern: prefix-sum/rational. Host builds f64 prefix sums of x and x^2
// and a prefix count of NaNs to exactly mirror scalar warmup/NaN semantics.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }

// ----------------------- Batch: one series × many params -----------------------
// out layout: row-major [n_combos, len]
extern "C" __global__ void var_batch_f32(
    const double* __restrict__ prefix_sum,     // len+1
    const double* __restrict__ prefix_sum_sq,  // len+1
    const int*    __restrict__ prefix_nan,     // len+1
    int len,
    int first_valid,
    const int*    __restrict__ periods,        // n_combos
    const float*  __restrict__ nbdev2,         // n_combos (nbdev^2)
    int n_combos,
    float*        __restrict__ out)            // [n_combos, len]
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float scale = nbdev2[combo];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = dev_nan();
        if (t >= warm) {
            const int start = t + 1 - period; // using len+1 prefixes
            const int bad = prefix_nan[t + 1] - prefix_nan[start];
            if (bad == 0) {
                const double sum  = prefix_sum[t + 1]    - prefix_sum[start];
                const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[start];
                const double den  = static_cast<double>(period);
                const double mean = sum / den;
                double var = (sum2 / den) - (mean * mean);
                if (var < 0.0) var = 0.0; // guard tiny negatives
                out_val = __double2float_rn(var) * scale;
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// -------- Many-series × one param (time-major) --------
// Prefix arrays are time-major and sized rows*cols + 1, with prefix at (t,s)
// stored at index (t*cols + s) + 1.
extern "C" __global__ void var_many_series_one_param_f32(
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,   // cols
    int series_len,   // rows
    const int*    __restrict__ first_valids,   // per series
    float nbdev2,
    float*        __restrict__ out_tm)         // time-major
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    const int warm = fv + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series; // time-major index
        float out_val = dev_nan();
        if (t >= warm) {
            const int start_pre_t = t - period; // t0 - 1
            // Use global prefix index 0 when start_pre_t < 0
            const int w_end = idx + 1;
            const int w_start = (start_pre_t >= 0)
                ? (start_pre_t * num_series + series + 1)
                : 0;
            const int bad = prefix_nan_tm[w_end] - prefix_nan_tm[w_start];
            if (bad == 0) {
                const double sum  = prefix_sum_tm[w_end]    - prefix_sum_tm[w_start];
                const double sum2 = prefix_sum_sq_tm[w_end] - prefix_sum_sq_tm[w_start];
                const double den  = static_cast<double>(period);
                const double mean = sum / den;
                double var = (sum2 / den) - (mean * mean);
                if (var < 0.0) var = 0.0;
                out_val = __double2float_rn(var) * nbdev2;
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}
