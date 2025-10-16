// CUDA kernels for Bollinger Bands Width (BBW) using SMA + standard deviation.
//
// Batch kernel: one series × many params. Each parameter combo (period, u_plus_d)
// maps to blockIdx.y; threads in X iterate over time. Computation uses host-built
// prefix sums of values and squares plus a prefix count of NaNs to exactly match
// CPU warmup/NaN semantics.
//
// Many-series kernel: many series × one param (time-major layout). Threads in X
// march across time, Y indexes series. First-valid per series determines warmup.

#include <cuda_runtime.h>
#include <math.h>

// Quiet NaN helper
__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

extern "C" __global__ void bbw_sma_prefix_f32(
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    const float*  __restrict__ uplusd,
    int n_combos,
    float*        __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float k = uplusd[combo]; // devup + devdn
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f32();
        if (t >= warm) {
            const int start = t + 1 - period; // using len+1 prefixes
            const int nan_count = prefix_nan[t + 1] - prefix_nan[start];
            if (nan_count == 0) {
                const double sum  = prefix_sum[t + 1]    - prefix_sum[start];
                const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[start];
                const double den = static_cast<double>(period);
                const double mean = sum / den;
                double var = sum2 / den - mean * mean;
                if (var < 0.0) var = 0.0;
                const double std = (var > 0.0) ? sqrt(var) : 0.0;
                // Intentionally allow division by zero to mirror scalar semantics
                out_val = __double2float_rn((static_cast<double>(k) * std) / mean);
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// Many-series, one-param (time-major). prefix_* arrays are time-major with length rows*cols,
// holding cumulative sums per series across time. first_valids provides per-series warmup base.
extern "C" __global__ void bbw_multi_series_one_param_tm_f32(
    const double* __restrict__ prefix_sum_tm,
    const double* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*    __restrict__ first_valids,
    float u_plus_d,
    float*        __restrict__ out_tm)
{
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series_idx;
        float out_val = nan_f32();
        if (t >= warm) {
            const int start = (t + 1 - period) * num_series + series_idx;
            const int nan_count = prefix_nan_tm[idx + 1] - prefix_nan_tm[start];
            if (nan_count == 0) {
                const double sum  = prefix_sum_tm[idx + 1]    - prefix_sum_tm[start];
                const double sum2 = prefix_sum_sq_tm[idx + 1] - prefix_sum_sq_tm[start];
                const double den = static_cast<double>(period);
                const double mean = sum / den;
                double var = sum2 / den - mean * mean;
                if (var < 0.0) var = 0.0;
                const double std = (var > 0.0) ? sqrt(var) : 0.0;
                out_val = __double2float_rn((static_cast<double>(u_plus_d) * std) / mean);
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}
