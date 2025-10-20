// CUDA kernels for Zscore (SMA mean + standard deviation).
//
// Each parameter combination (period, nbdev) is assigned to blockIdx.y. Threads
// in the x-dimension iterate over time indices and compute z-scores using
// precomputed prefix sums of the input data and squared data, along with a
// prefix count of NaNs to preserve CPU parity. All accumulation happens in
// float64 to minimise drift; the final results are written as float32 values.

#include <cuda_runtime.h>
#include <math.h>
#include "ds_float2.cuh"

// ----------------- Helpers -----------------
__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ bool nonpos_or_nan(float x) { return !(x > 0.0f); }

// Load dsf from float2 prefix array (x=hi, y=lo)
__device__ __forceinline__ dsf load_dsf_f2(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}

// ----------------- One-series × many-params (prefix-sum based, DS) -----------------
// Consumes float2 prefixes (double-single). Grid-x sweeps time; grid-y = combos.
extern "C" __global__ void zscore_sma_prefix_f32ds(
    const float*  __restrict__ data,             // [len]
    const float2* __restrict__ prefix_sum,       // [len+1] DS (x=hi,y=lo)
    const float2* __restrict__ prefix_sum_sq,    // [len+1] DS (x=hi,y=lo)
    const int*    __restrict__ prefix_nan,       // [len+1] prefix count of NaNs
    int len,
    int first_valid,
    const int*   __restrict__ periods,           // [n_combos]
    const float* __restrict__ nbdevs,            // [n_combos]
    int n_combos,
    float* __restrict__ out                      // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const float nbdev  = nbdevs[combo];
    if (period <= 0) return;

    const int   warm       = first_valid + period - 1;
    const int   row_offset = combo * len;
    const float invN       = 1.0f / (float)period;
    const float inv_nbdev  = (nbdev != 0.0f) ? (1.0f / nbdev) : 0.0f;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f32();

        if (t >= warm && nbdev != 0.0f) {
            const int end = t + 1;
            int start = end - period; if (start < 0) start = 0;
            const int nan_count = prefix_nan[end] - prefix_nan[start];
            if (nan_count == 0) {
                // DS window sums via prefix diffs
                const dsf s1 = ds_sub(load_dsf_f2(prefix_sum, end),    load_dsf_f2(prefix_sum, start));
                const dsf s2 = ds_sub(load_dsf_f2(prefix_sum_sq, end), load_dsf_f2(prefix_sum_sq, start));
                const dsf mean_ds = ds_scale(s1, invN);
                const dsf ex2_ds  = ds_scale(s2, invN);
                const dsf var_ds  = ds_sub(ex2_ds, ds_mul(mean_ds, mean_ds));
                const float mean  = ds_to_f(mean_ds);
                const float var   = ds_to_f(var_ds);
                if (var > 0.0f && isfinite(var)) {
                    const float sd = sqrtf(var);
                    const float denom_inv = (sd > 0.0f) ? (inv_nbdev / sd) : 0.0f;
                    const float x = data[t];
                    const float z = (x - mean) * denom_inv;
                    out_val = z;
                }
            }
        }

        out[row_offset + t] = out_val;
        t += stride;
    }
}

// Temporary: pack double prefix -> float2 (hi,lo) for DS kernels
extern "C" __global__
void pack_prefix_double_to_float2(const double* __restrict__ src,
                                  float2* __restrict__ dst,
                                  int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double v = src[i];
    float hi = (float)v;
    float lo = (float)(v - (double)hi);
    dst[i] = make_float2(hi, lo);
}

// ----------------- One-series × many-params (prefix-sum based) -----------------
extern "C" __global__ void zscore_sma_prefix_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    const double* __restrict__ prefix_sum_sq,
    const int* __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    const float* __restrict__ nbdevs,
    int n_combos,
    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const float nbdev = nbdevs[combo];
    if (period <= 0) {
        return;
    }

    const int warm = first_valid + period - 1;
    const int row_offset = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        float out_val = nan_f;

        if (t >= warm) {
            if (nbdev != 0.0f) {
                int start = t + 1 - period;
                if (start < 0) {
                    start = 0;
                }

                const int nan_count = prefix_nan[t + 1] - prefix_nan[start];
                if (nan_count == 0) {
                    const double sum = prefix_sum[t + 1] - prefix_sum[start];
                    const double sum2 = prefix_sum_sq[t + 1] - prefix_sum_sq[start];
                    const double mean = sum / static_cast<double>(period);
                    double variance = (sum2 / static_cast<double>(period)) - (mean * mean);
                    if (variance > 0.0) {
                        const double std_base = sqrt(variance);
                        const double denom = std_base * static_cast<double>(nbdev);
                        if (denom != 0.0 && !isnan(denom)) {
                            const double val = static_cast<double>(data[t]);
                            const double z = (val - mean) / denom;
                            out_val = static_cast<float>(z);
                        }
                    }
                }
            }
        }

        out[row_offset + t] = out_val;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------
// Time-major layout [t][series]. Each block handles one series (column).
// Thread 0 performs the sequential sliding-window scan; other threads help
// initialize the column with NaNs. Mean is SMA; deviation is population stddev.
// Output is the z-score: (x - mean) / (stddev * nbdev). If nbdev == 0 or
// any NaN in the window, result is NaN. Warmup and NaN semantics match scalar.
extern "C" __global__ void zscore_many_series_one_param_f32(
    const float* __restrict__ data_tm,    // [rows * cols], time-major
    const int* __restrict__ first_valids, // [cols]
    int period,
    float nbdev,
    int cols,
    int rows,
    float* __restrict__ out_tm            // [rows * cols], time-major
) {
    const int series = blockIdx.x;
    if (series >= cols || period <= 0) return;
    const int stride = cols;

    // Fill column with NaN cooperatively
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= rows) return;

    const int warm = first_valid + period - 1;
    if (nbdev == 0.0f) {
        // All outputs remain NaN by contract when denominator is zero
        return;
    }

    const double inv_n = 1.0 / (double)period;

    // Bootstrap raw sums over initial window [first_valid .. warm]
    double s1 = 0.0, s2 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, rows);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else { const double d = (double)v; s1 += d; s2 += d * d; }
    }

    if (warm < rows && nan_in_win == 0) {
        const double mean = s1 * inv_n;
        const double var = (s2 * inv_n) - (mean * mean);
        if (var > 1e-30) {
            const double sd_nb = sqrt(var) * (double)nbdev;
            const double x = (double)data_tm[warm * stride + series];
            out_tm[warm * stride + series] = (float)((x - mean) / sd_nb);
        }
        // else stays NaN
    }

    // Slide window forward
    for (int t = warm + 1; t < rows; ++t) {
        const int old_idx = t - period;
        const float old_v = data_tm[old_idx * stride + series];
        const float new_v = data_tm[t * stride + series];

        if (isnan(old_v) || isnan(new_v)) {
            // Rebuild over the current window
            s1 = 0.0; s2 = 0.0; nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else { const double d = (double)vv; s1 += d; s2 += d * d; }
            }
        } else {
            // O(1) update
            const double od = (double)old_v;
            const double nd = (double)new_v;
            s1 += nd - od;
            s2 += (nd * nd) - (od * od);
        }

        if (nan_in_win == 0) {
            const double mean = s1 * inv_n;
            const double var  = (s2 * inv_n) - (mean * mean);
            if (var > 1e-30) {
                const double sd_nb = sqrt(var) * (double)nbdev;
                const double x = (double)new_v;
                out_tm[t * stride + series] = (float)((x - mean) / sd_nb);
            } else {
                // leave NaN
            }
        } else {
            // leave NaN
        }
    }
}

