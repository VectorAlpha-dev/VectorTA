// CUDA kernels for Kurtosis (excess kurtosis) indicator.
//
// Semantics mirror src/indicators/kurtosis.rs scalar path:
// - Output is NaN before warm = first_valid + period - 1
// - If any NaN exists in a window, output is NaN for that index
// - Zero variance (m2 ~ 0) -> NaN
// - Accumulations in float64; outputs in float32

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ----------------- One-series × many-params (prefix-sum based) -----------------
// Uses prefix sums for x, x^2, x^3, x^4 and a prefix of NaN counts (where any
// x is NaN). Each block-y is a parameter row. Threads in x sweep time.
extern "C" __global__ void kurtosis_batch_f32(
    const double* __restrict__ ps_x,   // [len+1]
    const double* __restrict__ ps_x2,  // [len+1]
    const double* __restrict__ ps_x3,  // [len+1]
    const double* __restrict__ ps_x4,  // [len+1]
    const int*    __restrict__ ps_nan, // [len+1]
    int len,
    int first_valid,
    const int* __restrict__ periods,   // [n_combos]
    int n_combos,
    float* __restrict__ out            // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_n = 1.0 / (double)period;

    while (t < len) {
        float out_val = nan_f;
        if (t >= warm) {
            int end = t + 1;
            int start = end - period;
            if (start < 0) start = 0; // clamp for safety

            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {
                const double s1 = ps_x[end]  - ps_x[start];
                const double s2 = ps_x2[end] - ps_x2[start];
                const double s3 = ps_x3[end] - ps_x3[start];
                const double s4 = ps_x4[end] - ps_x4[start];

                const double mean = s1 * inv_n;
                // Central moments via raw moments (to avoid second pass):
                // m2 = E[x^2] - mean^2
                const double Ex2 = s2 * inv_n;
                const double m2  = Ex2 - mean * mean;
                if (m2 > 0.0) {
                    // m4 = E[(x-mean)^4] = E[x^4] - 4*mean*E[x^3] + 6*mean^2*E[x^2] - 3*mean^4
                    const double Ex3 = s3 * inv_n;
                    const double Ex4 = s4 * inv_n;
                    const double mean2 = mean * mean;
                    const double mean4 = mean2 * mean2;
                    const double m4 = Ex4 - 4.0 * mean * Ex3 + 6.0 * mean2 * Ex2 - 3.0 * mean4;

                    const double denom = m2 * m2;
                    if (denom > 0.0 && !isnan(denom)) {
                        const double g2 = (m4 / denom) - 3.0;
                        out_val = (float)g2;
                    }
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------
// Time-major layout [t][series]. Each block handles one series (column). Thread 0
// performs the sequential sliding-window scan while other threads help fill NaNs.
extern "C" __global__ void kurtosis_many_series_one_param_f32(
    const float* __restrict__ data_tm,   // [series_len * num_series], time-major
    const int* __restrict__ first_valids,// [num_series]
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm           // [series_len * num_series], time-major
) {
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;
    const int stride = num_series;

    // Fill column with NaNs cooperatively
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int warm = first_valid + period - 1;
    const double inv_n = 1.0 / (double)period;

    // Bootstrap raw sums over initial window [first_valid .. warm]
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    int nan_in_win = 0;
    const int init_end = min(warm + 1, series_len);
    for (int i = first_valid; i < init_end; ++i) {
        const float v = data_tm[i * stride + series];
        if (isnan(v)) { nan_in_win++; }
        else {
            const double d = (double)v;
            const double d2 = d * d;
            s1 += d;
            s2 += d2;
            s3 += d2 * d;
            s4 += d2 * d2;
        }
    }

    if (warm < series_len && nan_in_win == 0) {
        const double mean = s1 * inv_n;
        const double Ex2 = s2 * inv_n;
        const double m2  = Ex2 - mean * mean;
        float out0 = __int_as_float(0x7fffffff);
        if (m2 > 0.0) {
            const double Ex3 = s3 * inv_n;
            const double Ex4 = s4 * inv_n;
            const double mean2 = mean * mean;
            const double mean4 = mean2 * mean2;
            const double m4 = Ex4 - 4.0 * mean * Ex3 + 6.0 * mean2 * Ex2 - 3.0 * mean4;
            const double denom = m2 * m2;
            if (denom > 0.0 && !isnan(denom)) {
                out0 = (float)((m4 / denom) - 3.0);
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
            // Rebuild raw sums over the current window
            s1 = s2 = s3 = s4 = 0.0; nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float vv = data_tm[k * stride + series];
                if (isnan(vv)) { nan_in_win++; }
                else {
                    const double d = (double)vv;
                    const double d2 = d * d;
                    s1 += d; s2 += d2; s3 += d2 * d; s4 += d2 * d2;
                }
            }
        } else {
            // O(1) update of raw sums
            const double od = (double)old_v;
            const double nd = (double)new_v;
            const double od2 = od * od;
            const double nd2 = nd * nd;
            s1 += nd - od;
            s2 += nd2 - od2;
            s3 += nd2 * nd - od2 * od;
            s4 += nd2 * nd2 - od2 * od2;
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            const double mean = s1 * inv_n;
            const double Ex2 = s2 * inv_n;
            const double m2  = Ex2 - mean * mean;
            float outv = __int_as_float(0x7fffffff);
            if (m2 > 0.0) {
                const double Ex3 = s3 * inv_n;
                const double Ex4 = s4 * inv_n;
                const double mean2 = mean * mean;
                const double mean4 = mean2 * mean2;
                const double m4 = Ex4 - 4.0 * mean * Ex3 + 6.0 * mean2 * Ex2 - 3.0 * mean4;
                const double denom = m2 * m2;
                if (denom > 0.0 && !isnan(denom)) {
                    outv = (float)((m4 / denom) - 3.0);
                }
            }
            out_tm[t * stride + series] = outv;
        }
    }
}

