// CUDA kernels for Pearson correlation between High and Low (CORREL_HL)
//
// Semantics mirror src/indicators/correl_hl.rs (scalar path):
// - Output is NaN before warm = first_valid + period - 1
// - If any NaN exists in a window, output is NaN for that index
// - If either window variance is <= 0, output is 0.0
// - Accumulations in float64; outputs in float32

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ----------------- One-series × many-params (prefix-sum based) -----------------

// Each block-y is a parameter row. Threads along x sweep time indices using
// precomputed prefix sums for h, l, h^2, l^2, and h*l, plus a prefix of NaN counts
// (counting indices where high or low is NaN).
extern "C" __global__ void correl_hl_batch_f32(
    const double* __restrict__ ps_h,   // [len+1]
    const double* __restrict__ ps_h2,  // [len+1]
    const double* __restrict__ ps_l,   // [len+1]
    const double* __restrict__ ps_l2,  // [len+1]
    const double* __restrict__ ps_hl,  // [len+1]
    const int* __restrict__ ps_nan,    // [len+1]
    int len,
    int first_valid,
    const int* __restrict__ periods,   // [n_combos]
    int n_combos,
    float* __restrict__ out            // [n_combos * len]
){
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const double inv_pf = 1.0 / (double)period;

    while (t < len) {
        float out_val = nan_f;
        if (t >= warm) {
            const int end = t + 1;
            int start = end - period;
            if (start < 0) start = 0; // clamp for safety
            const int nan_count = ps_nan[end] - ps_nan[start];
            if (nan_count == 0) {
                const double sum_h  = ps_h[end]  - ps_h[start];
                const double sum_l  = ps_l[end]  - ps_l[start];
                const double sum_h2 = ps_h2[end] - ps_h2[start];
                const double sum_l2 = ps_l2[end] - ps_l2[start];
                const double sum_hl = ps_hl[end] - ps_hl[start];
                const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
                const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
                const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
                if (varh > 0.0 && varl > 0.0) {
                    const double denom = sqrt(varh) * sqrt(varl);
                    if (denom > 0.0 && !isnan(denom)) {
                        out_val = (float)(cov / denom);
                    } else {
                        out_val = 0.0f;
                    }
                } else {
                    out_val = 0.0f;
                }
            }
        }
        out[row_off + t] = out_val;
        t += stride;
    }
}

// ----------------- Many-series × one-param (time-major) -----------------

// Each block handles one series (column) in a time-major layout [t][series].
// Thread 0 in the block runs the sequential sliding window with O(1) updates.
extern "C" __global__ void correl_hl_many_series_one_param_f32(
    const float* __restrict__ high_tm, // [series_len * num_series], time-major
    const float* __restrict__ low_tm,  // [series_len * num_series], time-major
    const int* __restrict__ first_valids, // [num_series]
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm // [series_len * num_series], time-major
){
    const int series = blockIdx.x;
    if (series >= num_series || period <= 0) return;

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) return;

    const int stride = num_series;

    // Fill column with NaNs cooperatively
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Bootstrap running sums over [first_valid .. first_valid+period-1]
    const int init_start = first_valid;
    const int init_end = min(first_valid + period, series_len);
    double sum_h = 0.0, sum_l = 0.0, sum_h2 = 0.0, sum_l2 = 0.0, sum_hl = 0.0;
    int nan_in_win = 0;
    for (int i = init_start; i < init_end; ++i) {
        const float h = high_tm[i * stride + series];
        const float l = low_tm[i * stride + series];
        if (isnan(h) || isnan(l)) {
            nan_in_win += 1;
        } else {
            const double hd = (double)h;
            const double ld = (double)l;
            sum_h += hd;
            sum_l += ld;
            sum_h2 += hd * hd;
            sum_l2 += ld * ld;
            sum_hl += hd * ld;
        }
    }

    const double inv_pf = 1.0 / (double)period;
    const int warm = first_valid + period - 1;
    if (warm < series_len && nan_in_win == 0) {
        const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
        const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
        const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
        float out0 = 0.0f;
        if (varh > 0.0 && varl > 0.0) {
            const double denom = sqrt(varh) * sqrt(varl);
            out0 = (float)((denom > 0.0) ? (cov / denom) : 0.0);
        }
        out_tm[warm * stride + series] = out0;
    }

    // Slide window
    for (int t = warm + 1; t < series_len; ++t) {
        const int old_idx = t - period;
        const float old_h = high_tm[old_idx * stride + series];
        const float old_l = low_tm[old_idx * stride + series];
        const float new_h = high_tm[t * stride + series];
        const float new_l = low_tm[t * stride + series];

        if (isnan(old_h) || isnan(old_l) || isnan(new_h) || isnan(new_l)) {
            // Rebuild sums from scratch over the window [t-period+1 .. t]
            sum_h = sum_l = sum_h2 = sum_l2 = sum_hl = 0.0;
            nan_in_win = 0;
            const int start = t + 1 - period;
            for (int k = start; k <= t; ++k) {
                const float hh = high_tm[k * stride + series];
                const float ll = low_tm[k * stride + series];
                if (isnan(hh) || isnan(ll)) {
                    nan_in_win += 1;
                } else {
                    const double hd = (double)hh;
                    const double ld = (double)ll;
                    sum_h += hd;
                    sum_l += ld;
                    sum_h2 += hd * hd;
                    sum_l2 += ld * ld;
                    sum_hl += hd * ld;
                }
            }
        } else {
            // O(1) update
            const double oh = (double)old_h, ol = (double)old_l;
            const double nh = (double)new_h, nl = (double)new_l;
            sum_h += nh - oh;
            sum_l += nl - ol;
            sum_h2 += nh * nh - oh * oh;
            sum_l2 += nl * nl - ol * ol;
            sum_hl = fma(nh, nl, sum_hl - oh * ol);
        }

        if (nan_in_win != 0) {
            out_tm[t * stride + series] = __int_as_float(0x7fffffff);
        } else {
            const double cov  = sum_hl - (sum_h * sum_l) * inv_pf;
            const double varh = sum_h2 - (sum_h * sum_h) * inv_pf;
            const double varl = sum_l2 - (sum_l * sum_l) * inv_pf;
            float outv = 0.0f;
            if (varh > 0.0 && varl > 0.0) {
                const double denom = sqrt(varh) * sqrt(varl);
                outv = (float)((denom > 0.0) ? (cov / denom) : 0.0);
            }
            out_tm[t * stride + series] = outv;
        }
    }
}

