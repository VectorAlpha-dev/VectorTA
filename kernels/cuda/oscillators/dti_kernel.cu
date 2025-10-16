// CUDA kernels for Dynamic Trend Index (DTI) by William Blau
//
// Semantics mirror the Rust scalar implementation:
// - Warmup index: start = first_valid + 1
// - Outputs [0 .. start-1] are NaN
// - Triple EMA chains over x and |x| (numerator/denominator)
// - Division-by-zero or NaN denominator yields 0.0
// - For the batch kernel, x and |x| are precomputed once on host and shared across rows
// - Many-series kernel computes x on the fly from (high, low) in time-major layout

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef DTI_QNAN
#define DTI_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// -------------------- Batch (one series × many params) --------------------
// x, ax: precomputed on host starting at index `start` (<= start-1 values are unused)
extern "C" __global__ void dti_batch_f32(
    const float* __restrict__ x,          // length = series_len
    const float* __restrict__ ax,         // length = series_len
    const int*   __restrict__ r_arr,      // length = n_combos
    const int*   __restrict__ s_arr,      // length = n_combos
    const int*   __restrict__ u_arr,      // length = n_combos
    int series_len,
    int n_combos,
    int start,                             // start = first_valid + 1
    float* __restrict__ out               // length = n_combos * series_len (row-major)
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    const int r = r_arr[row];
    const int s = s_arr[row];
    const int u = u_arr[row];
    float* out_row = out + (size_t)row * series_len;

    if (UNLIKELY(r <= 0 || s <= 0 || u <= 0 || start < 1 || start > series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = DTI_QNAN;
        return;
    }

    // Prefill NaN prefix identical to scalar (..=first_valid)
    for (int i = 0; i < start; ++i) out_row[i] = DTI_QNAN;

    // EMA coefficients (double precision for accumulations)
    const double ar = 2.0 / (static_cast<double>(r) + 1.0);
    const double as_ = 2.0 / (static_cast<double>(s) + 1.0);
    const double au = 2.0 / (static_cast<double>(u) + 1.0);
    const double br = 1.0 - ar;
    const double bs = 1.0 - as_;
    const double bu = 1.0 - au;

    double e0_r = 0.0, e0_s = 0.0, e0_u = 0.0;
    double e1_r = 0.0, e1_s = 0.0, e1_u = 0.0;

    // Process from start..series_len-1
    for (int i = start; i < series_len; ++i) {
        const double xi  = static_cast<double>(x[i]);
        const double axi = static_cast<double>(ax[i]);

        // Residual-form EMA updates (e += a*(x - e))
        e0_r += ar * (xi - e0_r);
        e0_s += as_ * (e0_r - e0_s);
        e0_u += au * (e0_s - e0_u);

        e1_r += ar * (axi - e1_r);
        e1_s += as_ * (e1_r - e1_s);
        e1_u += au * (e1_s - e1_u);

        const double den = e1_u;
        out_row[i] = (den == den && den != 0.0) ? static_cast<float>(100.0 * (e0_u / den)) : 0.0f;
    }
}

// -------------------- Many series × one param (time-major) --------------------
// Inputs are time-major: a[t * num_series + s]
extern "C" __global__ void dti_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int r,
    int s,
    int u,
    float* __restrict__ out_tm // time-major
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(fv < 0 || fv >= series_len || r <= 0 || s <= 0 || u <= 0)) {
        // fill NaNs for the series
        for (int t = 0; t < series_len; ++t) {
            *(out_tm + (size_t)t * num_series + series) = DTI_QNAN;
        }
        return;
    }
    const int start = fv + 1; // first computable index
    if (UNLIKELY(start >= series_len)) {
        for (int t = 0; t < series_len; ++t) {
            *(out_tm + (size_t)t * num_series + series) = DTI_QNAN;
        }
        return;
    }

    // NaN prefix (..=fv)
    for (int t = 0; t < start; ++t) {
        *(out_tm + (size_t)t * num_series + series) = DTI_QNAN;
    }

    // EMA coefficients
    const double ar = 2.0 / (static_cast<double>(r) + 1.0);
    const double as_ = 2.0 / (static_cast<double>(s) + 1.0);
    const double au = 2.0 / (static_cast<double>(u) + 1.0);
    const double br = 1.0 - ar;
    const double bs = 1.0 - as_;
    const double bu = 1.0 - au;

    // EMA states
    double e0_r = 0.0, e0_s = 0.0, e0_u = 0.0;
    double e1_r = 0.0, e1_s = 0.0, e1_u = 0.0;

    // Previous high/low at fv
    double prev_h = static_cast<double>(*(high_tm + (size_t)fv * num_series + series));
    double prev_l = static_cast<double>(*(low_tm  + (size_t)fv * num_series + series));

    for (int t = start; t < series_len; ++t) {
        const double h = static_cast<double>(*(high_tm + (size_t)t * num_series + series));
        const double l = static_cast<double>(*(low_tm  + (size_t)t * num_series + series));
        const double dh = h - prev_h;
        const double dl = l - prev_l;
        prev_h = h;
        prev_l = l;

        // x = max(dh,0) - max(-dl,0)
        const double x_hmu = (dh > 0.0) ? dh : 0.0;
        const double x_lmd = (dl < 0.0) ? -dl : 0.0;
        const double xi = x_hmu - x_lmd;
        const double axi = fabs(xi);

        e0_r = ar * xi + br * e0_r;
        e0_s = as_ * e0_r + bs * e0_s;
        e0_u = au * e0_s + bu * e0_u;

        e1_r = ar * axi + br * e1_r;
        e1_s = as_ * e1_r + bs * e1_s;
        e1_u = au * e1_s + bu * e1_u;

        const double den = e1_u;
        *(out_tm + (size_t)t * num_series + series) = (den == den && den != 0.0)
            ? static_cast<float>(100.0 * (e0_u / den))
            : 0.0f;
    }
}

