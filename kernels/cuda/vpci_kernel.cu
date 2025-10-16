// CUDA kernels for Volume Price Confirmation Index (VPCI)
//
// Math pattern: prefix-sum/rational.
// Inputs are prefix sums of close (C), volume (V), and C*V, plus the raw
// volume series for the rolling VPCIS numerator. Warmup/NaN semantics match
// the scalar path: indices [0, warm) are NaN where warm = first_valid + long - 1.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }

__device__ __forceinline__ double zf64(double x) {
    return isfinite(x) ? x : 0.0;
}

__device__ __forceinline__ double zf32d(float x) {
    return isfinite(x) ? static_cast<double>(x) : 0.0;
}

// --- Batch: one series × many params ---
// Each thread handles one parameter row (short, long) over the full series length.
extern "C" __global__ void vpci_batch_f32(
    const double* __restrict__ pfx_c,   // len = series_len
    const double* __restrict__ pfx_v,   // len = series_len
    const double* __restrict__ pfx_cv,  // len = series_len
    const float*  __restrict__ volume,  // len = series_len
    const int*    __restrict__ shorts,  // len = n_rows
    const int*    __restrict__ longs,   // len = n_rows
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_vpci,       // len = n_rows * series_len
    float* __restrict__ out_vpcis       // len = n_rows * series_len
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int short_p = shorts[row];
    const int long_p  = longs[row];
    const int base    = row * series_len;
    float* __restrict__ y_vpci  = out_vpci  + base;
    float* __restrict__ y_vpcis = out_vpcis + base;

    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < long_p)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int warm = first_valid + long_p - 1;
    // Warmup prefix: set NaNs
    for (int i = 0; i < warm; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }

    const double inv_long  = 1.0 / (double)long_p;
    const double inv_short = 1.0 / (double)short_p;
    double sum_vpci_vol_short = 0.0;

    for (int i = warm; i < series_len; ++i) {
        // Prefix window differences (i >= long_p and i >= short_p by construction)
        const int idx_long_prev  = i - long_p;
        const int idx_short_prev = i - short_p;

        const double sc_l  = pfx_c[i]  - pfx_c[idx_long_prev];
        const double sv_l  = pfx_v[i]  - pfx_v[idx_long_prev];
        const double scv_l = pfx_cv[i] - pfx_cv[idx_long_prev];
        const double sc_s  = pfx_c[i]  - pfx_c[idx_short_prev];
        const double sv_s  = pfx_v[i]  - pfx_v[idx_short_prev];
        const double scv_s = pfx_cv[i] - pfx_cv[idx_short_prev];

        const double sma_l   = sc_l * inv_long;
        const double sma_s   = sc_s * inv_short;
        const double sma_v_l = sv_l * inv_long;
        const double sma_v_s = sv_s * inv_short;

        const double vwma_l = (sv_l != 0.0) ? (scv_l / sv_l) : NAN;
        const double vwma_s = (sv_s != 0.0) ? (scv_s / sv_s) : NAN;

        const double vpc = vwma_l - sma_l;
        const double vpr = (sma_s != 0.0)   ? (vwma_s / sma_s) : NAN;
        const double vm  = (sma_v_l != 0.0) ? (sma_v_s / sma_v_l) : NAN;

        const double vpci = vpc * vpr * vm;
        y_vpci[i] = (float)vpci;

        // Update VPCIS rolling numerator: sum(VPCI * Volume) over last `short_p`
        sum_vpci_vol_short += (isfinite(vpci) ? vpci : 0.0) * zf32d(volume[i]);
        if (i >= warm + short_p) {
            const int rm_idx = i - short_p;
            const double vpci_rm = (double)y_vpci[rm_idx];
            sum_vpci_vol_short -= (isfinite(vpci_rm) ? vpci_rm : 0.0) * zf32d(volume[rm_idx]);
        }

        const double denom = sma_v_s; // = SMA(Volume, short)
        if (denom != 0.0 && isfinite(denom)) {
            y_vpcis[i] = (float)((sum_vpci_vol_short * inv_short) / denom);
        } else {
            y_vpcis[i] = nan_f32();
        }
    }
}

// --- Many series × one param (time-major) ---
// Threads in X dimension index series; each thread scans time rows for its series.
extern "C" __global__ void vpci_many_series_one_param_f32(
    const double* __restrict__ pfx_c_tm,   // len = rows * cols (time-major)
    const double* __restrict__ pfx_v_tm,
    const double* __restrict__ pfx_cv_tm,
    const float*  __restrict__ volume_tm,  // raw volume time-major
    const int*    __restrict__ first_valids, // len = cols
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_vpci_tm,      // len = rows * cols
    float* __restrict__ out_vpcis_tm      // len = rows * cols
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int first = first_valids[series];
    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > rows || first < 0 || first >= rows)) {
        // Fill column with NaNs
        for (int r = 0; r < rows; ++r) {
            const int idx = r * cols + series;
            out_vpci_tm[idx]  = nan_f32();
            out_vpcis_tm[idx] = nan_f32();
        }
        return;
    }

    const int warm = first + long_p - 1;
    // Warmup prefix
    for (int r = 0; r < warm; ++r) {
        const int idx = r * cols + series;
        out_vpci_tm[idx]  = nan_f32();
        out_vpcis_tm[idx] = nan_f32();
    }

    const double inv_long  = 1.0 / (double)long_p;
    const double inv_short = 1.0 / (double)short_p;
    double sum_vpci_vol_short = 0.0;

    for (int r = warm; r < rows; ++r) {
        const int idx          = r * cols + series;
        const int idx_long_pr  = (r - long_p) * cols + series;
        const int idx_short_pr = (r - short_p) * cols + series;

        const double sc_l  = pfx_c_tm[idx]  - pfx_c_tm[idx_long_pr];
        const double sv_l  = pfx_v_tm[idx]  - pfx_v_tm[idx_long_pr];
        const double scv_l = pfx_cv_tm[idx] - pfx_cv_tm[idx_long_pr];
        const double sc_s  = pfx_c_tm[idx]  - pfx_c_tm[idx_short_pr];
        const double sv_s  = pfx_v_tm[idx]  - pfx_v_tm[idx_short_pr];
        const double scv_s = pfx_cv_tm[idx] - pfx_cv_tm[idx_short_pr];

        const double sma_l   = sc_l * inv_long;
        const double sma_s   = sc_s * inv_short;
        const double sma_v_l = sv_l * inv_long;
        const double sma_v_s = sv_s * inv_short;

        const double vwma_l = (sv_l != 0.0) ? (scv_l / sv_l) : NAN;
        const double vwma_s = (sv_s != 0.0) ? (scv_s / sv_s) : NAN;

        const double vpc = vwma_l - sma_l;
        const double vpr = (sma_s != 0.0)   ? (vwma_s / sma_s) : NAN;
        const double vm  = (sma_v_l != 0.0) ? (sma_v_s / sma_v_l) : NAN;

        const double vpci = vpc * vpr * vm;
        out_vpci_tm[idx] = (float)vpci;

        sum_vpci_vol_short += (isfinite(vpci) ? vpci : 0.0) * zf32d(volume_tm[idx]);
        if (r >= warm + short_p) {
            const int rm = (r - short_p) * cols + series;
            const double vpci_rm = (double)out_vpci_tm[rm];
            sum_vpci_vol_short -= (isfinite(vpci_rm) ? vpci_rm : 0.0) * zf32d(volume_tm[rm]);
        }

        const double denom = sma_v_s;
        if (denom != 0.0 && isfinite(denom)) {
            out_vpcis_tm[idx] = (float)((sum_vpci_vol_short * inv_short) / denom);
        } else {
            out_vpcis_tm[idx] = nan_f32();
        }
    }
}

