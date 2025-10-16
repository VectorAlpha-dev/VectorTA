// CUDA kernels for Vortex Indicator (VI)
//
// Computes VI+ and VI- given prefix sums of True Range (TR), VM+ (vp), and VM- (vm).
// Warmup/NaN semantics match the scalar path: indices [0, warm) are NaN where
// warm = first_valid + period - 1. Arithmetic in FP32 for throughput.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef VI_NAN
#define VI_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// --- Batch: one series × many params ---
// Inputs are prefix sums over time for a single series; each thread handles one row (param).
extern "C" __global__ void vi_batch_f32(
    const float* __restrict__ pfx_tr,   // len = series_len (time-major 1D)
    const float* __restrict__ pfx_vp,   // len = series_len
    const float* __restrict__ pfx_vm,   // len = series_len
    const int*   __restrict__ periods,  // len = n_rows
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_plus,       // len = n_rows * series_len
    float* __restrict__ out_minus       // len = n_rows * series_len
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int period = periods[row];
    const int base   = row * series_len;
    float* __restrict__ plus  = out_plus  + base;
    float* __restrict__ minus = out_minus + base;

    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) {
            plus[i]  = VI_NAN;
            minus[i] = VI_NAN;
        }
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        for (int i = 0; i < series_len; ++i) {
            plus[i]  = VI_NAN;
            minus[i] = VI_NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    // Prefill warmup region with NaN
    for (int i = 0; i < warm; ++i) {
        plus[i]  = VI_NAN;
        minus[i] = VI_NAN;
    }

    // Compute steady-state windowed ratios via prefix sums
    for (int i = warm; i < series_len; ++i) {
        // Note: if i < period, use full prefix; otherwise subtract prefix at i - period.
        const float tr_sum = (i >= period) ? (pfx_tr[i] - pfx_tr[i - period]) : pfx_tr[i];
        const float vp_sum = (i >= period) ? (pfx_vp[i] - pfx_vp[i - period]) : pfx_vp[i];
        const float vm_sum = (i >= period) ? (pfx_vm[i] - pfx_vm[i - period]) : pfx_vm[i];
        // Scalar behavior: do not special-case divide-by-zero; preserve IEEE results
        plus[i]  = vp_sum / tr_sum;
        minus[i] = vm_sum / tr_sum;
    }
}

// --- Many series × one param (time-major) ---
// Inputs are prefix sums laid out time-major: idx = row * num_series + series
extern "C" __global__ void vi_many_series_one_param_f32(
    const float* __restrict__ pfx_tr_tm,
    const float* __restrict__ pfx_vp_tm,
    const float* __restrict__ pfx_vm_tm,
    const int*   __restrict__ first_valids, // len = num_series
    int num_series,
    int series_len, // rows (time)
    int period,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int first = first_valids[series];
    // Column pointers (time-major stepping)
    float*       __restrict__ col_plus  = plus_tm  + series;
    float*       __restrict__ col_minus = minus_tm + series;

    if (UNLIKELY(period <= 0 || period > series_len || first < 0 || first >= series_len)) {
        for (int row = 0; row < series_len; ++row) {
            *col_plus  = VI_NAN;
            *col_minus = VI_NAN;
            col_plus  += num_series;
            col_minus += num_series;
        }
        return;
    }

    const int tail = series_len - first;
    if (UNLIKELY(tail < period)) {
        for (int row = 0; row < series_len; ++row) {
            *col_plus  = VI_NAN;
            *col_minus = VI_NAN;
            col_plus  += num_series;
            col_minus += num_series;
        }
        return;
    }

    const int warm = first + period - 1;

    // Warmup prefix: write NaN
    for (int row = 0; row < warm; ++row) {
        *col_plus  = VI_NAN;
        *col_minus = VI_NAN;
        col_plus  += num_series;
        col_minus += num_series;
    }

    // Main loop: compute per-row using prefix sums along this series
    for (int row = warm; row < series_len; ++row) {
        const int idx      = row * num_series + series;
        const int idx_prev = (row - period) * num_series + series;
        const float tr_sum = (row >= period) ? (pfx_tr_tm[idx] - pfx_tr_tm[idx_prev]) : pfx_tr_tm[idx];
        const float vp_sum = (row >= period) ? (pfx_vp_tm[idx] - pfx_vp_tm[idx_prev]) : pfx_vp_tm[idx];
        const float vm_sum = (row >= period) ? (pfx_vm_tm[idx] - pfx_vm_tm[idx_prev]) : pfx_vm_tm[idx];
        *col_plus  = vp_sum / tr_sum;
        *col_minus = vm_sum / tr_sum;
        col_plus  += num_series;
        col_minus += num_series;
    }
}

