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
// Inputs are prefix sums over time for a single series; grid-stride over rows (params).
extern "C" __global__ void vi_batch_f32(
    const float* __restrict__ pfx_tr,   // len = series_len (time-major 1D)
    const float* __restrict__ pfx_vp,   // len = series_len
    const float* __restrict__ pfx_vm,   // len = series_len
    const int*   __restrict__ periods,  // len = n_rows
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_plus,       // len = n_rows * series_len (row-major)
    float* __restrict__ out_minus       // len = n_rows * series_len (row-major)
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_rows;
         row += blockDim.x * gridDim.x)
    {
        const int period = periods[row];
        const int base   = row * series_len;
        float* __restrict__ plus  = out_plus  + base;
        float* __restrict__ minus = out_minus + base;

        // Validate once per row
        if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
            for (int i = 0; i < series_len; ++i) { plus[i] = VI_NAN; minus[i] = VI_NAN; }
            continue;
        }

        const int tail = series_len - first_valid;
        if (UNLIKELY(tail < period)) {
            for (int i = 0; i < series_len; ++i) { plus[i] = VI_NAN; minus[i] = VI_NAN; }
            continue;
        }

        const int warm = first_valid + period - 1;

        // Warmup (prefix) region is NaN
        for (int i = 0; i < warm; ++i) {
            plus[i]  = VI_NAN;
            minus[i] = VI_NAN;
        }

        // --- Steady state ---
        if (first_valid == 0) {
            // Edge element i == period-1 uses full prefix (no subtraction)
            const int i0 = period - 1;
            {
                const float tr_sum = pfx_tr[i0];
                const float inv    = 1.0f / tr_sum;       // one IEEE divide
                plus[i0]  = pfx_vp[i0] * inv;            // two fast multiplies
                minus[i0] = pfx_vm[i0] * inv;
            }

            // From here on: sliding window using pointer-increment addressing
            const float* __restrict__ tr_cur = pfx_tr + (i0 + 1);
            const float* __restrict__ vp_cur = pfx_vp + (i0 + 1);
            const float* __restrict__ vm_cur = pfx_vm + (i0 + 1);

            const float* __restrict__ tr_prev = tr_cur - period;
            const float* __restrict__ vp_prev = vp_cur - period;
            const float* __restrict__ vm_prev = vm_cur - period;

            for (int i = i0 + 1; i < series_len; ++i) {
                const float tr_sum = (*tr_cur++ - *tr_prev++);
                const float inv    = 1.0f / tr_sum;  // one divide
                plus[i]  = (*vp_cur++ - *vp_prev++) * inv;
                minus[i] = (*vm_cur++ - *vm_prev++) * inv;
            }
        } else {
            // warm >= period, so we always use the difference path
            const int i0 = warm;

            const float* __restrict__ tr_cur = pfx_tr + i0;
            const float* __restrict__ vp_cur = pfx_vp + i0;
            const float* __restrict__ vm_cur = pfx_vm + i0;

            const float* __restrict__ tr_prev = tr_cur - period;
            const float* __restrict__ vp_prev = vp_cur - period;
            const float* __restrict__ vm_prev = vm_cur - period;

            for (int i = i0; i < series_len; ++i) {
                const float tr_sum = (*tr_cur++ - *tr_prev++);
                const float inv    = 1.0f / tr_sum;  // one divide
                plus[i]  = (*vp_cur++ - *vp_prev++) * inv;
                minus[i] = (*vm_cur++ - *vm_prev++) * inv;
            }
        }
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
    // grid-stride over series for scalability/occupancy
    for (int series = blockIdx.x * blockDim.x + threadIdx.x;
         series < num_series;
         series += blockDim.x * gridDim.x)
    {
        const int first = first_valids[series];

        float* __restrict__ col_plus  = plus_tm  + series;
        float* __restrict__ col_minus = minus_tm + series;

        if (UNLIKELY(period <= 0 || period > series_len || first < 0 || first >= series_len)) {
            for (int row = 0; row < series_len; ++row) {
                *col_plus  = VI_NAN; *col_minus = VI_NAN;
                col_plus  += num_series; col_minus += num_series;
            }
            continue;
        }

        const int tail = series_len - first;
        if (UNLIKELY(tail < period)) {
            for (int row = 0; row < series_len; ++row) {
                *col_plus  = VI_NAN; *col_minus = VI_NAN;
                col_plus  += num_series; col_minus += num_series;
            }
            continue;
        }

        const int warm = first + period - 1;

        // Warmup prefix: write NaN
        for (int row = 0; row < warm; ++row) {
            *col_plus  = VI_NAN; *col_minus = VI_NAN;
            col_plus  += num_series; col_minus += num_series;
        }

        if (first == 0) {
            // Edge row: row == period-1 uses full prefix
            int row = period - 1;
            int idx = row * num_series + series;

            {
                const float tr_sum = pfx_tr_tm[idx];
                const float inv    = 1.0f / tr_sum;              // one divide
                *col_plus  = pfx_vp_tm[idx] * inv;
                *col_minus = pfx_vm_tm[idx] * inv;
                col_plus  += num_series; col_minus += num_series;
            }

            // Sliding window from next row onward
            int idx_cur  = idx + num_series;                        // (row+1, series)
            int idx_prev = (row + 1 - period) * num_series + series; // (0, series)
            for (row = row + 1; row < series_len; ++row) {
                const float tr_sum = pfx_tr_tm[idx_cur] - pfx_tr_tm[idx_prev];
                const float inv    = 1.0f / tr_sum;              // one divide
                *col_plus  = (pfx_vp_tm[idx_cur] - pfx_vp_tm[idx_prev]) * inv;
                *col_minus = (pfx_vm_tm[idx_cur] - pfx_vm_tm[idx_prev]) * inv;
                col_plus  += num_series; col_minus += num_series;
                idx_cur  += num_series; idx_prev += num_series;
            }
        } else {
            // warm >= period: always difference path
            int row      = warm;
            int idx_cur  = row * num_series + series;
            int idx_prev = (row - period) * num_series + series;

            for (; row < series_len; ++row) {
                const float tr_sum = pfx_tr_tm[idx_cur] - pfx_tr_tm[idx_prev];
                const float inv    = 1.0f / tr_sum;              // one divide
                *col_plus  = (pfx_vp_tm[idx_cur] - pfx_vp_tm[idx_prev]) * inv;
                *col_minus = (pfx_vm_tm[idx_cur] - pfx_vm_tm[idx_prev]) * inv;
                col_plus  += num_series; col_minus += num_series;
                idx_cur  += num_series; idx_prev += num_series;
            }
        }
    }
}

