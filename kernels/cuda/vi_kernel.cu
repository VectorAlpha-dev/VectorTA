





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



extern "C" __global__ void vi_batch_f32(
    const float* __restrict__ pfx_tr,
    const float* __restrict__ pfx_vp,
    const float* __restrict__ pfx_vm,
    const int*   __restrict__ periods,
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_plus,
    float* __restrict__ out_minus
) {

    if (gridDim.y > 1) {
        const int t   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        const int row = (int)blockIdx.y;
        if (t >= series_len || row >= n_rows) {
            return;
        }
        const size_t out_idx = (size_t)row * (size_t)series_len + (size_t)t;

        const int period = periods[row];
        if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int tail = series_len - first_valid;
        if (UNLIKELY(tail < period)) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int warm = first_valid + period - 1;
        if (t < warm) {
            out_plus[out_idx] = VI_NAN;
            out_minus[out_idx] = VI_NAN;
            return;
        }

        const int prev = t - period;
        const float tr_prev = (prev >= 0) ? pfx_tr[prev] : 0.0f;
        const float vp_prev = (prev >= 0) ? pfx_vp[prev] : 0.0f;
        const float vm_prev = (prev >= 0) ? pfx_vm[prev] : 0.0f;

        const float tr_sum = pfx_tr[t] - tr_prev;
        const float inv    = 1.0f / tr_sum;
        out_plus[out_idx]  = (pfx_vp[t] - vp_prev) * inv;
        out_minus[out_idx] = (pfx_vm[t] - vm_prev) * inv;
        return;
    }


    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = (size_t)n_rows * (size_t)series_len;
    if (tid >= total) {
        return;
    }
    const int row = (int)(tid / (size_t)series_len);
    const int t   = (int)(tid - (size_t)row * (size_t)series_len);

    const int period = periods[row];
    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    if (t < warm) {
        out_plus[tid] = VI_NAN;
        out_minus[tid] = VI_NAN;
        return;
    }

    const int prev = t - period;
    const float tr_prev = (prev >= 0) ? pfx_tr[prev] : 0.0f;
    const float vp_prev = (prev >= 0) ? pfx_vp[prev] : 0.0f;
    const float vm_prev = (prev >= 0) ? pfx_vm[prev] : 0.0f;

    const float tr_sum = pfx_tr[t] - tr_prev;
    const float inv    = 1.0f / tr_sum;
    out_plus[tid]  = (pfx_vp[t] - vp_prev) * inv;
    out_minus[tid] = (pfx_vm[t] - vm_prev) * inv;
}



extern "C" __global__ void vi_many_series_one_param_f32(
    const float* __restrict__ pfx_tr_tm,
    const float* __restrict__ pfx_vp_tm,
    const float* __restrict__ pfx_vm_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm
) {
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = (size_t)num_series * (size_t)series_len;
    if (tid >= total) {
        return;
    }

    const int series = (int)(tid % (size_t)num_series);
    const int row    = (int)(tid / (size_t)num_series);
    const size_t idx = (size_t)row * (size_t)num_series + (size_t)series;

    const int first = first_valids[series];
    if (UNLIKELY(period <= 0 || period > series_len || first < 0 || first >= series_len)) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int tail = series_len - first;
    if (UNLIKELY(tail < period)) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int warm = first + period - 1;
    if (row < warm) {
        plus_tm[idx] = VI_NAN;
        minus_tm[idx] = VI_NAN;
        return;
    }

    const int prev_row = row - period;
    if (prev_row >= 0) {
        const size_t idx_prev = (size_t)prev_row * (size_t)num_series + (size_t)series;
        const float tr_sum = pfx_tr_tm[idx] - pfx_tr_tm[idx_prev];
        const float inv    = 1.0f / tr_sum;
        plus_tm[idx]  = (pfx_vp_tm[idx] - pfx_vp_tm[idx_prev]) * inv;
        minus_tm[idx] = (pfx_vm_tm[idx] - pfx_vm_tm[idx_prev]) * inv;
    } else {
        const float tr_sum = pfx_tr_tm[idx];
        const float inv    = 1.0f / tr_sum;
        plus_tm[idx]  = pfx_vp_tm[idx] * inv;
        minus_tm[idx] = pfx_vm_tm[idx] * inv;
    }
}

