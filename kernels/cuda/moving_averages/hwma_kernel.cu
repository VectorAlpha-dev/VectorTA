// CUDA kernels for the Holt-Winters Moving Average (HWMA).
//
// The HWMA recurrence depends on prior level (f), velocity (v), and
// acceleration (a) states. Each kernel keeps these states in registers and
// streams sequentially through the series. A batch variant handles the
// single-series × many-parameter sweep, while the second kernel covers the
// many-series × one-parameter case using a time-major layout.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

extern "C" __global__
void hwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ nas,
                    const float* __restrict__ nbs,
                    const float* __restrict__ ncs,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    int first = first_valid;
    if (first < 0) {
        first = 0;
    }
    if (first > series_len) {
        first = series_len;
    }

    const float na = nas[combo];
    const float nb = nbs[combo];
    const float nc = ncs[combo];
    const float one_m_na = 1.0f - na;
    const float one_m_nb = 1.0f - nb;
    const float one_m_nc = 1.0f - nc;
    const float half = 0.5f;

    const int base = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    for (int t = 0; t < first; ++t) {
        out[base + t] = nan_f;
    }
    if (first >= series_len) {
        return;
    }

    float f = prices[first];
    float v = 0.0f;
    float a = 0.0f;

    for (int t = first; t < series_len; ++t) {
        const float price = prices[t];
        const float fv_sum = f + v + half * a;
        const float f_new = na * price + one_m_na * fv_sum;
        const float v_new = nb * (f_new - f) + one_m_nb * (v + a);
        const float a_new = nc * (v_new - v) + one_m_nc * a;
        out[base + t] = f_new + v_new + half * a_new;
        f = f_new;
        v = v_new;
        a = a_new;
    }
}

extern "C" __global__
void hwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     float na,
                                     float nb,
                                     float nc,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const float half = 0.5f;
    const float nan_f = __int_as_float(0x7fffffff);

    int first = first_valids[series_idx];
    if (first < 0) {
        first = 0;
    }
    if (first > series_len) {
        first = series_len;
    }

    for (int t = 0; t < first; ++t) {
        out_tm[t * stride + series_idx] = nan_f;
    }
    if (first >= series_len) {
        return;
    }

    const float one_m_na = 1.0f - na;
    const float one_m_nb = 1.0f - nb;
    const float one_m_nc = 1.0f - nc;

    const int first_idx = first * stride + series_idx;
    float f = prices_tm[first_idx];
    float v = 0.0f;
    float a = 0.0f;

    for (int t = first; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const float price = prices_tm[idx];
        const float fv_sum = f + v + half * a;
        const float f_new = na * price + one_m_na * fv_sum;
        const float v_new = nb * (f_new - f) + one_m_nb * (v + a);
        const float a_new = nc * (v_new - v) + one_m_nc * a;
        out_tm[idx] = f_new + v_new + half * a_new;
        f = f_new;
        v = v_new;
        a = a_new;
    }
}
