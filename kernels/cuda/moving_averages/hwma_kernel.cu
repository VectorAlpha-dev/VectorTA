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

    // Compute in FP64 to reduce accumulation error; store as FP32.
    double f = static_cast<double>(prices[first]);
    double v = 0.0;
    double a = 0.0;
    const double dna = static_cast<double>(na);
    const double dnb = static_cast<double>(nb);
    const double dnc = static_cast<double>(nc);
    const double dh = 0.5;

    for (int t = first; t < series_len; ++t) {
        const double price = static_cast<double>(prices[t]);
        const double s_prev = dh * a + (f + v);
        const double f_new = dna * price + (1.0 - dna) * s_prev;
        const double v_new = dnb * (f_new - f) + (1.0 - dnb) * (v + a);
        const double a_new = dnc * (v_new - v) + (1.0 - dnc) * a;
        const double s_new = dh * a_new + (f_new + v_new);
        out[base + t] = static_cast<float>(s_new);
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
    double f = static_cast<double>(prices_tm[first_idx]);
    double v = 0.0;
    double a = 0.0;
    const double dna = static_cast<double>(na);
    const double dnb = static_cast<double>(nb);
    const double dnc = static_cast<double>(nc);
    const double dh = 0.5;

    for (int t = first; t < series_len; ++t) {
        const int idx = t * stride + series_idx;
        const double price = static_cast<double>(prices_tm[idx]);
        const double s_prev = dh * a + (f + v);
        const double f_new = dna * price + (1.0 - dna) * s_prev;
        const double v_new = dnb * (f_new - f) + (1.0 - dnb) * (v + a);
        const double a_new = dnc * (v_new - v) + (1.0 - dnc) * a;
        const double s_new = dh * a_new + (f_new + v_new);
        out_tm[idx] = static_cast<float>(s_new);
        f = f_new;
        v = v_new;
        a = a_new;
    }
}
