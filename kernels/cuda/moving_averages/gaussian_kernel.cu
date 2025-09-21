// CUDA kernels for the Gaussian moving average.
//
// Both kernels execute the cascaded single-pole recurrence entirely on the GPU
// using FP32 inputs/outputs with FP64 intermediates for numerical parity with
// the scalar reference implementation. Each block processes one parameter
// combination (or one series, for the many-series variant) sequentially; only
// thread 0 performs arithmetic to keep the control flow identical to the CPU
// path.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ void gaussian_run_poles1(const float* __restrict__ prices,
                                                           float* __restrict__ out,
                                                           int start,
                                                           int series_len,
                                                           int stride,
                                                           int warm,
                                                           float nan_f,
                                                           double c0,
                                                           double c1) {
    double prev = 0.0;
    int idx = 0;
    for (int t = 0; t < series_len; ++t, idx += stride) {
        const double x = static_cast<double>(prices[idx]);
        prev = c1 * prev + c0 * x;
        if (t < start || t < warm || isnan(x)) {
            out[idx] = nan_f;
        } else {
            out[idx] = static_cast<float>(prev);
        }
    }
}


static __device__ __forceinline__ void gaussian_run_poles2(const float* __restrict__ prices,
                                                           float* __restrict__ out,
                                                           int start,
                                                           int series_len,
                                                           int stride,
                                                           int warm,
                                                           float nan_f,
                                                           double c0,
                                                           double c1,
                                                           double c2) {
    double prev1 = 0.0; // y[n-1]
    double prev0 = 0.0; // y[n-2]
    int idx = 0;
    for (int t = 0; t < series_len; ++t, idx += stride) {
        const double x = static_cast<double>(prices[idx]);
        const double y = c2 * prev0 + c1 * prev1 + c0 * x;
        prev0 = prev1;
        prev1 = y;
        if (t < start || t < warm || isnan(x)) {
            out[idx] = nan_f;
        } else {
            out[idx] = static_cast<float>(y);
        }
    }
}


static __device__ __forceinline__ void gaussian_run_poles3(const float* __restrict__ prices,
                                                           float* __restrict__ out,
                                                           int start,
                                                           int series_len,
                                                           int stride,
                                                           int warm,
                                                           float nan_f,
                                                           double c0,
                                                           double c1,
                                                           double c2,
                                                           double c3) {
    double p2 = 0.0; // y[n-1]
    double p1 = 0.0; // y[n-2]
    double p0 = 0.0; // y[n-3]
    int idx = 0;
    for (int t = 0; t < series_len; ++t, idx += stride) {
        const double x = static_cast<double>(prices[idx]);
        const double y = c3 * p0 + c2 * p1 + c1 * p2 + c0 * x;
        p0 = p1;
        p1 = p2;
        p2 = y;
        if (t < start || t < warm || isnan(x)) {
            out[idx] = nan_f;
        } else {
            out[idx] = static_cast<float>(y);
        }
    }
}


static __device__ __forceinline__ void gaussian_run_poles4(const float* __restrict__ prices,
                                                           float* __restrict__ out,
                                                           int start,
                                                           int series_len,
                                                           int stride,
                                                           int warm,
                                                           float nan_f,
                                                           double c0,
                                                           double c1,
                                                           double c2,
                                                           double c3,
                                                           double c4) {
    double p3 = 0.0; // y[n-1]
    double p2 = 0.0; // y[n-2]
    double p1 = 0.0; // y[n-3]
    double p0 = 0.0; // y[n-4]
    int idx = 0;
    for (int t = 0; t < series_len; ++t, idx += stride) {
        const double x = static_cast<double>(prices[idx]);
        const double y = (((c4 * p0) + (c3 * p1)) + (c2 * p2)) + (c1 * p3) + (c0 * x);
        p0 = p1;
        p1 = p2;
        p2 = p3;
        p3 = y;
        if (t < start || t < warm || isnan(x)) {
            out[idx] = nan_f;
        } else {
            out[idx] = static_cast<float>(y);
        }
    }
}


extern "C" __global__ void gaussian_batch_f32(const float* __restrict__ prices,
                                               const int* __restrict__ periods,
                                               const int* __restrict__ poles,
                                               const float* __restrict__ coeffs,
                                               int coeff_stride,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    const int period = periods[combo];
    const int pole = poles[combo];
    if (period < 2 || pole < 1 || pole > 4 || series_len <= 0) {
        return;
    }

    float* out_row = out + combo * series_len;
    const float nan_f = nanf("");
    for (int i = 0; i < series_len; ++i) {
        out_row[i] = nan_f;
    }

    int start = first_valid;
    if (start < 0) {
        start = 0;
    }
    if (start >= series_len) {
        return;
    }

    const float* coeff = coeffs + combo * coeff_stride;
    const double c0 = static_cast<double>(coeff[0]);
    const double c1 = static_cast<double>(coeff[1]);
    const double c2 = static_cast<double>(coeff[2]);
    const double c3 = static_cast<double>(coeff[3]);
    const double c4 = static_cast<double>(coeff[4]);

    int warm = first_valid + period;
    if (warm < 0) {
        warm = 0;
    }
    if (warm > series_len) {
        warm = series_len;
    }

    switch (pole) {
        case 1:
            gaussian_run_poles1(prices, out_row, start, series_len, 1, warm, nan_f, c0, c1);
            break;
        case 2:
            gaussian_run_poles2(prices, out_row, start, series_len, 1, warm, nan_f, c0, c1, c2);
            break;
        case 3:
            gaussian_run_poles3(prices, out_row, start, series_len, 1, warm, nan_f, c0, c1, c2, c3);
            break;
        case 4:
        default:
            gaussian_run_poles4(prices, out_row, start, series_len, 1, warm, nan_f, c0, c1, c2, c3, c4);
            break;
    }
}

extern "C" __global__ void gaussian_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ coeffs,
    int period,
    int poles,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    if (period < 2 || poles < 1 || poles > 4 || series_len <= 0) {
        return;
    }

    const float nan_f = nanf("");
    for (int t = 0; t < series_len; ++t) {
        const int idx = t * num_series + series_idx;
        out_tm[idx] = nan_f;
    }

    int start = first_valids[series_idx];
    if (start < 0) {
        start = 0;
    }
    if (start >= series_len) {
        return;
    }

    const double c0 = static_cast<double>(coeffs[0]);
    const double c1 = static_cast<double>(coeffs[1]);
    const double c2 = static_cast<double>(coeffs[2]);
    const double c3 = static_cast<double>(coeffs[3]);
    const double c4 = static_cast<double>(coeffs[4]);

    int warm = start + period;
    if (warm > series_len) {
        warm = series_len;
    }

    const float* price_series = prices_tm + series_idx;
    float* out_series = out_tm + series_idx;
    const int stride = num_series;

    switch (poles) {
        case 1:
            gaussian_run_poles1(price_series, out_series, start, series_len, stride, warm, nan_f, c0, c1);
            break;
        case 2:
            gaussian_run_poles2(price_series, out_series, start, series_len, stride, warm, nan_f, c0, c1, c2);
            break;
        case 3:
            gaussian_run_poles3(price_series, out_series, start, series_len, stride, warm, nan_f, c0, c1, c2, c3);
            break;
        case 4:
        default:
            gaussian_run_poles4(price_series, out_series, start, series_len, stride, warm, nan_f, c0, c1, c2, c3, c4);
            break;
    }
}
