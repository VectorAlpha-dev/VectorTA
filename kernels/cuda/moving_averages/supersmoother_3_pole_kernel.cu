// CUDA kernels for the 3-pole SuperSmoother filter.
//
// These variants mirror the VRAM-first design used across the moving average
// stack: a single price series evaluated against many period choices and a
// time-major kernel that processes many price series sharing one parameter.
// Each kernel operates in FP32 for IO while leveraging FP64 intermediates to
// keep the recursive filter numerically stable when compared to the f64 CPU
// reference implementation.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {

struct SupersmootherCoefs {
    double coef_source;
    double coef_prev1;
    double coef_prev2;
    double coef_prev3;
};

__device__ __forceinline__ SupersmootherCoefs make_coefs(int period) {
    const double inv_period = 1.0 / static_cast<double>(period);
    const double a = exp(-CUDART_PI * inv_period);
    const double b = 2.0 * a * cos(1.738 * CUDART_PI * inv_period);
    const double c = a * a;
    SupersmootherCoefs coefs;
    coefs.coef_source = 1.0 - c * c - b + b * c;
    coefs.coef_prev1 = b + c;
    coefs.coef_prev2 = -c - b * c;
    coefs.coef_prev3 = c * c;
    return coefs;
}

__device__ __forceinline__ void supersmoother_3_pole_row(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ out) {
    const float nan = CUDART_NAN_F;

    if (period <= 0 || series_len <= 0) {
        return;
    }

    const int bounded_first = first_valid < 0 ? 0 : first_valid;

    for (int t = 0; t < series_len; ++t) {
        if (t < bounded_first) {
            out[t] = nan;
        }
    }

    if (bounded_first >= series_len) {
        return;
    }

    const SupersmootherCoefs coefs = make_coefs(period);

    int idx0 = bounded_first;
    double y0 = static_cast<double>(prices[idx0]);
    out[idx0] = static_cast<float>(y0);

    int idx1 = bounded_first + 1;
    if (idx1 >= series_len) {
        return;
    }
    double y1 = static_cast<double>(prices[idx1]);
    out[idx1] = static_cast<float>(y1);

    int idx2 = bounded_first + 2;
    if (idx2 >= series_len) {
        return;
    }
    double y2 = static_cast<double>(prices[idx2]);
    out[idx2] = static_cast<float>(y2);

    for (int t = bounded_first + 3; t < series_len; ++t) {
        const double input = static_cast<double>(prices[t]);
        const double y_next = coefs.coef_source * input +
                              coefs.coef_prev1 * y2 +
                              coefs.coef_prev2 * y1 +
                              coefs.coef_prev3 * y0;
        out[t] = static_cast<float>(y_next);
        y0 = y1;
        y1 = y2;
        y2 = y_next;
    }
}

__device__ __forceinline__ void supersmoother_3_pole_row_strided(
    const float* __restrict__ prices,
    int series_len,
    int stride,
    int first_valid,
    int period,
    float* __restrict__ out) {
    const float nan = CUDART_NAN_F;

    if (period <= 0 || series_len <= 0) {
        return;
    }

    const int bounded_first = first_valid < 0 ? 0 : first_valid;

    for (int t = 0; t < series_len; ++t) {
        const int idx = t * stride;
        if (t < bounded_first) {
            out[idx] = nan;
        }
    }

    if (bounded_first >= series_len) {
        return;
    }

    const SupersmootherCoefs coefs = make_coefs(period);

    int t0 = bounded_first;
    int idx0 = t0 * stride;
    double y0 = static_cast<double>(prices[idx0]);
    out[idx0] = static_cast<float>(y0);

    int t1 = t0 + 1;
    if (t1 >= series_len) {
        return;
    }
    int idx1 = t1 * stride;
    double y1 = static_cast<double>(prices[idx1]);
    out[idx1] = static_cast<float>(y1);

    int t2 = t0 + 2;
    if (t2 >= series_len) {
        return;
    }
    int idx2 = t2 * stride;
    double y2 = static_cast<double>(prices[idx2]);
    out[idx2] = static_cast<float>(y2);

    for (int t = bounded_first + 3; t < series_len; ++t) {
        const int idx = t * stride;
        const double input = static_cast<double>(prices[idx]);
        const double y_next = coefs.coef_source * input +
                              coefs.coef_prev1 * y2 +
                              coefs.coef_prev2 * y1 +
                              coefs.coef_prev3 * y0;
        out[idx] = static_cast<float>(y_next);
        y0 = y1;
        y1 = y2;
        y2 = y_next;
    }
}

}  // namespace

extern "C" __global__ void supersmoother_3_pole_batch_f32(
    const float* __restrict__ prices,
    const int* __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || threadIdx.x != 0) {
        return;
    }

    const int period = periods[combo];
    float* out_row = out + combo * series_len;
    supersmoother_3_pole_row(prices, series_len, first_valid, period, out_row);
}

extern "C" __global__ void supersmoother_3_pole_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series || threadIdx.x != 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];
    const float* series_prices = prices_tm + series_idx;
    float* series_out = out_tm + series_idx;

    supersmoother_3_pole_row_strided(
        series_prices, series_len, stride, first_valid, period, series_out);
}
