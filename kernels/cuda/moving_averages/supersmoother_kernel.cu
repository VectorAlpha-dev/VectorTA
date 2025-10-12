// CUDA kernels for the SuperSmoother filter.
// Optimized: register rolling state, FMA, prefix-only NaN fill, pointer-stride time-major.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SUPERSMOOTHER_NAN
#define SUPERSMOOTHER_NAN (__int_as_float(0x7fffffff))
#endif

// Optional: define SS_FAST_MATH to use faster but slightly less accurate intrinsics.
static __device__ __forceinline__ void supersmoother_coeffs(float period, float* a, float* b, float* c) {
    const float PI = 3.14159265358979323846f;
    const float SQRT2 = 1.41421356237f;
    const float factor = (SQRT2 * PI) / period;

#ifdef SS_FAST_MATH
    // Approximate path (enable with -DSS_FAST_MATH); faster but less accurate
    const float a_val = __expf(-factor);
    const float b_val = 2.0f * a_val * __cosf(factor);
#else
    const float a_val = expf(-factor);
    const float b_val = 2.0f * a_val * cosf(factor);
#endif

    const float a_sq = a_val * a_val;
    const float c_val = 0.5f * (1.0f + a_sq - b_val);
    *a = a_val;
    *b = b_val;
    *c = c_val;
}

extern "C" __global__ void supersmoother_batch_f32(const float* __restrict__ prices,
                                                   const int*   __restrict__ periods,
                                                   int series_len,
                                                   int n_combos,
                                                   int first_valid,
                                                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* __restrict__ row_out = out + combo * series_len;

    // Validate early; if invalid, write full-row NaNs (preserve original behavior) and return.
    if (period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len) {
        for (int i = 0; i < series_len; ++i) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        for (int i = 0; i < series_len; ++i) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        for (int i = 0; i < series_len; ++i) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    // Prefix-only NaN fill: [0, warm)
    for (int i = 0; i < warm; ++i) row_out[i] = SUPERSMOOTHER_NAN;

    // Coefficients
    float a, b, c;
    supersmoother_coeffs((float)period, &a, &b, &c);
    const float a_sq = a * a;

    // Seeds
    float y_im2 = prices[warm];
    row_out[warm] = y_im2;

    if (warm + 1 >= series_len) return;

    float y_im1 = prices[warm + 1];
    row_out[warm + 1] = y_im1;

    // Main recurrence using register rolling state and FMA
    // y[i] = c*(x[i] + x[i-1]) + b*y[i-1] - (a^2)*y[i-2]
#pragma unroll 1
    for (int idx = warm + 2; idx < series_len; ++idx) {
        const float x_i    = prices[idx];
        const float x_im1  = prices[idx - 1];
        // Two FMAs: t = fmaf(b, y_im1, -a_sq*y_im2); y = fmaf(c, (x_i + x_im1), t);
        const float t  = fmaf(b, y_im1, -a_sq * y_im2);
        const float yi = fmaf(c, (x_i + x_im1), t);
        row_out[idx] = yi;
        y_im2 = y_im1;
        y_im1 = yi;
    }
}

extern "C" __global__ void supersmoother_many_series_one_param_f32(
    const float* __restrict__ prices_tm,   // time-major: [row][series]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    // Validate early; if invalid, write full-column NaNs and return.
    if (period <= 0 || period > series_len) {
        for (int row = 0; row < series_len; ++row) out_tm[row * num_series + series] = SUPERSMOOTHER_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        for (int row = 0; row < series_len; ++row) out_tm[row * num_series + series] = SUPERSMOOTHER_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        for (int row = 0; row < series_len; ++row) out_tm[row * num_series + series] = SUPERSMOOTHER_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        for (int row = 0; row < series_len; ++row) out_tm[row * num_series + series] = SUPERSMOOTHER_NAN;
        return;
    }

    // Column pointers with stride iteration to reduce per-iteration index math
    const int stride = num_series;
    const float* __restrict__ px = prices_tm + series;
    float*       __restrict__ py = out_tm    + series;

    // Prefix-only NaN fill: rows [0, warm)
    for (int row = 0; row < warm; ++row) py[row * stride] = SUPERSMOOTHER_NAN;

    // Coefficients
    float a, b, c;
    supersmoother_coeffs((float)period, &a, &b, &c);
    const float a_sq = a * a;

    // Seeds (time-major access)
    float y_im2 = px[warm * stride];
    py[warm * stride] = y_im2;

    if (warm + 1 >= series_len) return;

    float y_im1 = px[(warm + 1) * stride];
    py[(warm + 1) * stride] = y_im1;

    // Main recurrence; keep state in registers; row-wise coalesced across the warp
#pragma unroll 1
    for (int row = warm + 2; row < series_len; ++row) {
        const float x_i   = px[row * stride];
        const float x_im1 = px[(row - 1) * stride];
        const float t     = fmaf(b, y_im1, -a_sq * y_im2);
        const float yi    = fmaf(c, (x_i + x_im1), t);
        py[row * stride]  = yi;
        y_im2 = y_im1;
        y_im1 = yi;
    }
}
