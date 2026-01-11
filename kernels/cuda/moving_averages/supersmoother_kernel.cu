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

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the 2x2 affine state update:
//
//   y[i] = c*(x[i] + x[i-1]) + b*y[i-1] - a^2*y[i-2]
//
// State is s[i] = [y[i], y[i-1]] and:
//   s[i] = M*s[i-1] + [u[i], 0]
//   M = [[b, -a^2],
//        [1,   0 ]]
//   u[i] = c*(x[i] + x[i-1])
//
// - blockDim.x must be exactly 32
// - outputs:
//   - [0, warm) = NaN where warm = first_valid + period - 1
//   - y[warm] = x[warm], y[warm+1] = x[warm+1]
//   - i >= warm+2 computed by recurrence
extern "C" __global__ void supersmoother_batch_warp_scan_f32(const float* __restrict__ prices,
                                                            const int* __restrict__ periods,
                                                            int series_len,
                                                            int n_combos,
                                                            int first_valid,
                                                            float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0) return;
    if (threadIdx.x >= 32) return;

    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffffu;

    float* __restrict__ row_out = out + (size_t)combo * (size_t)series_len;

    if (first_valid < 0 || first_valid >= series_len) {
        for (int i = lane; i < series_len; i += 32) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > series_len) {
        for (int i = lane; i < series_len; i += 32) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period) {
        for (int i = lane; i < series_len; i += 32) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    const int warm = first_valid + period - 1;
    if (warm >= series_len) {
        for (int i = lane; i < series_len; i += 32) row_out[i] = SUPERSMOOTHER_NAN;
        return;
    }

    // Prefix-only NaN fill: [0, warm)
    for (int i = lane; i < warm; i += 32) row_out[i] = SUPERSMOOTHER_NAN;

    // Seeds
    if (lane == 0) {
        row_out[warm] = prices[warm];
        if (warm + 1 < series_len) row_out[warm + 1] = prices[warm + 1];
    }
    if (warm + 1 >= series_len) return;

    // Coefficients
    float a, b, c;
    supersmoother_coeffs((float)period, &a, &b, &c);
    const float a_sq = a * a;

    // Previous state at t0-1 = warm+1
    float s0_prev = 0.0f;
    float s1_prev = 0.0f;
    if (lane == 0) {
        s1_prev = prices[warm];     // y_{t0-2}
        s0_prev = prices[warm + 1]; // y_{t0-1}
    }
    s0_prev = __shfl_sync(mask, s0_prev, 0);
    s1_prev = __shfl_sync(mask, s1_prev, 0);

    // Constant state-transition matrix M
    const float m00 = b;
    const float m01 = -a_sq;
    const float m10 = 1.0f;
    const float m11 = 0.0f;

    const int t0 = warm + 2;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        float u = 0.0f;
        if (valid) {
            const float x0 = prices[t];
            const float x1 = prices[t - 1];
            u = c * (x0 + x1);
        }

        // Each lane starts with its step transform: (P, v), where
        // P = M and v = [u, 0]. For inactive lanes, use identity.
        float p00 = valid ? m00 : 1.0f;
        float p01 = valid ? m01 : 0.0f;
        float p10 = valid ? m10 : 0.0f;
        float p11 = valid ? m11 : 1.0f;
        float v0  = valid ? u   : 0.0f;
        float v1  = 0.0f;

        // Inclusive scan over composed transforms: (P_cur, v_cur) ∘ (P_prev, v_prev)
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float p00_prev = __shfl_up_sync(mask, p00, offset);
            const float p01_prev = __shfl_up_sync(mask, p01, offset);
            const float p10_prev = __shfl_up_sync(mask, p10, offset);
            const float p11_prev = __shfl_up_sync(mask, p11, offset);
            const float v0_prev  = __shfl_up_sync(mask, v0,  offset);
            const float v1_prev  = __shfl_up_sync(mask, v1,  offset);
            if (lane >= offset) {
                const float c00 = p00, c01 = p01, c10 = p10, c11 = p11;
                const float cv0 = v0,  cv1 = v1;

                const float n00 = fmaf(c00, p00_prev, c01 * p10_prev);
                const float n01 = fmaf(c00, p01_prev, c01 * p11_prev);
                const float n10 = fmaf(c10, p00_prev, c11 * p10_prev);
                const float n11 = fmaf(c10, p01_prev, c11 * p11_prev);

                const float nv0 = fmaf(c00, v0_prev, fmaf(c01, v1_prev, cv0));
                const float nv1 = fmaf(c10, v0_prev, fmaf(c11, v1_prev, cv1));

                p00 = n00; p01 = n01; p10 = n10; p11 = n11;
                v0  = nv0; v1  = nv1;
            }
        }

        // Apply prefix transform to previous state s_prev
        const float y0 = fmaf(p00, s0_prev, fmaf(p01, s1_prev, v0));
        const float y1 = fmaf(p10, s0_prev, fmaf(p11, s1_prev, v1));

        if (valid) {
            row_out[t] = y0;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        s0_prev = __shfl_sync(mask, y0, last_lane);
        s1_prev = __shfl_sync(mask, y1, last_lane);
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
