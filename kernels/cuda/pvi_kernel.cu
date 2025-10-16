// CUDA kernels for Positive Volume Index (PVI)
//
// Semantics mirror src/indicators/pvi.rs exactly:
// - Warmup: indices < first_valid are NaN.
// - At first_valid: PVI = initial_value.
// - Thereafter: if volume[t] > volume[t-1],
//       r = (close[t] - close[t-1]) / close[t-1]
//       pvi += r * pvi;
//   else pvi stays unchanged.
// - NaN handling: if any of {close[t], volume[t], prev_close, prev_volume} is NaN,
//   write NaN for this t and only update prev_* when current {close, volume} are finite.
//
// We expose two-stage batch path to reuse shared precompute across rows:
//   1) pvi_build_scale_f32: builds a row-invariant multiplicative scale[] with scale[first]=1.
//   2) pvi_apply_scale_batch_f32: for each parameter row (initial value),
//      writes out[row, t] = initial_value[row] * scale[t] when scale[t] is finite, else NaN.
// Also provide many-series × one-param time-major kernel.

#include <cuda_runtime.h>
#include <math.h>

// Helper: test for finite (avoid C++ isfinite macro conflicts)
__device__ __forceinline__ bool finite_f(float x) { return isfinite(x); }

// 1) Build scale vector: one thread scans sequentially.
extern "C" __global__ void pvi_build_scale_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ scale_out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // single-thread sequential

    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = 0; i < len; ++i) scale_out[i] = nanf("");
    if (fv >= len) return;

    scale_out[fv] = 1.0f; // unit scale at the seed

    // Keep prev_* as last valid observations; do not poison on NaN ticks
    double prev_close = (double)close[fv];
    double prev_vol = (double)volume[fv];
    double accum = 1.0; // multiplicative scale

    for (int i = fv + 1; i < len; ++i) {
        const float cf = close[i];
        const float vf = volume[i];
        if (finite_f(cf) && finite_f(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;
                accum = fma(r, accum, accum); // accum += r*accum; (matches CPU mul_add path)
            }
            scale_out[i] = (float)accum;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            scale_out[i] = nanf("");
            if (finite_f(cf) && finite_f(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}

// 2) Apply scale to many rows (one-series × many-params).
// One thread per time index; loops over rows to reduce kernel count and avoid grid-y overflow.
extern "C" __global__ void pvi_apply_scale_batch_f32(
    const float* __restrict__ scale,
    int len,
    int first_valid,
    const float* __restrict__ initial_values, // [rows]
    int rows,
    float* __restrict__ out)                 // [rows * len], row-major
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len || rows <= 0) return;

    const float s = scale[t];
    const bool s_is_finite = isfinite(s);
    const float nan_f = nanf("");

    if (t < first_valid) {
        for (int r = 0; r < rows; ++r) {
            out[r * len + t] = nan_f;
        }
        return;
    }
    if (t == first_valid) {
        for (int r = 0; r < rows; ++r) {
            out[r * len + t] = initial_values[r];
        }
        return;
    }

    if (!s_is_finite) {
        for (int r = 0; r < rows; ++r) {
            out[r * len + t] = nan_f;
        }
        return;
    }

    for (int r = 0; r < rows; ++r) {
        out[r * len + t] = initial_values[r] * s;
    }
}

// Many-series × one-param (time-major layout): each thread processes one series.
extern "C" __global__ void pvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, // [cols]
    float initial_value,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const float nan_f = nanf("");

    // Warm prefix
    for (int t = 0; t < fv && t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (fv >= rows) return;

    double pvi = (double)initial_value;
    out_tm[fv * cols + s] = (float)pvi;
    if (fv + 1 >= rows) return;

    double prev_close = (double)close_tm[fv * cols + s];
    double prev_vol = (double)volume_tm[fv * cols + s];

    for (int t = fv + 1; t < rows; ++t) {
        const float cf = close_tm[t * cols + s];
        const float vf = volume_tm[t * cols + s];
        if (isfinite(cf) && isfinite(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;
                pvi += r * pvi; // preserve op ordering as CPU
            }
            out_tm[t * cols + s] = (float)pvi;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            out_tm[t * cols + s] = nan_f;
            if (isfinite(cf) && isfinite(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}

