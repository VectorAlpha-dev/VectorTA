// CUDA kernels for Volume Price Trend (VPT)
//
// Semantics mirror src/indicators/vpt.rs exactly:
// - Warmup: indices 0..=first_valid are NaN (first_valid >= 1).
// - Seed: prev = v[first_valid] * ((p[first_valid] - p[first_valid-1]) / p[first_valid-1]).
// - Thereafter t = first_valid + 1..len-1:
//       cur = v[t] * ((p[t] - p[t-1]) / p[t-1])
//       out[t] = cur + prev; prev = out[t];
// - NaN handling: if any term involved in the increment is non‑finite, write NaN.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void vpt_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    // Single-thread sequential scan (dependency across time).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = nanf("");

    // Warmup prefix: NaN for indices 0..=fv
    const int warm_end = (fv < len) ? fv : (len - 1);
    for (int i = 0; i <= warm_end; ++i) out[i] = nan_f;
    if (fv + 1 >= len) return; // nothing else to compute

    // Seed prev increment at index fv
    const double p0 = (double)price[fv - 1];
    const double p1 = (double)price[fv];
    const double v1 = (double)volume[fv];
    double prev = (isfinite(p0) && isfinite(p1) && isfinite(v1) && p0 != 0.0)
                      ? v1 * ((p1 - p0) / p0)
                      : NAN;
    double p_prev = p1;

    for (int t = fv + 1; t < len; ++t) {
        const double pt = (double)price[t];
        const double vt = (double)volume[t];
        const double cur = (isfinite(p_prev) && isfinite(pt) && isfinite(vt) && p_prev != 0.0)
                               ? vt * ((pt - p_prev) / p_prev)
                               : NAN;
        const double val = cur + prev; // will be NaN if either is NaN
        out[t] = (float)val;
        prev = val;
        p_prev = pt;
    }
}

// Many-series × one-param (time-major layout): one thread per series scans sequentially.
extern "C" __global__ void vpt_many_series_one_param_f32(
    const float* __restrict__ price_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids, // per series, i >= 1
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const float nan_f = nanf("");

    // Warmup up to and including fv
    const int warm_end = (fv < rows) ? fv : (rows - 1);
    for (int t = 0; t <= warm_end; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (fv + 1 >= rows) return;

    // Seed prev increment at fv using price[fv-1], price[fv], volume[fv]
    const double p0 = (double)price_tm[(fv - 1) * cols + s];
    const double p1 = (double)price_tm[fv * cols + s];
    const double v1 = (double)volume_tm[fv * cols + s];
    double prev = (isfinite(p0) && isfinite(p1) && isfinite(v1) && p0 != 0.0)
                      ? v1 * ((p1 - p0) / p0)
                      : NAN;
    double p_prev = p1;

    for (int t = fv + 1; t < rows; ++t) {
        const double pt = (double)price_tm[t * cols + s];
        const double vt = (double)volume_tm[t * cols + s];
        const double cur = (isfinite(p_prev) && isfinite(pt) && isfinite(vt) && p_prev != 0.0)
                               ? vt * ((pt - p_prev) / p_prev)
                               : NAN;
        const double val = cur + prev;
        out_tm[t * cols + s] = (float)val;
        prev = val;
        p_prev = pt;
    }
}

