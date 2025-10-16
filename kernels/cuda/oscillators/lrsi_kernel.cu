// CUDA kernels for Laguerre RSI (LRSI)
//
// Semantics match the scalar Rust implementation in src/indicators/lrsi.rs:
// - Warmup: first_valid + 3 elements are NaN
// - NaN inputs do not advance state; post-warm outputs become NaN for those slots
// - sum_abs <= FLT_EPSILON => output 0.0
// - FP32 math, FMA ordering mirrors scalar for numerical stability

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

extern "C" __global__
void lrsi_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ alphas,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Initialize entire row with NaNs in parallel to guarantee warmup prefix
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // sequential time-scan by a single lane
    if (first_valid < 0 || first_valid >= series_len) return;

    const float alpha = alphas[combo];
    if (!(alpha > 0.0f && alpha < 1.0f)) return;
    const float gamma = 1.0f - alpha;
    const float mgamma = -gamma;

    const int warm = first_valid + 3; // need 4 valid values total
    if (warm >= series_len) return;

    // Initialize filter state from first valid price
    const float p0 = prices[first_valid];
    float l0 = p0, l1 = p0, l2 = p0, l3 = p0;

    for (int t = first_valid + 1; t < series_len; ++t) {
        const float p = prices[t];
        if (isnan(p)) {
            if (t >= warm) out[base + t] = NAN;
            continue;
        }

        // 4-stage Laguerre filter with FMAs
        const float t0 = fmaf(alpha, (p - l0), l0);
        const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
        const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
        const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));

        l0 = t0; l1 = t1; l2 = t2; l3 = t3;

        if (t >= warm) {
            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;
            if (sum_abs <= FLT_EPSILON) {
                out[base + t] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out[base + t] = cu / sum_abs; // inherently in [0,1] for well-formed inputs
            }
        }
    }
}

// Many-series (time-major), one parameter. Output is time-major [rows x cols].
extern "C" __global__
void lrsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    float alpha,
                                    int num_series,
                                    int series_len,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= num_series) return;
    if (!(alpha > 0.0f && alpha < 1.0f)) return;
    const float gamma = 1.0f - alpha;
    const float mgamma = -gamma;

    const int first = max(0, first_valids[s]);
    if (first >= series_len) return;
    const int warm = first + 3;
    if (warm >= series_len) return;

    const int cols = num_series;
    const int idx0 = first * cols + s;
    float l0 = prices_tm[idx0];
    float l1 = l0, l2 = l0, l3 = l0;

    // Ensure warmup prefix is NaN for this series
    for (int t = 0; t < warm; ++t) {
        out_tm[t * cols + s] = NAN;
    }

    for (int t = first + 1; t < series_len; ++t) {
        const int idx = t * cols + s;
        const float p = prices_tm[idx];
        if (isnan(p)) {
            if (t >= warm) out_tm[idx] = NAN;
            continue;
        }

        const float t0 = fmaf(alpha, (p - l0), l0);
        const float t1 = fmaf(gamma, l1, fmaf(mgamma, t0, l0));
        const float t2 = fmaf(gamma, l2, fmaf(mgamma, t1, l1));
        const float t3 = fmaf(gamma, l3, fmaf(mgamma, t2, l2));

        l0 = t0; l1 = t1; l2 = t2; l3 = t3;

        if (t >= warm) {
            const float d01 = t0 - t1;
            const float d12 = t1 - t2;
            const float d23 = t2 - t3;
            const float a01 = fabsf(d01);
            const float a12 = fabsf(d12);
            const float a23 = fabsf(d23);
            const float sum_abs = a01 + a12 + a23;
            if (sum_abs <= FLT_EPSILON) {
                out_tm[idx] = 0.0f;
            } else {
                const float cu = 0.5f * (d01 + a01 + d12 + a12 + d23 + a23);
                out_tm[idx] = cu / sum_abs;
            }
        }
    }
}

