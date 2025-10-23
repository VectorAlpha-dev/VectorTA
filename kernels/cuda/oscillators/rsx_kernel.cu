// CUDA kernels for RSX (Relative Strength Xtra)
//
// Math pattern: recurrence/IIR chain with warmup latch.
// Semantics (must match scalar RSX exactly):
// - Warmup index = first_valid + period - 1; write NaN at warmup and before.
// - Valid outputs start at warmup + 1.
// - If denominator proxy (v20_) <= 1e-10, output 50.0.
// - Clamp final value to [0, 100].
// - Ignore NaNs in input like scalar: the arithmetic propagates them; we only
//   guard warmup and bounds.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Branchless clamp helper used by both kernels
static __device__ __forceinline__ float clamp_0_100(float x) {
    x = fminf(x, 100.0f);
    x = fmaxf(x, 0.0f);
    return x;
}

// One series × many params (batch)
// Mapping: 1 thread = 1 param combo; warp broadcasts price[t]
// prices: length = series_len
// periods: length = n_combos
// out: layout rows=n_combos, cols=series_len (row-major)
extern "C" __global__
void rsx_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    if (period <= 0) {
        for (int t = 0; t < series_len; ++t) out[base + t] = NAN;
        return;
    }

    const int warm = first_valid + period - 1;

    // RSX state (FP32)
    float f0  = 0.0f;
    float f8  = 0.0f;   // set at t==warm
    bool  have_init = false;
    const float alpha = 3.0f / (float(period) + 2.0f);
    const float beta  = 1.0f - alpha;
    float f28 = 0.0f, f30 = 0.0f;
    float f38 = 0.0f, f40 = 0.0f;
    float f48 = 0.0f, f50 = 0.0f;
    float f58 = 0.0f, f60 = 0.0f;
    float f68 = 0.0f, f70 = 0.0f;
    float f78 = 0.0f, f80 = 0.0f;
    const float f88 = (period >= 6) ? float(period - 1) : 5.0f;
    float f90 = 1.0f;

    // Sequential in time per combo
    #pragma unroll 1
    for (int t = 0; t < series_len; ++t) {
        // Warp-broadcast price[t]: lane0 loads; __shfl_sync shares within warp
        unsigned mask = __activemask();
        float p = 0.0f;
        if ((threadIdx.x & 31) == 0) {
            p = __ldg(prices + t);
        }
        p = __shfl_sync(mask, p, 0);
        const float p100 = 100.0f * p;

        // Warmup semantics: prefix including warm index as NaN
        if (t <= warm) {
            out[base + t] = NAN;
            if (t == warm) { f8 = p100; have_init = true; }
            continue;
        }

        // After warmup
        f90 = (f88 <= f90) ? (f88 + 1.0f) : (f90 + 1.0f);
        const float prev = f8;
        f8 = p100;
        const float v8 = f8 - prev;

        // IIR cascade; compiler fuses to FMAs
        f28 = beta * f28 + alpha * v8;
        f30 = alpha * f28 + beta * f30;
        const float v_c = 1.5f * f28 - 0.5f * f30;

        f38 = beta * f38 + alpha * v_c;
        f40 = alpha * f38 + beta * f40;
        const float v10 = 1.5f * f38 - 0.5f * f40;

        f48 = beta * f48 + alpha * v10;
        f50 = alpha * f48 + beta * f50;
        const float v14 = 1.5f * f48 - 0.5f * f50;

        const float av = fabsf(v8);
        f58 = beta * f58 + alpha * av;
        f60 = alpha * f58 + beta * f60;
        const float v18 = 1.5f * f58 - 0.5f * f60;

        f68 = beta * f68 + alpha * v18;
        f70 = alpha * f68 + beta * f70;
        const float v1c = 1.5f * f68 - 0.5f * f70;

        f78 = beta * f78 + alpha * v1c;
        f80 = alpha * f78 + beta * f80;
        const float v20_ = 1.5f * f78 - 0.5f * f80;

        if (f88 >= f90 && f8 != prev) { f0 = 1.0f; }
        if (fabsf(f88 - f90) <= 1e-12f && f0 == 0.0f) { f90 = 0.0f; }

        float y = 50.0f;
        if (f88 < f90 && v20_ > 1e-10f && have_init) {
            y = clamp_0_100((v14 / v20_ + 1.0f) * 50.0f);
        }
        out[base + t] = y;
    }
}

// Many-series × one-param (time-major)
// prices_tm/out_tm layout: index = t * cols + s
extern "C" __global__
void rsx_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;
    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = NAN;
        return;
    }
    const int warm = fv + period - 1;

    // Warmup prefix incl. warm index as NaN to match scalar
    for (int t = 0; t <= warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    // State registers
    float f0 = 0.0f;
    float f8 = 100.0f * prices_tm[warm * cols + s];
    const float alpha = 3.0f / (float(period) + 2.0f);
    const float beta  = 1.0f - alpha;
    float f28 = 0.0f, f30 = 0.0f;
    float f38 = 0.0f, f40 = 0.0f;
    float f48 = 0.0f, f50 = 0.0f;
    float f58 = 0.0f, f60 = 0.0f;
    float f68 = 0.0f, f70 = 0.0f;
    float f78 = 0.0f, f80 = 0.0f;
    const float f88 = (period >= 6) ? float(period - 1) : 5.0f;
    float f90 = 1.0f;

    for (int t = warm + 1; t < rows; ++t) {
        f90 = (f88 <= f90) ? (f88 + 1.0f) : (f90 + 1.0f);

        const float prev = f8;
        const float cur  = prices_tm[t * cols + s];
        f8 = 100.0f * cur;
        const float v8 = f8 - prev;

        f28 = beta * f28 + alpha * v8;
        f30 = alpha * f28 + beta * f30;
        const float v_c = 1.5f * f28 - 0.5f * f30;

        f38 = beta * f38 + alpha * v_c;
        f40 = alpha * f38 + beta * f40;
        const float v10 = 1.5f * f38 - 0.5f * f40;

        f48 = beta * f48 + alpha * v10;
        f50 = alpha * f48 + beta * f50;
        const float v14 = 1.5f * f48 - 0.5f * f50;

        const float av = fabsf(v8);
        f58 = beta * f58 + alpha * av;
        f60 = alpha * f58 + beta * f60;
        const float v18 = 1.5f * f58 - 0.5f * f60;

        f68 = beta * f68 + alpha * v18;
        f70 = alpha * f68 + beta * f70;
        const float v1c = 1.5f * f68 - 0.5f * f70;

        f78 = beta * f78 + alpha * v1c;
        f80 = alpha * f78 + beta * f80;
        const float v20_ = 1.5f * f78 - 0.5f * f80;

        if (f88 >= f90 && f8 != prev) {
            f0 = 1.0f;
        }
        if (fabsf(f88 - f90) <= 1e-12f && f0 == 0.0f) {
            f90 = 0.0f;
        }

        float y = 50.0f;
        if (f88 < f90 && v20_ > 1e-10f) {
            y = clamp_0_100((v14 / v20_ + 1.0f) * 50.0f);
        }
        out_tm[t * cols + s] = y;
    }
}

