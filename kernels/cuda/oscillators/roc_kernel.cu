// CUDA kernels for Rate of Change (ROC)
//
// Math: roc[t] = (price[t] / price[t - period]) * 100 - 100
// Semantics:
// - Warmup region [0, first_valid + period) is NaN.
// - If previous value is 0.0 or NaN => output 0.0 (match scalar policy)
// - Mid-stream NaNs in current propagate naturally via division

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#if __CUDACC_VER_MAJOR__ >= 12
// CUDA 12+ / 13: nothing special required here
#endif

// Canonical quiet NaN in FP32 (avoid double->float NAN cast)
__device__ __forceinline__ float qnanf() { return nanf(""); }

// ================================
// One price series × many params
// out is row-major: base = combo * series_len
// ================================
extern "C" __global__
void roc_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    // Row base pointer (helps SASS hoist the base add)
    float* __restrict__ out_row = out + combo * series_len;

    // Load period once
    const int period = periods[combo];
    if (period <= 0) {
        // Fill entire row with NaN (rare path; keep parallelized)
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_row[t] = qnanf();
        }
        return;
    }

    // Warmup boundary (first valid output index)
    const int warm = first_valid + period;

    // If no valid outputs exist, just fill NaNs up to series_len
    if (warm >= series_len) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_row[t] = qnanf();
        }
        return;
    }

    // ----------------------------------------------------------------
    // 1) Warmup prefix: [0, warm) -> NaN (no need to touch the tail)
    // ----------------------------------------------------------------
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out_row[t] = qnanf();
    }

    // ----------------------------------------------------------------
    // 2) Valid range: [warm, series_len)
    //    Keep loads read-only; compiler will often use RO cache.
    //    For clarity and portability we do not depend on -use_fast_math.
    // ----------------------------------------------------------------
    for (int t = warm + threadIdx.x; t < series_len; t += blockDim.x) {
        // Read current and previous prices; use __ldg where available+safe
        float cur  =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices[t]);
#else
            prices[t];
#endif
        float prev =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices[t - period]);
#else
            prices[t - period];
#endif

        // Scalar policy: prev==0 or NaN -> output 0.0
        // (mid-stream NaNs in cur propagate naturally)
        if (prev == 0.0f || isnan(prev)) {
            out_row[t] = 0.0f;
        } else {
            // 100 * (cur/prev) - 100 == 100 * (cur*(1/prev) - 1)
            // FMA improves rounding slightly vs. separate mul+sub.
            const float inv_prev = 1.0f / prev;              // IEEE-accurate division
            const float rel      = fmaf(cur, inv_prev, -1.0f);
            out_row[t] = 100.0f * rel;
        }
    }
}

// ======================================================
// Many-series × one-param (time-major: idx = t*cols + s)
// Each thread handles one column (series) end-to-end.
// ======================================================
extern "C" __global__
void roc_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int cols,
                                   int rows,
                                   int period,
                                   float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0) {
        // Degenerate: fill entire column with NaN
        for (int t = 0; t < rows; ++t) {
            out_tm[t * cols + s] = qnanf();
        }
        return;
    }

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) {
            out_tm[t * cols + s] = qnanf();
        }
        return;
    }

    const int warm = fv + period;

    // Warmup prefix
    for (int t = 0; t < warm && t < rows; ++t) {
        out_tm[t * cols + s] = qnanf();
    }

    // Valid range
    for (int t = max(0, warm); t < rows; ++t) {
        const int idx  = t * cols + s;
        const float cur =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices_tm[idx]);
#else
            prices_tm[idx];
#endif
        const float prev =
#if __CUDA_ARCH__ >= 350
            __ldg(&prices_tm[(t - period) * cols + s]);
#else
            prices_tm[(t - period) * cols + s];
#endif

        if (prev == 0.0f || isnan(prev)) {
            out_tm[idx] = 0.0f;
        } else {
            const float inv_prev = 1.0f / prev;
            const float rel      = fmaf(cur, inv_prev, -1.0f);
            out_tm[idx] = 100.0f * rel;
        }
    }
}

