// CUDA kernels for RSI (Relative Strength Index)
//
// Math pattern: Wilder-style recurrence (IIR). Warmup matches scalar:
// - Warm index = first_valid + period. Elements [..warm] set to NaN.
// - Initial averages computed over first `period` deltas after first_valid.
// - If a non-finite delta is encountered during warmup, the initial RSI and
//   all subsequent outputs are NaN (propagate by keeping avg_g/avg_l as NaN).
// - For updates, denom==0 -> 50.0; outputs clamped to [0, 100] via arithmetic.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// One series × many params (batch)
// prices: length = series_len
// periods: length = n_combos
// out: rows=n_combos, cols=series_len (row-major)
extern "C" __global__
void rsi_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ periods,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int base = combo * series_len;

    // Initialize row to NaN for warmup semantics
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane sequential scan
    if (first_valid >= series_len) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period; // scalar places first RSI at idx0 = first + period
    if (warm >= series_len) return;

    const float inv_p = 1.0f / (float)period;
    const float beta  = 1.0f - inv_p;

    // Warmup over first `period` deltas
    float avg_g = 0.0f;
    float avg_l = 0.0f;
    bool has_nan = false;
    int i = first_valid + 1;
    const int warm_last = min(warm, series_len - 1);
    while (i <= warm_last) {
        const float d = prices[i] - prices[i - 1];
        if (!isfinite(d)) { has_nan = true; break; }
        if (d > 0.0f) avg_g += d; else if (d < 0.0f) avg_l -= d;
        ++i;
    }

    if (has_nan) {
        avg_g = NAN; avg_l = NAN;
        out[base + warm] = NAN; // initial RSI
    } else {
        avg_g *= inv_p; avg_l *= inv_p;
        const float denom = avg_g + avg_l;
        out[base + warm] = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
    }

    // Recursive updates (Wilder smoothing)
    for (int t = warm + 1; t < series_len; ++t) {
        const float d = prices[t] - prices[t - 1];
        // Batch scalar semantics: if a non-finite delta occurs after warmup,
        // mark averages NaN so the rest of the row stays NaN.
        if (!isfinite(d)) { avg_g = NAN; avg_l = NAN; out[base + t] = NAN; continue; }
        const float g = (d > 0.0f) ? d : 0.0f;
        const float l = (d < 0.0f) ? -d : 0.0f;
        avg_g = fmaf(beta, avg_g, inv_p * g);
        avg_l = fmaf(beta, avg_l, inv_p * l);
        const float denom = avg_g + avg_l;
        out[base + t] = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
    }
}

// Many-series × one-param (time-major)
// prices_tm/out_tm layout: index = t * cols + s
extern "C" __global__
void rsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int* __restrict__ first_valids,
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
    const int warm = fv + period; // first RSI at warm index

    for (int t = 0; t <= warm && t < rows; ++t) {
        out_tm[t * cols + s] = NAN;
    }
    if (warm >= rows) return;

    const float inv_p = 1.0f / (float)period;
    const float beta  = 1.0f - inv_p;

    // Warmup averages over first `period` deltas for this series
    float avg_g = 0.0f;
    float avg_l = 0.0f;
    bool has_nan = false;
    int t = fv + 1;
    const int warm_last = min(warm, rows - 1);
    while (t <= warm_last) {
        const float d = prices_tm[t * cols + s] - prices_tm[(t - 1) * cols + s];
        if (!isfinite(d)) { has_nan = true; break; }
        if (d > 0.0f) avg_g += d; else if (d < 0.0f) avg_l -= d;
        ++t;
    }

    if (has_nan) {
        avg_g = NAN; avg_l = NAN;
        out_tm[warm * cols + s] = NAN;
    } else {
        avg_g *= inv_p; avg_l *= inv_p;
        const float denom = avg_g + avg_l;
        out_tm[warm * cols + s] = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
    }

    // Recursive updates (match scalar single-series semantics for NaN deltas:
    // treat NaN deltas like zero change).
    for (int u = warm + 1; u < rows; ++u) {
        const float d = prices_tm[u * cols + s] - prices_tm[(u - 1) * cols + s];
        const float g = (d > 0.0f) ? d : 0.0f; // if d is NaN, comparisons are false -> g=0
        const float l = (d < 0.0f) ? -d : 0.0f; // if d is NaN -> l=0
        avg_g = fmaf(beta, avg_g, inv_p * g);
        avg_l = fmaf(beta, avg_l, inv_p * l);
        const float denom = avg_g + avg_l;
        out_tm[u * cols + s] = (denom == 0.0f) ? 50.0f : (100.0f * avg_g / denom);
    }
}

