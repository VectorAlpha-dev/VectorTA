// CUDA kernels for Aroon indicator (Up/Down)
//
// Semantics mirror src/indicators/aroon.rs (scalar path):
// - Warmup: values before (first_valid + length) are NaN
// - Window: scan [t - length .. t] (length+1 samples) for max(high), min(low)
// - NaN: any non-finite high/low in the window -> both outputs are NaN
// - Ties: use strict comparisons (>) for highs, (<) for lows so earlier idx wins
// - Percent: up = 100 - dist_hi * (100/length), down = 100 - dist_lo * (100/length)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool both_finite(float h, float l) {
    return isfinite(h) && isfinite(l);
}

extern "C" __global__
void aroon_batch_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const int* __restrict__ lengths,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out_up,
                     float* __restrict__ out_down) {
    const int combo = blockIdx.y * gridDim.x + blockIdx.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    if (length <= 0 || first_valid < 0 || first_valid >= series_len) return;

    const int base = combo * series_len;
    // Fill both outputs with NaN cooperatively
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_up[base + i] = NAN;
        out_down[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // one thread scans sequentially per combo

    const float scale = 100.0f / (float)length;
    const int warm = first_valid + length;
    if (warm >= series_len) return;

    for (int t = warm; t < series_len; ++t) {
        const int start = t - length;
        // initialize with first element
        float h0 = high[start];
        float l0 = low[start];
        if (!both_finite(h0, l0)) {
            out_up[base + t] = NAN;
            out_down[base + t] = NAN;
            continue;
        }
        float best_h = h0;
        float best_l = l0;
        int best_h_off = 0;
        int best_l_off = 0;

        int off = 1;
        const int window = length + 1; // inclusive window size
        bool valid = true;
        while (off < window) {
            const float h = high[start + off];
            const float l = low[start + off];
            if (!both_finite(h, l)) { valid = false; break; }
            if (h > best_h) { best_h = h; best_h_off = off; }
            if (l < best_l) { best_l = l; best_l_off = off; }
            ++off;
        }
        if (!valid) {
            out_up[base + t] = NAN;
            out_down[base + t] = NAN;
            continue;
        }
        const int dist_hi = length - best_h_off;
        const int dist_lo = length - best_l_off;
        const float up = (dist_hi == 0) ? 100.0f : (dist_hi >= length ? 0.0f : fmaf(-(float)dist_hi, scale, 100.0f));
        const float dn = (dist_lo == 0) ? 100.0f : (dist_lo >= length ? 0.0f : fmaf(-(float)dist_lo, scale, 100.0f));
        out_up[base + t] = up;
        out_down[base + t] = dn;
    }
}

// Many-series × one-param (time-major). Each block handles one series with a single thread.
extern "C" __global__
void aroon_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const int* __restrict__ first_valids,
                                     int length,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_up_tm,
                                     float* __restrict__ out_down_tm) {
    const int s = blockIdx.x;
    if (s >= num_series || length <= 0) return;
    const int first = first_valids[s];
    if (first < 0 || first >= series_len) return;

    const int stride = num_series;
    // Fill with NaNs
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_up_tm[t * stride + s] = NAN;
        out_down_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float scale = 100.0f / (float)length;
    const int warm = first + length;
    if (warm >= series_len) return;

    for (int t = warm; t < series_len; ++t) {
        const int start = t - length;
        float h0 = high_tm[start * stride + s];
        float l0 = low_tm[start * stride + s];
        if (!both_finite(h0, l0)) {
            out_up_tm[t * stride + s] = NAN;
            out_down_tm[t * stride + s] = NAN;
            continue;
        }
        float best_h = h0;
        float best_l = l0;
        int best_h_off = 0;
        int best_l_off = 0;
        int off = 1;
        const int window = length + 1;
        bool valid = true;
        while (off < window) {
            const float h = high_tm[(start + off) * stride + s];
            const float l = low_tm[(start + off) * stride + s];
            if (!both_finite(h, l)) { valid = false; break; }
            if (h > best_h) { best_h = h; best_h_off = off; }
            if (l < best_l) { best_l = l; best_l_off = off; }
            ++off;
        }
        if (!valid) {
            out_up_tm[t * stride + s] = NAN;
            out_down_tm[t * stride + s] = NAN;
            continue;
        }
        const int dist_hi = length - best_h_off;
        const int dist_lo = length - best_l_off;
        const float up = (dist_hi == 0) ? 100.0f : (dist_hi >= length ? 0.0f : fmaf(-(float)dist_hi, scale, 100.0f));
        const float dn = (dist_lo == 0) ? 100.0f : (dist_lo >= length ? 0.0f : fmaf(-(float)dist_lo, scale, 100.0f));
        out_up_tm[t * stride + s] = up;
        out_down_tm[t * stride + s] = dn;
    }
}

