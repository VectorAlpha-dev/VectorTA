// CUDA kernels for Aroon Oscillator (AROONOSC).
//
// Semantics mirror the scalar Rust implementation in src/indicators/aroonosc.rs:
// - Warmup index per series/row: warm = first_valid + length
// - Before warm: outputs remain NaN
// - After warm: compute indices of highest high and lowest low over the last
//   (length + 1) bars and emit 100/length * (idx_high - idx_low), clamped to [-100, 100].
// - Inputs are FP32; outputs are FP32.
// - We do not attempt deque-based O(1) sliding extrema on GPU; a straightforward
//   O(window) rescan is used to match scalar behavior and keep kernels simple.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// One-series × many-params (batch). One CUDA block per parameter row.
extern "C" __global__
void aroonosc_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const int* __restrict__ lengths,
                        int series_len,
                        int first_valid,
                        int n_combos,
                        float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    // Initialize this row to NaN in parallel
    const int base = combo * series_len;
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single lane performs sequential scan

    const int L = lengths[combo];
    if (L <= 0) return;
    const int warm = first_valid + L; // window-1 == L
    if (warm >= series_len) return;

    const float scale = 100.0f / (float)L;
    for (int t = warm; t < series_len; ++t) {
        const int start = t - L;
        int hi_idx = start;
        int lo_idx = start;
        float hi_val = high[start];
        float lo_val = low[start];
        // Rescan window [start..t]
        for (int j = start + 1; j <= t; ++j) {
            const float h = high[j];
            if (h > hi_val) { hi_val = h; hi_idx = j; }
            const float l = low[j];
            if (l < lo_val) { lo_val = l; lo_idx = j; }
        }
        float v = (float)(hi_idx - lo_idx) * scale;
        if (v > 100.0f) v = 100.0f;
        if (v < -100.0f) v = -100.0f;
        out[base + t] = v;
    }
}

// Many-series × one-param, time-major layout: [t][series]
extern "C" __global__
void aroonosc_many_series_one_param_f32(const float* __restrict__ high_tm,
                                        const float* __restrict__ low_tm,
                                        const int* __restrict__ first_valids,
                                        int num_series,
                                        int series_len,
                                        int length,
                                        float* __restrict__ out_tm) {
    const int s = blockIdx.x; // one block per series
    if (s >= num_series || series_len <= 0) return;

    // Initialize this series to NaN in parallel
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    if (length <= 0) return;
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm = fv + length; // window-1 == length
    if (warm >= series_len) return;

    const float scale = 100.0f / (float)length;
    const int stride = num_series; // time-major
    for (int t = warm; t < series_len; ++t) {
        const int start = t - length;
        int hi_idx = start;
        int lo_idx = start;
        float hi_val = high_tm[start * stride + s];
        float lo_val = low_tm[start * stride + s];
        for (int j = start + 1; j <= t; ++j) {
            const float h = high_tm[j * stride + s];
            if (h > hi_val) { hi_val = h; hi_idx = j; }
            const float l = low_tm[j * stride + s];
            if (l < lo_val) { lo_val = l; lo_idx = j; }
        }
        float v = (float)(hi_idx - lo_idx) * scale;
        if (v > 100.0f) v = 100.0f;
        if (v < -100.0f) v = -100.0f;
        out_tm[t * stride + s] = v;
    }
}

