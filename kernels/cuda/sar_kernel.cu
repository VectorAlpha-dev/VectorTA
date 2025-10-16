// CUDA kernels for Parabolic SAR (Stop and Reverse)
//
// Math pattern: recurrence/IIR with per-sample dependency, no shared
// precomputation between rows. Warmup semantics match the scalar Rust
// implementation in src/indicators/sar.rs:
// - Find first index where both high and low are finite (first_valid)
// - Write NaN up to first_valid (inclusive of index first_valid)
// - At index (first_valid + 1), write initial SAR (prev high or prev low
//   per trend) and continue sequentially thereafter.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
static __forceinline__ __device__ float f32_nan() { return __int_as_float(0x7fffffff); }

// --------------------- Batch: one series × many params ---------------------
// Inputs:
//  - high, low: series (length = len)
//  - len: number of bars
//  - first_valid: first index where both high and low are finite
//  - accs, maxes: parameter arrays (length = n_rows)
//  - out: row-major [row][t] (n_rows × len)
extern "C" __global__ void sar_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    const float* __restrict__ accs,
    const float* __restrict__ maxes,
    int n_rows,
    float* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= n_rows) return;

    const int base = row * len;
    const int warm = first_valid + 1; // per scalar semantics

    // Fill warmup prefix with NaN cooperatively across block.x
    for (int t = threadIdx.x; t < min(warm, len); t += blockDim.x) {
        out[base + t] = f32_nan();
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // sequential stateful scan
    if (warm >= len) return;

    // Load params
    float acceleration = accs[row];
    float maximum = maxes[row];
    if (!(acceleration > 0.0f) || !(maximum > 0.0f)) {
        // keep row as NaN if params invalid (matches scalar error paths)
        return;
    }

    const int i0 = first_valid;
    const int i1 = warm;

    float h0 = high[i0];
    float h1 = high[i1];
    float l0 = low[i0];
    float l1 = low[i1];

    // Initial trend and state (Wilder)
    bool trend_up = (h1 > h0);
    float sar = trend_up ? l0 : h0;
    float ep = trend_up ? h1 : l1; // extreme point (highest high or lowest low)
    float acc = acceleration;       // acceleration factor (AF)

    // Write initial SAR at warm index
    out[base + i1] = sar;

    float low_prev2 = l0;
    float low_prev  = l1;
    float high_prev2 = h0;
    float high_prev  = h1;

    // Main sequential loop
    for (int i = i1 + 1; i < len; ++i) {
        const float hi = high[i];
        const float lo = low[i];

        // next_sar = sar + acc * (ep - sar)
        float next_sar = fmaf(acc, (ep - sar), sar);

        if (trend_up) {
            if (lo < next_sar) {
                // Reversal to downtrend
                trend_up = false;
                next_sar = ep; // per Wilder, prior EP becomes SAR at reversal
                ep = lo;
                acc = acceleration; // reset AF
            } else {
                // Continue uptrend: extend EP/AF and clamp to prior lows
                if (hi > ep) {
                    ep = hi;
                    acc = fminf(acc + acceleration, maximum);
                }
                next_sar = fminf(next_sar, fminf(low_prev, low_prev2));
            }
        } else { // downtrend
            if (hi > next_sar) {
                // Reversal to uptrend
                trend_up = true;
                next_sar = ep;
                ep = hi;
                acc = acceleration;
            } else {
                // Continue downtrend: extend EP/AF and clamp to prior highs
                if (lo < ep) {
                    ep = lo;
                    acc = fminf(acc + acceleration, maximum);
                }
                next_sar = fmaxf(next_sar, fmaxf(high_prev, high_prev2));
            }
        }

        out[base + i] = next_sar;
        sar = next_sar;

        // Shift previous two highs/lows
        low_prev2 = low_prev;
        low_prev = lo;
        high_prev2 = high_prev;
        high_prev = hi;
    }
}

// --------------- Many-series × one-param (time‑major layout) ---------------
// Inputs (time-major):
//  - high_tm, low_tm: arrays length = rows*cols, index = t * cols + s
//  - first_valids: per-series first valid index (length = cols). Negative => no data
//  - cols: number of series (columns)
//  - rows: number of time steps (rows)
//  - acceleration, maximum: parameters
//  - out_tm: time-major output [t][s]
extern "C" __global__ void sar_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    float acceleration,
    float maximum,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series index (column)
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            out_tm[t * cols + s] = f32_nan();
        }
        return;
    }

    const int warm = fv + 1;

    // Fill warmup prefix with NaN cooperatively across block.x
    for (int t = threadIdx.x; t < min(warm, rows); t += blockDim.x) {
        out_tm[t * cols + s] = f32_nan();
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (warm >= rows) return;

    // Initial state from bars fv and warm
    const int i0 = fv;
    const int i1 = warm;
    float h0 = high_tm[i0 * cols + s];
    float h1 = high_tm[i1 * cols + s];
    float l0 = low_tm[i0 * cols + s];
    float l1 = low_tm[i1 * cols + s];

    bool trend_up = (h1 > h0);
    float sar = trend_up ? l0 : h0;
    float ep = trend_up ? h1 : l1;
    float acc = acceleration;

    out_tm[i1 * cols + s] = sar;

    float low_prev2 = l0;
    float low_prev  = l1;
    float high_prev2 = h0;
    float high_prev  = h1;

    for (int i = i1 + 1; i < rows; ++i) {
        const float hi = high_tm[i * cols + s];
        const float lo = low_tm[i * cols + s];

        float next_sar = fmaf(acc, (ep - sar), sar);
        if (trend_up) {
            if (lo < next_sar) {
                trend_up = false;
                next_sar = ep;
                ep = lo;
                acc = acceleration;
            } else {
                if (hi > ep) { ep = hi; acc = fminf(acc + acceleration, maximum); }
                next_sar = fminf(next_sar, fminf(low_prev, low_prev2));
            }
        } else {
            if (hi > next_sar) {
                trend_up = true;
                next_sar = ep;
                ep = hi;
                acc = acceleration;
            } else {
                if (lo < ep) { ep = lo; acc = fminf(acc + acceleration, maximum); }
                next_sar = fmaxf(next_sar, fmaxf(high_prev, high_prev2));
            }
        }
        out_tm[i * cols + s] = next_sar;
        sar = next_sar;

        low_prev2 = low_prev;
        low_prev = lo;
        high_prev2 = high_prev;
        high_prev = hi;
    }
}

