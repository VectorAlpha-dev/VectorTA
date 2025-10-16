// CUDA kernels for Klinger Volume Oscillator (KVO)
//
// Math pattern: recurrence/IIR per row/series.
// - Batch (one series × many params): precompute VF (volume force) on host once
//   and pass it to the kernel. Each row (short/long pair) is processed by a
//   single thread that performs a sequential time scan for EMA updates.
// - Many-series × one-param (time-major): compute VF and EMA inside the kernel
//   per series in a single sequential pass.
//
// Semantics (match scalar implementation):
// - Warmup: outputs are NaN up to (first_valid + 1). At index (first_valid + 1),
//   both EMAs are seeded to the first VF value and the output is 0.0.
// - Accumulations use float64 for numeric stability; outputs are float32.

#include <cuda_runtime.h>
#include <math.h>

// Helper to write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// -------------------------- Batch: one series × many params --------------------------
// Inputs:
//  - vf:          precomputed VF stream (length = len)
//  - len:         number of elements
//  - first_valid: index of first fully valid OHLCV bar
//  - shorts/longs: period arrays of length n_combos
//  - out:         row-major [combo][t] (n_combos x len)
extern "C" __global__ void kvo_batch_f32(
    const float* __restrict__ vf,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    int n_combos,
    float* __restrict__ out)
{
    const int combo = blockIdx.y; // row index
    if (combo >= n_combos) return;

    const int s = shorts[combo];
    const int l = longs[combo];
    if (s <= 0 || l < s) return;

    const int warm = first_valid + 1; // matches scalar warmup for KVO
    const int row_off = combo * len;

    // Fill warmup prefix with NaN in parallel across block.x
    for (int t = threadIdx.x; t < min(warm, len); t += blockDim.x) {
        out[row_off + t] = f32_nan();
    }
    __syncthreads();

    // Only one thread performs the sequential EMA scan to respect dependency
    if (threadIdx.x != 0) return;
    if (warm >= len) return;

    const double alpha_s = 2.0 / (double)(s + 1);
    const double alpha_l = 2.0 / (double)(l + 1);

    double ema_s = 0.0;
    double ema_l = 0.0;

    // Seed at warm index
    const double seed = (double)vf[warm];
    ema_s = seed;
    ema_l = seed;
    out[row_off + warm] = 0.0f; // first difference is zero by definition

    // Sequential scan
    for (int t = warm + 1; t < len; ++t) {
        const double vfi = (double)vf[t];
        ema_s += (vfi - ema_s) * alpha_s;
        ema_l += (vfi - ema_l) * alpha_l;
        out[row_off + t] = (float)(ema_s - ema_l);
    }
}

// ----------------------- Many-series × one-param (time-major) -----------------------
// Inputs (time-major):
//  - high_tm, low_tm, close_tm, volume_tm: arrays of length rows*cols in time-major order
//  - first_valids: per-series first valid index (length = cols). Negative => no valid data
//  - cols: number of series (columns)
//  - rows: number of time steps (rows)
//  - short_p, long_p: EMA periods (long_p >= short_p >= 1)
//  - out_tm: output buffer, time-major [t][series]
extern "C" __global__ void kvo_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column index
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        // Fill with NaN if no valid data
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            out_tm[t * cols + s] = f32_nan();
        }
        return;
    }

    const int warm = fv + 1;

    // Fill warmup prefix with NaN in parallel across block.x
    for (int t = threadIdx.x; t < min(warm, rows); t += blockDim.x) {
        out_tm[t * cols + s] = f32_nan();
    }
    __syncthreads();

    if (threadIdx.x != 0) return;
    if (warm >= rows) return;

    const double alpha_s = 2.0 / (double)(short_p + 1);
    const double alpha_l = 2.0 / (double)(long_p + 1);

    // Seed state from bar fv
    double ema_s = 0.0;
    double ema_l = 0.0;

    // Initialize recurrence state for VF computation
    double prev_hlc = (double)high_tm[fv * cols + s]
                    + (double)low_tm[fv * cols + s]
                    + (double)close_tm[fv * cols + s];
    double prev_dm = (double)high_tm[fv * cols + s] - (double)low_tm[fv * cols + s];
    int trend = -1; // -1 for down, 0 down, 1 up (mirrors scalar init)
    double cm = 0.0;

    // First consumable VF at t = warm
    {
        const int t = warm;
        const double h = (double)high_tm[t * cols + s];
        const double l = (double)low_tm[t * cols + s];
        const double c = (double)close_tm[t * cols + s];
        const double v = (double)volume_tm[t * cols + s];
        const double hlc = h + l + c;
        const double dm = h - l;
        if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
        else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
        cm += dm;
        const double temp = fabs(((dm / cm) * 2.0) - 1.0);
        const double sign = (trend == 1) ? 1.0 : -1.0;
        const double vf = v * temp * 100.0 * sign;
        ema_s = vf;
        ema_l = vf;
        out_tm[t * cols + s] = 0.0f; // seed difference
        prev_hlc = hlc;
        prev_dm = dm;
    }

    // Continue sequentially for t > warm
    for (int t = warm + 1; t < rows; ++t) {
        const double h = (double)high_tm[t * cols + s];
        const double l = (double)low_tm[t * cols + s];
        const double c = (double)close_tm[t * cols + s];
        const double v = (double)volume_tm[t * cols + s];
        const double hlc = h + l + c;
        const double dm = h - l;
        if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
        else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
        cm += dm;
        const double temp = fabs(((dm / cm) * 2.0) - 1.0);
        const double sign = (trend == 1) ? 1.0 : -1.0;
        const double vf = v * temp * 100.0 * sign;
        ema_s += (vf - ema_s) * alpha_s;
        ema_l += (vf - ema_l) * alpha_l;
        out_tm[t * cols + s] = (float)(ema_s - ema_l);
        prev_hlc = hlc;
        prev_dm = dm;
    }
}

