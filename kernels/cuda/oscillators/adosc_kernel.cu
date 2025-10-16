// CUDA kernels for Chaikin Accumulation/Distribution Oscillator (ADOSC)
//
// Math pattern: recurrence/IIR.
// - First build ADL (Accum/Dist Line) once: prefix sum of MFV where
//   MFM = ((close-low) - (high-close)) / (high-low) and MFV = MFM * volume.
// - Then, for each (short,long) pair, run two EMAs over ADL and subtract.
//
// Semantics to mirror scalar path:
// - No warmup NaNs; ADOSC starts from index 0 with value 0.0 (short==long at t=0).
// - Division by zero in MFM when (high==low) yields 0.0 contribution (keeps ADL).
// - NaNs propagate naturally from inputs via arithmetic; no special masking here.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Build ADL: adl[t] = adl[t-1] + mfm(t) * volume(t)
extern "C" __global__ void adosc_adl_f32(const float* __restrict__ high,
                                          const float* __restrict__ low,
                                          const float* __restrict__ close,
                                          const float* __restrict__ volume,
                                          int series_len,
                                          float* __restrict__ adl_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return; // single thread does the sequential scan
    }
    if (series_len <= 0) return;

    const float h0 = high[0];
    const float l0 = low[0];
    const float c0 = close[0];
    const float v0 = volume[0];
    const float hl0 = h0 - l0;
    const float mfm0 = (hl0 != 0.0f) ? (((c0 - l0) - (h0 - c0)) / hl0) : 0.0f;
    float sum_ad = mfm0 * v0;
    adl_out[0] = sum_ad;

    for (int i = 1; i < series_len; ++i) {
        const float h = high[i];
        const float l = low[i];
        const float c = close[i];
        const float v = volume[i];
        const float hl = h - l;
        const float mfm = (hl != 0.0f) ? (((c - l) - (h - c)) / hl) : 0.0f;
        const float mfv = mfm * v;
        sum_ad += mfv;
        adl_out[i] = sum_ad;
    }
}

// Batch over many (short,long) pairs using a precomputed ADL.
// Each block handles one pair and scans time sequentially.
extern "C" __global__ void adosc_batch_from_adl_f32(const float* __restrict__ adl,
                                                     const int* __restrict__ short_periods,
                                                     const int* __restrict__ long_periods,
                                                     int series_len,
                                                     int n_combos,
                                                     float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return; // single-thread per block does the scan

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) {
        // Leave row as zeros; semantics mirror scalar safeguards (no warmup NaNs).
        return;
    }
    const float a_s = 2.0f / (float)(sp + 1);
    const float a_l = 2.0f / (float)(lp + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    const int base = combo * series_len;
    // i = 0 bootstrap
    float s_ema = adl[0];
    float l_ema = adl[0];
    out[base + 0] = s_ema - l_ema; // 0.0f
    // i = 1..N-1
    for (int i = 1; i < series_len; ++i) {
        const float x = adl[i];
        s_ema = a_s * x + oms * s_ema;
        l_ema = a_l * x + oml * l_ema;
        out[base + i] = s_ema - l_ema;
    }
}

// Many-series × one-param (time-major inputs/outputs)
// Layout: [rows][cols] with time-major (index = t*cols + s)
extern "C" __global__ void adosc_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,    // number of series
    int rows,    // length per series (time)
    int short_p,
    int long_p,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols) return;
    if (threadIdx.x != 0) return;

    if (short_p <= 0 || long_p <= 0 || short_p >= long_p) {
        // do nothing; outputs remain unspecified by caller if not zeroed
        return;
    }

    const float a_s = 2.0f / (float)(short_p + 1);
    const float a_l = 2.0f / (float)(long_p + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    // Build ADL on the fly for this series and compute EMAs
    int idx0 = 0 * cols + s;
    float h0 = high_tm[idx0];
    float l0 = low_tm[idx0];
    float c0 = close_tm[idx0];
    float v0 = volume_tm[idx0];
    float hl0 = h0 - l0;
    float mfm0 = (hl0 != 0.0f) ? (((c0 - l0) - (h0 - c0)) / hl0) : 0.0f;
    float sum_ad = mfm0 * v0;
    float s_ema = sum_ad;
    float l_ema = sum_ad;
    out_tm[idx0] = s_ema - l_ema; // 0.0f

    for (int t = 1; t < rows; ++t) {
        const int idx = t * cols + s;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        const float v = volume_tm[idx];
        const float hl = h - l;
        const float mfm = (hl != 0.0f) ? (((c - l) - (h - c)) / hl) : 0.0f;
        const float mfv = mfm * v;
        sum_ad += mfv;
        const float x = sum_ad;
        s_ema = a_s * x + oms * s_ema;
        l_ema = a_l * x + oml * l_ema;
        out_tm[idx] = s_ema - l_ema;
    }
}

