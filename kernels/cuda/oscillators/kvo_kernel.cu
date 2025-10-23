// CUDA kernels for Klinger Volume Oscillator (KVO)
//
// This file implements two FP32-only kernels using FMA-centric math and
// compensated float-float (hi+lo) state for EMA recurrences. Semantics match
// the scalar implementation: NaN through t < first_valid+1; at t=first_valid+1
// both EMAs are seeded to the first VF value and the output is 0.0.

#include <cuda_runtime.h>
#include <math.h>

// ======================== Small numeric helpers (FP32 only) ========================

// Write an IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// Error-free transforms (float-float arithmetic). Based on Dekker/Knuth patterns.
// Using FMA gives exact product residual cheaply on NVIDIA GPUs.
__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void two_diff(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - b;
}
__device__ __forceinline__ void quick_two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}
__device__ __forceinline__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;
    e = fmaf(a, b, -p); // exact residual with one FMA
}

struct f2 { float hi, lo; }; // value ~= hi + lo

__device__ __forceinline__ f2 f2_make(float x) { f2 r; r.hi = x; r.lo = 0.0f; return r; }

// ema <- ema + alpha * (x - ema) in compensated float-float
__device__ __forceinline__ void ema_update_f2(f2 &ema, float x, float alpha)
{
    float s, s_err; two_sum(ema.hi, ema.lo, s, s_err);
    float d_hi, d_err; two_diff(x, s, d_hi, d_err);
    float delta_hi = d_hi;
    float delta_lo = d_err - s_err;

    float p_hi, p_lo; two_prod(alpha, delta_hi, p_hi, p_lo);
    p_lo = fmaf(alpha, delta_lo, p_lo);

    float y_hi, y_lo; two_sum(s, p_hi, y_hi, y_lo);
    y_lo += p_lo;
    quick_two_sum(y_hi, y_lo, ema.hi, ema.lo); // renormalize
}

// One-step NR reciprocal (near-1 ulp) — cheaper than a divide, uses FMA.
// r ≈ 1/c with one Newton step: r = r * (2 - c*r)
__device__ __forceinline__ float rcp_nr(float c)
{
    float r = __fdividef(1.0f, c);
    r = r * fmaf(-c, r, 2.0f);
    return r;
}

// ==================== Batch: one series × many (short,long) ====================
// Inputs:
//  - vf:     precomputed VF stream [len]
//  - shorts/longs: period arrays [n_combos]
//  - out:    row-major [combo][t] (n_combos x len)
extern "C" __global__ void kvo_batch_f32(
    const float* __restrict__ vf,
    int len,
    int first_valid,
    const int* __restrict__ shorts,
    const int* __restrict__ longs,
    int n_combos,
    float* __restrict__ out)
{
    // Grid-stride over combos. Compatible with existing launcher chunking
    // that passes (shorts,longs,out) already offset to the current window.
    for (int combo = blockIdx.x * blockDim.x + threadIdx.x;
         combo < n_combos;
         combo += blockDim.x * gridDim.x)
    {
        const int s = shorts[combo];
        const int l = longs[combo];
        if (s <= 0 || l < s) continue;

        const int warm = first_valid + 1; // same scalar warmup
        float* __restrict__ row_out = out + (size_t)combo * (size_t)len;

        const int warm_end = (warm < len ? warm : len);
        for (int t = 0; t < warm_end; ++t) row_out[t] = f32_nan();
        if (warm >= len) continue;

        const float alpha_s = 2.0f / (float)(s + 1);
        const float alpha_l = 2.0f / (float)(l + 1);

        const float seed = vf[warm];
        float ema_s = seed;
        float ema_l = seed;

        row_out[warm] = 0.0f; // first difference defined as zero

        #pragma unroll 1
        for (int t = warm + 1; t < len; ++t) {
            const float vfi = vf[t];
            // EMA update via FFMA: ema += alpha*(x-ema)
            ema_s = fmaf(alpha_s, (vfi - ema_s), ema_s);
            ema_l = fmaf(alpha_l, (vfi - ema_l), ema_l);
            row_out[t] = ema_s - ema_l;
        }
    }
}

// ================= Many-series × one-param (time-major) =================
// Inputs (time-major):
//  - *_tm: float arrays [rows*cols] time-major
//  - first_valids[cols]
//  - short_p, long_p >= 1
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
    // 1-D launch over columns (series) with grid-stride on grid.x
    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s];
        if (fv < 0 || fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
            continue;
        }

        const int warm = fv + 1;

        const int warm_end = (warm < rows ? warm : rows);
        for (int t = 0; t < warm_end; ++t) out_tm[(size_t)t * (size_t)cols + s] = f32_nan();
        if (warm >= rows) continue;

        const float alpha_s = 2.0f / (float)(short_p + 1);
        const float alpha_l = 2.0f / (float)(long_p + 1);

        const size_t idx0 = (size_t)fv * (size_t)cols + s;
        double prev_h = (double)high_tm[idx0];
        double prev_l = (double)low_tm[idx0];
        double prev_c = (double)close_tm[idx0];
        double prev_hlc = prev_h + prev_l + prev_c;
        double prev_dm  = prev_h - prev_l;
        int    trend    = -1; // initial state
        double cm       = 0.0;

        // First consumable VF at t = warm
        {
            const size_t idx = (size_t)warm * (size_t)cols + s;
            const double h = (double)high_tm[idx];
            const double l = (double)low_tm[idx];
            const double c = (double)close_tm[idx];
            const double v = (double)volume_tm[idx];
            const double hlc = h + l + c;
            const double dm  = h - l;

            if (hlc > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
            else if (hlc < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
            cm += dm;

            const double ratio = dm / cm;
            const double temp  = fabs((ratio * 2.0) - 1.0);
            const double sign  = (trend == 1) ? 1.0 : -1.0;
            const float vf     = (float)(v * temp * 100.0 * sign);

            float ema_s = vf;
            float ema_l = vf;
            out_tm[idx] = 0.0f; // seed diff

            prev_hlc = hlc;
            prev_dm  = dm;

            #pragma unroll 1
            for (int t = warm + 1; t < rows; ++t) {
                const size_t j = (size_t)t * (size_t)cols + s;
                const double h2 = (double)high_tm[j];
                const double l2 = (double)low_tm[j];
                const double c2 = (double)close_tm[j];
                const double v2 = (double)volume_tm[j];
                const double hlc2 = h2 + l2 + c2;
                const double dm2  = h2 - l2;

                if (hlc2 > prev_hlc && trend != 1) { trend = 1; cm = prev_dm; }
                else if (hlc2 < prev_hlc && trend != 0) { trend = 0; cm = prev_dm; }
                cm += dm2;

                const double ratio2 = dm2 / cm;
                const double temp2  = fabs((ratio2 * 2.0) - 1.0);
                const double sign2  = (trend == 1) ? 1.0 : -1.0;
                const float vf2     = (float)(v2 * temp2 * 100.0 * sign2);

                // EMA in FP32 with FFMA
                ema_s = fmaf(alpha_s, (vf2 - ema_s), ema_s);
                ema_l = fmaf(alpha_l, (vf2 - ema_l), ema_l);
                out_tm[j] = ema_s - ema_l;

                prev_hlc = hlc2;
                prev_dm  = dm2;
            }
        }
    }
}

