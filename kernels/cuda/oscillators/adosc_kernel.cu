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

// --- Small helpers (header-local) --------------------------------------------

struct KahanF32 {
    float sum;
    float c;
};

__device__ __forceinline__ void kahan_add(KahanF32& s, float x) {
    // Neumaier form: stable and cheap
    float y = x - s.c;
    float t = s.sum + y;
    s.c = (t - s.sum) - y;
    s.sum = t;
}

// mfm = ((close-low) - (high-close)) / (high-low)
// Use the original algebra to match CPU/scalar numerics more closely.
__device__ __forceinline__ float mfm_from_hlc(float h, float l, float c) {
    const float hl = h - l;
    if (hl == 0.0f) return 0.0f;
    const float num = (c - l) - (h - c);
    return num / hl;
}

// -----------------------------------------------------------------------------
// Build ADL: adl[t] = adl[t-1] + mfm(t) * volume(t)
// - Sequential by nature, but use compensated summation to avoid drift.
// -----------------------------------------------------------------------------
extern "C" __global__ void adosc_adl_f32(const float* __restrict__ high,
                                         const float* __restrict__ low,
                                         const float* __restrict__ close,
                                         const float* __restrict__ volume,
                                         int series_len,
                                         float* __restrict__ adl_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;   // single-thread scan is fine here
    if (series_len <= 0) return;

    // t = 0 bootstrap
    const float mfm0 = mfm_from_hlc(high[0], low[0], close[0]);
    KahanF32 acc { mfm0 * volume[0], 0.0f };
    adl_out[0] = acc.sum;

    // t = 1..N-1
    for (int i = 1; i < series_len; ++i) {
        const float mfv = mfm_from_hlc(high[i], low[i], close[i]) * volume[i];
        kahan_add(acc, mfv);
        adl_out[i] = acc.sum;
    }
}

// -----------------------------------------------------------------------------
// One price series × many (short,long) pairs from precomputed ADL.
// - Primary, optimized path: one *thread* per parameter pair.
// - Threads advance time in lock-step; each warp broadcasts adl[i] implicitly.
// - Leaves rows untouched for invalid (sp,lp) to match your current semantics.
// -----------------------------------------------------------------------------
extern "C" __global__ void adosc_batch_from_adl_f32(const float* __restrict__ adl,
                                                    const int*   __restrict__ short_periods,
                                                    const int*   __restrict__ long_periods,
                                                    int series_len,
                                                    int n_combos,
                                                    float* __restrict__ out)
{
    if (series_len <= 0) return;

    // One warp per combo. Each lane advances 1 timestep; warp scan emits 32 outputs per iteration.
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const int combo = (int)(blockIdx.x * warps_per_block + warp);
    if (combo >= n_combos) return;

    const int sp = short_periods[combo];
    const int lp = long_periods[combo];
    if (sp <= 0 || lp <= 0 || sp >= lp) {
        // Semantics: leave row as-is (caller validates in practice)
        return;
    }

    const float a_s = 2.0f / (float)(sp + 1);
    const float a_l = 2.0f / (float)(lp + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    float* out_row = out + (size_t)combo * (size_t)series_len;

    // t=0: both EMAs are seeded with ADL[0], so ADOSC = 0
    if (lane == 0) out_row[0] = 0.0f;
    float s_ema = adl[0];
    float l_ema = adl[0];

    const unsigned mask = 0xffffffffu;

    // Rolling update: process 32 timesteps per iteration, starting at t=1
    for (int t0 = 1; t0 < series_len; t0 += 32) {
        const int t = t0 + (int)lane;

        // EMA recurrence: y = (1-a)*y + a*x => y = A*y + B
        float As = 1.0f;
        float Bs = 0.0f;
        float Al = 1.0f;
        float Bl = 0.0f;
        if (t < series_len) {
            const float x = adl[t];
            As = oms;
            Bs = a_s * x;
            Al = oml;
            Bl = a_l * x;
        }

        // Inclusive warp scan composing (A,B) left-to-right.
        // Composition: (A1,B1) o (A2,B2) = (A1*A2, A1*B2 + B1).
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float As_prev = __shfl_up_sync(mask, As, offset);
            const float Bs_prev = __shfl_up_sync(mask, Bs, offset);
            const float Al_prev = __shfl_up_sync(mask, Al, offset);
            const float Bl_prev = __shfl_up_sync(mask, Bl, offset);
            if (lane >= (unsigned)offset) {
                const float As_cur = As;
                const float Bs_cur = Bs;
                const float Al_cur = Al;
                const float Bl_cur = Bl;
                As = As_cur * As_prev;
                Bs = __fmaf_rn(As_cur, Bs_prev, Bs_cur);
                Al = Al_cur * Al_prev;
                Bl = __fmaf_rn(Al_cur, Bl_prev, Bl_cur);
            }
        }

        const float ys = __fmaf_rn(As, s_ema, Bs);
        const float yl = __fmaf_rn(Al, l_ema, Bl);

        if (t < series_len) {
            out_row[t] = ys - yl;
        }

        // Advance to next tile using the last valid lane.
        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        s_ema = __shfl_sync(mask, ys, last_lane);
        l_ema = __shfl_sync(mask, yl, last_lane);
    }
}

// -----------------------------------------------------------------------------
// Many series × one (short,long) pair (time-major layout).
// - Threads are mapped to series with a grid-stride loop.
// - ADL is built per-series with Kahan; EMA uses FMA.
// -----------------------------------------------------------------------------
extern "C" __global__ void adosc_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,    // number of series
    int rows,    // length per series (time)
    int short_p,
    int long_p,
    float* __restrict__ out_tm)
{
    if (short_p <= 0 || long_p <= 0 || short_p >= long_p) return;
    if (rows <= 0 || cols <= 0) return;

    const float a_s = 2.0f / (float)(short_p + 1);
    const float a_l = 2.0f / (float)(long_p + 1);
    const float oms = 1.0f - a_s;
    const float oml = 1.0f - a_l;

    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    // Grid-stride over series
    for (int s = tid; s < cols; s += totalThreads) {
        int idx0 = /* t=0 */ 0 * cols + s;
        const float mfm0 = mfm_from_hlc(high_tm[idx0], low_tm[idx0], close_tm[idx0]);
        KahanF32 acc { mfm0 * volume_tm[idx0], 0.0f };

        float s_ema = acc.sum;
        float l_ema = acc.sum;
        out_tm[idx0] = 0.0f;  // short==long at t=0

        for (int t = 1; t < rows; ++t) {
            const int idx = t * cols + s;
            const float mfv = mfm_from_hlc(high_tm[idx], low_tm[idx], close_tm[idx]) * volume_tm[idx];
            kahan_add(acc, mfv);
            const float x = acc.sum;
            s_ema = fmaf(a_s, x, oms * s_ema);
            l_ema = fmaf(a_l, x, oml * l_ema);
            out_tm[idx] = s_ema - l_ema;
        }
    }
}

