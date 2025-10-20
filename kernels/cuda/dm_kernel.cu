// CUDA kernels for Directional Movement (DM)
// Optimized: FP32 I/O, FP32 compute with compensated updates (no FP64).
// - Warmup: compensated summation (Kahan/Neumaier).
// - Smoothing: compensated EMA using FMA-based product error recovery.
// - NaN prefix fill only (avoid redundant stores).
// - __restrict__ and (optionally) __ldg for read-only loads.
//
// References:
//   - Ogita, Rump, Oishi (2005): Error-free transformations; TwoSum/TwoProductFMA.
//   - CUDA Math API (__fmaf_rn) & Programming Guide (read-only cache, restrict).
//
// Semantics preserved from original:
// - FP32 IO
// - Division-free (no ATR)
// - Inputs may contain NaNs before first_valid
// - Write NaN before warm index; write first values at warm index, then smooth forward

#include <cuda_runtime.h>
#include <math.h>

#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif

// ---- Small device utilities -------------------------------------------------

// Canonical quiet-NaN bit pattern (IEEE 754).
__device__ __forceinline__ float qnan() {
    return __int_as_float(0x7fc00000);
}

// Read-only cached load helper. Safe because inputs are const.
// Modern toolchains often infer the read-only path automatically from const+restrict,
// but __ldg can still be used explicitly without extra build options.
template <typename T>
__device__ __forceinline__ T ro_load(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Fill first `len` entries of a row/column with NaN (prefix only).
__device__ __forceinline__ void fill_nan_prefix(float* ptr, int len) {
    const float nanv = qnan();
    for (int i = 0; i < len; ++i) ptr[i] = nanv;
}

// Compute per-step directional move (branchless / predicated).
__device__ __forceinline__ void dm_step(float ch, float cl, float& prev_h, float& prev_l,
                                        float& plus_val, float& minus_val)
{
    const float dp = ch - prev_h;
    const float dm = prev_l - cl;
    prev_h = ch;
    prev_l = cl;

    const float ap = (dp > 0.0f) ? dp : 0.0f;
    const float am = (dm > 0.0f) ? dm : 0.0f;

    // Keep only the larger positive delta
    const bool take_p = (ap > am);
    plus_val  = take_p ? ap : 0.0f;
    minus_val = take_p ? 0.0f : am;
}

// Compensated adder for warmup accumulation (Kahan/Neumaier).
struct CompSum {
    float s;   // running sum
    float c;   // compensation (low part)
    __device__ __forceinline__ void init() { s = 0.0f; c = 0.0f; }
    __device__ __forceinline__ void add(float x) {
        // Kahan/Neumaier style compensated addition
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};

// Compensated EMA update: s = s*(1 - rp) + x, with product error recovery.
// Uses FMA to obtain exact rounding error of the product in one extra op.
struct CompEMA {
    float s;  // running state (high)
    float c;  // compensation (low)
    __device__ __forceinline__ void init(float s0) { s = s0; c = 0.0f; }
    __device__ __forceinline__ void update(float one_minus_rp, float x) {
        // prod = s*(1-rp), perr = exact rounding error of the product (TwoProductFMA)
        float prod = s * one_minus_rp;
        float perr = __fmaf_rn(s, one_minus_rp, -prod);
        // Now do compensated add of (x + perr)
        float y = (x + perr) - c;
        float t = prod + y;
        c = (t - prod) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};

// ---- Kernel 1: one price series, many period combos (row-major outputs) -----

extern "C" __global__
void dm_batch_f32(const float* __restrict__ high,
                  const float* __restrict__ low,
                  const int*   __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ plus_out,
                  float* __restrict__ minus_out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* plus_row  = plus_out  + combo * series_len;
    float* minus_row = minus_out + combo * series_len;

    const int p = periods[combo];
    if (p <= 0) {
        // Fill entire row with NaN since no valid outputs exist for this combo
        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }
    if (first_valid < 0 || first_valid + p - 1 >= series_len) {
        fill_nan_prefix(plus_row, series_len);
        fill_nan_prefix(minus_row, series_len);
        return;
    }

    const int i0 = first_valid;
    const int warm_end = i0 + p - 1; // index where first outputs are written

    // Only the prefix [0, warm_end) must be NaN by semantics.
    if (warm_end > 0) {
        fill_nan_prefix(plus_row,  warm_end);
        fill_nan_prefix(minus_row, warm_end);
    }

    // Initialize from first valid price
    float prev_h = ro_load(high + i0);
    float prev_l = ro_load(low  + i0);

    // Warmup accumulation over (p - 1) steps with compensated summation
    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int i = i0 + 1; i <= warm_end; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }

    // First outputs at warm_end
    plus_row [warm_end] = wplus.value();
    minus_row[warm_end] = wminus.value();

    // Early out if no smoothing steps remain
    if (warm_end + 1 >= series_len) return;

    const float rp = 1.0f / (float)p;
    const float one_minus_rp = 1.0f - rp;

    // Compensated EMA forward
    CompEMA splus, sminus;
    splus.init(plus_row [warm_end]);
    sminus.init(minus_row[warm_end]);

    for (int i = warm_end + 1; i < series_len; ++i) {
        const float ch = ro_load(high + i);
        const float cl = ro_load(low  + i);

        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_row [i] = splus.value();
        minus_row[i] = sminus.value();
    }
}

// ---- Kernel 2: many series (columns) with one period, time-major layout -----

extern "C" __global__
void dm_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ plus_tm,
    float* __restrict__ minus_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index (column)
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (period <= 0 || fv < 0 || fv + period - 1 >= rows) {
        // Entire column is NaN
        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            plus_tm [idx] = qnan();
            minus_tm[idx] = qnan();
        }
        return;
    }

    // Helper to compute flat index in time-major arrays
    auto at = [&](int t) { return t * cols + s; };

    const int i0 = fv;
    const int warm_end = i0 + period - 1;

    // Only fill the prefix that must be NaN
    for (int t = 0; t < warm_end; ++t) {
        const int idx = at(t);
        plus_tm [idx] = qnan();
        minus_tm[idx] = qnan();
    }

    float prev_h = ro_load(high_tm + at(i0));
    float prev_l = ro_load(low_tm  + at(i0));

    // Warmup with compensated sums
    CompSum wplus, wminus; wplus.init(); wminus.init();
    for (int t = i0 + 1; t <= warm_end; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);
        if (pv != 0.0f) wplus.add(pv);
        if (mv != 0.0f) wminus.add(mv);
    }

    plus_tm [at(warm_end)] = wplus.value();
    minus_tm[at(warm_end)] = wminus.value();

    if (warm_end + 1 >= rows) return;

    const float rp = 1.0f / (float)period;
    const float one_minus_rp = 1.0f - rp;

    CompEMA splus, sminus;
    splus.init(plus_tm [at(warm_end)]);
    sminus.init(minus_tm[at(warm_end)]);

    for (int t = warm_end + 1; t < rows; ++t) {
        const float ch = ro_load(high_tm + at(t));
        const float cl = ro_load(low_tm  + at(t));
        float pv, mv;
        dm_step(ch, cl, prev_h, prev_l, pv, mv);

        splus.update(one_minus_rp, pv);
        sminus.update(one_minus_rp, mv);

        plus_tm [at(t)] = splus.value();
        minus_tm[at(t)] = sminus.value();
    }
}
