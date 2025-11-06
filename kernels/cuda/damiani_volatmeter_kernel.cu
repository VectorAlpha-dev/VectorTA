// CUDA kernels for Damiani Volatmeter (dual-volatility: ATR ratio + stddev ratio with lag)
// Optimized for FP32-only math with float-float (float2) prefix sums to avoid FP64 in hot paths.
//
// Math category: recurrence/time-scan per-parameter. Each block handles one
// parameter row (batch) or one series (many-series). We scan sequentially in
// thread 0 to preserve scalar semantics. Standard deviation windows are
// computed from compensated prefix sums (S, SS) represented as float2 (hi,lo)
// with NaN→0 policy to avoid per-thread rings and to reuse work across rows/series.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LDG
#  if __CUDA_ARCH__ >= 350
#    define LDG(p) __ldg(p)
#  else
#    define LDG(p) (*(p))
#  endif
#endif

__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }
__device__ __forceinline__ bool finite_f32(float x){ return isfinite(x); }

// Kahan compensated add for FP32 (used for ATR seeding)
__device__ __forceinline__ void kahan_add(float &sum, float &comp, float x){
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

// ----- float-float (float2) helpers -----
// Error-free transforms for float-float arithmetic (hi,lo) in a float2
// References: Dekker/Muller/Shewchuk (2Sum/Fast2Sum/2Prod)
__device__ __forceinline__ float2 ff_two_sum(float a, float b){
    float s  = a + b;
    float bb = s - a;
    float e  = (a - (s - bb)) + (b - bb);
    return make_float2(s, e);
}

__device__ __forceinline__ float2 ff_add(float2 x, float2 y){
    float2 t = ff_two_sum(x.x, y.x);
    float e  = t.y + x.y + y.y;
    return ff_two_sum(t.x, e);
}

__device__ __forceinline__ float2 ff_neg(float2 x){ return make_float2(-x.x, -x.y); }
__device__ __forceinline__ float2 ff_sub(float2 x, float2 y){ return ff_add(x, ff_neg(y)); }

__device__ __forceinline__ float2 ff_two_prod(float a, float b){
    float p = a * b;
    float e = fmaf(a, b, -p); // exact error term via one FMA
    return make_float2(p, e);
}

__device__ __forceinline__ float2 ff_scale(float2 x, float s){
    // scale both parts and renormalize
    return ff_two_sum(x.x * s, x.y * s);
}

__device__ __forceinline__ float2 ff_mul(float2 x, float2 y){
    // (xh+xl)*(yh+yl) = xh*yh + (xh*yl + xl*yh) + xl*yl
    float2 p  = ff_two_prod(x.x, y.x);
    float cross = x.x * y.y + x.y * y.x;
    float2 s  = ff_two_sum(p.x, cross);
    float e   = p.y + s.y + x.y * y.y;
    return ff_two_sum(s.x, e);
}

__device__ __forceinline__ float ff_to_f32(float2 x){ return x.x + x.y; }

// Guard small/non-finite denominators
__device__ __forceinline__ float safe_pos_den(float x){
    const float EPS = 1.1920929e-7f; // ~FLT_EPSILON
    return (finite_f32(x) && x > 0.0f) ? x : EPS;
}

// Stddev from compensated prefix sums over [t-win+1, t]
__device__ __forceinline__ float std_from_ff_prefix(const float2 s_t, const float2 s_prev,
                                                    const float2 ss_t, const float2 ss_prev,
                                                    int win)
{
    const float inv_n = 1.0f / (float)win;
    const float2 sum   = ff_sub(s_t,  s_prev);
    const float2 sumsq = ff_sub(ss_t, ss_prev);
    const float2 mean   = ff_scale(sum,   inv_n);
    const float2 mean2  = ff_mul(mean, mean);
    const float2 ex2    = ff_scale(sumsq, inv_n);
    const float2 var_ff = ff_sub(ex2, mean2);
    const float var = fmaxf(ff_to_f32(var_ff), 0.0f);
    return sqrtf(var);
}

// One-series × many-params (batch) — close-only path
// Inputs:
//  - prices: length = series_len (time)
//  - first_valid: first non-NaN index in prices
//  - vis_atr/vis_std/sed_atr/sed_std/threshold: length = n_combos
//  - s_prefix/ss_prefix: float2 prefix sums (hi,lo) with NaN→0 policy, length = series_len
//  - tr: precomputed True Range vs previous close (close-only path), length = series_len
// Output layout:
//  - out has 2*n_combos rows stacked: row*2 = vol, row*2+1 = anti; each row has series_len columns
extern "C" __global__
void damiani_volatmeter_batch_f32(const float* __restrict__ prices,
                                  int series_len,
                                  int first_valid,
                                  const int* __restrict__ vis_atr,
                                  const int* __restrict__ vis_std,
                                  const int* __restrict__ sed_atr,
                                  const int* __restrict__ sed_std,
                                  const float* __restrict__ threshold,
                                  int n_combos,
                                  const float2* __restrict__ s_prefix,
                                  const float2* __restrict__ ss_prefix,
                                  const float* __restrict__ tr,
                                  float* __restrict__ out)
{
    const int row = blockIdx.x;
    if (row >= n_combos || series_len <= 0) return;

    const int p_vis_atr = vis_atr[row];
    const int p_vis_std = vis_std[row];
    const int p_sed_atr = sed_atr[row];
    const int p_sed_std = sed_std[row];
    const float th = threshold[row];

    const int needed = max(max(max(p_vis_atr, p_vis_std), max(p_sed_atr, p_sed_std)), 3);
    if (first_valid >= series_len) return;

    const int base_vol  = (row * 2 + 0) * series_len;
    const int base_anti = (row * 2 + 1) * series_len;

    const int warm_end = min(series_len, first_valid + needed - 1);
    if (threadIdx.x != 0) return;

    // Wilder ATR state (FP32) with Kahan seed for initial window
    float atr_vis = NAN, atr_sed = NAN;
    float sum_vis = 0.0f, c_vis = 0.0f;
    float sum_sed = 0.0f, c_sed = 0.0f;

    // Keep lag history in registers; treat history as 0 before first writes
    float vh1 = 0.0f, vh2 = 0.0f, vh3 = 0.0f;
    const float lag_s = 0.5f;

    for (int t = first_valid; t < series_len; ++t) {
        const float tr_t = LDG(&tr[t]);
        const int k = t - first_valid;

        // Wilder ATR for vis
        if (k < p_vis_atr) {
            kahan_add(sum_vis, c_vis, tr_t);
            if (k == p_vis_atr - 1) atr_vis = sum_vis / (float)p_vis_atr;
        } else if (finite_f32(atr_vis)) {
            const float alpha = 1.0f / (float)p_vis_atr;
            atr_vis = fmaf(atr_vis, (1.0f - alpha), tr_t * alpha);
        }

        // Wilder ATR for sed
        if (k < p_sed_atr) {
            kahan_add(sum_sed, c_sed, tr_t);
            if (k == p_sed_atr - 1) atr_sed = sum_sed / (float)p_sed_atr;
        } else if (finite_f32(atr_sed)) {
            const float alpha = 1.0f / (float)p_sed_atr;
            atr_sed = fmaf(atr_sed, (1.0f - alpha), tr_t * alpha);
        }

        // Start outputs once every window is satisfied (includes warm_end)
        if (k >= needed - 1) {
            const float inv_sed = 1.0f / safe_pos_den(atr_sed);
            const float base    = atr_vis * inv_sed;
            const float vol_t   = fmaf(lag_s, (vh1 - vh3), base);
            out[base_vol + t]   = vol_t;
            // update lag ring after writing
            vh3 = vh2; vh2 = vh1; vh1 = vol_t;

            // Anti only when both stddev windows are full
            if (k >= max(p_vis_std, p_sed_std) - 1) {
                const int prev_v = t - p_vis_std;
                const int prev_s = t - p_sed_std;

                const float2 S_t   = s_prefix[t];
                const float2 SS_t  = ss_prefix[t];
                const float2 S_pv  = (prev_v >= 0) ? s_prefix[prev_v]  : make_float2(0.f,0.f);
                const float2 SS_pv = (prev_v >= 0) ? ss_prefix[prev_v] : make_float2(0.f,0.f);
                const float2 S_ps  = (prev_s >= 0) ? s_prefix[prev_s]  : make_float2(0.f,0.f);
                const float2 SS_ps = (prev_s >= 0) ? ss_prefix[prev_s] : make_float2(0.f,0.f);

                const float std_v = std_from_ff_prefix(S_t, S_pv, SS_t, SS_pv, p_vis_std);
                const float std_s = std_from_ff_prefix(S_t, S_ps, SS_t, SS_ps, p_sed_std);

                const float anti_t = th - (std_v / safe_pos_den(std_s));
                out[base_anti + t] = anti_t;
            }
        }
    }
    // Re-assert NaN prefix exactly through warm_end for both outputs
    for (int t = 0; t <= warm_end && t < series_len; ++t) {
        out[base_vol + t] = nan_f32();
        out[base_anti + t] = nan_f32();
    }
}

// Many-series × one-param (time-major). Uses HLC for ATR and close for StdDev.
// Layouts:
//  - high_tm/low_tm/close_tm: length = num_series * series_len, index = t*num_series + series
//  - first_valids: per-series first valid index (based on close only)
//  - s_tm/ss_tm: float2 prefix sums (NaN→0) with same layout/length as inputs
//  - out_tm: 2 matrices stacked (vol then anti): dims = series_len x (2*num_series)
extern "C" __global__
void damiani_volatmeter_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int vis_atr,
    int vis_std,
    int sed_atr,
    int sed_std,
    float threshold,
    const int* __restrict__ first_valids,
    const float2* __restrict__ s_tm,
    const float2* __restrict__ ss_tm,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x;
    if (series >= num_series || series_len <= 0) return;

    const int fv = max(0, first_valids[series]);
    const int needed = max(max(max(vis_atr, vis_std), max(sed_atr, sed_std)), 3);
    const int warm_end = min(series_len, fv + needed - 1);

    const int stride = num_series;
    if (threadIdx.x != 0) return;

    float atr_vis = NAN, atr_sed = NAN;
    float sum_vis = 0.0f, c_vis = 0.0f;
    float sum_sed = 0.0f, c_sed = 0.0f;
    const float lag_s = 0.5f;
    float prev_close = NAN;
    bool have_prev = false;

    // lag history in registers
    float vh1 = 0.0f, vh2 = 0.0f, vh3 = 0.0f;

    for (int t = fv; t < series_len; ++t) {
        const int idx = t * stride + series;
        const int k   = t - fv;
        const float h = LDG(&high_tm[idx]);
        const float l = LDG(&low_tm[idx]);
        const float c = LDG(&close_tm[idx]);

        float tr;
        if (have_prev && finite_f32(c)) {
            const float tr1 = h - l;
            const float tr2 = fabsf(h - prev_close);
            const float tr3 = fabsf(l - prev_close);
            tr = fmaxf(tr1, fmaxf(tr2, tr3));
        } else {
            tr = 0.0f;
        }
        if (finite_f32(c)) { prev_close = c; have_prev = true; }

        // Wilder ATR vis
        if (k < vis_atr) {
            kahan_add(sum_vis, c_vis, tr);
            if (k == vis_atr - 1) atr_vis = sum_vis / (float)vis_atr;
        } else if (finite_f32(atr_vis)) {
            const float alpha = 1.0f / (float)vis_atr;
            atr_vis = fmaf(atr_vis, (1.0f - alpha), tr * alpha);
        }

        // Wilder ATR sed
        if (k < sed_atr) {
            kahan_add(sum_sed, c_sed, tr);
            if (k == sed_atr - 1) atr_sed = sum_sed / (float)sed_atr;
        } else if (finite_f32(atr_sed)) {
            const float alpha = 1.0f / (float)sed_atr;
            atr_sed = fmaf(atr_sed, (1.0f - alpha), tr * alpha);
        }

        if (k >= needed - 1) {
            const int out_row = t * (2 * stride);
            const float inv_sed = 1.0f / safe_pos_den(atr_sed);
            const float base    = atr_vis * inv_sed;
            const float vol_t   = fmaf(lag_s, (vh1 - vh3), base);
            out_tm[out_row + series] = vol_t;

            vh3 = vh2; vh2 = vh1; vh1 = vol_t;

            // Anti when both std windows full
            if (k >= max(vis_std, sed_std) - 1) {
                const int prev_v = t - vis_std;
                const int prev_s = t - sed_std;

                const int pv_idx = (prev_v >= 0) ? (prev_v * stride + series) : -1;
                const int ps_idx = (prev_s >= 0) ? (prev_s * stride + series) : -1;

                const float2 S_t   = s_tm[idx];
                const float2 SS_t  = ss_tm[idx];
                const float2 S_pv  = (pv_idx >= 0) ? s_tm[pv_idx]  : make_float2(0.f,0.f);
                const float2 SS_pv = (pv_idx >= 0) ? ss_tm[pv_idx] : make_float2(0.f,0.f);
                const float2 S_ps  = (ps_idx >= 0) ? s_tm[ps_idx]  : make_float2(0.f,0.f);
                const float2 SS_ps = (ps_idx >= 0) ? ss_tm[ps_idx] : make_float2(0.f,0.f);

                const float std_v = std_from_ff_prefix(S_t, S_pv, SS_t, SS_pv, vis_std);
                const float std_s = std_from_ff_prefix(S_t, S_ps, SS_t, SS_ps, sed_std);
                out_tm[out_row + (stride + series)] = threshold - (std_v / safe_pos_den(std_s));
            }
        }
    }
    // Re-assert NaN prefix exactly through warm_end for this series
    for (int t = 0; t <= warm_end && t < series_len; ++t) {
        out_tm[t * (2 * stride) + series] = nan_f32();
        out_tm[t * (2 * stride) + (stride + series)] = nan_f32();
    }
}
