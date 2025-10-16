// CUDA kernels for Damiani Volatmeter (dual-volatility: ATR ratio + stddev ratio with lag)
//
// Math category: recurrence/time-scan per-parameter. Each block handles one
// parameter row (batch) or one series (many-series). We scan sequentially in
// thread 0 to preserve scalar semantics. Standard deviation windows are
// computed from precomputed prefix sums (S, SS) with NaN→0 policy to avoid
// per-thread ring buffers and to reuse work across rows/series.

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

// One-series × many-params (batch) — close-only path
// Inputs:
//  - prices: length = series_len (time)
//  - first_valid: first non-NaN index in prices
//  - vis_atr/vis_std/sed_atr/sed_std/threshold: length = n_combos
//  - s_prefix/ss_prefix: f64 prefix sums of prices (NaN→0 policy), length = series_len
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
                                  const double* __restrict__ s_prefix,
                                  const double* __restrict__ ss_prefix,
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

    // ATR Wilder seeds and running values
    double atr_vis = NAN;
    double atr_sed = NAN;
    double sum_vis = 0.0;
    double sum_sed = 0.0;
    const double vis_atr_f = (double)p_vis_atr;
    const double sed_atr_f = (double)p_sed_atr;

    // Lag history for vol (p1 and p3). We build from actual output values
    // to preserve exact dependency.
    float vh1 = NAN, vh2 = NAN, vh3 = NAN;
    const float lag_s = 0.5f;

    bool have_prev = false; float prev_c = NAN;
    for (int t = first_valid; t < series_len; ++t) {
        const float c = LDG(&prices[t]);
        float tr_t;
        if (have_prev && isfinite(c)) { tr_t = fabsf(c - prev_c); } else { tr_t = 0.0f; }
        if (isfinite(c)) { prev_c = c; have_prev = true; }
        const int k = t - first_valid; // relative index since first_valid

        // Wilder ATR for vis
        if (k < p_vis_atr) {
            sum_vis += (double)tr_t;
            if (k == p_vis_atr - 1) {
                atr_vis = (sum_vis / vis_atr_f);
            }
        } else if (!isnan(atr_vis)) {
            // atr = ((n-1)*atr + tr) / n
            atr_vis = (((vis_atr_f - 1.0) * atr_vis) + (double)tr_t) / vis_atr_f;
        }

        // Wilder ATR for sed
        if (k < p_sed_atr) {
            sum_sed += (double)tr_t;
            if (k == p_sed_atr - 1) {
                atr_sed = (sum_sed / sed_atr_f);
            }
        } else if (!isnan(atr_sed)) {
            atr_sed = (((sed_atr_f - 1.0) * atr_sed) + (double)tr_t) / sed_atr_f;
        }

        // Start outputs once every window is satisfied (includes warm_end)
        if (k >= needed - 1) {
            const float p1 = (t >= 1 && !isnan(out[base_vol + t - 1])) ? out[base_vol + t - 1] : 0.0f;
            const float p3 = (t >= 3 && !isnan(out[base_vol + t - 3])) ? out[base_vol + t - 3] : 0.0f;
            const double sed_safe = (!isnan(atr_sed) && atr_sed != 0.0) ? atr_sed : (atr_sed + 1.1920929e-7);

            const float vol_t = (float)((atr_vis / sed_safe) + (double)lag_s * (double)(p1 - p3));
            out[base_vol + t] = vol_t;
            vh3 = vh2; vh2 = vh1; vh1 = vol_t;

            // Anti only when both stddev windows are full
            if (k >= max(p_vis_std, p_sed_std) - 1) {
                const int prev_v = t - p_vis_std;
                const int prev_s = t - p_sed_std;
                // S, SS are prefix sums of (NaN→0) price
                double sum_v  = LDG(&s_prefix[t]);
                double sum_v2 = LDG(&ss_prefix[t]);
                double sum_s_  = sum_v;
                double sum_s2_ = sum_v2;
                if (prev_v >= 0) {
                    sum_v  -= LDG(&s_prefix[prev_v]);
                    sum_v2 -= LDG(&ss_prefix[prev_v]);
                }
                if (prev_s >= 0) {
                    sum_s_  -= LDG(&s_prefix[prev_s]);
                    sum_s2_ -= LDG(&ss_prefix[prev_s]);
                }
                const double mean_v = sum_v / (double)p_vis_std;
                const double var_v  = (sum_v2 / (double)p_vis_std) - mean_v * mean_v;
                const float std_v   = (float)sqrt(fmax(var_v, 0.0));

                const double mean_s = sum_s_ / (double)p_sed_std;
                const double var_s  = (sum_s2_ / (double)p_sed_std) - mean_s * mean_s;
                const float std_s   = (float)sqrt(fmax(var_s, 0.0));

                const float den = (std_s != 0.0f) ? std_s : (std_s + 1.1920929e-7f);
                const float anti_t = th - (std_v / den);
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
//  - s_tm/ss_tm: f64 prefix sums of close (NaN→0) with same layout/length as inputs
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
    const double* __restrict__ s_tm,
    const double* __restrict__ ss_tm,
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
    double sum_vis = 0.0, sum_sed = 0.0;
    const double vis_atr_f = (double)vis_atr;
    const double sed_atr_f = (double)sed_atr;
    const float lag_s = 0.5f;
    float prev_close = NAN;
    bool have_prev = false;

    for (int t = fv; t < series_len; ++t) {
        const int idx = t * stride + series;
        const int k = t - fv; // relative index since first_valid
        const float h = LDG(&high_tm[idx]);
        const float l = LDG(&low_tm[idx]);
        const float c = LDG(&close_tm[idx]);

        float tr;
        if (have_prev && isfinite(c)) {
            const float tr1 = h - l;
            const float tr2 = fabsf(h - prev_close);
            const float tr3 = fabsf(l - prev_close);
            tr = fmaxf(tr1, fmaxf(tr2, tr3));
        } else {
            tr = 0.0f;
        }
        if (isfinite(c)) { prev_close = c; have_prev = true; }

        if (k < vis_atr) {
            sum_vis += (double)tr;
            if (k == vis_atr - 1) atr_vis = (float)(sum_vis / vis_atr_f);
        } else if (!isnan(atr_vis)) {
            atr_vis = (float)((((vis_atr_f - 1.0) * (double)atr_vis) + (double)tr) / vis_atr_f);
        }

        if (k < sed_atr) {
            sum_sed += (double)tr;
            if (k == sed_atr - 1) atr_sed = (float)(sum_sed / sed_atr_f);
        } else if (!isnan(atr_sed)) {
            atr_sed = (float)((((sed_atr_f - 1.0) * (double)atr_sed) + (double)tr) / sed_atr_f);
        }

        if (k >= needed - 1) {
            const int out_row = t * (2 * stride);
            const float p1 = (t >= 1 && !isnan(out_tm[(t - 1) * (2 * stride) + series]))
                                 ? out_tm[(t - 1) * (2 * stride) + series]
                                 : 0.0f;
            const float p3 = (t >= 3 && !isnan(out_tm[(t - 3) * (2 * stride) + series]))
                                 ? out_tm[(t - 3) * (2 * stride) + series]
                                 : 0.0f;
            const double sed_safe = (!isnan(atr_sed) && atr_sed != 0.0) ? atr_sed : (atr_sed + 1.1920929e-7);
            const float vol_t = (float)((atr_vis / sed_safe) + (double)lag_s * (double)(p1 - p3));
            out_tm[out_row + series] = vol_t;

            // Anti when both std windows full
            if (k >= max(vis_std, sed_std) - 1) {
                const int prev_v = t - vis_std;
                const int prev_s = t - sed_std;
                double sum_v  = LDG(&s_tm[idx]);
                double sum_v2 = LDG(&ss_tm[idx]);
                double sum_s_  = sum_v;
                double sum_s2_ = sum_v2;
                if (prev_v >= 0) {
                    const int pv_idx = prev_v * stride + series;
                    sum_v  -= LDG(&s_tm[pv_idx]);
                    sum_v2 -= LDG(&ss_tm[pv_idx]);
                }
                if (prev_s >= 0) {
                    const int ps_idx = prev_s * stride + series;
                    sum_s_  -= LDG(&s_tm[ps_idx]);
                    sum_s2_ -= LDG(&ss_tm[ps_idx]);
                }
                const double mean_v = sum_v / (double)vis_std;
                const double var_v  = (sum_v2 / (double)vis_std) - mean_v * mean_v;
                const float std_v   = (float)sqrt(fmax(var_v, 0.0));
                const double mean_s = sum_s_ / (double)sed_std;
                const double var_s  = (sum_s2_ / (double)sed_std) - mean_s * mean_s;
                const float std_s   = (float)sqrt(fmax(var_s, 0.0));
                const float den = (std_s != 0.0f) ? std_s : (std_s + 1.1920929e-7f);
                out_tm[out_row + (stride + series)] = threshold - (std_v / den);
            }
        }
    }
    // Re-assert NaN prefix exactly through warm_end for this series
    for (int t = 0; t <= warm_end && t < series_len; ++t) {
        out_tm[t * (2 * stride) + series] = nan_f32();
        out_tm[t * (2 * stride) + (stride + series)] = nan_f32();
    }
}
