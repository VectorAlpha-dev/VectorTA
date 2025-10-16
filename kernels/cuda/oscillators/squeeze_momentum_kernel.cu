// CUDA kernels for Squeeze Momentum Indicator (SMI)
//
// One-series × many-params (batch): each block processes one combo (row),
// single thread performs the sequential time scan, other threads init NaNs.
// Many-series × one-param (time-major): each thread handles one series.
//
// Semantics match src/indicators/squeeze_momentum.rs classic scalar path:
// - Warmup indices:
//     warm_sq = max(lbb, lkc) - 1
//     warm_mo = lkc - 1
//     warm_si = warm_mo + 1
//   Write NaN prior to warmups for respective outputs.
// - Squeeze classification uses BB mean/std (SMA + stddev) vs KC mid +/- mkc*ATR
// - RAW = close - 0.25*(highest+lowest) - 0.5*kc_sma
// - Momentum = last fitted value from OLS of RAW over window [t-lkc+1..t]
//   with x in [1..lkc]. We compute S0 = sum(raw), S1 = sum(j*raw) each step
//   using a simple O(lkc) loop for clarity. (Correctness-first; performance
//   acceptable for moderate lkc and batch sizes.)
// - Signal (lagged 1): ±1 accelerating, ±2 decelerating, mirrors scalar logic.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef SMI_QNAN_F
#define SMI_QNAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }

// Compute TR at index i given high/low/close, matching scalar logic
static __device__ __forceinline__ double true_range_idx(
    int i, const float* __restrict__ high, const float* __restrict__ low, const float* __restrict__ close
) {
    const double h = (double)high[i];
    const double l = (double)low[i];
    if (i == 0) {
        return fabs(h - l);
    } else {
        const double pc = (double)close[i - 1];
        const double tr1 = fabs(h - l);
        const double tr2 = fabs(h - pc);
        const double tr3 = fabs(l - pc);
        return fmax(fmax(tr1, tr2), tr3);
    }
}

extern "C" __global__ void squeeze_momentum_batch_f32(
    // Inputs (one series)
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ lbb_arr,
    const float* __restrict__ mbb_arr,
    const int*   __restrict__ lkc_arr,
    const float* __restrict__ mkc_arr,
    int series_len,
    int n_combos,
    int first_valid,
    // Outputs (row-major)
    float* __restrict__ out_sq,
    float* __restrict__ out_mo,
    float* __restrict__ out_si
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Parallel prefill with NaNs; single thread performs the scan
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_sq[base + i] = SMI_QNAN_F;
        out_mo[base + i] = SMI_QNAN_F;
        out_si[base + i] = SMI_QNAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const int lbb = lbb_arr[combo];
    const double mbb = (double)mbb_arr[combo];
    const int lkc = lkc_arr[combo];
    const double mkc = (double)mkc_arr[combo];

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(lbb <= 0 || lkc <= 0)) return;

    const int warm_sq = max(lbb, lkc) - 1;
    const int warm_mo = lkc - 1;
    const int warm_si = warm_mo + 1;

    // Precompute constants for BB and OLS
    const double inv_lbb = 1.0 / (double)lbb;
    const double inv_lkc = 1.0 / (double)lkc;
    const double p = (double)lkc;
    const double sum_x  = 0.5 * p * (p + 1.0);
    const double sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    const double denom  = p * sum_x2 - sum_x * sum_x;
    const double inv_den= 1.0 / denom;
    const double x_last_minus_xbar = p - sum_x * inv_lkc; // (p - (p+1)/2)

    // Rolling sums for BB and KC/TR
    int start_bb = first_valid + lbb - 1;
    int start_kc = first_valid + lkc - 1;
    if (start_bb < series_len) {
        double sum_bb = 0.0, sumsq_bb = 0.0;
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = (double)close[j];
            sum_bb += v;
            sumsq_bb = fma(v, v, sumsq_bb);
        }
        // We'll keep sum_bb/sumsq_bb updated while scanning
        // To avoid re-doing warmup work, we set i to start_bb-1 and advance once before use
    }
    double sum_bb = 0.0, sumsq_bb = 0.0;
    if (start_bb < series_len) {
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = (double)close[j];
            sum_bb += v;
            sumsq_bb = fma(v, v, sumsq_bb);
        }
    }
    double sum_kc = 0.0, sum_tr = 0.0;
    if (start_kc < series_len) {
        for (int j = start_kc + 1 - lkc; j <= start_kc; ++j) {
            sum_kc += (double)close[j];
            sum_tr += true_range_idx(j, high, low, close);
        }
    }

    // Evaluate seed finiteness semantics to mirror host SMA/stddev builders
    bool bb_seed_ok = true; for (int j = 0; j < lbb && j < series_len; ++j) { if (!is_finite_f(close[j])) { bb_seed_ok = false; break; } }
    bool kc_seed_ok = true; for (int j = 0; j < lkc && j < series_len; ++j) { if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { kc_seed_ok = false; break; } }

    // Sequential scan
    double prev_momentum = NAN;
    for (int i = first_valid; i < series_len; ++i) {
        // Advance rolling sums when past first full windows
        if (i > start_bb) {
            const double c_new = (double)close[i];
            const double c_old = (double)close[i - lbb];
            sum_bb += c_new - c_old;
            sumsq_bb = fma(c_new, c_new, sumsq_bb - c_old * c_old);
        }
        if (i > start_kc) {
            const double c_new = (double)close[i];
            const double c_old = (double)close[i - lkc];
            sum_kc += c_new - c_old;
            // TR new/old
            const double tr_new = true_range_idx(i, high, low, close);
            const int old_idx = i - lkc;
            const double tr_old = true_range_idx(old_idx, high, low, close);
            sum_tr += tr_new - tr_old;
        }

        // Squeeze classification (requires both windows and warmup)
        if (i >= start_bb && i >= start_kc && i >= warm_sq) {
            // Additional finiteness checks to mirror CPU's precomputed SMA/stddev semantics:
            // - Seed windows at the head of the series must be fully finite for SMA/STDDEV to ever become finite.
            if (bb_seed_ok && kc_seed_ok) {
            const double mean_bb = sum_bb * inv_lbb;
            const double var_bb = fma(sumsq_bb * inv_lbb, 1.0, -mean_bb * mean_bb);
            const double dev_bb = sqrt(fmax(0.0, var_bb));
            const double upper_bb = mean_bb + mbb * dev_bb;
            const double lower_bb = mean_bb - mbb * dev_bb;

            const double kc_mid = sum_kc * inv_lkc;
            const double tr_avg = sum_tr * inv_lkc;
            const double upper_kc = kc_mid + mkc * tr_avg;
            const double lower_kc = kc_mid - mkc * tr_avg;

            const bool on  = (lower_bb > lower_kc) && (upper_bb < upper_kc);
            const bool off = (lower_bb < lower_kc) && (upper_bb > upper_kc);
            out_sq[base + i] = on ? -1.0f : (off ? 1.0f : 0.0f);
            } else {
                out_sq[base + i] = SMI_QNAN_F;
            }
        }

        // RAW and momentum when KC window ready
        if (i >= start_kc && kc_seed_ok) {
            // Highest/Lowest over current KC window (naive O(lkc) scan for correctness)
            double highest = -INFINITY, lowest = INFINITY;
            const int win_start = i - lkc + 1;
            for (int j = win_start; j <= i; ++j) {
                const double h = (double)high[j];
                const double l = (double)low[j];
                if (h > highest) highest = h;
                if (l < lowest) lowest = l;
            }
            const double kc_mid = sum_kc * inv_lkc;
            const double c_i = (double)close[i];
            const double raw_i = c_i - 0.25 * (highest + lowest) - 0.5 * kc_mid;

            // Compute S0, S1 over RAW window [i-lkc+1..i]
            double S0 = 0.0, S1 = 0.0;
            double j = 1.0;
            for (int t = win_start; t <= i; ++t, j += 1.0) {
                const double y = (double)close[t] - 0.25 * ((double)high[t] + (double)low[t]) - 0.5 * (sum_kc /* placeholder */);
                // The placeholder above is wrong if used; compute RAW properly per t with its kc_mid_t.
            }
            // For correctness we recompute RAW per t using a local SMA of close over [t-lkc+1..t].
            // This costs O(lkc^2) if done naively; instead, compute RAW series for the window once:
            S0 = 0.0; S1 = 0.0; j = 1.0;
            for (int t = win_start; t <= i; ++t, j += 1.0) {
                // kc_mid at t: SMA(close[t-lkc+1..t])
                double sum_c = 0.0;
                const int ts = t - lkc + 1;
                for (int u = ts; u <= t; ++u) sum_c += (double)close[u];
                const double kc_mid_t = sum_c * inv_lkc;
                // Highest/lowest at t (reuse naive scan)
                double hh = -INFINITY, ll = INFINITY;
                for (int u = ts; u <= t; ++u) { double hhv=(double)high[u]; double llv=(double)low[u]; if (hhv>hh) hh=hhv; if (llv<ll) ll=llv; }
                const double raw_t = (double)close[t] - 0.25 * (hh + ll) - 0.5 * kc_mid_t;
                S0 += raw_t;
                S1 = fma(j, raw_t, S1);
            }
            const double b = (-sum_x * S0 + p * S1) * inv_den;
            const double ybar = S0 * inv_lkc;
            const double yhat_last = fma(b, x_last_minus_xbar, ybar);
            out_mo[base + i] = (float)yhat_last;

            // Signal (lagged 1)
            if (i >= 1) {
                const double prev = (double)out_mo[base + (i - 1)];
                if (isfinite(prev) && isfinite(yhat_last)) {
                    float sig = 0.0f;
                    if (yhat_last > 0.0) {
                        sig = (yhat_last > prev) ? 1.0f : 2.0f;
                    } else {
                        sig = (yhat_last < prev) ? -1.0f : -2.0f;
                    }
                    out_si[base + i] = sig;
                } else if (i >= warm_si) {
                    out_si[base + i] = SMI_QNAN_F;
                }
            }
        }
    }
}

// Many-series × one param (time-major). Grid.y = num_series, block.x processes rows.
extern "C" __global__ void squeeze_momentum_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int lbb,
    float mbb,
    int lkc,
    float mkc,
    float* __restrict__ out_sq_tm,
    float* __restrict__ out_mo_tm,
    float* __restrict__ out_si_tm
) {
    const int series = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;
    if (tid != 0) return; // one thread per series performs scan

    const int fv = first_valids[series];
    if (UNLIKELY(fv < 0 || fv >= series_len || lbb <= 0 || lkc <= 0)) {
        // Fill NaNs for entire series
        float* sq = out_sq_tm + series;
        float* mo = out_mo_tm + series;
        float* si = out_si_tm + series;
        for (int r = 0; r < series_len; ++r) {
            *sq = SMI_QNAN_F; *mo = SMI_QNAN_F; *si = SMI_QNAN_F;
            sq += num_series; mo += num_series; si += num_series;
        }
        return;
    }

    const int warm_sq = max(lbb, lkc) - 1;
    const int warm_mo = lkc - 1;
    const int warm_si = warm_mo + 1;
    const double inv_lbb = 1.0 / (double)lbb;
    const double inv_lkc = 1.0 / (double)lkc;
    const double p = (double)lkc;
    const double sum_x  = 0.5 * p * (p + 1.0);
    const double sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    const double inv_den= 1.0 / (p * sum_x2 - sum_x * sum_x);
    const double x_last_minus_xbar = p - sum_x * inv_lkc;

    // Helpers to index time-major arrays
    auto H = [&](int t){ return (double)high_tm[(size_t)t * num_series + series]; };
    auto L = [&](int t){ return (double)low_tm[(size_t)t * num_series + series]; };
    auto C = [&](int t){ return (double)close_tm[(size_t)t * num_series + series]; };
    auto TR = [&](int t){
        if (t == 0) return fabs(H(t) - L(t));
        const double pc = C(t - 1);
        const double tr1 = fabs(H(t) - L(t));
        const double tr2 = fabs(H(t) - pc);
        const double tr3 = fabs(L(t) - pc);
        return fmax(fmax(tr1, tr2), tr3);
    };

    // Rolling sums for BB/KC at current t
    double sum_bb = 0.0, sumsq_bb = 0.0;
    double sum_kc = 0.0, sum_tr = 0.0;
    const int start_bb = fv + lbb - 1;
    const int start_kc = fv + lkc - 1;
    if (start_bb < series_len) {
        for (int j = start_bb + 1 - lbb; j <= start_bb; ++j) {
            const double v = C(j);
            sum_bb += v; sumsq_bb = fma(v, v, sumsq_bb);
        }
    }
    if (start_kc < series_len) {
        for (int j = start_kc + 1 - lkc; j <= start_kc; ++j) {
            sum_kc += C(j);
            sum_tr += TR(j);
        }
    }

    float* sq = out_sq_tm + series;
    float* mo = out_mo_tm + series;
    float* si = out_si_tm + series;

    // Prefill NaNs up to warmups
    for (int r = 0; r < warm_sq && r < series_len; ++r) { sq[r * num_series] = SMI_QNAN_F; }
    for (int r = 0; r < warm_mo && r < series_len; ++r) { mo[r * num_series] = SMI_QNAN_F; }
    for (int r = 0; r < warm_si && r < series_len; ++r) { si[r * num_series] = SMI_QNAN_F; }

    // Seed finiteness (head of series)
    bool bb_seed_ok = true; for (int j = 0; j < lbb && j < series_len; ++j) { float cc = (float)C(j); if (!is_finite_f(cc)) { bb_seed_ok=false; break; } }
    bool kc_seed_ok = true; for (int j = 0; j < lkc && j < series_len; ++j) { float ch=(float)H(j); float cl=(float)L(j); float cc=(float)C(j); if (!is_finite_f(ch)||!is_finite_f(cl)||!is_finite_f(cc)) { kc_seed_ok=false; break; } }

    for (int t = fv; t < series_len; ++t) {
        if (t > start_bb) {
            const double c_new = C(t);
            const double c_old = C(t - lbb);
            sum_bb += c_new - c_old;
            sumsq_bb = fma(c_new, c_new, sumsq_bb - c_old * c_old);
        }
        if (t > start_kc) {
            sum_kc += C(t) - C(t - lkc);
            sum_tr += TR(t) - TR(t - lkc);
        }

        if (t >= start_bb && t >= start_kc && t >= warm_sq) {
            const double mean_bb = sum_bb * inv_lbb;
            const double var_bb = fma(sumsq_bb * inv_lbb, 1.0, -mean_bb * mean_bb);
            const double dev_bb = sqrt(fmax(0.0, var_bb));
            const double upper_bb = mean_bb + (double)mbb * dev_bb;
            const double lower_bb = mean_bb - (double)mbb * dev_bb;
            const double kc_mid = sum_kc * inv_lkc;
            const double tr_avg = sum_tr * inv_lkc;
            const double upper_kc = kc_mid + (double)mkc * tr_avg;
            const double lower_kc = kc_mid - (double)mkc * tr_avg;
            const bool on = (lower_bb > lower_kc) && (upper_bb < upper_kc);
            const bool off= (lower_bb < lower_kc) && (upper_bb > upper_kc);
            sq[t * num_series] = on ? -1.0f : (off ? 1.0f : 0.0f);
        }

        if (t >= start_kc) {
            // Highest/Lowest in [t-lkc+1 .. t]
            const int ws = t - lkc + 1;
            double highest = -INFINITY, lowest = INFINITY;
            for (int j = ws; j <= t; ++j) { const double h=H(j), l=L(j); if (h>highest) highest=h; if (l<lowest) lowest=l; }

            // Momentum via OLS on RAW (recompute S0/S1 over the window)
            double S0 = 0.0, S1 = 0.0, j = 1.0;
            for (int u = ws; u <= t; ++u, j += 1.0) {
                // kc_mid at u
                double sum_c = 0.0; const int us = u - lkc + 1;
                for (int v = us; v <= u; ++v) sum_c += C(v);
                const double kc_mid_u = sum_c * inv_lkc;
                double hh=-INFINITY, ll=INFINITY; for (int v=us; v<=u; ++v){ double hv=H(v), lv=L(v); if (hv>hh) hh=hv; if (lv<ll) ll=lv; }
                const double raw_u = C(u) - 0.25 * (hh + ll) - 0.5 * kc_mid_u;
                S0 += raw_u; S1 = fma(j, raw_u, S1);
            }
            const double b = (-sum_x * S0 + p * S1) * inv_den;
            const double ybar = S0 * inv_lkc;
            const double yhat_last = fma(b, x_last_minus_xbar, ybar);
            mo[t * num_series] = (float)yhat_last;
            if (t >= 1) {
                const double prev = (double)mo[(t - 1) * num_series];
                if (isfinite(prev) && isfinite(yhat_last)) {
                    float sig = 0.0f;
                    if (yhat_last > 0.0) sig = (yhat_last > prev) ? 1.0f : 2.0f; else sig = (yhat_last < prev) ? -1.0f : -2.0f;
                    si[t * num_series] = sig;
                } else if (t >= warm_si) {
                    si[t * num_series] = SMI_QNAN_F;
                }
            }
        }
    }
}
