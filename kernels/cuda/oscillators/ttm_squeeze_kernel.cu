// CUDA kernels for TTM Squeeze
//
// One-series × many-params (batch): each block processes one parameter combo (row).
// A single thread per block performs the sequential time scan; other threads prefill NaNs.
//
// Many-series × one-param (time-major): each block.y is a series; one thread performs the scan.
//
// Semantics aim to match src/indicators/ttm_squeeze.rs (scalar/streaming):
// - Warmup index = first_valid + length - 1.
// - Before warmup: write NaN for both momentum and squeeze.
// - Squeeze levels (0=NoSqz, 1=Low, 2=Mid, 3=High) computed via squared-compare to avoid sqrt:
//     bbv = (bb_mult^2) * var
//     thresholds = (kc_*^2) * (dkc^2) where dkc = SMA(TrueRange, length)
//   if bbv > t_low => 0; else if bbv <= t_high => 3; else if bbv <= t_mid => 2; else => 1.
// - Momentum = yhat_last from OLS over window [i-length+1..i] of y = close - avg( (highest+lowest)/2, SMA(close) )
//   We compute sums naively per step for clarity/correctness; performance is acceptable for typical lengths.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef TTM_QNAN_F
#define TTM_QNAN_F (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) { return isfinite(x); }

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

extern "C" __global__ void ttm_squeeze_batch_f32(
    // Inputs (one series)
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    // Per-combo params
    const int*   __restrict__ length_arr,
    const float* __restrict__ bb_mult_arr,
    const float* __restrict__ kc_high_arr,
    const float* __restrict__ kc_mid_arr,
    const float* __restrict__ kc_low_arr,
    // Shared
    int series_len,
    int n_combos,
    int first_valid,
    // Outputs (row-major): momentum, squeeze
    float* __restrict__ out_momentum,
    float* __restrict__ out_squeeze
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int base = combo * series_len;

    // Prefill NaNs in parallel
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_momentum[base + i] = TTM_QNAN_F;
        out_squeeze[base + i] = TTM_QNAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const int    L  = length_arr[combo];
    const double bb = (double)bb_mult_arr[combo];
    const double kh = (double)kc_high_arr[combo];
    const double km = (double)kc_mid_arr[combo];
    const double kl = (double)kc_low_arr[combo];
    if (UNLIKELY(L <= 0 || first_valid < 0 || first_valid >= series_len)) return;

    const int warm = first_valid + L - 1;
    const double invL = 1.0 / (double)L;
    const double bb_sq = bb * bb;
    const double kh_sq = kh * kh;
    const double km_sq = km * km;
    const double kl_sq = kl * kl;

    // Precompute x sums for OLS with x = 0..L-1
    const double n = (double)L;
    const double sx  = 0.5 * n * (n - 1.0);
    const double sx2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    const double den = n * sx2 - sx * sx;
    const double inv_den = 1.0 / den;

    // Head seed check: require finite values in first L to seed SMA/STDDEV/TR properly
    bool seed_ok = true;
    for (int j = 0; j < L && j < series_len; ++j) {
        if (!is_finite_f(close[j]) || !is_finite_f(high[j]) || !is_finite_f(low[j])) { seed_ok = false; break; }
    }

    // Main scan
    for (int i = warm; i < series_len; ++i) {
        // Window bounds
        const int start = i - L + 1;

        // Sums for mean/var of close (BB) and TR SMA (KC)
        double sumc = 0.0, sumc2 = 0.0, sumtr = 0.0;
        double highest = -INFINITY, lowest = INFINITY;
        for (int j = start; j <= i; ++j) {
            const double c = (double)close[j];
            const double h = (double)high[j];
            const double l = (double)low[j];
            sumc += c; sumc2 = fma(c, c, sumc2);
            const double trj = true_range_idx(j, high, low, close);
            sumtr += trj;
            if (h > highest) highest = h;
            if (l < lowest)  lowest  = l;
        }

        if (!seed_ok) { out_momentum[base + i] = TTM_QNAN_F; out_squeeze[base + i] = TTM_QNAN_F; continue; }

        const double mean = sumc * invL;
        const double var  = fma(sumc2 * invL, 1.0, -mean * mean);
        const double var_pos = (var > 0.0) ? var : 0.0;
        const double dkc  = sumtr * invL; // average TR
        const double dkc2 = dkc * dkc;

        // Squeeze classification via squared compare
        const double bbv = bb_sq * var_pos;
        const double t_low  = kl_sq * dkc2;
        const double t_mid  = km_sq * dkc2;
        const double t_high = kh_sq * dkc2;
        out_squeeze[base + i] = (bbv > t_low) ? 0.0f : ((bbv <= t_high) ? 3.0f : ((bbv <= t_mid) ? 2.0f : 1.0f));

        // Momentum via OLS on y = close - avg(midpoint, mean)
        const double midpoint = 0.5 * (highest + lowest);
        const double avg = 0.5 * (midpoint + mean);
        double S0 = 0.0, S1 = 0.0; // Σy, Σx*y with x=0..L-1
        double x = 0.0;
        for (int j = start; j <= i; ++j, x += 1.0) {
            const double y = (double)close[j] - avg;
            S0 += y; S1 = fma(x, y, S1);
        }
        const double slope = (n * S1 - sx * S0) * inv_den;
        const double intercept = (S0 - slope * sx) * (1.0 / n);
        const double yhat_last = intercept + slope * (n - 1.0);
        out_momentum[base + i] = (float)yhat_last;
    }
}

// Many-series × one param (time-major). Each series scanned by one thread.
extern "C" __global__ void ttm_squeeze_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int length,
    float bb_mult,
    float kc_high,
    float kc_mid,
    float kc_low,
    float* __restrict__ out_momentum_tm,
    float* __restrict__ out_squeeze_tm
) {
    const int s = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;
    if (tid != 0) return;

    const int fv = first_valids[s];
    if (UNLIKELY(length <= 0 || fv < 0 || fv >= series_len)) {
        // Fill NaNs
        float* mo = out_momentum_tm + s;
        float* sq = out_squeeze_tm + s;
        for (int t = 0; t < series_len; ++t) { *mo = TTM_QNAN_F; *sq = TTM_QNAN_F; mo += num_series; sq += num_series; }
        return;
    }
    const int L = length;
    const int warm = fv + L - 1;
    const double invL = 1.0 / (double)L;
    const double bb_sq = (double)bb_mult * (double)bb_mult;
    const double kh_sq = (double)kc_high * (double)kc_high;
    const double km_sq = (double)kc_mid  * (double)kc_mid;
    const double kl_sq = (double)kc_low  * (double)kc_low;

    const double n = (double)L;
    const double sx  = 0.5 * n * (n - 1.0);
    const double sx2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    const double inv_den = 1.0 / (n * sx2 - sx * sx);

    auto H = [&](int t){ return (double)high_tm[(size_t)t * num_series + s]; };
    auto Lw= [&](int t){ return (double)low_tm [(size_t)t * num_series + s]; };
    auto C = [&](int t){ return (double)close_tm[(size_t)t * num_series + s]; };
    auto TR = [&](int t){
        if (t == 0) return fabs(H(t) - Lw(t));
        const double pc = C(t - 1);
        const double tr1 = fabs(H(t) - Lw(t));
        const double tr2 = fabs(H(t) - pc);
        const double tr3 = fabs(Lw(t) - pc);
        return fmax(fmax(tr1, tr2), tr3);
    };

    // Seed finiteness at head
    bool seed_ok = true; for (int j = 0; j < L && j < series_len; ++j) { float cc=(float)C(j); float ch=(float)H(j); float cl=(float)Lw(j); if (!is_finite_f(cc)||!is_finite_f(ch)||!is_finite_f(cl)) { seed_ok=false; break; } }

    float* mo = out_momentum_tm + s;
    float* sq = out_squeeze_tm + s;
    for (int t = 0; t < warm && t < series_len; ++t) { mo[t * num_series] = TTM_QNAN_F; sq[t * num_series] = TTM_QNAN_F; }

    for (int i = warm; i < series_len; ++i) {
        const int start = i - L + 1;
        double sumc = 0.0, sumc2 = 0.0, sumtr = 0.0;
        double highest = -INFINITY, lowest = INFINITY;
        for (int j = start; j <= i; ++j) {
            const double c = C(j);
            const double h = H(j);
            const double l = Lw(j);
            sumc += c; sumc2 = fma(c, c, sumc2);
            sumtr += TR(j);
            if (h > highest) highest = h;
            if (l < lowest)  lowest  = l;
        }
        if (!seed_ok) { mo[i * num_series] = TTM_QNAN_F; sq[i * num_series] = TTM_QNAN_F; continue; }

        const double mean = sumc * invL;
        const double var  = fma(sumc2 * invL, 1.0, -mean * mean);
        const double var_pos = (var > 0.0) ? var : 0.0;
        const double dkc  = sumtr * invL;
        const double dkc2 = dkc * dkc;
        const double bbv = bb_sq * var_pos;
        const double t_low  = kl_sq * dkc2;
        const double t_mid  = km_sq * dkc2;
        const double t_high = kh_sq * dkc2;
        sq[i * num_series] = (float)((bbv > t_low) ? 0.0 : ((bbv <= t_high) ? 3.0 : ((bbv <= t_mid) ? 2.0 : 1.0)));

        const double midpoint = 0.5 * (highest + lowest);
        const double avg = 0.5 * (midpoint + mean);
        double S0 = 0.0, S1 = 0.0, x = 0.0;
        for (int j = start; j <= i; ++j, x += 1.0) { const double y = C(j) - avg; S0 += y; S1 = fma(x, y, S1); }
        const double slope = (n * S1 - sx * S0) * inv_den;
        const double intercept = (S0 - slope * sx) * (1.0 / n);
        const double yhat_last = intercept + slope * (n - 1.0);
        mo[i * num_series] = (float)yhat_last;
    }
}

