// CUDA kernels for KDJ (Stochastic Oscillator with K, D, J lines).
//
// Batch kernel (one series × many params):
//   - Uses precomputed sparse tables (st_max/st_min + log2/offsets + nan_psum)
//     built on host (same as WILLR) to compute HH/LL in O(1) per query.
//   - For smoothing, provides two fused fast paths with exact scalar semantics:
//       • SMA→SMA: averages ignore NaNs within the window (divide by count)
//       • EMA→EMA: initializes EMA with average of first window’s valid values
//                  and updates only when the current sample is finite
//   - Other MA combinations are not handled here; host should short-circuit.
//
// Many-series × one-param (time-major):
//   - Computes rolling HH/LL with a naive O(fast_k) scan (fast_k is small).
//   - Provides the same SMA→SMA and EMA→EMA fused paths.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef KDJ_QNAN
#define KDJ_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Map MA type to integers (host side must agree)
// 0 = SMA, 1 = EMA

// Helper: compute stochastic value at index t using sparse tables (batch path)
__device__ __forceinline__ float stoch_from_tables(
    int t,
    int fast_k,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
) {
    const int start = t - fast_k + 1;
    // Any NaN in window → NaN
    if (nan_psum[t + 1] - nan_psum[start] != 0) return KDJ_QNAN;

    const int window = fast_k;
    const int k = log2_tbl[window];
    const int offset = 1 << k;
    const int level_base = level_offsets[k];
    const int idx_a = level_base + start;
    const int idx_b = level_base + (t + 1 - offset);
    const double h = (double)fmaxf(st_max[idx_a], st_max[idx_b]);
    const double l = (double)fminf(st_min[idx_a], st_min[idx_b]);
    const double c = (double)close[t];
    if (!isfinite(h) || !isfinite(l) || !isfinite(c)) return KDJ_QNAN;
    const double den = h - l;
    if (!(den == den) || den == 0.0) return KDJ_QNAN;
    return (float)(100.0 * ((c - l) / den));
}

extern "C" __global__ void kdj_batch_f32(
    // Inputs
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    const int*   __restrict__ fast_k_arr,
    const int*   __restrict__ slow_k_arr,
    const int*   __restrict__ slow_d_arr,
    const int*   __restrict__ k_ma_types,
    const int*   __restrict__ d_ma_types,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    // Outputs (row-major; each row = one combo)
    float* __restrict__ out_k,
    float* __restrict__ out_d,
    float* __restrict__ out_j
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Parallel init of NaN prefixes for K/D/J
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_k[base + i] = KDJ_QNAN;
        out_d[base + i] = KDJ_QNAN;
        out_j[base + i] = KDJ_QNAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single lane performs sequential scan per combo

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    const int fk = fast_k_arr[combo];
    const int sk = slow_k_arr[combo];
    const int sd = slow_d_arr[combo];
    if (UNLIKELY(fk <= 0 || sk <= 0 || sd <= 0)) return;

    // Guard sparse table arrays range
    if (UNLIKELY(level_count <= 0)) return;

    const int stoch_warm = first_valid + fk - 1;
    const int k_warm     = stoch_warm + sk - 1;
    const int d_warm     = k_warm + sd - 1;
    if (UNLIKELY(stoch_warm >= series_len)) return;

    const int k_ma = k_ma_types[combo];
    const int d_ma = d_ma_types[combo];

    // Naive stoch using H/L slices for exact parity (fast_k is usually small)
    auto stoch_naive_batch = [&](int t)->float {
        const int start = t - fk + 1;
        double h = -INFINITY, l = INFINITY;
        for (int i = start; i <= t; ++i) {
            const double hi = (double)high[i];
            const double lo = (double)low[i];
            if (!(hi == hi) || !(lo == lo)) return KDJ_QNAN;
            h = fmax(h, hi); l = fmin(l, lo);
        }
        const double c = (double)close[t];
        if (!(c == c)) return KDJ_QNAN;
        const double den = h - l; if (den == 0.0 || !isfinite(den)) return KDJ_QNAN;
        return (float)(100.0 * ((c - l) / den));
    };

    // ---- SMA -> SMA fused path (ignore NaNs in window) ----
    if (k_ma == 0 && d_ma == 0) {
        // K is defined starting at k_warm; D at d_warm; J at d_warm
        for (int t = k_warm; t < series_len; ++t) {
            // Average stoch over last sk samples [t-sk+1..t]
            double sum_k = 0.0; int cnt_k = 0;
            const int k_start = t - sk + 1;
            // Note: k_start >= stoch_warm by construction
            for (int u = 0; u < sk; ++u) {
                const int ti = k_start + u;
                float s = stoch_naive_batch(ti);
                if (s == s) { sum_k += (double)s; cnt_k += 1; }
            }
            const float kv = (cnt_k > 0) ? (float)(sum_k / (double)cnt_k) : KDJ_QNAN;
            out_k[base + t] = kv;

            if (t >= d_warm) {
                double sum_d = 0.0; int cnt_d = 0;
                const int d_start = t - sd + 1;
                for (int v = 0; v < sd; ++v) {
                    const int tj = d_start + v;
                    const float kk = out_k[base + tj];
                    if (kk == kk) { sum_d += (double)kk; cnt_d += 1; }
                }
                const float dv = (cnt_d > 0) ? (float)(sum_d / (double)cnt_d) : KDJ_QNAN;
                out_d[base + t] = dv;
                out_j[base + t] = (kv == kv && dv == dv) ? (3.0f * kv - 2.0f * dv) : KDJ_QNAN;
            }
        }
        return;
    }

    // ---- EMA -> EMA fused path ----
    if (k_ma == 1 && d_ma == 1) {
        const double alpha_k = 2.0 / (double(sk) + 1.0);
        const double om_k    = 1.0 - alpha_k;
        const double alpha_d = 2.0 / (double(sd) + 1.0);
        const double om_d    = 1.0 - alpha_d;

        double ema_k = NAN;
        double ema_d = NAN;
        double sum_init_k = 0.0; int cnt_init_k = 0;
        double sum_init_d = 0.0; int cnt_init_d = 0;

        // Seed ema_k at t = k_warm with average of valid stoch in [k_warm - sk + 1 .. k_warm]
        if (k_warm < series_len) {
            const int k_start = k_warm - sk + 1;
            for (int ti = k_start; ti <= k_warm; ++ti) {
                float s = stoch_from_tables(ti, fk, close, log2_tbl, level_offsets, st_max, st_min, nan_psum);
                if (s == s) { sum_init_k += (double)s; cnt_init_k += 1; }
            }
            ema_k = (cnt_init_k > 0) ? (sum_init_k / (double)cnt_init_k) : NAN;
            out_k[base + k_warm] = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        // Advance until d_warm, still accumulating init D average
        for (int t = k_warm + 1; t <= d_warm && t < series_len; ++t) {
            float s = stoch_naive_batch(t);
            if (s == s && ema_k == ema_k) {
                ema_k = alpha_k * (double)s + om_k * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s; // bootstrap if ema_k was NaN
            }
            out_k[base + t] = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        // Seed ema_d at t = d_warm
        if (d_warm < series_len) {
            ema_d = (cnt_init_d > 0) ? (sum_init_d / (double)cnt_init_d) : NAN;
            out_d[base + d_warm] = (float)ema_d;
            const float kv = out_k[base + d_warm];
            out_j[base + d_warm] = (kv == kv && ema_d == ema_d) ? (3.0f * kv - 2.0f * (float)ema_d) : KDJ_QNAN;
        }

        for (int t = d_warm + 1; t < series_len; ++t) {
            float s = stoch_naive_batch(t);
            if (s == s && ema_k == ema_k) {
                ema_k = alpha_k * (double)s + om_k * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s;
            }
            out_k[base + t] = (float)ema_k;

            if (ema_k == ema_k && ema_d == ema_d) {
                ema_d = alpha_d * ema_k + om_d * ema_d;
            } else if (ema_k == ema_k && !(ema_d == ema_d)) {
                ema_d = ema_k;
            }
            out_d[base + t] = (float)ema_d;
            out_j[base + t] = (ema_k == ema_k && ema_d == ema_d) ? (3.0f * (float)ema_k - 2.0f * (float)ema_d) : KDJ_QNAN;
        }
        return;
    }

    // Unsupported MA combination: leave NaNs (host should avoid this path)
}

// ---------------- Many-series × one-param (time-major) ----------------
// Inputs are time-major: x[t * num_series + s]
extern "C" __global__ void kdj_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int fast_k,
    int slow_k,
    int slow_d,
    int k_ma_type,
    int d_ma_type,
    float* __restrict__ k_out_tm,
    float* __restrict__ d_out_tm,
    float* __restrict__ j_out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    // Initialize outputs to NaN for entire series
    for (int t = 0; t < series_len; ++t) {
        *(k_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
        *(d_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
        *(j_out_tm + (size_t)t * num_series + series) = KDJ_QNAN;
    }
    if (UNLIKELY(fv < 0 || fv >= series_len || fast_k <= 0 || slow_k <= 0 || slow_d <= 0)) return;

    const int stoch_warm = fv + fast_k - 1;
    const int k_warm     = stoch_warm + slow_k - 1;
    const int d_warm     = k_warm + slow_d - 1;
    if (UNLIKELY(stoch_warm >= series_len)) return;

    auto load_tm = [num_series, series](const float* base, int t)->float {
        return *(base + (size_t)t * num_series + series);
    };

    auto stoch_naive = [&](int t)->float {
        // compute over [t-fast_k+1 .. t]
        int start = t - fast_k + 1;
        double h = -INFINITY, l = INFINITY;
        for (int i = start; i <= t; ++i) {
            double hi = (double)load_tm(high_tm, i);
            double lo = (double)load_tm(low_tm,  i);
            if (!(hi == hi) || !(lo == lo)) return KDJ_QNAN; // NaN in window
            h = fmax(h, hi);
            l = fmin(l, lo);
        }
        const double c = (double)load_tm(close_tm, t);
        if (!(c == c)) return KDJ_QNAN;
        const double den = h - l;
        if (den == 0.0 || !isfinite(den)) return KDJ_QNAN;
        return (float)(100.0 * ((c - l) / den));
    };

    if (k_ma_type == 0 && d_ma_type == 0) {
        for (int t = k_warm; t < series_len; ++t) {
            double sum_k = 0.0; int cnt_k = 0;
            const int k_start = t - slow_k + 1;
            for (int u = 0; u < slow_k; ++u) {
                const int ti = k_start + u;
                float s = stoch_naive(ti);
                if (s == s) { sum_k += (double)s; cnt_k += 1; }
            }
            const float kv = (cnt_k > 0) ? (float)(sum_k / (double)cnt_k) : KDJ_QNAN;
            *(k_out_tm + (size_t)t * num_series + series) = kv;

            if (t >= d_warm) {
                double sum_d = 0.0; int cnt_d = 0;
                const int d_start = t - slow_d + 1;
                for (int v = 0; v < slow_d; ++v) {
                    const int tj = d_start + v;
                    float kk = *(k_out_tm + (size_t)tj * num_series + series);
                    if (kk == kk) { sum_d += (double)kk; cnt_d += 1; }
                }
                const float dv = (cnt_d > 0) ? (float)(sum_d / (double)cnt_d) : KDJ_QNAN;
                *(d_out_tm + (size_t)t * num_series + series) = dv;
                *(j_out_tm + (size_t)t * num_series + series) = (kv == kv && dv == dv) ? (3.0f * kv - 2.0f * dv) : KDJ_QNAN;
            }
        }
        return;
    }

    if (k_ma_type == 1 && d_ma_type == 1) {
        const double ak = 2.0 / (double(slow_k) + 1.0);
        const double ok = 1.0 - ak;
        const double ad = 2.0 / (double(slow_d) + 1.0);
        const double od = 1.0 - ad;
        double ema_k = NAN, ema_d = NAN;
        double sum_init_k = 0.0; int cnt_init_k = 0;
        double sum_init_d = 0.0; int cnt_init_d = 0;

        if (k_warm < series_len) {
            const int ks = k_warm - slow_k + 1;
            for (int ti = ks; ti <= k_warm; ++ti) {
                float s = stoch_naive(ti);
                if (s == s) { sum_init_k += s; cnt_init_k += 1; }
            }
            ema_k = (cnt_init_k > 0) ? (sum_init_k / (double)cnt_init_k) : NAN;
            *(k_out_tm + (size_t)k_warm * num_series + series) = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        for (int t = k_warm + 1; t <= d_warm && t < series_len; ++t) {
            float s = stoch_naive(t);
            if (s == s && ema_k == ema_k) {
                ema_k = ak * (double)s + ok * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s;
            }
            *(k_out_tm + (size_t)t * num_series + series) = (float)ema_k;
            if (ema_k == ema_k) { sum_init_d += ema_k; cnt_init_d += 1; }
        }

        if (d_warm < series_len) {
            ema_d = (cnt_init_d > 0) ? (sum_init_d / (double)cnt_init_d) : NAN;
            *(d_out_tm + (size_t)d_warm * num_series + series) = (float)ema_d;
            const float kv = *(k_out_tm + (size_t)d_warm * num_series + series);
            *(j_out_tm + (size_t)d_warm * num_series + series) = (kv == kv && ema_d == ema_d) ? (3.0f * kv - 2.0f * (float)ema_d) : KDJ_QNAN;
        }

        for (int t = d_warm + 1; t < series_len; ++t) {
            float s = stoch_naive(t);
            if (s == s && ema_k == ema_k) {
                ema_k = ak * (double)s + ok * ema_k;
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = (double)s;
            }
            *(k_out_tm + (size_t)t * num_series + series) = (float)ema_k;

            if (ema_k == ema_k && ema_d == ema_d) {
                ema_d = ad * ema_k + od * ema_d;
            } else if (ema_k == ema_k && !(ema_d == ema_d)) {
                ema_d = ema_k;
            }
            *(d_out_tm + (size_t)t * num_series + series) = (float)ema_d;
            *(j_out_tm + (size_t)t * num_series + series) = (ema_k == ema_k && ema_d == ema_d) ? (3.0f * (float)ema_k - 2.0f * (float)ema_d) : KDJ_QNAN;
        }
        return;
    }
}
