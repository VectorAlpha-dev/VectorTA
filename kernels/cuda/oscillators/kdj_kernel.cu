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

// --- Helpers: compensated FP32 sum (Kahan) and sparse-table %K with precomputed fk state ---
__device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    // Compensated add: sum += x with small-error compensation (all FP32)
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// stoch_from_tables specialized for a fixed fast_k (k, offset, level_base precomputed)
__device__ __forceinline__ float stoch_from_tables_fk(
    int t, int fk, int k_log2, int offset, int level_base,
    const float* __restrict__ close,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
){
    const int start = t - fk + 1;
    // Any NaN in window -> NaN
    if (nan_psum[t + 1] - nan_psum[start]) return KDJ_QNAN;

    const int idx_a = level_base + start;
    const int idx_b = level_base + (t + 1 - offset);
    const float h = fmaxf(st_max[idx_a], st_max[idx_b]);
    const float l = fminf(st_min[idx_a], st_min[idx_b]);
    const float c = close[t];
    const float den = h - l;
    // Note: NaN checks fold into (x==x); den<=0 covers zero-span ranges
    if (!(h==h) || !(l==l) || !(c==c) || den <= 0.0f) return KDJ_QNAN;
    return (c - l) * (100.0f / den);
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

    // Parallel init of NaNs (also clears scratch in out_j)
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_k[base + i] = KDJ_QNAN;
        out_d[base + i] = KDJ_QNAN;
        out_j[base + i] = KDJ_QNAN; // later used as scratch for S[t]
    }
    __syncthreads();

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;

    const int fk = fast_k_arr[combo];
    const int sk = slow_k_arr[combo];
    const int sd = slow_d_arr[combo];
    if (UNLIKELY(fk <= 0 || sk <= 0 || sd <= 0)) return;

    if (UNLIKELY(level_count <= 0)) return;

    const int stoch_warm = first_valid + fk - 1;
    if (UNLIKELY(stoch_warm >= series_len)) return;

    const int k_ma = k_ma_types[combo];
    const int d_ma = d_ma_types[combo];

    // Precompute sparse-table constants for this fk
    const int k_log2 = log2_tbl[fk];
    if (UNLIKELY(k_log2 < 0 || k_log2 >= level_count)) return;
    const int offset     = 1 << k_log2;
    const int level_base = level_offsets[k_log2];

    // Warp-0 cooperatively precomputes S[t] = raw stochastic for all t >= stoch_warm
    // Store into out_j[base + t] as a temporary scratch buffer.
    const int lane = threadIdx.x & 31;
    if ((threadIdx.x >> 5) == 0) {
        // Fill pre-warm with NaN (usually small)
        for (int t = lane; t < stoch_warm; t += 32) {
            out_j[base + t] = KDJ_QNAN;
        }
        for (int t = stoch_warm + lane; t < series_len; t += 32) {
            out_j[base + t] = stoch_from_tables_fk(
                t, fk, k_log2, offset, level_base, close, st_max, st_min, nan_psum
            );
        }
    }
    __syncthreads();

    // All smoothing is inherently sequential; run it in lane 0 for this combo
    if (threadIdx.x != 0) return;

    const int k_warm = stoch_warm + sk - 1;
    const int d_warm = k_warm     + sd - 1;

    // ---- SMA -> SMA fused path (ignore NaNs within windows) ----
    if (k_ma == 0 && d_ma == 0) {
        if (k_warm >= series_len) return;

        // Seed K at t = k_warm over S[t-sk+1..t]
        float sum_k = 0.0f, ck = 0.0f; int cnt_k = 0;
        {
            const int start = k_warm - sk + 1;
            for (int ti = start; ti <= k_warm; ++ti) {
                const float s = out_j[base + ti];
                if (s == s) { kahan_add(s, sum_k, ck); ++cnt_k; }
            }
            const float kv = (cnt_k > 0) ? (sum_k / (float)cnt_k) : KDJ_QNAN;
            out_k[base + k_warm] = kv;
        }

        // Prepare D sliding accumulators
        float sum_d = 0.0f, cd = 0.0f; int cnt_d = 0;
        const float kv0 = out_k[base + k_warm];
        if (kv0 == kv0) { kahan_add(kv0, sum_d, cd); ++cnt_d; }

        // Slide forward
        for (int t = k_warm + 1; t < series_len; ++t) {
            // Update K window: +S[t], -S[t-sk]
            const float s_new = out_j[base + t];
            if (s_new == s_new) { kahan_add(s_new, sum_k, ck); ++cnt_k; }
            const int t_old = t - sk;
            if (t_old >= stoch_warm) {
                const float s_old = out_j[base + t_old];
                if (s_old == s_old) { kahan_add(-s_old, sum_k, ck); --cnt_k; }
            }
            const float kv = (cnt_k > 0) ? (sum_k / (float)cnt_k) : KDJ_QNAN;
            out_k[base + t] = kv;

            // Update D window over K: +K[t], -K[t-sd]
            if (kv == kv) { kahan_add(kv, sum_d, cd); ++cnt_d; }
            const int t_old_k = t - sd;
            if (t_old_k >= k_warm) {
                const float kk_old = out_k[base + t_old_k];
                if (kk_old == kk_old) { kahan_add(-kk_old, sum_d, cd); --cnt_d; }
            }

            if (t >= d_warm) {
                const float dv = (cnt_d > 0) ? (sum_d / (float)cnt_d) : KDJ_QNAN;
                out_d[base + t] = dv;
                out_j[base + t] = (kv == kv && dv == dv) ? (3.0f * kv - 2.0f * dv) : KDJ_QNAN;
            }
        }
        return;
    }

    // ---- EMA -> EMA fused path (seed averages, update only on finite samples) ----
    if (k_ma == 1 && d_ma == 1) {
        if (k_warm >= series_len) return;

        const float alpha_k = 2.0f / (float(sk) + 1.0f);
        const float one_mk  = 1.0f - alpha_k;
        const float alpha_d = 2.0f / (float(sd) + 1.0f);
        const float one_md  = 1.0f - alpha_d;

        // Seed ema_k at t=k_warm with average of finite S over [k_warm-sk+1..k_warm]
        float sum0 = 0.0f, c0 = 0.0f; int cnt0 = 0;
        {
            const int start = k_warm - sk + 1;
            for (int ti = start; ti <= k_warm; ++ti) {
                const float s = out_j[base + ti];
                if (s == s) { kahan_add(s, sum0, c0); ++cnt0; }
            }
        }
        float ema_k = (cnt0 > 0) ? (sum0 / (float)cnt0) : KDJ_QNAN;
        out_k[base + k_warm] = ema_k;

        // Accumulate initial D seed over K[k_warm..d_warm]
        float sum_d = (ema_k == ema_k) ? ema_k : 0.0f;
        int   cnt_d = (ema_k == ema_k) ? 1 : 0;

        // Advance K until d_warm, accumulating for D’s seed
        for (int t = k_warm + 1; t <= d_warm && t < series_len; ++t) {
            const float s = out_j[base + t];
            if (s == s && ema_k == ema_k) {
                ema_k = fmaf(alpha_k, s, one_mk * ema_k);
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = s; // bootstrap
            }
            out_k[base + t] = ema_k;
            if (ema_k == ema_k) { sum_d += ema_k; ++cnt_d; }
        }

        // Seed ema_d at t=d_warm
        float ema_d = (d_warm < series_len && cnt_d > 0) ? (sum_d / (float)cnt_d) : KDJ_QNAN;
        if (d_warm < series_len) {
            out_d[base + d_warm] = ema_d;
            const float kv = out_k[base + d_warm];
            out_j[base + d_warm] = (kv == kv && ema_d == ema_d) ? (3.0f * kv - 2.0f * ema_d) : KDJ_QNAN;
        }

        // Full EMA update
        for (int t = d_warm + 1; t < series_len; ++t) {
            const float s = out_j[base + t];
            if (s == s && ema_k == ema_k) {
                ema_k = fmaf(alpha_k, s, one_mk * ema_k);
            } else if (s == s && !(ema_k == ema_k)) {
                ema_k = s;
            }
            out_k[base + t] = ema_k;

            if (ema_k == ema_k && ema_d == ema_d) {
                ema_d = fmaf(alpha_d, ema_k, one_md * ema_d);
            } else if (ema_k == ema_k && !(ema_d == ema_d)) {
                ema_d = ema_k;
            }
            out_d[base + t] = ema_d;
            out_j[base + t] = (ema_k == ema_k && ema_d == ema_d) ? (3.0f * ema_k - 2.0f * ema_d) : KDJ_QNAN;
        }
        return;
    }

    // Unsupported MA combination: leave NaNs
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
    const int start = t - fast_k + 1;
    float h = -INFINITY, l = INFINITY;

        // Row pointers that we walk by +num_series
        size_t idx = (size_t)start * (size_t)num_series + (size_t)series;
        const float* __restrict__ ph = high_tm;
        const float* __restrict__ pl = low_tm;

        for (int i = start; i <= t; ++i, idx += (size_t)num_series) {
            const float hi = ph[idx];
            const float lo = pl[idx];
            if (!(hi==hi) || !(lo==lo)) return KDJ_QNAN;
            h = fmaxf(h, hi);
            l = fminf(l, lo);
        }
        const float c = *(close_tm + (size_t)t * (size_t)num_series + series);
        if (!(c==c)) return KDJ_QNAN;
        const float den = h - l;
        return (den > 0.0f) ? (c - l) * (100.0f / den) : KDJ_QNAN;
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
