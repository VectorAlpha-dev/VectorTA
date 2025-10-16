// CUDA kernels for Reverse RSI (price level that would yield a target RSI)
//
// Semantics mirror the scalar implementation in src/indicators/reverse_rsi.rs:
// - Warmup index: warm_idx = first_valid + (2*rsi_length-1) - 1
// - Outputs before warm_idx are NaN
// - Warmup seed uses SMA of gains/losses over ema_len = 2*n - 1, which yields
//   alpha = 2/(ema_len+1) == 1/n (Wilder smoothing equivalence)
// - Subsequent outputs use EMA recurrence on up/down with alpha/beta
// - Let n = rsi_length, L = rsi_level (0<L<100).
//   rs_target = L/(100-L), rs_coeff = (n-1)*rs_target, neg_scale=(100-L)/L
//   x = rs_coeff*down_ema - (n-1)*up_ema
//   scale = (x >= 0 ? 1.0 : neg_scale)
//   out[i] = price[i] + x * scale
// - If out[i] is non-finite and x < 0.0, write 0.0 (matches scalar guard)
// - Deltas treat NaNs as no-change: diff=0.0 if cur or prev is not finite

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef RRSI_NAN
#define RRSI_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ bool is_finite_f(float x) {
    return isfinite(x);
}

extern "C" __global__ void reverse_rsi_batch_f32(
    const float* __restrict__ prices,   // one series (FP32)
    const int*   __restrict__ lengths,  // rsi_length per combo
    const float* __restrict__ levels,   // rsi_level per combo (0<L<100)
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out            // length = n_combos * series_len
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int n = lengths[combo];
    const float L = levels[combo];
    float* out_row = out + combo * series_len;

    // Basic validation; mirror scalar/CPU guard semantics by producing NaNs
    if (UNLIKELY(n <= 0 || !(L > 0.0f && L < 100.0f) || !isfinite(L))) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }
    const int ema_len = (2 * n) - 1;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail <= ema_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = RRSI_NAN;
        return;
    }

    // Prefill NaN up to warm_idx
    const int warm_end = first_valid + ema_len; // exclusive
    const int warm_idx = warm_end - 1;
    for (int i = 0; i < warm_idx; ++i) out_row[i] = RRSI_NAN;

    // Precompute constants
    const double nd = static_cast<double>(n);
    const double n_minus_1 = nd - 1.0;
    const double inv = 100.0 - static_cast<double>(L);
    const double rs_target = static_cast<double>(L) / inv;   // L / (100-L)
    const double neg_scale = inv / static_cast<double>(L);   // (100-L)/L
    const double rs_coeff = n_minus_1 * rs_target;
    const double alpha = 2.0 / (static_cast<double>(ema_len) + 1.0); // == 1/n
    const double beta = 1.0 - alpha;

    // Determine if all remaining values are finite; if so, we can skip per-step checks
    bool all_finite = true;
    for (int i = first_valid; i < series_len; ++i) {
        if (!is_finite_f(prices[i])) { all_finite = false; break; }
    }

    // Warmup sums across ema_len samples starting at first_valid
    double sum_up = 0.0;
    double sum_dn = 0.0;
    double prev = 0.0; // matches scalar: first delta uses prev=0.0
    for (int i = first_valid; i < warm_end; ++i) {
        const float c = prices[i];
        double d = 0.0;
        if (all_finite) {
            d = static_cast<double>(c) - prev;
        } else {
            if (is_finite_f(c) && isfinite(prev)) {
                d = static_cast<double>(c) - prev;
            } else {
                d = 0.0;
            }
        }
        sum_up += fmax(0.0, d);
        sum_dn += fmax(0.0, -d);
        prev = static_cast<double>(c);
    }

    double up_ema = sum_up / static_cast<double>(ema_len);
    double dn_ema = sum_dn / static_cast<double>(ema_len);

    // First output at warm_idx
    {
        const double base = static_cast<double>(prices[warm_idx]);
        const double x0 = rs_coeff * dn_ema - n_minus_1 * up_ema;
        const double m0 = (x0 >= 0.0) ? 1.0 : 0.0;
        const double scale0 = neg_scale + m0 * (1.0 - neg_scale);
        const double v0 = base + x0 * scale0;
        out_row[warm_idx] = (isfinite(v0) || x0 >= 0.0) ? static_cast<float>(v0) : 0.0f;
    }

    // Main loop
    double prev_d = static_cast<double>(prices[warm_idx]);
    for (int i = warm_end; i < series_len; ++i) {
        const float c = prices[i];
        double d = 0.0;
        if (all_finite) {
            d = static_cast<double>(c) - prev_d;
        } else {
            if (is_finite_f(c) && isfinite(prev_d)) {
                d = static_cast<double>(c) - prev_d;
            } else {
                d = 0.0;
            }
        }
        const double up = fmax(0.0, d);
        const double dn = fmax(0.0, -d);
        up_ema = beta * up_ema + alpha * up;
        dn_ema = beta * dn_ema + alpha * dn;
        const double x = rs_coeff * dn_ema - n_minus_1 * up_ema;
        const double m = (x >= 0.0) ? 1.0 : 0.0;
        const double scale = neg_scale + m * (1.0 - neg_scale);
        const double v = static_cast<double>(c) + x * scale;
        out_row[i] = (isfinite(v) || x >= 0.0) ? static_cast<float>(v) : 0.0f;
        prev_d = static_cast<double>(c);
    }
}

// Many-series × one-param, time-major layout
// prices_tm: [row * num_series + series]
extern "C" __global__ void reverse_rsi_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids, // per series
    int num_series,
    int series_len,
    int rsi_length,
    float rsi_level,
    float* __restrict__ out_tm // time-major
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];
    if (UNLIKELY(rsi_length <= 0 || !(rsi_level > 0.0f && rsi_level < 100.0f) || fv < 0 || fv >= series_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }
    const int ema_len = (2 * rsi_length) - 1;
    const int tail = series_len - fv;
    if (UNLIKELY(tail <= ema_len)) {
        float* o = out_tm + series;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = RRSI_NAN;
        return;
    }

    const int warm_end = fv + ema_len; // exclusive
    const int warm_idx = warm_end - 1;

    // Prefill NaNs
    {
        float* o = out_tm + series;
        for (int r = 0; r < warm_idx; ++r, o += num_series) *o = RRSI_NAN;
    }

    // Constants
    const double nd = static_cast<double>(rsi_length);
    const double n_minus_1 = nd - 1.0;
    const double inv = 100.0 - static_cast<double>(rsi_level);
    const double rs_target = static_cast<double>(rsi_level) / inv;
    const double rs_coeff = n_minus_1 * rs_target;
    const double neg_scale = inv / static_cast<double>(rsi_level);
    const double alpha = 2.0 / (static_cast<double>(ema_len) + 1.0);
    const double beta = 1.0 - alpha;

    // all_finite check for this series
    bool all_finite = true;
    for (int r = fv; r < series_len; ++r) {
        const float v = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        if (!is_finite_f(v)) { all_finite = false; break; }
    }

    // Warmup
    double sum_up = 0.0, sum_dn = 0.0;
    double prev = 0.0;
    for (int r = fv; r < warm_end; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        double d = 0.0;
        if (all_finite) {
            d = static_cast<double>(cf) - prev;
        } else {
            if (is_finite_f(cf) && isfinite(prev)) {
                d = static_cast<double>(cf) - prev;
            } else {
                d = 0.0;
            }
        }
        sum_up += fmax(0.0, d);
        sum_dn += fmax(0.0, -d);
        prev = static_cast<double>(cf);
    }
    double up_ema = sum_up / static_cast<double>(ema_len);
    double dn_ema = sum_dn / static_cast<double>(ema_len);

    // First output
    {
        const double base = static_cast<double>(*(prices_tm + static_cast<size_t>(warm_idx) * num_series + series));
        const double x0 = rs_coeff * dn_ema - n_minus_1 * up_ema;
        const double m0 = (x0 >= 0.0) ? 1.0 : 0.0;
        const double scale0 = neg_scale + m0 * (1.0 - neg_scale);
        const double v0 = base + x0 * scale0;
        *(out_tm + static_cast<size_t>(warm_idx) * num_series + series) =
            (isfinite(v0) || x0 >= 0.0) ? static_cast<float>(v0) : 0.0f;
    }

    // Main loop
    double prevd = static_cast<double>(*(prices_tm + static_cast<size_t>(warm_idx) * num_series + series));
    for (int r = warm_end; r < series_len; ++r) {
        const float cf = *(prices_tm + static_cast<size_t>(r) * num_series + series);
        double d = 0.0;
        if (all_finite) {
            d = static_cast<double>(cf) - prevd;
        } else {
            if (is_finite_f(cf) && isfinite(prevd)) {
                d = static_cast<double>(cf) - prevd;
            } else {
                d = 0.0;
            }
        }
        const double up = fmax(0.0, d);
        const double dn = fmax(0.0, -d);
        up_ema = beta * up_ema + alpha * up;
        dn_ema = beta * dn_ema + alpha * dn;
        const double x = rs_coeff * dn_ema - n_minus_1 * up_ema;
        const double m = (x >= 0.0) ? 1.0 : 0.0;
        const double scale = neg_scale + m * (1.0 - neg_scale);
        const double v = static_cast<double>(cf) + x * scale;
        *(out_tm + static_cast<size_t>(r) * num_series + series) =
            (isfinite(v) || x >= 0.0) ? static_cast<float>(v) : 0.0f;
        prevd = static_cast<double>(cf);
    }
}

