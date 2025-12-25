// CUDA kernels for Percentage Price Oscillator (PPO)
//
// Math patterns supported:
// - SMA mode (ma_mode = 0): prefix-sum/rational. Host provides f64 prefix sums
//   over the valid segment starting at first_valid. Kernel computes window sums
//   in O(1) per output and writes NaN for the warmup prefix (first_valid+slow-1).
// - EMA mode (ma_mode = 1): per-row sequential recurrence that mirrors the
//   scalar classic EMA seeding (first slow samples form slow SMA; fast seeded
//   from its SMA and then advanced until alignment; then standard EMA updates).
//
// Outputs are f32; critical accumulations use f64 when applicable.

#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>

// Write IEEE-754 quiet NaN as f32
__device__ __forceinline__ float f32_nan() { return __int_as_float(0x7fffffff); }

// --- high-accuracy dual-FP32 ("float-float") helpers -----------------
// Value ≈ hi + lo with near-double precision using error-free transforms.
struct Float2 { float hi, lo; };

__device__ __forceinline__ Float2 f2_make(float a)    { return {a, 0.0f}; }

// Error-free two-sum (Knuth / QD Alg.4); returns normalized (hi, lo)
__device__ __forceinline__ Float2 f2_two_sum(float a, float b) {
    Float2 r;
    float s  = a + b;
    float bp = s - a;
    float e  = (a - (s - bp)) + (b - bp);
    // quick-two-sum normalize
    float t  = s + e;
    r.lo     = e - (t - s);
    r.hi     = t;
    return r;
}

// Add Float2 + float
__device__ __forceinline__ Float2 f2_add_f(Float2 a, float b) {
    Float2 s = f2_two_sum(a.hi, b);
    s.lo += a.lo;
    // renormalize
    float t = s.hi + s.lo;
    s.lo = s.lo - (t - s.hi);
    s.hi = t;
    return s;
}

// Multiply Float2 by float: (a.hi*b) + (a.lo*b) with FMA residual
__device__ __forceinline__ Float2 f2_mul_f(Float2 a, float b) {
    float ph = a.hi * b;
    float pe = fmaf(a.hi, b, -ph) + a.lo * b;
    float t  = ph + pe;
    Float2 r = { t, pe - (t - ph) };
    return r;
}

// FMA into Float2: a*b + c, where c is Float2
__device__ __forceinline__ Float2 f2_fma(float a, float b, Float2 c) {
    float ph = fmaf(a, b, c.hi);        // primary sum in hi
    float pe = fmaf(a, b, - (ph - c.hi)) + c.lo; // error accumulation
    float t  = ph + pe;
    Float2 r = { t, pe - (t - ph) };
    return r;
}

// Divide Float2 by int via reciprocal (one-step refinement)
__device__ __forceinline__ Float2 f2_div_int(Float2 a, int den) {
    float d      = (float)den;
    float inv_d  = 1.0f / d;
    // first quotient from collapsed (hi+lo)
    float q0     = (a.hi + a.lo) * inv_d;
    // one Newton-like refinement on (hi,lo)
    float r      = (a.hi + a.lo) - q0 * d; // residual
    float q1     = r * inv_d;
    Float2 q     = f2_make(q0 + q1);
    return q;
}

// Ratio (a/b) as float with first-order correction using low parts
__device__ __forceinline__ float f2_ratio(Float2 num, Float2 den) {
    float N  = num.hi + num.lo;
    float D  = den.hi + den.lo;
    float invD = 1.0f / D;
    float y  = N * invD;
    // Correct using low parts: y ≈ (N + δN) / (D + δD) ≈ y + (δN - y*δD)/D
    float corr = (num.lo - y * den.lo) * invD;
    return y + corr;
}

// Warp reductions (min/max) over a 32-lane warp
__device__ __forceinline__ int warp_max_i(int v, unsigned mask) {
    const int lane = (int)(threadIdx.x & 31);
    for (int ofs = 16; ofs; ofs >>= 1) {
        const int src_lane = lane + ofs;
        const int other = __shfl_down_sync(mask, v, ofs);
        if (src_lane < 32 && (mask & (1u << src_lane))) v = max(v, other);
    }
    return v;
}
__device__ __forceinline__ int warp_min_i(int v, unsigned mask) {
    const int lane = (int)(threadIdx.x & 31);
    for (int ofs = 16; ofs; ofs >>= 1) {
        const int src_lane = lane + ofs;
        const int other = __shfl_down_sync(mask, v, ofs);
        if (src_lane < 32 && (mask & (1u << src_lane))) v = min(v, other);
    }
    return v;
}

// -----------------------------------------------------------------------------
// One-series × many-params (EMA only), warp-cooperative fast path.
// Each warp processes up to 32 (fast,slow) combos on the same price series.
// Lane 0 loads x_t and broadcasts via __shfl_sync to eliminate redundant loads.
// Row-major out: [combo][t]. Prefix [0..start_idx) filled with NaN per combo.
extern "C" __global__ void ppo_batch_ema_manyparams_f32(
    const float* __restrict__ data,   // length len
    int len,
    int first_valid,
    const int* __restrict__ fasts,    // length n_combos
    const int* __restrict__ slows,    // length n_combos
    int n_combos,
    float* __restrict__ out)          // row-major: [combo][t]
{
    if (len <= 0 || n_combos <= 0) return;

    const unsigned lane  = threadIdx.x & 31;
    const unsigned warp  = threadIdx.x >> 5;
    const unsigned wpb   = blockDim.x >> 5;                // warps per block
    if (wpb == 0) return;

    const int combos_per_block = (int)(wpb * 32);
    const int base_combo = (int)blockIdx.y * combos_per_block + (int)warp * 32;
    const int combo      = base_combo + (int)lane;

    // Keep all lanes participating in __ballot_sync so we can form a correct warp mask.
    // IMPORTANT: Threads whose lane is *not* in `mask` must not call warp intrinsics with `mask`.
    const unsigned full_mask  = __activemask();
    const bool     valid_lane = (combo < n_combos);
    const unsigned mask       = __ballot_sync(full_mask, valid_lane);
    if (mask == 0u) return; // no valid lanes in this warp
    if (!valid_lane) return; // avoid calling warp intrinsics with a mask that excludes this lane

    // Load params (per lane)
    int fast = 0, slow = 0;
    if (valid_lane) {
        fast = fasts[combo];
        slow = slows[combo];
    }
    // Sanity: non-positive periods -> lane becomes inactive
    const bool periods_ok = valid_lane && (fast > 0) && (slow > 0);
    const float nanf = f32_nan();

    // Compute per-lane geometry
    int start_idx = 0;
    if (periods_ok) start_idx = first_valid + slow - 1;

    // Write prefix NaNs for this combo (one lane owns one row).
    if (periods_ok) {
        const int row_off = combo * len;
        for (int t = 0; t < min(start_idx, len); ++t) {
            out[row_off + t] = nanf;
        }
    }

    // If lane invalid or start already beyond series, it still participates in shuffles but does nothing afterwards
    // Compute warp-wide bounds to orchestrate shared steps
    int warp_slow_min = periods_ok ? slow : INT_MAX;
    int warp_slow_max = periods_ok ? slow : 0;
    int warp_fast_min = periods_ok ? fast : INT_MAX;

    warp_slow_min = warp_min_i(warp_slow_min, mask);
    warp_slow_max = warp_max_i(warp_slow_max, mask);
    warp_fast_min = warp_min_i(warp_fast_min, mask);

    // --- Seed sums over first `slow` values (per lane) using broadcasted x ---
    Float2 slow_sum = f2_make(0.0f);
    Float2 fast_sum = f2_make(0.0f);
    int overlap = 0;
    if (periods_ok) overlap = slow - fast; // may be negative (mirror scalar seeding)

    for (int k = 0; k < warp_slow_max && k + first_valid < len; ++k) {
        float v = 0.0f;
        if (lane == 0u) v = data[first_valid + k];
        v = __shfl_sync(mask, v, 0);
        if (periods_ok) {
            if (k < slow) {
                slow_sum = f2_add_f(slow_sum, v);
                if (k >= overlap) fast_sum = f2_add_f(fast_sum, v);
            }
        }
    }

    // Convert sums to EMA seeds
    Float2 fast_ema = f2_make(0.0f), slow_ema = f2_make(0.0f);
    float fa = 0.0f, fb = 0.0f, sa = 0.0f, sb = 0.0f;
    int row_off = 0;
    if (periods_ok) {
        fast_ema = f2_div_int(fast_sum, max(fast, 1));
        slow_ema = f2_div_int(slow_sum, max(slow, 1));
        fa = 2.0f / (float)(fast + 1);
        fb = 1.0f - fa;
        sa = 2.0f / (float)(slow + 1);
        sb = 1.0f - sa;
        row_off = combo * len;
    }

    // Advance fast EMA from (first_valid + fast) to (first_valid + slow - 1) per lane.
    // Iterate i over [first_valid + warp_fast_min, first_valid + warp_slow_max - 1]
    const int i_begin = first_valid + warp_fast_min;
    const int i_end   = first_valid + warp_slow_max - 1;
    for (int i = i_begin; i <= i_end && i < len; ++i) {
        float x = 0.0f;
        if (lane == 0u) x = data[i];
        x = __shfl_sync(mask, x, 0);
        if (periods_ok) {
            if (i >= first_valid + fast && i <= first_valid + slow - 1) {
                // fast_ema = fa*x + fb*fast_ema
                Float2 tmp = f2_mul_f(fast_ema, fb);
                fast_ema = f2_fma(fa, x, tmp);
            }
        }
    }

    // First PPO at start_idx
    if (periods_ok && start_idx < len) {
        float y0 = nanf;
        float den = slow_ema.hi + slow_ema.lo;
        if (isfinite(den) && den != 0.0f) {
            float ratio = f2_ratio(fast_ema, slow_ema);
            y0 = ratio * 100.0f - 100.0f;
        }
        out[row_off + start_idx] = y0;
    }

    // Main time scan: from t = min(start) + 1 to len-1
    int warp_start_min = periods_ok ? start_idx : INT_MAX;
    warp_start_min = warp_min_i(warp_start_min, mask);
    for (int t = warp_start_min + 1; t < len; ++t) {
        float x = 0.0f;
        if (lane == 0u) x = data[t];
        x = __shfl_sync(mask, x, 0);
        if (periods_ok && t > start_idx) {
            // EMA recurrences
            fast_ema = f2_fma(fa, x, f2_mul_f(fast_ema, fb));
            slow_ema = f2_fma(sa, x, f2_mul_f(slow_ema, sb));
            // PPO
            float y = nanf;
            float den = slow_ema.hi + slow_ema.lo;
            if (isfinite(den) && den != 0.0f) {
                float ratio = f2_ratio(fast_ema, slow_ema);
                y = ratio * 100.0f - 100.0f;
            }
            out[row_off + t] = y;
        }
    }
}

// One-series × many-params (row-major out: [combo][t])
// data: len
// prefix_sum: len+1 (only used for SMA mode; can be len=1 dummy otherwise)
// fasts, slows: length n_combos
// ma_mode: 0=SMA, 1=EMA
extern "C" __global__ void ppo_batch_f32(
    const float* __restrict__ data,
    const double* __restrict__ prefix_sum,
    int len,
    int first_valid,
    const int* __restrict__ fasts,
    const int* __restrict__ slows,
    int ma_mode,
    int n_combos,
    float* __restrict__ out)
{
    if (len <= 0) return;
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int fast = fasts[combo];
    const int slow = slows[combo];
    if (fast <= 0 || slow <= 0) return;
    const int warm_idx = first_valid + max(fast, slow) - 1; // require both MAs warmed up
    const int row_off = combo * len;
    const float nanf = f32_nan();

    if (ma_mode == 0) {
        // SMA path: parallel over time using prefix sums
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (t < len) {
            float y = nanf;
            if (t >= warm_idx) {
                const int tr = t + 1;
                const double s_fast = prefix_sum[tr] - prefix_sum[tr - fast];
                const double s_slow = prefix_sum[tr] - prefix_sum[tr - slow];
                if (isfinite(s_fast) && isfinite(s_slow) && s_slow != 0.0) {
                    const double ratio = (s_fast * (double)slow) / (s_slow * (double)fast);
                    y = (float)(ratio * 100.0 - 100.0);
                } else {
                    y = nanf;
                }
            }
            out[row_off + t] = y;
            t += stride;
        }
        return;
    }

    // EMA path: only thread 0 performs sequential scan; others help prefix NaN init
    // Initialize prefix [0..start_idx) to NaN in parallel
    const int start_idx = first_valid + slow - 1; // classic EMA start index (slow warmup)
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < min(start_idx, len); idx += gridDim.x * blockDim.x) {
        out[row_off + idx] = nanf;
    }
    __syncthreads();

    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (start_idx >= len) return;

    const double fa = 2.0 / (double)(fast + 1);
    const double sa = 2.0 / (double)(slow + 1);
    const double fb = 1.0 - fa;
    const double sb = 1.0 - sa;

    // Seed: sums over the first `slow` valid samples; fast_sum over overlap tail
    double slow_sum = 0.0;
    double fast_sum = 0.0;
    const int overlap = slow - fast;
    for (int k = 0; k < slow; ++k) {
        const double v = (double)data[first_valid + k];
        slow_sum += v;
        if (k >= overlap) fast_sum += v;
    }

    double fast_ema = fast_sum / (double)fast;
    double slow_ema = slow_sum / (double)slow;

    // Advance fast EMA until alignment at start_idx
    for (int i = first_valid + fast; i <= start_idx; ++i) {
        const double x = (double)data[i];
        fast_ema = fa * x + fb * fast_ema;
    }

    // First PPO at start_idx
    float y0 = nanf;
    if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
        const double ratio = fast_ema / slow_ema;
        y0 = (float)(ratio * 100.0 - 100.0);
    }
    out[row_off + start_idx] = y0;

    // Main loop
    for (int j = start_idx + 1; j < len; ++j) {
        const double x = (double)data[j];
        fast_ema = fa * x + fb * fast_ema;
        slow_ema = sa * x + sb * slow_ema;
        float y = nanf;
        if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
            const double ratio = fast_ema / slow_ema;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out[row_off + j] = y;
    }
}

// Many-series × one-param (time-major)
// prices_tm: rows x cols (time-major: idx = t*cols + s)
// prefix_sum_tm: (rows*cols)+1, time-major running sums per series (only SMA). May be dummy for EMA.
// first_valids: length cols
// fast, slow, ma_mode
extern "C" __global__ void ppo_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const double* __restrict__ prefix_sum_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    int fast,
    int slow,
    int ma_mode,
    float* __restrict__ out_tm)
{
    if (cols <= 0 || rows <= 0) return;
    const int s = blockIdx.y * blockDim.y + threadIdx.y; // series/column
    if (s >= cols) return;
    const int fv = max(0, first_valids[s]);
    const int warm_idx = fv + max(fast, slow) - 1;
    const float nanf = f32_nan();

    if (ma_mode == 0) {
        // SMA path: parallelize over time using per-series time-major prefix sums.
        // Clamp the left window boundary to max(first_valid-1, -1) so that we can
        // safely address prefix_sum_tm (ps[0] is the global zero).
        const int tx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        for (int t = tx; t < rows; t += stride) {
            float y = nanf;
            if (t >= warm_idx) {
                const int wr = (t * cols + s) + 1;
                const int lfast_t = max(t - fast, fv - 1);
                const int lslow_t = max(t - slow, fv - 1);
                const int wl_fast = (lfast_t >= 0) ? (lfast_t * cols + s) + 1 : 0;
                const int wl_slow = (lslow_t >= 0) ? (lslow_t * cols + s) + 1 : 0;
                const double s_fast = prefix_sum_tm[wr] - prefix_sum_tm[wl_fast];
                const double s_slow = prefix_sum_tm[wr] - prefix_sum_tm[wl_slow];
                if (isfinite(s_fast) && isfinite(s_slow) && s_slow != 0.0) {
                    const double ratio = (s_fast * (double)slow) / (s_slow * (double)fast);
                    y = (float)(ratio * 100.0 - 100.0);
                }
            }
            out_tm[t * cols + s] = y;
        }
        return;
    }

    // EMA path: let lane 0 (threadIdx.x==0 && threadIdx.y==0) do the sequential scan for series s
    if (!(threadIdx.x == 0)) return; // only one thread along x for recurrence

    // Prefix NaN init
    const int start_idx = fv + slow - 1;
    for (int t = 0; t < min(start_idx, rows); ++t) {
        out_tm[t * cols + s] = nanf;
    }
    if (start_idx >= rows) return;

    const double fa = 2.0 / (double)(fast + 1);
    const double sa = 2.0 / (double)(slow + 1);
    const double fb = 1.0 - fa;
    const double sb = 1.0 - sa;

    // Seed from first `slow` values of this series
    double slow_sum = 0.0;
    double fast_sum = 0.0;
    const int overlap = slow - fast;
    for (int k = 0; k < slow; ++k) {
        const double v = (double)prices_tm[(fv + k) * cols + s];
        slow_sum += v;
        if (k >= overlap) fast_sum += v;
    }
    double fast_ema = fast_sum / (double)fast;
    double slow_ema = slow_sum / (double)slow;

    for (int i = fv + fast; i <= start_idx; ++i) {
        const double x = (double)prices_tm[i * cols + s];
        fast_ema = fa * x + fb * fast_ema;
    }
    float y0 = nanf;
    if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
        const double ratio = fast_ema / slow_ema;
        y0 = (float)(ratio * 100.0 - 100.0);
    }
    out_tm[start_idx * cols + s] = y0;

    for (int t = start_idx + 1; t < rows; ++t) {
        const double x = (double)prices_tm[t * cols + s];
        fast_ema = fa * x + fb * fast_ema;
        slow_ema = sa * x + sb * slow_ema;
        float y = nanf;
        if (isfinite(fast_ema) && isfinite(slow_ema) && slow_ema != 0.0) {
            const double ratio = fast_ema / slow_ema;
            y = (float)(ratio * 100.0 - 100.0);
        }
        out_tm[t * cols + s] = y;
    }
}

// Elementwise PPO from precomputed MA series (batch):
// fast_ma: nf x len (row-major), slow_ma: ns x len (row-major)
// out: (nf*ns) x len, with row mapping r = fi*ns + si
extern "C" __global__ void ppo_from_ma_batch_f32(
    const float* __restrict__ fast_ma,
    const float* __restrict__ slow_ma,
    int len,
    int nf,
    int ns,
    int first_valid,
    const int* __restrict__ slow_periods,
    int row_start,
    float* __restrict__ out)
{
    const int r = row_start + blockIdx.y; // global output row
    if (r >= nf * ns) return;
    const int fi = r / ns;
    const int si = r - fi * ns;
    const int fast_off = fi * len;
    const int slow_off = si * len;
    const int stride = gridDim.x * blockDim.x;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const float nanf = f32_nan();
    const int warm = first_valid + slow_periods[si] - 1;
    for (int t = t0; t < len; t += stride) {
        const float sf = slow_ma[slow_off + t];
        const float ff = fast_ma[fast_off + t];
        float y = nanf;
        if (t >= warm && isfinite(sf) && isfinite(ff) && sf != 0.0f) {
            y = (ff / sf) * 100.0f - 100.0f;
        }
        out[r * len + t] = y;
    }
}

// Elementwise PPO from precomputed MA series (many-series, time-major):
// fast_ma_tm and slow_ma_tm: rows x cols (time-major)
extern "C" __global__ void ppo_from_ma_many_series_one_param_time_major_f32(
    const float* __restrict__ fast_ma_tm,
    const float* __restrict__ slow_ma_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    int slow,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;
    const int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float nanf = f32_nan();
    const int warm = first_valids[s] + slow - 1;
    for (int t = t0; t < rows; t += stride) {
        const float sf = slow_ma_tm[t * cols + s];
        const float ff = fast_ma_tm[t * cols + s];
        float y = nanf;
        if (t >= warm && isfinite(sf) && isfinite(ff) && sf != 0.0f) {
            y = (ff / sf) * 100.0f - 100.0f;
        }
        out_tm[t * cols + s] = y;
    }
}
