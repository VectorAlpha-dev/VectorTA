// CUDA kernels for Gator Oscillator (GATOR)
//
// Math pattern: three EMA recurrences (jaws/teeth/lips) with forward shifts.
// Outputs:
//  - upper = abs(jaws - teeth)
//  - lower = -abs(teeth - lips)
//  - upper_change = upper - upper_prev
//  - lower_change = -(lower - lower_prev)
//
// Warmup semantics (must match scalar implementation):
//   let first = first_valid (index of first finite input)
//   upper_warm = first + max(jl, tl) + max(js, ts) - 1
//   lower_warm = first + max(tl, ll) + max(ts, ls) - 1
//   upper_change_warm = upper_warm + 1
//   lower_change_warm = lower_warm + 1
//   Values before their warmup indices are NaN.
//
// Optimization notes (drop-in from the provided guide):
// - Remove FP64 from hot loop; use FP32 FMA for EMA updates.
// - Heuristic dual-FP32 (double-single) accumulator for very long periods
//   to reduce drift without resorting to FP64. Enabled by default with
//   DS_LEN_THRESHOLD (4096) and requires no build flags.
// - Keep shared-memory rings to avoid local-memory spills due to dynamic
//   indexing by shifts.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Define a quiet-NaN constant for prefill
#ifndef GATOR_NAN_F
#define GATOR_NAN_F (__int_as_float(0x7fffffff))
#endif

// ---- Utility ---------------------------------------------------------------
static __forceinline__ __device__ float fin_or_prev(float x, float prev) {
    // Keep exact scalar semantics: carry previous finite value forward
    return isfinite(x) ? x : prev;
}

// Stable FP32 EMA update (one rounding via FMA):
// ema = ema + a*(x - ema)
static __forceinline__ __device__ float ema_update_f32(float ema, float a, float x) {
    return fmaf(a, (x - ema), ema);
}

// ---- Dual-FP32 (double-single) helpers ------------------------------------
// Based on Dekker/Knuth + FMA: represent a number as hi+lo (both float).
// See NVIDIA forum posts by N. Juffa for validated ds add/mul patterns.
struct dsfloat { float hi, lo; };

static __forceinline__ __device__ dsfloat ds_make(float x) {
    dsfloat r; r.hi = x; r.lo = 0.0f; return r;
}

static __forceinline__ __device__ dsfloat ds_add(dsfloat a, dsfloat b) {
    // TwoSum(a.hi, b.hi) + (a.lo + b.lo)
    float s  = a.hi + b.hi;
    float bp = s - a.hi;
    float t  = ((b.hi - bp) + (a.hi - (s - bp))) + a.lo + b.lo;
    float hi = s + t;
    float lo = t - (hi - s);
    dsfloat r; r.hi = hi; r.lo = lo; return r;
}

static __forceinline__ __device__ dsfloat ds_mul_f(dsfloat a, float b) {
    // TwoProdFMA(a.hi, b) + a.lo*b
    float p   = a.hi * b;
    float err = fmaf(a.hi, b, -p);     // exact mul error via FMA
    float lo  = a.lo * b;
    float s   = p + lo;
    float bp  = s - p;
    float t   = ((lo - bp) + (p - (s - bp))) + err;
    float hi  = s + t;
    float l   = t - (hi - s);
    dsfloat r; r.hi = hi; r.lo = l; return r;
}

// s = (1 - a)*s + a*x  in dual-FP32
static __forceinline__ __device__ void ema_update_ds(dsfloat &s, float a, float x) {
    // term1 = s * (1 - a)
    dsfloat term1 = ds_mul_f(s, 1.0f - a);
    // term2 = a * x (as ds)
    float ax_hi = a * x;
    float ax_lo = fmaf(a, x, -ax_hi);
    dsfloat term2; term2.hi = ax_hi; term2.lo = ax_lo;
    s = ds_add(term1, term2);
}

// Heuristic: when periods are very long, flip to dual-FP32 for stability.
#ifndef DS_LEN_THRESHOLD
#define DS_LEN_THRESHOLD 4096
#endif

// ---- Kernel 1: one series, many param combos ------------------------------

extern "C" __global__ void gatorosc_batch_f32(
    const float* __restrict__ data, // one series
    const int    len,
    const int    first_valid,
    const int*   __restrict__ jlens,
    const int*   __restrict__ jshifts,
    const int*   __restrict__ tlens,
    const int*   __restrict__ tshifts,
    const int*   __restrict__ llens,
    const int*   __restrict__ lshifts,
    const int    n_combos,
    const int    ring_len_max, // == max(js,ts,ls) + 1 for this launch
    float* __restrict__ out_upper,        // [n_combos][len]
    float* __restrict__ out_lower,        // [n_combos][len]
    float* __restrict__ out_upper_change, // [n_combos][len]
    float* __restrict__ out_lower_change  // [n_combos][len]
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int jl = jlens[combo];
    const int js = jshifts[combo];
    const int tl = tlens[combo];
    const int ts = tshifts[combo];
    const int ll = llens[combo];
    const int ls = lshifts[combo];
    if (jl <= 0 || tl <= 0 || ll <= 0) return;

    const int upper_needed = max(jl, tl) + max(js, ts);
    const int lower_needed = max(tl, ll) + max(ts, ls);
    const int uwarm = first_valid + max(upper_needed - 1, 0);
    const int lwarm = first_valid + max(lower_needed - 1, 0);
    const int ucwarm = uwarm + 1;
    const int lcwarm = lwarm + 1;

    float* __restrict__ upper = out_upper + (size_t)combo * len;
    float* __restrict__ lower = out_lower + (size_t)combo * len;
    float* __restrict__ uchn  = out_upper_change + (size_t)combo * len;
    float* __restrict__ lchn  = out_lower_change + (size_t)combo * len;

    // Use only the first warp. The wrapper should launch blockDim.x as a
    // multiple of 32, ideally exactly 32 (one warp per combo row).
    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;
    const unsigned mask = 0xffffffffu;

    // Alphas in FP32 (no FP64)
    const float ja   = 2.0f / (float)(jl + 1);
    const float ta   = 2.0f / (float)(tl + 1);
    const float la   = 2.0f / (float)(ll + 1);

    // Shared-memory rings: [j | t | l], each of length ring_len_max
    extern __shared__ float s[];
    float* jring = s;
    float* tring = s + ring_len_max;
    float* lring = s + 2 * ring_len_max;
    const int rlen = ring_len_max;

    // Fallback path: keep the original sequential implementation for the rare
    // "very long period" DS mode or for unexpected launch parameters.
    const int maxlen = max(jl, max(tl, ll));
    const bool use_ds = (maxlen >= DS_LEN_THRESHOLD);
    // The warp-scan path below requires a full warp. Keep the sequential fallback as
    // the safe default whenever blockDim.x < 32 or the ring is not 32-aligned.
    if (use_ds || blockDim.x < 32 || first_valid >= len || rlen < 32 || (rlen & 31) != 0 || js < 0 || ts < 0 || ls < 0 ||
        js >= rlen || ts >= rlen || ls >= rlen) {
        if (lane != 0) return;

        // Full NaN prefill (rare path; keep semantics obvious).
        for (int i = 0; i < len; ++i) {
            upper[i] = GATOR_NAN_F;
            lower[i] = GATOR_NAN_F;
            uchn[i]  = GATOR_NAN_F;
            lchn[i]  = GATOR_NAN_F;
        }

        if (first_valid >= len) return;

        // Initialize EMA states at first_valid
        float seed = isfinite(data[first_valid]) ? data[first_valid] : 0.0f;

        float  jema_f = seed, tema_f = seed, lema_f = seed;
        dsfloat jema_ds = ds_make(seed), tema_ds = ds_make(seed), lema_ds = ds_make(seed);

        // Pre-fill rings with the seed state
        for (int k = 0; k < rlen; ++k) {
            jring[k] = seed; tring[k] = seed; lring[k] = seed;
        }

        float u_prev = 0.0f, l_prev = 0.0f;
        bool have_u = false, have_l = false;
        int rpos = 0;

        for (int i = first_valid; i < len; ++i) {
            const float xi = data[i];

            if (!use_ds) {
                const float x = fin_or_prev(xi, jema_f);
                jema_f = ema_update_f32(jema_f, ja, x);
                tema_f = ema_update_f32(tema_f, ta, x);
                lema_f = ema_update_f32(lema_f, la, x);

                jring[rpos] = jema_f;
                tring[rpos] = tema_f;
                lring[rpos] = lema_f;
            } else {
                const float x = fin_or_prev(xi, jema_ds.hi);
                ema_update_ds(jema_ds, ja, x);
                ema_update_ds(tema_ds, ta, x);
                ema_update_ds(lema_ds, la, x);

                jring[rpos] = jema_ds.hi;
                tring[rpos] = tema_ds.hi;
                lring[rpos] = lema_ds.hi;
            }

            int jj = rpos - js; if (jj < 0) jj += rlen;
            int tt = rpos - ts; if (tt < 0) tt += rlen;
            int llp = rpos - ls; if (llp < 0) llp += rlen;

            if (i >= uwarm) {
                const float u = fabsf(jring[jj] - tring[tt]);
                upper[i] = u;
                if (i == uwarm) { u_prev = u; have_u = true; }
                else if (i >= ucwarm && have_u) { uchn[i] = u - u_prev; u_prev = u; }
            }
            if (i >= lwarm) {
                const float l = -fabsf(tring[tt] - lring[llp]);
                lower[i] = l;
                if (i == lwarm) { l_prev = l; have_l = true; }
                else if (i >= lcwarm && have_l) { lchn[i] = -(l - l_prev); l_prev = l; }
            }

            rpos += 1; if (rpos == rlen) rpos = 0;
        }
        return;
    }

    // ---- Fast path: warp-cooperative scan over time (32-wide tiles) --------

    // Initialize EMA states at first_valid
    float seed = isfinite(data[first_valid]) ? data[first_valid] : 0.0f;

    // Pre-fill shared rings with the seed state (covers look-backs before first_valid)
    for (int k = lane; k < rlen; k += 32) {
        jring[k] = seed;
        tring[k] = seed;
        lring[k] = seed;
    }

    // Warmup NaN prefixes (avoid full-row memset)
    const int up_pref  = (uwarm  < len) ? uwarm  : len;
    const int lo_pref  = (lwarm  < len) ? lwarm  : len;
    const int uc_pref  = (ucwarm < len) ? ucwarm : len;
    const int lc_pref  = (lcwarm < len) ? lcwarm : len;
    for (int i = lane; i < up_pref; i += 32) { upper[i] = GATOR_NAN_F; }
    for (int i = lane; i < lo_pref; i += 32) { lower[i] = GATOR_NAN_F; }
    for (int i = lane; i < uc_pref; i += 32) { uchn[i]  = GATOR_NAN_F; }
    for (int i = lane; i < lc_pref; i += 32) { lchn[i]  = GATOR_NAN_F; }

    // States carried between tiles (broadcast in-warp).
    float prev_j = seed;
    float prev_t = seed;
    float prev_l = seed;

    float prev_u = 0.0f;   // upper[t-1] when t >= uwarm+1
    float prev_lo = 0.0f;  // lower[t-1] when t >= lwarm+1

    // Ring write base. With rlen a multiple of 32, rbase is always a multiple of 32,
    // so the lane stores never wrap within the tile.
    int rbase = 0;

    const float oma_j = 1.0f - ja;
    const float oma_t = 1.0f - ta;
    const float oma_l = 1.0f - la;

    for (int t0 = first_valid; t0 < len; t0 += 32) {
        const int t = t0 + lane;
        const int remaining = len - t0;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        const int tile_end  = t0 + last_lane;

        // Load price (out-of-range lanes use NaN -> identity updates).
        const float xi = (t < len) ? data[t] : GATOR_NAN_F;
        const bool xi_finite = (t < len) && isfinite(xi);

        // ---------------- Jaws EMA (NaN -> hold) ----------------
        float Aj = xi_finite ? oma_j : 1.0f;
        float Bj = xi_finite ? (ja * xi) : 0.0f;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float A_prev = __shfl_up_sync(mask, Aj, off);
            const float B_prev = __shfl_up_sync(mask, Bj, off);
            if (lane >= off) {
                const float A_cur = Aj;
                const float B_cur = Bj;
                Aj = A_cur * A_prev;
                Bj = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float pj = __shfl_sync(mask, prev_j, 0);
        const float yj = fmaf(Aj, pj, Bj);
        prev_j = __shfl_sync(mask, yj, last_lane);

        // x(t) for teeth/lips: finite(price) ? price : jaws_prev (jaws at t-1)
        float jaws_prev = __shfl_up_sync(mask, yj, 1);
        if (lane == 0) jaws_prev = pj;
        const float x_eff = xi_finite ? xi : jaws_prev;

        // ---------------- Teeth EMA ----------------
        float At = (t < len) ? oma_t : 1.0f;
        float Bt = (t < len) ? (ta * x_eff) : 0.0f;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float A_prev = __shfl_up_sync(mask, At, off);
            const float B_prev = __shfl_up_sync(mask, Bt, off);
            if (lane >= off) {
                const float A_cur = At;
                const float B_cur = Bt;
                At = A_cur * A_prev;
                Bt = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float pt = __shfl_sync(mask, prev_t, 0);
        const float yt = fmaf(At, pt, Bt);
        prev_t = __shfl_sync(mask, yt, last_lane);

        // ---------------- Lips EMA ----------------
        float Al = (t < len) ? oma_l : 1.0f;
        float Bl = (t < len) ? (la * x_eff) : 0.0f;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float A_prev = __shfl_up_sync(mask, Al, off);
            const float B_prev = __shfl_up_sync(mask, Bl, off);
            if (lane >= off) {
                const float A_cur = Al;
                const float B_cur = Bl;
                Al = A_cur * A_prev;
                Bl = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float pl = __shfl_sync(mask, prev_l, 0);
        const float yl = fmaf(Al, pl, Bl);
        prev_l = __shfl_sync(mask, yl, last_lane);

        // Write EMA states into the shared rings.
        const int rpos = rbase + lane; // no wrap within tile
        jring[rpos] = yj;
        tring[rpos] = yt;
        lring[rpos] = yl;
        __syncwarp();

        // Emit outputs for this tile.
        float u = GATOR_NAN_F;
        float lo = GATOR_NAN_F;
        if (t < len) {
            int jj = rpos - js; if (jj < 0) jj += rlen;
            int tt = rpos - ts; if (tt < 0) tt += rlen;
            int llp = rpos - ls; if (llp < 0) llp += rlen;

            if (t >= uwarm) {
                u = fabsf(jring[jj] - tring[tt]);
                upper[t] = u;
            }
            if (t >= lwarm) {
                lo = -fabsf(tring[tt] - lring[llp]);
                lower[t] = lo;
            }

            if (t >= ucwarm) {
                float up = __shfl_up_sync(mask, u, 1);
                if (lane == 0) up = prev_u;
                uchn[t] = u - up;
            }
            if (t >= lcwarm) {
                float lp = __shfl_up_sync(mask, lo, 1);
                if (lane == 0) lp = prev_lo;
                lchn[t] = lp - lo;
            }
        }

        // Carry last values for tile-boundary diffs.
        if (tile_end >= uwarm) {
            prev_u = __shfl_sync(mask, u, last_lane);
        }
        if (tile_end >= lwarm) {
            prev_lo = __shfl_sync(mask, lo, last_lane);
        }

        // Advance ring write base (rlen is a multiple of 32).
        rbase += 32;
        if (rbase == rlen) rbase = 0;
    }
}

// ---- Kernel 2: many series, one param combo --------------------------------

extern "C" __global__ void gatorosc_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids, // per series
    const int    cols,
    const int    rows,
    const int    jl,
    const int    js,
    const int    tl,
    const int    ts,
    const int    ll,
    const int    ls,
    const int    ring_len,
    float* __restrict__ out_upper_tm,
    float* __restrict__ out_lower_tm,
    float* __restrict__ out_upper_change_tm,
    float* __restrict__ out_lower_change_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series index
    if (s >= cols) return;

    const int first_valid = first_valids[s];
    const int upper_needed = max(jl, tl) + max(js, ts);
    const int lower_needed = max(tl, ll) + max(ts, ls);
    const int uwarm = first_valid + max(upper_needed - 1, 0);
    const int lwarm = first_valid + max(lower_needed - 1, 0);
    const int ucwarm = uwarm + 1;
    const int lcwarm = lwarm + 1;

    // Prefill column with NaNs
    for (int t = 0; t < rows; ++t) {
        out_upper_tm[(size_t)t * cols + s] = GATOR_NAN_F;
        out_lower_tm[(size_t)t * cols + s] = GATOR_NAN_F;
        out_upper_change_tm[(size_t)t * cols + s] = GATOR_NAN_F;
        out_lower_change_tm[(size_t)t * cols + s] = GATOR_NAN_F;
    }

    if (first_valid >= rows || jl <= 0 || tl <= 0 || ll <= 0) return;

    // FP32 alphas
    const float ja = 2.0f / (float)(jl + 1);
    const float ta = 2.0f / (float)(tl + 1);
    const float la = 2.0f / (float)(ll + 1);

    // Shared memory layout per-thread: [j | t | l] each length=ring_len
    extern __shared__ float smem[];
    float* base  = smem + (size_t)threadIdx.x * 3 * ring_len;
    float* jring = base;
    float* tring = base + ring_len;
    float* lring = base + 2 * ring_len;
    int rpos = 0;

    // Seed EMA states at first_valid
    float seed = isfinite(prices_tm[(size_t)first_valid * cols + s]) ? prices_tm[(size_t)first_valid * cols + s] : 0.0f;

    const int maxlen = max(jl, max(tl, ll));
    const bool use_ds = (maxlen >= DS_LEN_THRESHOLD);

    float  jema_f = seed, tema_f = seed, lema_f = seed;
    dsfloat jema_ds = ds_make(seed), tema_ds = ds_make(seed), lema_ds = ds_make(seed);

    // Initialize rings
    for (int k = 0; k < ring_len; ++k) { jring[k] = seed; tring[k] = seed; lring[k] = seed; }

    float u_prev = 0.0f, l_prev = 0.0f; bool have_u = false, have_l = false;

    for (int t = first_valid; t < rows; ++t) {
        const float xv = prices_tm[(size_t)t * cols + s];

        if (!use_ds) {
            const float x = fin_or_prev(xv, jema_f);
            jema_f = ema_update_f32(jema_f, ja, x);
            tema_f = ema_update_f32(tema_f, ta, x);
            lema_f = ema_update_f32(lema_f, la, x);

            jring[rpos] = jema_f;
            tring[rpos] = tema_f;
            lring[rpos] = lema_f;
        } else {
            const float x = fin_or_prev(xv, jema_ds.hi);
            ema_update_ds(jema_ds, ja, x);
            ema_update_ds(tema_ds, ta, x);
            ema_update_ds(lema_ds, la, x);

            jring[rpos] = jema_ds.hi;
            tring[rpos] = tema_ds.hi;
            lring[rpos] = lema_ds.hi;
        }

        int jj = rpos - js; if (jj < 0) jj += ring_len;
        int tt = rpos - ts; if (tt < 0) tt += ring_len;
        int llp = rpos - ls; if (llp < 0) llp += ring_len;

        if (t >= uwarm) {
            const float u = fabsf(jring[jj] - tring[tt]);
            out_upper_tm[(size_t)t * cols + s] = u;
            if (t == uwarm) { u_prev = u; have_u = true; }
            else if (t >= ucwarm && have_u) {
                out_upper_change_tm[(size_t)t * cols + s] = u - u_prev;
                u_prev = u;
            }
        }
        if (t >= lwarm) {
            const float l = -fabsf(tring[tt] - lring[llp]);
            out_lower_tm[(size_t)t * cols + s] = l;
            if (t == lwarm) { l_prev = l; have_l = true; }
            else if (t >= lcwarm && have_l) {
                out_lower_change_tm[(size_t)t * cols + s] = -(l - l_prev);
                l_prev = l;
            }
        }

        rpos += 1; if (rpos == ring_len) rpos = 0;
    }
}
