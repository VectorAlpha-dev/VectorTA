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

    // Prefill NaNs with all threads
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        upper[i] = GATOR_NAN_F; lower[i] = GATOR_NAN_F; uchn[i] = GATOR_NAN_F; lchn[i] = GATOR_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    if (first_valid >= len) return;

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
    int rpos = 0;

    // Initialize EMA states at first_valid
    float seed = isfinite(data[first_valid]) ? data[first_valid] : 0.0f;

    // Heuristic: dual‑FP32 only for very long periods
    const int maxlen = max(jl, max(tl, ll));
    const bool use_ds = (maxlen >= DS_LEN_THRESHOLD);

    // State
    float jema_f = seed, tema_f = seed, lema_f = seed;
    dsfloat jema_ds = ds_make(seed), tema_ds = ds_make(seed), lema_ds = ds_make(seed);

    // Pre-fill rings with the seed state
    for (int k = 0; k < rlen; ++k) { 
        jring[k] = seed; tring[k] = seed; lring[k] = seed; 
    }

    float u_prev = 0.0f, l_prev = 0.0f; 
    bool have_u = false, have_l = false;

    // Main scan
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

