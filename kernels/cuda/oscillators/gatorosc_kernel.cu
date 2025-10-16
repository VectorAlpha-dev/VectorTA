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

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef GATOR_NAN_F
#define GATOR_NAN_F (__int_as_float(0x7fffffff))
#endif

static __forceinline__ __device__ float fin_or_prev(float x, float prev) {
    return isfinite(x) ? x : prev;
}

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
    // One block per combo, single-thread sequential scan (thread 0 does the work).
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

    float* upper = out_upper + (size_t)combo * len;
    float* lower = out_lower + (size_t)combo * len;
    float* uchn  = out_upper_change + (size_t)combo * len;
    float* lchn  = out_lower_change + (size_t)combo * len;

    // Prefill NaNs
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        upper[i] = GATOR_NAN_F; lower[i] = GATOR_NAN_F; uchn[i] = GATOR_NAN_F; lchn[i] = GATOR_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    // Alphas
    const double ja = 2.0 / (double)(jl + 1);
    const double ta = 2.0 / (double)(tl + 1);
    const double la = 2.0 / (double)(ll + 1);
    const double joma = 1.0 - ja;
    const double toma = 1.0 - ta;
    const double loma = 1.0 - la;

    // Shared-memory rings: [j | t | l] each of length ring_len_max
    extern __shared__ float s[];
    float* jring = s;
    float* tring = s + ring_len_max;
    float* lring = s + 2 * ring_len_max;
    const int rlen = ring_len_max; // >= required for this combo
    int rpos = 0;

    // Initialize EMA states at first_valid
    if (first_valid >= len) return;
    double jema = (double) (isfinite(data[first_valid]) ? data[first_valid] : 0.0f);
    double tema = jema;
    double lema = jema;
    for (int k = 0; k < rlen; ++k) { jring[k] = (float)jema; tring[k] = (float)tema; lring[k] = (float)lema; }

    float u_prev = 0.0f, l_prev = 0.0f; bool have_u = false, have_l = false;

    for (int i = first_valid; i < len; ++i) {
        const float xi = data[i];
        const double x = (double)fin_or_prev(xi, (float)jema);
        jema = joma * jema + ja * x;
        tema = toma * tema + ta * x;
        lema = loma * lema + la * x;

        jring[rpos] = (float)jema;
        tring[rpos] = (float)tema;
        lring[rpos] = (float)lema;

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

    // Alphas
    const double ja = 2.0 / (double)(jl + 1);
    const double ta = 2.0 / (double)(tl + 1);
    const double la = 2.0 / (double)(ll + 1);
    const double joma = 1.0 - ja;
    const double toma = 1.0 - ta;
    const double loma = 1.0 - la;

    // Shared memory layout per-thread: [j | t | l] each length=ring_len
    extern __shared__ float smem[];
    float* base = smem + (size_t)threadIdx.x * 3 * ring_len;
    float* jring = base;
    float* tring = base + ring_len;
    float* lring = base + 2 * ring_len;
    int rpos = 0;
    for (int k = 0; k < ring_len; ++k) { jring[k] = 0.0f; tring[k] = 0.0f; lring[k] = 0.0f; }

    // Seed EMA states at first_valid
    double jema = (double)(isfinite(prices_tm[(size_t)first_valid * cols + s]) ? prices_tm[(size_t)first_valid * cols + s] : 0.0f);
    double tema = jema;
    double lema = jema;
    for (int k = 0; k < ring_len; ++k) { jring[k] = (float)jema; tring[k] = (float)tema; lring[k] = (float)lema; }

    float u_prev = 0.0f, l_prev = 0.0f; bool have_u = false, have_l = false;
    for (int t = first_valid; t < rows; ++t) {
        const float xv = prices_tm[(size_t)t * cols + s];
        const double x = (double)fin_or_prev(xv, (float)jema);
        jema = joma * jema + ja * x;
        tema = toma * tema + ta * x;
        lema = loma * lema + la * x;

        jring[rpos] = (float)jema;
        tring[rpos] = (float)tema;
        lring[rpos] = (float)lema;

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
