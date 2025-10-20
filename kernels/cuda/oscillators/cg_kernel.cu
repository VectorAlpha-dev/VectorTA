// CUDA kernels for Center of Gravity (CG)
//
// Optimized O(n) kernels using a numerically robust sliding-window update
// specialized for the Ehlers COG formula:
//   CG(i) = - sum_{k=0..W-1} (k+1) * P_{i-k} / sum_{k=0..W-1} P_{i-k}
// with W = period - 1. First valid output index is (first_valid + period).
// When the denominator is non-finite or ~0, output 0.0f (Ehlers policy).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef CG_NAN
#define CG_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// ==== helpers ================================================================
#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// device-side compensated add: Kahan
__device__ __forceinline__ void kahan_add(float x, float& sum, float& c) {
    float y = x - c;
    float t = sum + y;
    c      = (t - sum) - y;
    sum    = t;
}

__device__ __forceinline__ bool bad(float x) { return !isfinite(x); }

// small epsilon guard (match scalar behavior)
#ifndef CG_EPS
#define CG_EPS (1.1920929e-7f) // ~FLT_EPSILON
#endif

// ==== 1) one price series – many params =====================================
// Thread-per-combo, O(n) per combo via sliding-window update.
extern "C" __global__ void cg_batch_f32(const float* __restrict__ prices,
                                         const int*   __restrict__ periods,
                                         int series_len,
                                         int n_combos,
                                         int first_valid,
                                         float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_ptr   = out + combo * series_len;

    // Basic validation (preserves scalar semantics)
    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm = first_valid + period; // first computed index
    const int W    = period - 1;           // window length used by the dot
    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;

    if (W <= 0) {
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }

    // --- initialize running sums at i = warm --------------------------------
    // S = sum P, N = sum (k+1)*P  (k=0..W-1; P_{i-k})
    float S = 0.0f, cS = 0.0f;   // Kahan for S
    float N = 0.0f, cN = 0.0f;   // Kahan for N
    int   bad_count = 0;

    {
        const int i = warm;
        for (int k = 0; k < W; ++k) {
            const float p = prices[i - k];
            const bool  b = bad(p);
            bad_count += b;
            const float v = b ? 0.0f : p;
            kahan_add(v, S, cS);
            kahan_add((float)(k + 1) * v, N, cN);
        }
        if (bad_count || fabsf(S) <= CG_EPS) out_ptr[i] = 0.0f;
        else                                 out_ptr[i] = -N / S;
    }

    // --- slide the window: for i -> i+1 -------------------------------------
    for (int i = warm + 1; i < series_len; ++i) {
        const float p_new = prices[i];
        const float p_old = prices[i - W];

        const bool b_new = bad(p_new);
        const bool b_old = bad(p_old);
        bad_count += (int)b_new - (int)b_old;

        const float v_new = b_new ? 0.0f : p_new;
        const float v_old = b_old ? 0.0f : p_old;

        // S_{i} -> S_{i+1}
        kahan_add( v_new, S, cS);
        kahan_add(-v_old, S, cS);
        // N_{i+1} = N_{i} + S_{i+1} - W * v_old
        kahan_add(S - (float)W * v_old, N, cN);

        if (bad_count || fabsf(S) <= CG_EPS) out_ptr[i] = 0.0f;
        else                                 out_ptr[i] = -N / S;
    }
}

// ==== 2) many series – one param (time-major: [row * num_series + series]) ===
// Thread-per-series, O(n) per series via the same sliding update.
extern "C" __global__ void cg_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        float* o = col_out;
        for (int row = 0; row < series_len; ++row, o += num_series) *o = CG_NAN;
        return;
    }

    const int warm = first_valid + period;
    const int W    = period - 1;

    // NaN prefix
    {
        float* o = col_out;
        for (int row = 0; row < warm; ++row, o += num_series) *o = CG_NAN;
    }

    if (W <= 0) {
        float* o = col_out + (size_t)warm * num_series;
        for (int row = warm; row < series_len; ++row, o += num_series) *o = 0.0f;
        return;
    }

    // Running sums for this series
    float S = 0.0f, cS = 0.0f;
    float N = 0.0f, cN = 0.0f;
    int   bad_count = 0;

    // initialize at row = warm
    {
        const int row = warm;
        for (int k = 0; k < W; ++k) {
            const float p = *(col_in + ((size_t)row - (size_t)k) * num_series);
            const bool  b = bad(p);
            bad_count += b;
            const float v = b ? 0.0f : p;
            kahan_add(v, S, cS);
            kahan_add((float)(k + 1) * v, N, cN);
        }
        float* dst = col_out + (size_t)row * num_series;
        *dst = (bad_count || fabsf(S) <= CG_EPS) ? 0.0f : (-N / S);
    }

    // slide for subsequent rows
    for (int row = warm + 1; row < series_len; ++row) {
        const float p_new = *(col_in + (size_t)row * num_series);
        const float p_old = *(col_in + (size_t)(row - W) * num_series);

        const bool b_new = bad(p_new);
        const bool b_old = bad(p_old);
        bad_count += (int)b_new - (int)b_old;

        const float v_new = b_new ? 0.0f : p_new;
        const float v_old = b_old ? 0.0f : p_old;

        kahan_add( v_new, S, cS);
        kahan_add(-v_old, S, cS);
        kahan_add(S - (float)W * v_old, N, cN);

        float* dst = col_out + (size_t)row * num_series;
        *dst = (bad_count || fabsf(S) <= CG_EPS) ? 0.0f : (-N / S);
    }
}

// ==== Optional prefix-sum path for one series – many params ==================
// Build prefix arrays for a single price series.
// P[i] = sum_{t=0..i} P_t (non-finite treated as 0)
// Q[i] = sum_{t=0..i} (t+1) * P_t
// B[i] = count_{t=0..i} !isfinite(P_t)
extern "C" __global__ void cg_build_prefix_f32(const float* __restrict__ prices,
                                               int series_len,
                                               float* __restrict__ P,
                                               float* __restrict__ Q,
                                               int*   __restrict__ B) {
    // Single-thread scan is simple, fast enough for typical lengths
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;

    float ps = 0.0f, qs = 0.0f;
    int   bc = 0;
    for (int i = 0; i < series_len; ++i) {
        const float p = prices[i];
        const bool  b = bad(p);
        bc += (int)b;
        const float v = b ? 0.0f : p;
        ps += v;
        qs += (float)(i + 1) * v;
        P[i] = ps; Q[i] = qs; B[i] = bc;
    }
}

// Uses the prefix arrays to compute all combos in O(1) per row, no sliding loop.
// Semantics (warmup and bad handling) match the sliding version.
extern "C" __global__ void cg_batch_from_prefix_f32(const float* __restrict__ P,
                                                    const float* __restrict__ Q,
                                                    const int*   __restrict__ B,
                                                    const int*   __restrict__ periods,
                                                    int series_len,
                                                    int n_combos,
                                                    int first_valid,
                                                    float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* out_ptr   = out + combo * series_len;

    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN; return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN; return;
    }

    const int warm = first_valid + period;
    const int W    = period - 1;
    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;
    if (W <= 0) { for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f; return; }

    for (int i = warm; i < series_len; ++i) {
        const int a = i - W + 1;
        const int b = i;

        // prefix helpers
        const float sumP = P[b] - (a > 0 ? P[a - 1] : 0.0f);
        const float sumQ = Q[b] - (a > 0 ? Q[a - 1] : 0.0f);
        const int   sumB = B[b] - (a > 0 ? B[a - 1] : 0);

        if (sumB > 0 || fabsf(sumP) <= CG_EPS) { out_ptr[i] = 0.0f; continue; }
        // N_i = (i + 2) * sumP - sumQ
        const float N = (float)(i + 2) * sumP - sumQ;
        out_ptr[i] = -N / sumP;
    }
}

