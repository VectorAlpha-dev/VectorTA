// CUDA kernels for Chande Kroll Stop (CKSP)
// Optimized: safer init (no inter-block race), first-sample hoist, q==1 fast path,
// optional TR precompute kernel + variant that consumes precomputed TR.
// FP32 throughout; RMA uses FMA; compensated warmup sum only.
//
// Semantics mirror src/indicators/cksp.rs exactly:
// - Inputs: high, low, close (f32)
// - Parameters per combo: p (ATR period), x (multiplier), q (rolling window)
// - ATR is an RMA with alpha = 1/p and a sum warmup for the first p samples.
// - Rolling max/min via monotonic ring deques (capacity q+1).
// - Warmup & NaN: write NaN until index = first_valid + p + q - 1.
//
// Entry points provided:
//  - cksp_batch_f32:         one series × many parameter rows (grid = {x=1, y=n_combos})
//  - cksp_batch_f32_pretr:   same as above, but consumes precomputed TR (recommended for many rows)
//  - cksp_many_series_one_param_f32: many series × one param (time-major layout)
//  - tr_from_hlc_f32:        parallel TR precompute for one series
//
// Shared-memory layout per CTA (row/combo):
//   int   h_idx[cap_max]
//   int   l_idx[cap_max]
//   int   ls_idx[cap_max]
//   int   ss_idx[cap_max]
//   float ls_val[cap_max]
//   float ss_val[cap_max]
//
// Dynamic shared mem bytes per block = (4 * cap_max * sizeof(int) + 2 * cap_max * sizeof(float)) = 24 * cap_max.
//
// Launch notes for cksp_batch_*:
//   grid  = dim3(1, n_combos, 1)   // IMPORTANT: x=1 to avoid inter-block races
//   block = dim3(256) or 512
//   shmem = 24 * cap_max bytes     // cap_max = max(q)+1 across rows
//
// For cap_max needing >48KB dynamic shared memory, enable opt-in (runtime attribute).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// Helpers for ring buffer index arithmetic
__device__ __forceinline__ int rb_inc(int idx, int cap) { return (idx + 1) >= cap ? 0 : idx + 1; }
__device__ __forceinline__ int rb_dec(int idx, int cap) { return (idx == 0) ? (cap - 1) : (idx - 1); }

// -------------------------------------------------------------------------------------------------
// Optional: Parallel True Range precompute (one series)
// TR[i] = max(hi-lo, |hi - close[i-1]|, |lo - close[i-1]|)
// For i == start (start = max(0, first_valid)), TR[i] = hi - lo.
// Earlier indices may be written but are ignored by CKSP warmup logic.
//
extern "C" __global__
void tr_from_hlc_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const float* __restrict__ close,
                     int series_len,
                     int first_valid,
                     float* __restrict__ tr_out) {
    if (series_len <= 0) return;
    const int start = (first_valid < 0 ? 0 : first_valid);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        const float hi = high[i];
        const float lo = low[i];
        if (i <= start) {
            tr_out[i] = hi - lo;  // i == start is the only one that matters; earlier values aren't used.
        } else {
            const float pc = close[i - 1];
            const float hl = hi - lo;
            const float hc = fabsf(hi - pc);
            const float lc = fabsf(lo - pc);
            tr_out[i] = fmaxf(hl, fmaxf(hc, lc));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Internal core step for cksp (shared by batch kernels)
// Computes CKSP for a single (p, x, q) row over one series given either:
//   - raw (high, low, close)  -> compute TR on the fly (first-sample hoist)
//   - precomputed tr[]        -> pass tr != nullptr
//
template<bool UsePrecomputedTR>
__device__ void cksp_core_row(const float* __restrict__ high,
                              const float* __restrict__ low,
                              const float* __restrict__ close,
                              const float* __restrict__ tr_opt,  // may be nullptr if !UsePrecomputedTR
                              int series_len,
                              int first_valid,
                              int p,
                              float x,
                              int q,
                              int cap_max,
                              float* __restrict__ out_long_row,
                              float* __restrict__ out_short_row) {
    if (series_len <= 0 || p <= 0 || q <= 0) return;
    const int start = (first_valid < 0 ? 0 : first_valid);
    if (start >= series_len) return;

    // Parallel NaN init within this block
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_long_row[i]  = CUDART_NAN_F;
        out_short_row[i] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;  // single producer per row

    // Shared memory for ring deques
    extern __shared__ __align__(16) unsigned char shraw[];
    int*   h_idx  = (int*)shraw;
    int*   l_idx  = h_idx + cap_max;
    int*   ls_idx = l_idx + cap_max;
    int*   ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;

    const int cap  = q + 1;
    const int warm = start + p + q - 1;

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    // ATR RMA state
    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;
    // Compensated warmup sum for the first p samples (small cost, better accuracy)
    float sum_tr = 0.0f, c_tr = 0.0f;

    // ---- process the first sample (i = start) outside the loop (no isfinite check needed)
    {
        const int i = start;
        const float hi = high[i];
        const float lo = low[i];
        const float tr = UsePrecomputedTR ? tr_opt[i] : (hi - lo); // first sample TR = hi - lo

        // Warmup (k = 0)
        // Kahan compensation on warmup sum
        float y = tr - c_tr;
        float t = sum_tr + y;
        c_tr = (t - sum_tr) - y;
        sum_tr = t;
        if (p == 1) rma = tr;  // immediate RMA

        // Initialize high/low deques unless q==1 fast path
        if (q > 1) {
            // push i into h deque
            while (h_head != h_tail) {
                const int last = rb_dec(h_tail, cap);
                const int last_i = h_idx[last];
                if (high[last_i] <= hi) h_tail = last; else break;
            }
            int next_tail = rb_inc(h_tail, cap);
            if (next_tail == h_head) h_head = rb_inc(h_head, cap); // drop oldest if full
            h_idx[h_tail] = i; h_tail = next_tail;

            // push i into l deque
            while (l_head != l_tail) {
                const int last = rb_dec(l_tail, cap);
                const int last_i = l_idx[last];
                if (low[last_i] >= lo) l_tail = last; else break;
            }
            next_tail = rb_inc(l_tail, cap);
            if (next_tail == l_head) l_head = rb_inc(l_head, cap);
            l_idx[l_tail] = i; l_tail = next_tail;
        }
    }

    // ---- main loop
    if (q == 1) {
        // Fast path: rolling max/min over window=1 is just current sample.
        int k = 1;  // already did k=0
        float prev_close = close[start]; // for on-the-fly TR when not precomputed
        for (int i = start + 1; i < series_len; ++i, ++k) {
            const float hi = high[i];
            const float lo = low[i];
            float tr = UsePrecomputedTR ? tr_opt[i]
                                        : fmaxf(hi - lo, fmaxf(fabsf(hi - prev_close), fabsf(lo - prev_close)));
            prev_close = close[i];

            if (k < p) {
                // warmup sum with compensation
                float y = tr - c_tr;
                float t = sum_tr + y;
                c_tr = (t - sum_tr) - y;
                sum_tr = t;
                if (k == p - 1) rma = sum_tr / (float)p;
            } else {
                rma = fmaf(alpha, tr - rma, rma); // EMA recursion
            }

            if (i >= warm) {
                out_long_row[i]  = fmaf(-x, rma, hi); // hi - x*rma
                out_short_row[i] = fmaf(+x, rma, lo); // lo + x*rma
            }
        }
        return;
    }

    // General path: q > 1 (deque maintenance)
    int k = 1; // have processed one sample (k=0)
    float prev_close = close[start];
    for (int i = start + 1; i < series_len; ++i, ++k) {
        const float hi = high[i];
        const float lo = low[i];
        float tr = UsePrecomputedTR ? tr_opt[i]
                                    : fmaxf(hi - lo, fmaxf(fabsf(hi - prev_close), fabsf(lo - prev_close)));
        prev_close = close[i];

        // Warmup then recursion
        if (k < p) {
            float y = tr - c_tr;
            float t = sum_tr + y;
            c_tr = (t - sum_tr) - y;
            sum_tr = t;
            if (k == p - 1) rma = sum_tr / (float)p;
        } else {
            rma = fmaf(alpha, tr - rma, rma);
        }

        // Rolling MAX of HIGH over q
        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_i = h_idx[last];
            if (high[last_i] <= hi) h_tail = last; else break;
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap);
        h_idx[h_tail] = i; h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_i = h_idx[h_head];
            if (front_i + q <= i) h_head = rb_inc(h_head, cap); else break;
        }
        const float mh = high[h_idx[h_head]];

        // Rolling MIN of LOW over q
        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_i = l_idx[last];
            if (low[last_i] >= lo) l_tail = last; else break;
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = i; l_tail = next_tail;
        while (l_head != l_tail) {
            const int front_i = l_idx[l_head];
            if (front_i + q <= i) l_head = rb_inc(l_head, cap); else break;
        }
        const float ml = low[l_idx[l_head]];

        if (i >= warm) {
            const float ls0 = fmaf(-x, rma, mh); // mh - x*rma
            const float ss0 = fmaf(+x, rma, ml); // ml + x*rma

            // Rolling MAX over ls0
            while (ls_head != ls_tail) {
                const int last = rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) ls_tail = last; else break;
            }
            next_tail = rb_inc(ls_tail, cap);
            if (next_tail == ls_head) ls_head = rb_inc(ls_head, cap);
            ls_idx[ls_tail] = i; ls_val[ls_tail] = ls0; ls_tail = next_tail;
            while (ls_head != ls_tail) {
                const int front_i = ls_idx[ls_head];
                if (front_i + q <= i) ls_head = rb_inc(ls_head, cap); else break;
            }
            out_long_row[i] = ls_val[ls_head];

            // Rolling MIN over ss0
            while (ss_head != ss_tail) {
                const int last = rb_dec(ss_tail, cap);
                if (ss_val[last] >= ss0) ss_tail = last; else break;
            }
            next_tail = rb_inc(ss_tail, cap);
            if (next_tail == ss_head) ss_head = rb_inc(ss_head, cap);
            ss_idx[ss_tail] = i; ss_val[ss_tail] = ss0; ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_i = ss_idx[ss_head];
                if (front_i + q <= i) ss_head = rb_inc(ss_head, cap); else break;
            }
            out_short_row[i] = ss_val[ss_head];
        }
    }
}

extern "C" __global__
void cksp_batch_f32(const float* __restrict__ high,
                    const float* __restrict__ low,
                    const float* __restrict__ close,
                    int series_len,
                    int first_valid,
                    const int* __restrict__ p_list,
                    const float* __restrict__ x_list,
                    const int* __restrict__ q_list,
                    int n_combos,
                    int cap_max,
                    float* __restrict__ out_long,
                    float* __restrict__ out_short) {
    const int row = blockIdx.y;
    if (row >= n_combos || series_len <= 0) return;
    if (blockIdx.x != 0) return; // grid.x must be 1

    const int base = row * series_len;
    cksp_core_row<false>(
        high, low, close, /*tr_opt*/nullptr, series_len, first_valid,
        p_list[row], x_list[row], q_list[row], cap_max,
        /*row output views*/ out_long + base, out_short + base
    );
}

// -------------------------------------------------------------------------------------------------
// One series × many rows, but with precomputed TR (recommended when combos is large).
//
extern "C" __global__
void cksp_batch_f32_pretr(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          const float* __restrict__ tr,   // <- precomputed TR
                          int series_len,
                          int first_valid,
                          const int* __restrict__ p_list,
                          const float* __restrict__ x_list,
                          const int* __restrict__ q_list,
                          int n_combos,
                          int cap_max,
                          float* __restrict__ out_long,
                          float* __restrict__ out_short) {
    const int row = blockIdx.y;
    if (row >= n_combos || series_len <= 0) return;
    if (blockIdx.x != 0) return; // grid.x must be 1

    const int base = row * series_len;
    cksp_core_row<true>(
        high, low, close, tr, series_len, first_valid,
        p_list[row], x_list[row], q_list[row], cap_max,
        out_long + base, out_short + base
    );
}

// Many-series × one param (time-major: [t][series])
extern "C" __global__
void cksp_many_series_one_param_f32(const float* __restrict__ high_tm,
                                    const float* __restrict__ low_tm,
                                    const float* __restrict__ close_tm,
                                    const int* __restrict__ first_valids,
                                    int num_series,
                                    int series_len,
                                    int p,
                                    float x,
                                    int q,
                                    int cap_max,
                                    float* __restrict__ out_long_tm,
                                    float* __restrict__ out_short_tm) {
    const int s = blockIdx.x; // one block per series
    if (s >= num_series || series_len <= 0 || p <= 0 || q <= 0) return;
    const int stride = num_series; // time-major
    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    if (fv >= series_len) return;
    const int warm = fv + p + q - 1;
    const int cap  = q + 1;

    // Parallel NaN init for this series
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_long_tm[t * stride + s]  = CUDART_NAN_F;
        out_short_tm[t * stride + s] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    extern __shared__ __align__(16) unsigned char shraw[];
    int*   h_idx  = (int*)shraw;
    int*   l_idx  = h_idx + cap_max;
    int*   ls_idx = l_idx + cap_max;
    int*   ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;
    float sum_tr = 0.0f, c_tr = 0.0f;

    // t = fv (first sample): TR = hi - lo ; initialize deques if q>1
    {
        const int t = fv;
        const float hi = high_tm[t * stride + s];
        const float lo = low_tm [t * stride + s];
        const float tr = hi - lo;
        // warmup k=0
        float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
        if (p == 1) rma = tr;

        if (q > 1) {
            while (h_head != h_tail) {
                const int last = rb_dec(h_tail, cap);
                const int last_t = h_idx[last];
                const float last_v = high_tm[last_t * stride + s];
                if (last_v <= hi) h_tail = last; else break;
            }
            int next_tail = rb_inc(h_tail, cap);
            if (next_tail == h_head) h_head = rb_inc(h_head, cap);
            h_idx[h_tail] = t; h_tail = next_tail;

            while (l_head != l_tail) {
                const int last = rb_dec(l_tail, cap);
                const int last_t = l_idx[last];
                const float last_v = low_tm[last_t * stride + s];
                if (last_v >= lo) l_tail = last; else break;
            }
            next_tail = rb_inc(l_tail, cap);
            if (next_tail == l_head) l_head = rb_inc(l_head, cap);
            l_idx[l_tail] = t; l_tail = next_tail;
        }
    }

    // main loop
    if (q == 1) {
        int k = 1;
        float prev_close = close_tm[fv * stride + s];
        for (int t = fv + 1; t < series_len; ++t, ++k) {
            const float hi = high_tm [t * stride + s];
            const float lo = low_tm  [t * stride + s];
            const float clp= prev_close;
            prev_close     = close_tm[t * stride + s];

            const float tr = fmaxf(hi - lo, fmaxf(fabsf(hi - clp), fabsf(lo - clp)));
            if (k < p) {
                float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
                if (k == p - 1) rma = sum_tr / (float)p;
            } else {
                rma = fmaf(alpha, tr - rma, rma);
            }
            if (t >= warm) {
                out_long_tm [t * stride + s] = fmaf(-x, rma, hi);
                out_short_tm[t * stride + s] = fmaf(+x, rma, lo);
            }
        }
        return;
    }

    int k = 1;
    float prev_close = close_tm[fv * stride + s];
    for (int t = fv + 1; t < series_len; ++t, ++k) {
        const float hi = high_tm[t * stride + s];
        const float lo = low_tm [t * stride + s];
        const float clp= prev_close;
        prev_close     = close_tm[t * stride + s];

        const float tr = fmaxf(hi - lo, fmaxf(fabsf(hi - clp), fabsf(lo - clp)));
        if (k < p) {
            float y = tr - c_tr; float tt = sum_tr + y; c_tr = (tt - sum_tr) - y; sum_tr = tt;
            if (k == p - 1) rma = sum_tr / (float)p;
        } else {
            rma = fmaf(alpha, tr - rma, rma);
        }

        // rolling MAX high over q
        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_t = h_idx[last];
            const float last_v = high_tm[last_t * stride + s];
            if (last_v <= hi) h_tail = last; else break;
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap);
        h_idx[h_tail] = t; h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_t = h_idx[h_head];
            if (front_t + q <= t) h_head = rb_inc(h_head, cap); else break;
        }
        const float mh = high_tm[h_idx[h_head] * stride + s];

        // rolling MIN low over q
        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_t = l_idx[last];
            const float last_v = low_tm[last_t * stride + s];
            if (last_v >= lo) l_tail = last; else break;
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = t; l_tail = next_tail;
        while (l_head != l_tail) {
            const int front_t = l_idx[l_head];
            if (front_t + q <= t) l_head = rb_inc(l_head, cap); else break;
        }
        const float ml = low_tm[l_idx[l_head] * stride + s];

        if (t >= warm) {
            const float ls0 = fmaf(-x, rma, mh);
            const float ss0 = fmaf(+x, rma, ml);

            while (ls_head != ls_tail) {
                const int last = rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) ls_tail = last; else break;
            }
            next_tail = rb_inc(ls_tail, cap);
            if (next_tail == ls_head) ls_head = rb_inc(ls_head, cap);
            ls_idx[ls_tail] = t; ls_val[ls_tail] = ls0; ls_tail = next_tail;
            while (ls_head != ls_tail) {
                const int front_t = ls_idx[ls_head];
                if (front_t + q <= t) ls_head = rb_inc(ls_head, cap); else break;
            }
            out_long_tm[t * stride + s] = ls_val[ls_head];

            while (ss_head != ss_tail) {
                const int last = rb_dec(ss_tail, cap);
                if (ss_val[last] >= ss0) ss_tail = last; else break;
            }
            next_tail = rb_inc(ss_tail, cap);
            if (next_tail == ss_head) ss_head = rb_inc(ss_head, cap);
            ss_idx[ss_tail] = t; ss_val[ss_tail] = ss0; ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_t = ss_idx[ss_head];
                if (front_t + q <= t) ss_head = rb_inc(ss_head, cap); else break;
            }
            out_short_tm[t * stride + s] = ss_val[ss_head];
        }
    }
}

