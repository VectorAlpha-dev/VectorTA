// CUDA kernels for Chande Kroll Stop (CKSP)
//
// Semantics mirror the scalar Rust implementation in src/indicators/cksp.rs:
// - Inputs: high, low, close (f32)
// - Parameters per combo: p (ATR period), x (multiplier), q (rolling window)
// - ATR is an RMA with alpha = 1/p and a sum warmup for the first p samples.
// - Rolling max/min are implemented with monotonic ring deques of capacity q+1.
// - Warmup and NaN behavior: write NaN until index = first_valid + p + q - 1.
//
// Two entry points are provided:
// - cksp_batch_f32: one series × many parameter rows (grid.y = combos)
// - cksp_many_series_one_param_f32: many series × one param (time-major layout)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// Helpers for ring buffer index arithmetic
__device__ __forceinline__ int rb_inc(int idx, int cap) { return (idx + 1) >= cap ? 0 : idx + 1; }
__device__ __forceinline__ int rb_dec(int idx, int cap) { return (idx == 0) ? (cap - 1) : (idx - 1); }

// Shared-memory layout helpers
// We allocate dynamic shared memory sized for the maximum q across all rows.
// Layout per CTA (combination):
//   int  h_idx[cap]
//   int  l_idx[cap]
//   int  ls_idx[cap]
//   int  ss_idx[cap]
//   float ls_val[cap]
//   float ss_val[cap]

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
    if (series_len <= 0) return;
    const int row = blockIdx.y;
    if (row >= n_combos) return;

    // Initialize outputs for this row to NaN in parallel
    const int base = row * series_len;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < series_len; i += blockDim.x * gridDim.x) {
        out_long[base + i] = CUDART_NAN_F;
        out_short[base + i] = CUDART_NAN_F;
    }
    __syncthreads();

    if (threadIdx.x != 0 || blockIdx.x != 0) return; // single producer per row

    const int p = p_list[row];
    const float x = x_list[row];
    const int q = q_list[row];
    if (p <= 0 || q <= 0) return;
    const int cap = q + 1;
    const int warm = (first_valid < 0 ? 0 : first_valid) + p + q - 1;
    if (first_valid >= series_len) return;

    extern __shared__ __align__(16) unsigned char shraw[];
    int* h_idx = (int*)shraw;
    int* l_idx = h_idx + cap_max;
    int* ls_idx = l_idx + cap_max;
    int* ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    float sum_tr = 0.0f;
    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;
    float prev_close = CUDART_NAN_F;

    // Sequential scan
    for (int i = max(0, first_valid); i < series_len; ++i) {
        const float hi = high[i];
        const float lo = low[i];
        const float cl = close[i];

        // True range
        float tr;
        if (!isfinite(prev_close)) {
            tr = hi - lo;
        } else {
            const float hl = hi - lo;
            const float hc = fabsf(hi - prev_close);
            const float lc = fabsf(lo - prev_close);
            tr = fmaxf(hl, fmaxf(hc, lc));
        }
        prev_close = cl;

        // RMA warmup, then recursion
        const int k = i - first_valid;
        if (k < p) {
            sum_tr += tr;
            if (k == p - 1) {
                rma = sum_tr / (float)p;
            }
        } else {
            // rma += alpha * (tr - rma)
            rma = fmaf(alpha, tr - rma, rma);
        }

        // Rolling MAX of HIGH over q using monotonic deque (indices only)
        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_i = h_idx[last];
            if (high[last_i] <= hi) {
                h_tail = last; // pop_back
            } else {
                break;
            }
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap); // full, drop oldest
        h_idx[h_tail] = i;
        h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_i = h_idx[h_head];
            if (front_i + q <= i) {
                h_head = rb_inc(h_head, cap);
            } else {
                break;
            }
        }
        const float mh = high[h_idx[h_head]];

        // Rolling MIN of LOW over q
        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_i = l_idx[last];
            if (low[last_i] >= lo) {
                l_tail = last;
            } else {
                break;
            }
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = i;
        l_tail = next_tail;
        while (l_head != l_tail) {
            const int front_i = l_idx[l_head];
            if (front_i + q <= i) {
                l_head = rb_inc(l_head, cap);
            } else {
                break;
            }
        }
        const float ml = low[l_idx[l_head]];

        if (i >= warm) {
            const float ls0 = fmaf(-x, rma, mh); // mh - x*rma
            const float ss0 = fmaf(+x, rma, ml); // ml + x*rma

            // Rolling MAX over ls0
            while (ls_head != ls_tail) {
                const int last = rb_dec(ls_tail, cap);
                if (ls_val[last] <= ls0) {
                    ls_tail = last;
                } else {
                    break;
                }
            }
            next_tail = rb_inc(ls_tail, cap);
            if (next_tail == ls_head) ls_head = rb_inc(ls_head, cap);
            ls_idx[ls_tail] = i;
            ls_val[ls_tail] = ls0;
            ls_tail = next_tail;
            while (ls_head != ls_tail) {
                const int front_i = ls_idx[ls_head];
                if (front_i + q <= i) {
                    ls_head = rb_inc(ls_head, cap);
                } else {
                    break;
                }
            }
            out_long[base + i] = ls_val[ls_head];

            // Rolling MIN over ss0
            while (ss_head != ss_tail) {
                const int last = rb_dec(ss_tail, cap);
                if (ss_val[last] >= ss0) {
                    ss_tail = last;
                } else {
                    break;
                }
            }
            next_tail = rb_inc(ss_tail, cap);
            if (next_tail == ss_head) ss_head = rb_inc(ss_head, cap);
            ss_idx[ss_tail] = i;
            ss_val[ss_tail] = ss0;
            ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_i = ss_idx[ss_head];
                if (front_i + q <= i) {
                    ss_head = rb_inc(ss_head, cap);
                } else {
                    break;
                }
            }
            out_short[base + i] = ss_val[ss_head];
        }
    }
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
    const int warm = fv + p + q - 1;

    // Init outputs for this series to NaN in parallel
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_long_tm[t * stride + s] = CUDART_NAN_F;
        out_short_tm[t * stride + s] = CUDART_NAN_F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (fv >= series_len) return;

    extern __shared__ __align__(16) unsigned char shraw[];
    int* h_idx = (int*)shraw;
    int* l_idx = h_idx + cap_max;
    int* ls_idx = l_idx + cap_max;
    int* ss_idx = ls_idx + cap_max;
    float* ls_val = (float*)(ss_idx + cap_max);
    float* ss_val = ls_val + cap_max;
    const int cap = q + 1;

    int h_head = 0, h_tail = 0;
    int l_head = 0, l_tail = 0;
    int ls_head = 0, ls_tail = 0;
    int ss_head = 0, ss_tail = 0;

    float sum_tr = 0.0f;
    float rma = 0.0f;
    const float alpha = 1.0f / (float)p;
    float prev_close = CUDART_NAN_F;

    for (int t = fv; t < series_len; ++t) {
        const float hi = high_tm[t * stride + s];
        const float lo = low_tm[t * stride + s];
        const float cl = close_tm[t * stride + s];

        float tr;
        if (!isfinite(prev_close)) {
            tr = hi - lo;
        } else {
            const float hl = hi - lo;
            const float hc = fabsf(hi - prev_close);
            const float lc = fabsf(lo - prev_close);
            tr = fmaxf(hl, fmaxf(hc, lc));
        }
        prev_close = cl;

        const int k = t - fv;
        if (k < p) {
            sum_tr += tr;
            if (k == p - 1) rma = sum_tr / (float)p;
        } else {
            rma = fmaf(alpha, tr - rma, rma);
        }

        // rolling MAX of high over q
        while (h_head != h_tail) {
            const int last = rb_dec(h_tail, cap);
            const int last_t = h_idx[last];
            const float last_v = high_tm[last_t * stride + s];
            if (last_v <= hi) h_tail = last; else break;
        }
        int next_tail = rb_inc(h_tail, cap);
        if (next_tail == h_head) h_head = rb_inc(h_head, cap);
        h_idx[h_tail] = t;
        h_tail = next_tail;
        while (h_head != h_tail) {
            const int front_t = h_idx[h_head];
            if (front_t + q <= t) h_head = rb_inc(h_head, cap); else break;
        }
        const float mh = high_tm[h_idx[h_head] * stride + s];

        // rolling MIN of low over q
        while (l_head != l_tail) {
            const int last = rb_dec(l_tail, cap);
            const int last_t = l_idx[last];
            const float last_v = low_tm[last_t * stride + s];
            if (last_v >= lo) l_tail = last; else break;
        }
        next_tail = rb_inc(l_tail, cap);
        if (next_tail == l_head) l_head = rb_inc(l_head, cap);
        l_idx[l_tail] = t;
        l_tail = next_tail;
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
            ls_idx[ls_tail] = t;
            ls_val[ls_tail] = ls0;
            ls_tail = next_tail;
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
            ss_idx[ss_tail] = t;
            ss_val[ss_tail] = ss0;
            ss_tail = next_tail;
            while (ss_head != ss_tail) {
                const int front_t = ss_idx[ss_head];
                if (front_t + q <= t) ss_head = rb_inc(ss_head, cap); else break;
            }
            out_short_tm[t * stride + s] = ss_val[ss_head];
        }
    }
}

