// CUDA kernels for Aroon indicator (Up/Down)
//
// Optimized O(n) per-length implementation using monotonic deques for
// sliding-window max(high) and min(low). Preserves scalar semantics:
// - Warmup: indices before (first_valid + length) are NaN.
// - Window: inclusive [t - length .. t] (length+1 samples).
// - NaN: any non‑finite high/low in the window -> both outputs NaN.
// - Ties: "earlier index wins" (strict < for max-pop, strict > for min-pop).
// - FP32 only; use fmaf with integer guards for exact 0/100 edges.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __forceinline__ __device__ bool both_finite(float h, float l) {
    return isfinite(h) && isfinite(l);
}

// ------------------------------
// One price series × many params
// ------------------------------
extern "C" __global__
void aroon_batch_f32(const float* __restrict__ high,   // single price series
                     const float* __restrict__ low,    // single price series
                     const int*   __restrict__ lengths,
                     int series_len,
                     int first_valid,
                     int n_combos,
                     float* __restrict__ out_up,       // [n_combos * series_len]
                     float* __restrict__ out_down) {   // [n_combos * series_len]
    // Each block handles 1 combo; support y-chunked launches.
    const int combo = blockIdx.x + blockIdx.y * gridDim.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    if (length <= 0 || first_valid < 0 || first_valid >= series_len) return;

    const int base = combo * series_len;
    const int W = length + 1;
    const int warm = first_valid + length;  // first index with a full window
    if (warm >= series_len) {
        // If warm never reached, fill all with NaN cooperatively and exit.
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_up  [base + i] = NAN;
            out_down[base + i] = NAN;
        }
        return;
    }

    // Fill only the warmup region [0, warm) with NaN cooperatively.
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_up  [base + i] = NAN;
        out_down[base + i] = NAN;
    }
    __syncthreads();

    // Dynamic shared memory: two int deques of capacity W (host ensures shmem size).
    extern __shared__ int s_deques[];
    int* __restrict__ dq_max = s_deques;        // indices for max(high), ring cap=W
    int* __restrict__ dq_min = s_deques + W;    // indices for min(low),  ring cap=W

    // Only one thread performs the streaming scan (sequential dependency).
    if (threadIdx.x != 0) return;

    // Deque heads/tails using [head, tail) index ranges.
    int h_head = 0, h_tail = 0;  // max(high) ring [head, tail)
    int l_head = 0, l_tail = 0;  // min(low)  ring [head, tail)

    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;  // last index with !both_finite

    // Stream through the series once.
    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;
        // Evict indices that fall left of the window from deque fronts.
        while (h_tail > h_head && dq_max[h_head % W] < start) ++h_head;
        while (l_tail > l_head && dq_min[l_head % W] < start) ++l_head;

        const float h = high[t];
        const float l = low[t];

        // Track non-finite and only push finite samples after eviction.
        if (!both_finite(h, l)) {
            last_bad = t;
        } else {
            // Push t into MAX deque for highs: pop strictly smaller to keep earliest on ties.
            while (h_tail > h_head) {
                const int idx = dq_max[(h_tail - 1) % W];
                if (high[idx] < h) --h_tail; else break;
            }
            dq_max[h_tail % W] = t; ++h_tail;
            // Push t into MIN deque for lows: pop strictly larger to keep earliest on ties.
            while (l_tail > l_head) {
                const int idx = dq_min[(l_tail - 1) % W];
                if (low[idx] > l) --l_tail; else break;
            }
            dq_min[l_tail % W] = t; ++l_tail;
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up  [base + t] = NAN;
                out_down[base + t] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max[h_head % W] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min[l_head % W] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up  [base + t] = NAN;
                    out_down[base + t] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;             // == length - best_h_off
                    const int dist_lo = t - idx_lo;
                    // Integer guards for exact edges; otherwise use one FMA.
                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up  [base + t] = up;
                    out_down[base + t] = dn;
                }
            }
        }
    }
}

// -------------------------------------------
// Many series × one param (time-major layout)
// -------------------------------------------
extern "C" __global__
void aroon_many_series_one_param_f32(const float* __restrict__ high_tm,   // [t][s]
                                     const float* __restrict__ low_tm,    // [t][s]
                                     const int*   __restrict__ first_valids,
                                     int length,
                                     int num_series,
                                     int series_len,
                                     float* __restrict__ out_up_tm,       // [t][s]
                                     float* __restrict__ out_down_tm) {   // [t][s]
    const int s = blockIdx.x;
    if (s >= num_series || length <= 0) return;

    const int first = first_valids[s];
    if (first < 0 || first >= series_len) {
        // Fill entire column with NaN (cooperatively) and exit.
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_up_tm  [t * num_series + s] = NAN;
            out_down_tm[t * num_series + s] = NAN;
        }
        return;
    }

    const int W = length + 1;
    const int warm = first + length;
    const int stride = num_series;

    // Prefill warmup region with NaN cooperatively.
    for (int t = threadIdx.x; t < (warm < series_len ? warm : series_len); t += blockDim.x) {
        out_up_tm  [t * stride + s] = NAN;
        out_down_tm[t * stride + s] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    extern __shared__ int s_deques[];
    int* __restrict__ dq_max = s_deques;      // capacity == W (ring)
    int* __restrict__ dq_min = s_deques + W;  // capacity == W (ring)

    int h_head = 0, h_tail = 0; // [head, tail)
    int l_head = 0, l_tail = 0; // [head, tail)
    const float scale = 100.0f / (float)length;
    int last_bad = -0x3fffffff;

    for (int t = 0; t < series_len; ++t) {
        const int start = t - length;
        while (h_tail > h_head && dq_max[h_head % W] < start) ++h_head;
        while (l_tail > l_head && dq_min[l_head % W] < start) ++l_head;

        const float h = high_tm[t * stride + s];
        const float l = low_tm [t * stride + s];

        if (!both_finite(h, l)) {
            last_bad = t;
        } else {
            // Push into max/min deques (strict inequalities to keep earlier ties).
            while (h_tail > h_head) {
                const int idx = dq_max[(h_tail - 1) % W];
                if (high_tm[idx * stride + s] < h) --h_tail; else break;
            }
            dq_max[h_tail % W] = t; ++h_tail;

            while (l_tail > l_head) {
                const int idx = dq_min[(l_tail - 1) % W];
                if (low_tm[idx * stride + s] > l) --l_tail; else break;
            }
            dq_min[l_tail % W] = t; ++l_tail;
        }

        if (t >= warm) {
            if (last_bad >= start) {
                out_up_tm  [t * stride + s] = NAN;
                out_down_tm[t * stride + s] = NAN;
            } else {
                const int idx_hi = (h_tail > h_head) ? dq_max[h_head % W] : -1;
                const int idx_lo = (l_tail > l_head) ? dq_min[l_head % W] : -1;
                if (idx_hi < 0 || idx_lo < 0) {
                    out_up_tm  [t * stride + s] = NAN;
                    out_down_tm[t * stride + s] = NAN;
                } else {
                    const int dist_hi = t - idx_hi;
                    const int dist_lo = t - idx_lo;
                    const float up = (dist_hi == 0) ? 100.0f
                                     : (dist_hi >= length ? 0.0f
                                     : fmaf(-(float)dist_hi, scale, 100.0f));
                    const float dn = (dist_lo == 0) ? 100.0f
                                     : (dist_lo >= length ? 0.0f
                                     : fmaf(-(float)dist_lo, scale, 100.0f));
                    out_up_tm  [t * stride + s] = up;
                    out_down_tm[t * stride + s] = dn;
                }
            }
        }
    }
}
