// CUDA kernels for SRSI (Stochastic RSI)
//
// Math pattern: recurrence + sliding-window extrema with SMA smoothing.
// - Batch (one series × many params): reuse precomputed RSI (host), per-param
//   deques for min/max and SMA rings for K/D. Single block per combo.
// - Many-series, one param (time-major): per-series RSI built on the fly
//   (Wilder), deques for extrema, SMA rings for K/D. One block per series.
//
// Semantics parity with scalar:
// - Warmup indices: rsi_warmup = first_valid + rsi_period;
//                   stoch_warmup = rsi_warmup + stoch - 1;
//                   k_warmup = stoch_warmup + k - 1;
//                   d_warmup = k_warmup + d - 1;
// - Outputs are NaN before their warmups; division by zero yields 50.0 for
//   Fast %K; Slow K/D written only from their warmups onward.
// - FP32 arithmetic; use minimal FMAs where natural.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Simple monotonic deque storing (idx, value) pairs in ring buffers residing
// in dynamic shared memory. The caller provides capacity (window size).
struct Deque {
    int*   idx;   // capacity elements
    float* val;   // capacity elements
    int    cap;   // window length
    int    head;  // points to first valid element
    int    tail;  // points to next write position
};

__device__ __forceinline__ void dq_init(Deque* d, int* idx_buf, float* val_buf, int cap) {
    d->idx = idx_buf; d->val = val_buf; d->cap = cap; d->head = 0; d->tail = 0;
}
__device__ __forceinline__ bool dq_empty(const Deque* d) { return d->head == d->tail; }
__device__ __forceinline__ int dq_dec(const Deque* d, int x) { return (x == 0 ? d->cap - 1 : x - 1); }
__device__ __forceinline__ int dq_inc(const Deque* d, int x) { return (x + 1 == d->cap ? 0 : x + 1); }
__device__ __forceinline__ void dq_expire(Deque* d, int start_idx) {
    if (!dq_empty(d)) {
        int front = d->idx[d->head];
        if (front < start_idx) { d->head = dq_inc(d, d->head); }
    }
}
// push for max-deque: keep values non-increasing
__device__ __forceinline__ void dq_push_max(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head) {
            if (d->val[pos] >= v) break;
            t = pos; pos = dq_dec(d, pos);
        }
        // Check head element as well
        if (pos == d->head && !dq_empty(d) && d->val[pos] < v) {
            t = d->head;
            d->head = dq_inc(d, d->head);
        }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}
// push for min-deque: keep values non-decreasing
__device__ __forceinline__ void dq_push_min(Deque* d, int idx, float v) {
    int t = d->tail;
    if (!dq_empty(d)) {
        int pos = dq_dec(d, t);
        while (pos != d->head) {
            if (d->val[pos] <= v) break;
            t = pos; pos = dq_dec(d, pos);
        }
        if (pos == d->head && !dq_empty(d) && d->val[pos] > v) {
            t = d->head;
            d->head = dq_inc(d, d->head);
        }
    }
    d->idx[t] = idx; d->val[t] = v; d->tail = dq_inc(d, t);
}

// ------------------- Batch: one series × many params -----------------------
// Inputs:
//  - rsi: precomputed RSI values (FP32), length = series_len
//  - stoch_periods, k_periods, d_periods: per-combo arrays (length = n_combos)
//  - series_len, first_valid, rsi_period
// Outputs:
//  - out_k, out_d: row-major [n_combos x series_len]
extern "C" __global__
void srsi_batch_f32(const float* __restrict__ rsi,
                    const int* __restrict__ stoch_periods,
                    const int* __restrict__ k_periods,
                    const int* __restrict__ d_periods,
                    int series_len,
                    int first_valid,
                    int rsi_period,
                    int n_combos,
                    float* __restrict__ out_k,
                    float* __restrict__ out_d) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int sp = stoch_periods[combo];
    const int kp = k_periods[combo];
    const int dp = d_periods[combo];
    if (sp <= 0 || kp <= 0 || dp <= 0) return;

    const int rsi_warmup   = first_valid + rsi_period;
    const int stoch_warmup = rsi_warmup + sp - 1;
    const int k_warmup     = stoch_warmup + kp - 1;
    const int d_warmup     = k_warmup + dp - 1;
    if (rsi_warmup >= series_len) return;

    float* row_k = out_k + combo * series_len;
    float* row_d = out_d + combo * series_len;

    // Initialize rows to NaN (warmup semantics)
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        row_k[i] = NAN;
        row_d[i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // sequential scan per combo

    extern __shared__ unsigned char smem[];
    // Layout: [max_idx sp]*int | [max_val sp]*float | [min_idx sp]*int | [min_val sp]*float |
    //         [ring_k kp]*float | [ring_d dp]*float
    int*   max_idx = (int*)smem;
    float* max_val = (float*)(max_idx + sp);
    int*   min_idx = (int*)(max_val + sp);
    float* min_val = (float*)(min_idx + sp);
    float* ring_k  = (float*)(min_val + sp);
    float* ring_d  = (float*)(ring_k + kp);

    Deque dq_max, dq_min;
    dq_init(&dq_max, max_idx, max_val, sp);
    dq_init(&dq_min, min_idx, min_val, sp);

    // Rolling sums for SMA of Fast %K (slow K) and Slow %D
    float sum_k = 0.0f, sum_d = 0.0f;
    int head_k = 0, head_d = 0, cnt_k = 0, cnt_d = 0;

    // Prime deques with first (sp-1) RSI values after rsi_warmup
    const int base = rsi_warmup;
    for (int t = 0; t < sp - 1 && (base + t) < series_len; ++t) {
        const float rv = rsi[base + t];
        dq_push_max(&dq_max, base + t, rv);
        dq_push_min(&dq_min, base + t, rv);
    }

    for (int i = stoch_warmup; i < series_len; ++i) {
        // Push current RSI into deques, then expire old
        const float rv = rsi[i];
        dq_push_max(&dq_max, i, rv);
        dq_push_min(&dq_min, i, rv);
        const int win_start = i - sp + 1;
        dq_expire(&dq_max, win_start);
        dq_expire(&dq_min, win_start);

        float hi = dq_empty(&dq_max) ? rv : dq_max.val[dq_max.head];
        float lo = dq_empty(&dq_min) ? rv : dq_min.val[dq_min.head];
        float fk;
        if (isfinite(hi) && isfinite(lo) && hi > lo) {
            fk = ((rv - lo) * 100.0f) / (hi - lo);
        } else {
            fk = 50.0f;
        }

        if (cnt_k < kp) { sum_k += fk; ring_k[head_k] = fk; head_k = (head_k + 1) % kp; ++cnt_k; }
        else             { sum_k += fk - ring_k[head_k]; ring_k[head_k] = fk; head_k = (head_k + 1) % kp; }

        if (i >= k_warmup) {
            const float sk = sum_k / (float)kp;
            row_k[i] = sk;
            if (cnt_d < dp) { sum_d += sk; ring_d[head_d] = sk; head_d = (head_d + 1) % dp; ++cnt_d; }
            else             { sum_d += sk - ring_d[head_d]; ring_d[head_d] = sk; head_d = (head_d + 1) % dp; }
            if (i >= d_warmup) { row_d[i] = sum_d / (float)dp; }
        }
    }
}

// --------------- Many-series: one param (time-major inputs) ----------------
// prices_tm: time-major [rows x cols], stride = cols
// first_valids: per-series first non-NaN time index
// Outputs are time-major arrays for K and D
extern "C" __global__
void srsi_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    int cols,
                                    int rows,
                                    int rsi_period,
                                    int stoch_period,
                                    int k_period,
                                    int d_period,
                                    const int* __restrict__ first_valids,
                                    float* __restrict__ k_out_tm,
                                    float* __restrict__ d_out_tm) {
    const int s = blockIdx.x; // series index
    if (s >= cols) return;
    if (rsi_period <= 0 || stoch_period <= 0 || k_period <= 0 || d_period <= 0) return;

    const int stride = cols;
    int first = first_valids[s]; if (first < 0) first = 0; if (first >= rows) return;
    const int rsi_warmup   = first + rsi_period;
    const int stoch_warmup = rsi_warmup + stoch_period - 1;
    const int k_warmup     = stoch_warmup + k_period - 1;
    const int d_warmup     = k_warmup + d_period - 1;

    // Initialize warmup to NaN in parallel
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        if (t < k_warmup) { k_out_tm[t * stride + s] = NAN; }
        if (t < d_warmup) { d_out_tm[t * stride + s] = NAN; }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    // Wilder RSI: seed average gain/loss over first rsi_period deltas
    float avg_gain = 0.0f, avg_loss = 0.0f;
    float prev = prices_tm[first * stride + s];
    for (int i = first + 1; i <= first + rsi_period && i < rows; ++i) {
        float cur = prices_tm[i * stride + s];
        float ch = cur - prev; prev = cur;
        if (ch > 0.0f) avg_gain += ch; else avg_loss += -ch;
    }
    avg_gain /= (float)rsi_period; avg_loss /= (float)rsi_period;
    float rsi_prev = 50.0f;
    if (rsi_warmup < rows) {
        rsi_prev = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
    }
    const float alpha = 1.0f / (float)rsi_period;

    // Deques and rings from dynamic shared memory
    extern __shared__ unsigned char smem2[];
    int*   max_idx = (int*)smem2;
    float* max_val = (float*)(max_idx + stoch_period);
    int*   min_idx = (int*)(max_val + stoch_period);
    float* min_val = (float*)(min_idx + stoch_period);
    float* ring_k  = (float*)(min_val + stoch_period);
    float* ring_d  = (float*)(ring_k + k_period);
    Deque dq_max, dq_min; dq_init(&dq_max, max_idx, max_val, stoch_period); dq_init(&dq_min, min_idx, min_val, stoch_period);

    // Prime deques with first (sp-1) RSI values
    for (int t = rsi_warmup; t < rsi_warmup + stoch_period - 1 && t < rows; ++t) {
        float x = prices_tm[t * stride + s];
        // Wilder update to next RSI
        float gain = 0.0f, loss = 0.0f; float prevp = prices_tm[(t-1) * stride + s];
        float ch = x - prevp; if (ch > 0.0f) gain = ch; else loss = -ch;
        avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
        avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
        float rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));
        dq_push_max(&dq_max, t, rsi);
        dq_push_min(&dq_min, t, rsi);
        rsi_prev = rsi;
    }

    float sum_k = 0.0f, sum_d = 0.0f; int head_k = 0, head_d = 0, cnt_k = 0, cnt_d = 0;
    for (int t = stoch_warmup; t < rows; ++t) {
        float x = prices_tm[t * stride + s];
        float prevp = prices_tm[(t - 1) * stride + s];
        float ch = x - prevp; float gain = (ch > 0.0f ? ch : 0.0f); float loss = (ch < 0.0f ? -ch : 0.0f);
        avg_gain = fmaf(gain - avg_gain, alpha, avg_gain);
        avg_loss = fmaf(loss - avg_loss, alpha, avg_loss);
        float rsi = (avg_loss == 0.0f) ? 100.0f : (100.0f - 100.0f / (1.0f + avg_gain / avg_loss));

        dq_push_max(&dq_max, t, rsi);
        dq_push_min(&dq_min, t, rsi);
        const int start = t - stoch_period + 1; dq_expire(&dq_max, start); dq_expire(&dq_min, start);

        float hi = dq_empty(&dq_max) ? rsi : dq_max.val[dq_max.head];
        float lo = dq_empty(&dq_min) ? rsi : dq_min.val[dq_min.head];
        float fk = (isfinite(hi) && isfinite(lo) && hi > lo) ? ((rsi - lo) * 100.0f) / (hi - lo) : 50.0f;

        if (cnt_k < k_period) { sum_k += fk; ring_k[head_k] = fk; head_k = (head_k + 1) % k_period; ++cnt_k; }
        else                   { sum_k += fk - ring_k[head_k]; ring_k[head_k] = fk; head_k = (head_k + 1) % k_period; }
        if (t >= k_warmup) {
            float sk = sum_k / (float)k_period; k_out_tm[t * stride + s] = sk;
            if (cnt_d < d_period) { sum_d += sk; ring_d[head_d] = sk; head_d = (head_d + 1) % d_period; ++cnt_d; }
            else                   { sum_d += sk - ring_d[head_d]; ring_d[head_d] = sk; head_d = (head_d + 1) % d_period; }
            if (t >= d_warmup) { d_out_tm[t * stride + s] = sum_d / (float)d_period; }
        }
        rsi_prev = rsi;
    }
}

