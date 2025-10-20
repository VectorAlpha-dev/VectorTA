// CUDA kernels for SafeZoneStop (Wilder DM-based stop levels)
// Optimized: dual-FP32 accumulator (no FP64 in hot path), thread-per-row/series mapping,
// small-window fast path (lb <= 4), and preserved monotonic deque ABI.
//
// Semantics preserved: warmup/NaN handling, direction, time indexing.
// Outputs remain FP32.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// -----------------------------
// Dual-FP32 (double-single) utils
// -----------------------------
struct dsf32 { float hi, lo; };  // value ~= hi + lo

__device__ __forceinline__ void two_sum_f(float a, float b, float& s, float& e) {
    // Error-free transform of a+b = s + e (Knuth/Dekker)
    float t = a + b;
    float bp = t - a;
    e = (a - (t - bp)) + (b - bp);
    s = t;
}

__device__ __forceinline__ dsf32 ds_from_sum(float s, float c) {
    dsf32 r;
    two_sum_f(s, c, r.hi, r.lo);
    return r;
}

__device__ __forceinline__ dsf32 ds_fma(const dsf32 y, float a, float x) {
    // Compute a*y + x in (hi,lo) with FMA-based error compensation.
    float p1 = a * y.hi;              // rounded product
    float e1 = fmaf(a, y.hi, -p1);    // product error
    float p2 = a * y.lo;
    float e2 = fmaf(a, y.lo, -p2);

    float s, es;       two_sum_f(p1, p2, s, es);
    float s2, ex;      two_sum_f(s,  x,  s2, ex);

    float corr = (e1 + e2) + (es + ex);
    float hi, lo;      two_sum_f(s2, corr, hi, lo);
    return {hi, lo};
}

// Cand = extreme(prev_extreme +/- mult * dm) with dsf32 dm
__device__ __forceinline__ float cand_long(float prev_low, float mult, const dsf32 dm) {
    // prev_low - mult*(dm.hi + dm.lo), with two FMAs for accuracy
    float t = fmaf(-mult, dm.hi, prev_low);
    return fmaf(-mult, dm.lo, t);
}
__device__ __forceinline__ float cand_short(float prev_high, float mult, const dsf32 dm) {
    float t = fmaf( mult, dm.hi, prev_high);
    return fmaf( mult, dm.lo, t);
}

// Small-window reducer (n <= 4)
__device__ __forceinline__ float reduce_max4(const float r[4], int n) {
    float m = r[0];
    if (n > 1) m = fmaxf(m, r[1]);
    if (n > 2) m = fmaxf(m, r[2]);
    if (n > 3) m = fmaxf(m, r[3]);
    return m;
}
__device__ __forceinline__ float reduce_min4(const float r[4], int n) {
    float m = r[0];
    if (n > 1) m = fminf(m, r[1]);
    if (n > 2) m = fminf(m, r[2]);
    if (n > 3) m = fminf(m, r[3]);
    return m;
}

extern "C" __global__
void safezonestop_batch_f32(const float* __restrict__ high,
                            const float* __restrict__ low,
                            const float* __restrict__ dm_raw, // length=len, zeros before/at `first`
                            int len,
                            int first,
                            const int*  __restrict__ periods,
                            const float* __restrict__ mults,
                            const int*  __restrict__ lookbacks,
                            int n_rows,
                            int dir_long,
                            // Optional workspace for a per-row monotonic deque (q_idx/q_val).
                            // Layout: row-major, stride = lb_cap per row.
                            int*   __restrict__ q_idx,
                            float* __restrict__ q_val,
                            int lb_cap,
                            float* __restrict__ out)
{
    // ---- Mapping: prefer one thread per row; but keep backward-compat with old grid.y launch ----
    int row = blockIdx.x * blockDim.x + threadIdx.x;             // new mapping (thread-per-row)
    if (gridDim.x == 1 && blockDim.x == 1) { row = blockIdx.y; } // compatibility for existing wrapper
    if (row >= n_rows) return;

    const int   period = periods[row];
    const float mult_f = mults[row];
    const int   lb     = lookbacks[row];
    const float nan_f  = CUDART_NAN_F;

    const int base = row * len;

    // Validate
    if (len <= 0 || period <= 0 || lb <= 0 || first < 0 || first >= len) {
        for (int i = 0; i < len; ++i) out[base + i] = nan_f;
        return;
    }

    const int end0 = first + period;
    const int warm = first + ((period > lb) ? period : lb) - 1;
    if (end0 >= len) {
        for (int i = 0; i < len; ++i) out[base + i] = nan_f;
        return;
    }

    // Prefix NaNs up to warm
    for (int i = 0; i <= warm && i < len; ++i) out[base + i] = nan_f;
    if (warm >= len - 1) return;

    // ---- Bootstrap Wilder smoothing: sum dm_raw[first+1 .. end0] with Neumaier compensation ----
    float sum = 0.0f, c = 0.0f;
    for (int j = first + 1; j <= end0; ++j) {
        float x = dm_raw[j];
        float t = sum + x;
        // Neumaier compensation: robust even if |x| > |sum|
        if (fabsf(sum) >= fabsf(x)) c += (sum - t) + x;
        else                        c += (x   - t) + sum;
        sum = t;
    }
    dsf32 dm = ds_from_sum(sum, c);

    // Stable alpha: alpha = 1 - 1/period computed with FMA (reduces cancellation)
    const float invp = 1.0f / (float)period;
    const float alpha_f = fmaf(-invp, 1.0f, 1.0f);  // 1 - invp, single rounding

    // ---- Rolling window strategy selection ----
    const bool have_deque = (q_idx != nullptr) && (q_val != nullptr) && (lb_cap >= (lb + 1));
    const bool small_win  = (lb <= 4);  // tiny, register-only ring

    // Setup deque views (if used)
    int *qidx = nullptr; float *qv = nullptr;
    int q_head = 0, q_tail = 0, q_len = 0;
    if (have_deque) {
        qidx = q_idx + row * lb_cap;
        qv   = q_val + row * lb_cap;
    }
    auto ring_inc = [&](int x) { int y = x + 1; return (y == lb_cap) ? 0 : y; };
    auto ring_dec = [&](int x) { return (x == 0) ? (lb_cap - 1) : (x - 1); };

    // Small-window register ring
    float ringv[4]; int rpos = 0, rcount = 0;

    // ---- Emit at i = end0 using bootstrapped dm ----
    if (dir_long) {
        // LONG path
        float prev_lm1 = low[end0 - 1];                // low[i-1] at i=end0
        float cand = cand_long(prev_lm1, mult_f, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
            if (end0 >= warm) out[base + end0] = reduce_max4(ringv, rcount);
        } else if (have_deque) {
            int start = end0 + 1 - lb;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; }
            qidx[q_tail] = end0; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (end0 >= warm && q_len > 0) out[base + end0] = qv[q_head];
        } else {
            if (end0 >= warm) out[base + end0] = cand;  // degenerate (window size 1)
        }

        float prev_l = low[end0];

        // ---- Main scan ----
        for (int i = end0 + 1; i < len; ++i) {
            float drm = dm_raw[i];
            dm = ds_fma(dm, alpha_f, drm);            // Wilder EMA in dsf32

            float cand_i = cand_long(prev_l, mult_f, dm);

            if (small_win) {
                ringv[rpos] = cand_i; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
                if (i >= warm) out[base + i] = reduce_max4(ringv, rcount);
            } else if (have_deque) {
                int start = i + 1 - lb;
                while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
                while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand_i) { q_tail = last; --q_len; } else break; }
                qidx[q_tail] = i; qv[q_tail] = cand_i; q_tail = ring_inc(q_tail); ++q_len;
                if (i >= warm && q_len > 0) out[base + i] = qv[q_head];
            } else {
                if (i >= warm) out[base + i] = cand_i;
            }

            prev_l = low[i];
        }
    } else {
        // SHORT path
        float prev_hm1 = high[end0 - 1];
        float cand = cand_short(prev_hm1, mult_f, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
            if (end0 >= warm) out[base + end0] = reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = end0 + 1 - lb;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; }
            qidx[q_tail] = end0; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (end0 >= warm && q_len > 0) out[base + end0] = qv[q_head];
        } else {
            if (end0 >= warm) out[base + end0] = cand;
        }

        float prev_h = high[end0];

        for (int i = end0 + 1; i < len; ++i) {
            float drm = dm_raw[i];
            dm = ds_fma(dm, alpha_f, drm);

            float cand_i = cand_short(prev_h, mult_f, dm);

            if (small_win) {
                ringv[rpos] = cand_i; rpos = (rpos + 1 == lb ? 0 : rpos + 1); if (rcount < lb) ++rcount;
                if (i >= warm) out[base + i] = reduce_min4(ringv, rcount);
            } else if (have_deque) {
                int start = i + 1 - lb;
                while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
                while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand_i) { q_tail = last; --q_len; } else break; }
                qidx[q_tail] = i; qv[q_tail] = cand_i; q_tail = ring_inc(q_tail); ++q_len;
                if (i >= warm && q_len > 0) out[base + i] = qv[q_head];
            } else {
                if (i >= warm) out[base + i] = cand_i;
            }

            prev_h = high[i];
        }
    }
}

extern "C" __global__
void safezonestop_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    int cols,              // number of series (time-major width)
    int rows,              // series length (time dimension)
    int period,
    float mult,
    int max_lookback,
    const int* __restrict__ first_valids, // per-series first index where (high,low) are finite
    int dir_long,
    // Deque workspace per series (optional). When provided, capacity must be >= max_lookback+1.
    int*   __restrict__ q_idx_tm,
    float* __restrict__ q_val_tm,
    int lb_cap,
    float* __restrict__ out_tm)
{
    // ---- Mapping: one thread per series ----
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int first = first_valids[s];
    const int len   = rows;
    const float nan_f = CUDART_NAN_F;

    auto at = [cols](const float* buf, int t, int ss) { return buf[t * cols + ss]; };

    if (first < 0 || first >= len || period <= 0 || max_lookback <= 0) {
        for (int t = 0; t < len; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    const int end0 = first + period;
    const int warm = first + ((period > max_lookback) ? period : max_lookback) - 1;
    if (end0 >= len) {
        for (int t = 0; t < len; ++t) out_tm[t * cols + s] = nan_f;
        return;
    }

    for (int t = 0; t <= warm && t < len; ++t) out_tm[t * cols + s] = nan_f;
    if (warm >= len - 1) return;

    // Bootstrap Wilder smoothing from raw bars using Neumaier-compensated float
    float sum = 0.0f, c = 0.0f;
    float prev_h = at(high_tm, first, s);
    float prev_l = at(low_tm,  first, s);
    for (int t = first + 1; t <= end0; ++t) {
        float h = at(high_tm, t, s);
        float l = at(low_tm,  t, s);
        float up = h - prev_h;
        float dn = prev_l - l;
        float up_pos = (up > 0.0f) ? up : 0.0f;
        float dn_pos = (dn > 0.0f) ? dn : 0.0f;
        float drm = dir_long ? ((dn_pos > up_pos) ? dn_pos : 0.0f)
                             : ((up_pos > dn_pos) ? up_pos : 0.0f);
        float tsum = sum + drm;
        if (fabsf(sum) >= fabsf(drm)) c += (sum - tsum) + drm; else c += (drm - tsum) + sum;
        sum = tsum;
        prev_h = h; prev_l = l;
    }
    dsf32 dm = ds_from_sum(sum, c);

    const float invp = 1.0f / (float)period;
    const float alpha_f = fmaf(-invp, 1.0f, 1.0f);

    // Rolling window selection
    const bool have_deque = (q_idx_tm != nullptr) && (q_val_tm != nullptr) && (lb_cap >= (max_lookback + 1));
    const bool small_win  = (max_lookback <= 4);

    int *qidx = nullptr; float *qv = nullptr; int q_head = 0, q_tail = 0, q_len = 0;
    if (have_deque) {
        qidx = q_idx_tm + s * lb_cap;
        qv   = q_val_tm + s * lb_cap;
    }
    auto ring_inc = [&](int x) { int y = x + 1; return (y == lb_cap) ? 0 : y; };
    auto ring_dec = [&](int x) { return (x == 0) ? (lb_cap - 1) : (x - 1); };

    float ringv[4]; int rpos = 0, rcount = 0;

    // Emit at i = end0
    {
        int i = end0;
        float cand = dir_long ? cand_long(at(low_tm, i - 1, s),  mult, dm)
                              : cand_short(at(high_tm, i - 1, s), mult, dm);
        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == max_lookback ? 0 : rpos + 1); if (rcount < max_lookback) ++rcount;
            if (i >= warm) out_tm[i * cols + s] = dir_long ? reduce_max4(ringv, rcount)
                                                           : reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = i + 1 - max_lookback;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            if (dir_long) { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; } }
            else          { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; } }
            qidx[q_tail] = i; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (i >= warm && q_len > 0) out_tm[i * cols + s] = qv[q_head];
        } else {
            if (i >= warm) out_tm[i * cols + s] = cand;
        }
    }

    // Continue scan
    float prev_h_i = at(high_tm, end0, s);
    float prev_l_i = at(low_tm,  end0, s);

    for (int i = end0 + 1; i < len; ++i) {
        float h = at(high_tm, i, s);
        float l = at(low_tm,  i, s);
        float up = h - prev_h_i;
        float dn = prev_l_i - l;
        float up_pos = (up > 0.0f) ? up : 0.0f;
        float dn_pos = (dn > 0.0f) ? dn : 0.0f;
        float drm = dir_long ? ((dn_pos > up_pos) ? dn_pos : 0.0f)
                             : ((up_pos > dn_pos) ? up_pos : 0.0f);

        dm = ds_fma(dm, alpha_f, drm);

        float cand = dir_long ? cand_long(prev_l_i, mult, dm)
                              : cand_short(prev_h_i, mult, dm);

        if (small_win) {
            ringv[rpos] = cand; rpos = (rpos + 1 == max_lookback ? 0 : rpos + 1); if (rcount < max_lookback) ++rcount;
            if (i >= warm) out_tm[i * cols + s] = dir_long ? reduce_max4(ringv, rcount)
                                                           : reduce_min4(ringv, rcount);
        } else if (have_deque) {
            int start = i + 1 - max_lookback;
            while (q_len > 0) { int idx_front = qidx[q_head]; if (idx_front < start) { q_head = ring_inc(q_head); --q_len; } else break; }
            if (dir_long) { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] <= cand) { q_tail = last; --q_len; } else break; } }
            else          { while (q_len > 0) { int last = ring_dec(q_tail); if (qv[last] >= cand) { q_tail = last; --q_len; } else break; } }
            qidx[q_tail] = i; qv[q_tail] = cand; q_tail = ring_inc(q_tail); ++q_len;
            if (i >= warm && q_len > 0) out_tm[i * cols + s] = qv[q_head];
        } else {
            if (i >= warm) out_tm[i * cols + s] = cand;
        }

        prev_h_i = h; prev_l_i = l;
    }
}
