// CUDA kernels for Average Sentiment Oscillator (ASO)
// Optimized FP32 versions (CUDA 13 / Ada+ friendly)
//
// Highlights:
//  - Hoist sparse-table invariants out of hot loop
//  - Branch-free mode mixing via weights
//  - Kahan compensated sliding sums (robust FP32)
//  - Runtime pow2 fast path for ring/deque modulo
//
// Semantics preserved:
//  - FP32 compute; warmup indices are NaN
//  - Division-by-zero -> reciprocal 1.0 (scale = 50.0)
//  - NaN propagation follows arithmetic

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// ----------------- Helpers -----------------
__device__ __forceinline__ float inv_or_one(const float x) {
    return (x != 0.0f) ? __fdividef(1.0f, x) : 1.0f;
}

__device__ __forceinline__ void mode_weights(const int mode, float& w_intra, float& w_group) {
    // mode: 0 = avg(intra,group), 1 = intra, 2 = group
    // Replace per-sample branching with fixed weights.
    if (mode == 0) { w_intra = 0.5f; w_group = 0.5f; }
    else if (mode == 2) { w_intra = 0.0f; w_group = 1.0f; }
    else { w_intra = 1.0f; w_group = 0.0f; }
}

// Kahan compensated add: sum += y (compensated in c)
__device__ __forceinline__ void kahan_add(float y, float& sum, float& c) {
    float t  = y - c;
    float ns = sum + t;
    c        = (ns - sum) - t;
    sum      = ns;
}

// Fast modulo for power-of-two period
struct ModHelper {
    const int period;
    const bool is_pow2;
    const int mask; // period-1 when pow2
    __device__ __forceinline__ ModHelper(int p)
        : period(p), is_pow2((p & (p - 1)) == 0), mask(p - 1) {}
    __device__ __forceinline__ int inc_wrap(int x) const {
        if (is_pow2) return (x + 1) & mask;
        int nx = x + 1; return (nx == period) ? 0 : nx;
    }
    __device__ __forceinline__ int dec_wrap(int x) const {
        if (is_pow2) return (x - 1) & mask;
        return (x == 0) ? (period - 1) : (x - 1);
    }
    __device__ __forceinline__ int mod(int x) const {
        return is_pow2 ? (x & mask) : (x % period);
    }
};

// ----------------- Batch: one series × many params -----------------
// Dynamic shared memory per block: 2 * period * sizeof(float)
//   [0..period-1]   -> ring_b (float)
//   [period..2P-1]  -> ring_e (float)
extern "C" __global__ void aso_batch_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    const int*   __restrict__ modes,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    float* __restrict__ out_bulls,
    float* __restrict__ out_bears)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    const int base = combo * series_len;

    // Invalid paths: fill the whole row with NaNs (rare; keep semantics obvious).
    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_bulls[base + i] = NAN;
            out_bears[base + i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan();
        return;
    }

    const int warm = first_valid + period - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }

    // Validate sparse-table level once (before we let most threads return).
    const int k = log2_tbl[period];
    if (UNLIKELY(k < 0 || k >= level_count)) {
        fill_all_nan();
        return;
    }

    // Precompute mode weights once per combo (avoids inner-loop branches).
    const int  mode = modes[combo];
    float w_intra, w_group;
    mode_weights(mode, w_intra, w_group);

    // Prefill only the warmup prefix. The scan below writes every index >= warm.
    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_bulls[base + i] = NAN;
        out_bears[base + i] = NAN;
    }
    __syncthreads();

    // Single-lane sequential scan per combo (dependencies across t).
    if (threadIdx.x != 0) return;

    // Sparse table invariants hoisted out of the loop.
    const int offset   = 1 << k;
    const int lvl_base = level_offsets[k];
    const float* __restrict__ st_max_lvl = st_max + lvl_base;
    const float* __restrict__ st_min_lvl = st_min + lvl_base;

    // Dynamic shared memory rings
    extern __shared__ float smem[];
    float* ring_b = smem;
    float* ring_e = smem + period;

    // Initialize rings
    for (int i = 0; i < period; ++i) { ring_b[i] = 0.0f; ring_e[i] = 0.0f; }

    ModHelper mh(period);
    int   head   = 0;
    int   filled = 0;
    float sum_b  = 0.0f, cb = 0.0f; // Kahan sum + compensation
    float sum_e  = 0.0f, ce = 0.0f;

    // Prepare rolling sparse-table indices and gopen index
    int start     = warm - period + 1;
    int idx_a     = start;
    int idx_b     = warm + 1 - offset;
    int gopen_idx = start;

    // Main time loop
    for (int t = warm; t < series_len; ++t, ++start, ++idx_a, ++idx_b, ++gopen_idx) {
        const float o = open[t];
        const float h = high[t];
        const float l = low[t];
        const float c = close[t];

        // Intrabar (per-bar) component
        const float intrarange    = h - l;
        const float scale1        = 50.0f * inv_or_one(intrarange);
        const float intrabarbulls = fmaf((c - l) + (h - o), scale1, 0.0f);
        const float intrabarbears = fmaf((h - c) + (o - l), scale1, 0.0f);

        // Group component from sparse tables (O(1) RMQ)
        const float gh    = fmaxf(st_max_lvl[idx_a], st_max_lvl[idx_b]);
        const float gl    = fminf(st_min_lvl[idx_a], st_min_lvl[idx_b]);
        const float gopen = open[gopen_idx];
        const float gr    = gh - gl;
        const float scale2        = 50.0f * inv_or_one(gr);
        const float groupbulls    = fmaf((c - gl) + (gh - gopen), scale2, 0.0f);
        const float groupbears    = fmaf((gh - c) + (gopen - gl), scale2, 0.0f);

        // Mix intrabar / group via weights (no branch in the hot loop)
        const float b = fmaf(w_intra, intrabarbulls, w_group * groupbulls);
        const float e = fmaf(w_intra, intrabarbears, w_group * groupbears);

        const float old_b = (filled == period) ? ring_b[head] : 0.0f;
        const float old_e = (filled == period) ? ring_e[head] : 0.0f;

        // Kahan-compensated sliding updates
        kahan_add(b - old_b, sum_b, cb);
        kahan_add(e - old_e, sum_e, ce);

        ring_b[head] = b;
        ring_e[head] = e;
        head = mh.inc_wrap(head);
        if (filled < period) ++filled;

        const float n = (float)filled;
        out_bulls[base + t] = __fdividef(sum_b, n);
        out_bears[base + t] = __fdividef(sum_e, n);
    }
}

// ----------------- Many series × one param (time-major) -----------------
// Dynamic shared memory per block:
//   floats: ring_b[period], ring_e[period]
//   ints:   dq_min_idx[period], dq_max_idx[period]
extern "C" __global__ void aso_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, // per series
    int cols,    // number of series (columns)
    int rows,    // number of rows (time)
    int period,
    int mode,
    float* __restrict__ out_bulls_tm,
    float* __restrict__ out_bears_tm)
{
    const int s = blockIdx.x; // series index (column)
    if (s >= cols) return;

    auto fill_all_nan = [&]() {
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            const int idx = t * cols + s;
            out_bulls_tm[idx] = NAN;
            out_bears_tm[idx] = NAN;
        }
    };

    if (UNLIKELY(period <= 0)) {
        fill_all_nan();
        return;
    }

    const int fv   = first_valids[s];
    if (UNLIKELY(fv < 0 || fv >= rows)) {
        fill_all_nan();
        return;
    }
    const int warm = fv + period - 1;
    if (UNLIKELY(warm >= rows)) {
        fill_all_nan();
        return;
    }

    // Prefill only the warmup prefix; the scan below writes every index >= warm.
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        const int idx = t * cols + s;
        out_bulls_tm[idx] = NAN;
        out_bears_tm[idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane per series (time dependency)

    float w_intra, w_group;
    mode_weights(mode, w_intra, w_group);

    extern __shared__ unsigned char smem_uc[];
    float* ring_b     = reinterpret_cast<float*>(smem_uc);
    float* ring_e     = ring_b + period;
    int*   dq_min_idx = reinterpret_cast<int*>(ring_e + period);
    int*   dq_max_idx = dq_min_idx + period;

    for (int i = 0; i < period; ++i) {
        ring_b[i] = 0.0f; ring_e[i] = 0.0f;
        dq_min_idx[i] = 0; dq_max_idx[i] = 0;
    }

    ModHelper mh(period);
    int head = 0, filled = 0;
    float sum_b = 0.0f, cb = 0.0f;
    float sum_e = 0.0f, ce = 0.0f;

    // Monotonic deques (store indices into time-major arrays)
    int min_head = 0, min_tail = 0, min_len = 0;
    int max_head = 0, max_tail = 0, max_len = 0;

    // Carry time-major indices forward to avoid repeated multiplies
    int idx       = fv * cols + s;                         // t*cols + s
    int start_idx = idx - (period - 1) * cols;             // (t - period + 1)*cols + s

    for (int t = fv; t < rows; ++t, idx += cols, start_idx += cols) {
        const float o = open_tm[idx];
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];

        // push low into monotonic-min deque
        while (min_len > 0) {
            int back = mh.dec_wrap(min_tail);
            int j    = dq_min_idx[back];
            float lj = low_tm[j * cols + s];
            if (l <= lj) { min_tail = back; --min_len; } else { break; }
        }
        if (min_len == period) { min_head = mh.inc_wrap(min_head); --min_len; }
        dq_min_idx[min_tail] = t; min_tail = mh.inc_wrap(min_tail); ++min_len;

        // push high into monotonic-max deque
        while (max_len > 0) {
            int back = mh.dec_wrap(max_tail);
            int j    = dq_max_idx[back];
            float hj = high_tm[j * cols + s];
            if (h >= hj) { max_tail = back; --max_len; } else { break; }
        }
        if (max_len == period) { max_head = mh.inc_wrap(max_head); --max_len; }
        dq_max_idx[max_tail] = t; max_tail = mh.inc_wrap(max_tail); ++max_len;

        if (t >= warm) {
            const int start = t - period + 1;
            while (min_len > 0 && dq_min_idx[min_head] < start) { min_head = mh.inc_wrap(min_head); --min_len; }
            while (max_len > 0 && dq_max_idx[max_head] < start) { max_head = mh.inc_wrap(max_head); --max_len; }

            const float gl    = low_tm[dq_min_idx[min_head] * cols + s];
            const float gh    = high_tm[dq_max_idx[max_head] * cols + s];
            const float gopen = open_tm[start_idx];

            // Intrabar
            const float intrarange    = h - l;
            const float scale1        = 50.0f * inv_or_one(intrarange);
            const float intrabarbulls = fmaf((c - l) + (h - o), scale1, 0.0f);
            const float intrabarbears = fmaf((h - c) + (o - l), scale1, 0.0f);

            // Group
            const float gr            = gh - gl;
            const float scale2        = 50.0f * inv_or_one(gr);
            const float groupbulls    = fmaf((c - gl) + (gh - gopen), scale2, 0.0f);
            const float groupbears    = fmaf((gh - c) + (gopen - gl), scale2, 0.0f);

            const float b = fmaf(w_intra, intrabarbulls, w_group * groupbulls);
            const float e = fmaf(w_intra, intrabarbears, w_group * groupbears);

            const float old_b = (filled == period) ? ring_b[head] : 0.0f;
            const float old_e = (filled == period) ? ring_e[head] : 0.0f;

            kahan_add(b - old_b, sum_b, cb);
            kahan_add(e - old_e, sum_e, ce);

            ring_b[head] = b; ring_e[head] = e;
            head = mh.inc_wrap(head);
            if (filled < period) ++filled;

            const float n = (float)filled;
            out_bulls_tm[idx] = __fdividef(sum_b, n);
            out_bears_tm[idx] = __fdividef(sum_e, n);
        }
    }
}

