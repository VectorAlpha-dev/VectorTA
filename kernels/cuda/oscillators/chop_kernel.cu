// CUDA kernels for Choppiness Index (CHOP)
//
// Semantics mirror src/indicators/chop.rs (scalar):
// - Warmup prefix NaNs: indices [0 .. first_valid + period - 1)
// - TR uses hi/lo/prev_close; ATR is RMA over `drift`
// - SUM(ATR(1), period) maintained via a ring buffer (batch) or via prefix
//   sums provided by the host (many-series variant)
// - High/Low range over [t - period + 1, t] computed via sparse tables
//   (batch) or a direct window scan (many-series variant)
// - range <= 0 or sum_atr <= 0 or any window NaN => output NaN

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

// --------------- Batch: one series × many params ------------------
// Inputs:
//  - high, low, close: price slices (len = series_len)
//  - periods, drifts, scalars: per-combo arrays (len = n_combos)
//  - log2_tbl, level_offsets, st_max, st_min, nan_psum: sparse-table data for H/L windows
//  - first_valid: index of first non-NaN triple (H,L,C)
//  - series_len, level_count, n_combos
// Output:
//  - out: row-major [combo * series_len + t]
//
// Shared memory: ring buffer of size max_period (floats)
extern "C" __global__ void chop_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    const int*   __restrict__ drifts,
    const float* __restrict__ scalars,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    int max_period,
    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int base = combo * series_len;

    // Initialize row with NaNs in parallel to match warmup semantics
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    // Single thread performs the sequential scan per row
    if (threadIdx.x != 0) return;

    const int period = periods[combo];
    const int drift  = drifts[combo];
    const float scalar = scalars[combo];
    if (UNLIKELY(period <= 0 || drift <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) return;

    // Ring buffer for rolling SUM(ATR(1), period)
    extern __shared__ unsigned char __smem[];
    float* ring = reinterpret_cast<float*>(__smem);
    for (int i = 0; i < period && i < max_period; ++i) ring[i] = 0.0f;
    float rolling_sum_atr = 0.0f;
    int ring_idx = 0;

    const float inv_drift = 1.0f / static_cast<float>(drift);
    const float inv_logp = 1.0f / log10f(static_cast<float>(period));
    const float scale_over_logp = scalar * inv_logp;

    // RMA(ATR) state
    float rma_atr = NAN;
    float sum_tr = 0.0f;

    // prev_close initialization
    float prev_close = close[first_valid];

    const int warm = first_valid + period - 1;
    for (int t = first_valid; t < series_len; ++t) {
        const float hi = high[t];
        const float lo = low[t];
        const float cl = close[t];
        const int rel = t - first_valid;

        // True Range
        float tr;
        if (rel == 0) {
            tr = hi - lo;
        } else {
            const float a = hi - lo;
            const float b = fabsf(hi - prev_close);
            const float c = fabsf(lo - prev_close);
            tr = fmaxf(a, fmaxf(b, c));
        }

        if (rel < drift) {
            sum_tr += tr;
            if (rel == drift - 1) {
                rma_atr = sum_tr * inv_drift;
            }
        } else {
            rma_atr += inv_drift * (tr - rma_atr);
        }
        prev_close = cl;

        // Current ATR sample for SUM(ATR(1), period)
        float current_atr = (rel < drift) ? ((rel == drift - 1) ? rma_atr : NAN) : rma_atr;
        const float oldest = ring[ring_idx];
        rolling_sum_atr -= oldest;
        const float add = (current_atr == current_atr) ? current_atr : 0.0f; // NaN -> 0
        ring[ring_idx] = add;
        rolling_sum_atr += add;
        ring_idx = (ring_idx + 1) % period;

        if (rel >= period - 1) {
            const int start = t - period + 1;
            // NaN guard: if any NaN in [start, t] for H/L, emit NaN
            if (nan_psum[t + 1] - nan_psum[start] != 0) {
                out[base + t] = NAN;
                continue;
            }
            // Query sparse tables for H/L range
            const int len = period;
            const int k = log2_tbl[len];
            if (UNLIKELY(k < 0 || k >= level_count)) {
                out[base + t] = NAN;
                continue;
            }
            const int offset = 1 << k;
            const int level_base = level_offsets[k];
            const int idx_a = level_base + start;
            const int idx_b = level_base + (t + 1 - offset);
            const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
            const float lmin = fminf(st_min[idx_a], st_min[idx_b]);
            const float range = hmax - lmin;

            if (!(range > 0.0f) || !(rolling_sum_atr > 0.0f)) {
                out[base + t] = NAN;
            } else {
                const float y = scale_over_logp * (log10f(rolling_sum_atr) - log10f(range));
                out[base + t] = y;
            }
        }
    }
}

// --------------- Many-series × one param (time-major) ---------------
// Each thread handles one series (column) and scans time sequentially.
// Inputs:
//  - high_tm, low_tm: time-major (row * cols + series)
//  - atr_psum_tm: prefix sums of ATR(1) values per series (time-major), shape (rows+1)×cols
//  - first_valids: per-series first valid index (based on H/L/C)
//  - cols (# series), rows (# time steps), period, scalar
extern "C" __global__ void chop_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ atr_psum_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const float inv_logp = 1.0f / log10f((float)period);
    const float scale_over_logp = scalar * inv_logp;

    const int first = first_valids[s];
    if (UNLIKELY(first < 0 || first >= rows)) {
        // write NaNs
        for (int r = 0; r < rows; ++r) out_tm[(size_t)r * cols + s] = NAN;
        return;
    }
    if (UNLIKELY(period <= 0 || period > rows - first)) {
        for (int r = 0; r < rows; ++r) out_tm[(size_t)r * cols + s] = NAN;
        return;
    }

    const int warm = first + period - 1;
    for (int r = 0; r < warm; ++r) out_tm[(size_t)r * cols + s] = NAN;

    for (int r = warm; r < rows; ++r) {
        // Sum ATR over the last `period` using prefix sums
        const float sum_atr = atr_psum_tm[(size_t)(r + 1) * cols + s]
                            - atr_psum_tm[(size_t)(r + 1 - period) * cols + s];

        // Window H/L extremes via direct scan (period is typically small)
        float hmax = -INFINITY;
        float lmin = INFINITY;
        bool nan_in_window = false;
        const size_t start = (size_t)(r - period + 1) * cols + s;
        for (int k = 0; k < period; ++k) {
            const float h = high_tm[start + (size_t)k * cols];
            const float l = low_tm[start + (size_t)k * cols];
            if (!(h == h) || !(l == l)) { nan_in_window = true; break; }
            hmax = fmaxf(hmax, h);
            lmin = fminf(lmin, l);
        }

        if (nan_in_window || !(sum_atr > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }
        const float range = hmax - lmin;
        if (!(range > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }
        const float y = scale_over_logp * (log10f(sum_atr) - log10f(range));
        out_tm[(size_t)r * cols + s] = y;
    }
}

