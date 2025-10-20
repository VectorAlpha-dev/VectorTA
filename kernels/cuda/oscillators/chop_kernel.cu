// CUDA kernels for Choppiness Index (CHOP)
//
// Semantics mirror src/indicators/chop.rs (scalar):
// - Warmup prefix NaNs
// - TR uses hi/lo/prev_close; ATR is RMA over `drift`
// - SUM(ATR(1), period) via ring (batch) or prefix sums (many-series)
// - High/Low range via sparse tables (batch) or direct scan (many-series)
// - range <= 0 or sum_atr <= 0 or window NaN => output NaN

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Small register ring threshold chosen for common TA periods (e.g., 10–50).
#ifndef CHOP_REG_RING_MAX
#define CHOP_REG_RING_MAX 64
#endif

// Neumaier compensated update: maintain sum_hi + sum_lo for accuracy
static __forceinline__ __device__
void kbn_update(float delta, float& sum_hi, float& sum_lo) {
    float t = sum_hi + delta;
    float c = (fabsf(sum_hi) >= fabsf(delta)) ? (sum_hi - t) + delta : (delta - t) + sum_hi;
    sum_hi = t;
    sum_lo += c;
}

// --------------- Batch: one series × many params ------------------
// Uses register ring when period <= CHOP_REG_RING_MAX; otherwise shared memory.
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
    float* __restrict__ row_out = out + base;

    // Initialize row with NaNs in parallel (warmup semantics).
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        row_out[i] = NAN;
    }
    __syncthreads();

    // Single thread performs sequential scan per row.
    if (threadIdx.x != 0) return;

    const int period = periods[combo];
    const int drift  = drifts[combo];
    const float scalar = scalars[combo];

    if (UNLIKELY(period <= 0 || drift <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) return;

    // Precompute invariants
    const float inv_drift = 1.0f / (float)drift;

    // Switch to base-2 logs: scale = scalar / log2(period)
    const float inv_log2p = 1.0f / log2f((float)period);
    const float scale_over_log2p = scalar * inv_log2p;

    // Sparse table invariants (depend only on `period`)
    const int k = log2_tbl[period];
    if (UNLIKELY(k < 0 || k >= level_count)) return; // row already NaN-inited
    const int offset = 1 << k;
    const int level_base = level_offsets[k];

    // If the series has no NaNs at all after first_valid, we can skip the per-window guard
    const bool series_has_nan = (nan_psum[series_len] - nan_psum[first_valid]) != 0;

    // RMA(ATR) state
    float rma_atr = NAN;
    float sum_tr = 0.0f;

    // Rolling SUM(ATR) via ring + compensated accumulation
    int ring_idx = 0;
    float sum_hi = 0.0f, sum_lo = 0.0f; // Neumaier compensation

    // Ring storage: registers for small periods, shared mem otherwise
    float ring_reg[CHOP_REG_RING_MAX]; // only used when period <= CHOP_REG_RING_MAX
    extern __shared__ unsigned char __smem[];
    float* ring_smem = reinterpret_cast<float*>(__smem);

    if (period <= CHOP_REG_RING_MAX) {
        #pragma unroll
        for (int i = 0; i < CHOP_REG_RING_MAX; ++i) {
            if (i < period) ring_reg[i] = 0.0f;
        }
    } else {
        for (int i = 0; i < period && i < max_period; ++i) ring_smem[i] = 0.0f;
    }

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

        // RMA(ATR) update
        if (rel < drift) {
            sum_tr += tr;
            if (rel == drift - 1) {
                rma_atr = sum_tr * inv_drift;
            }
        } else {
            // rma_atr += inv_drift * (tr - rma_atr) with FMA for precision/throughput
            rma_atr = fmaf(inv_drift, (tr - rma_atr), rma_atr);
        }
        prev_close = cl;

        // Current ATR sample to feed the rolling SUM(ATR, period)
        const float current_atr = (rel < drift) ? ((rel == drift - 1) ? rma_atr : NAN) : rma_atr;
        const float add = (current_atr == current_atr) ? current_atr : 0.0f; // NaN -> 0

        // Pop oldest, push newest, update compensated sum
        float oldest = 0.0f;
        if (period <= CHOP_REG_RING_MAX) {
            oldest = ring_reg[ring_idx];
            ring_reg[ring_idx] = add;
        } else {
            oldest = ring_smem[ring_idx];
            ring_smem[ring_idx] = add;
        }
        // ring_idx = (ring_idx + 1) % period;
        ring_idx += 1;
        if (ring_idx == period) ring_idx = 0;

        const float delta = add - oldest;
        kbn_update(delta, sum_hi, sum_lo);
        const float rolling_sum_atr = sum_hi + sum_lo;

        if (rel >= period - 1) {
            const int start = t - period + 1;

            // Optional NaN guard only if series actually has NaNs
            if (series_has_nan) {
                if (nan_psum[t + 1] - nan_psum[start] != 0) {
                    row_out[t] = NAN;
                    continue;
                }
            }

            // Query sparse tables for H/L in O(1)
            const int idx_a = level_base + start;
            const int idx_b = level_base + (t + 1 - offset);
            const float hmax = fmaxf(st_max[idx_a], st_max[idx_b]);
            const float lmin = fminf(st_min[idx_a], st_min[idx_b]);
            const float range = hmax - lmin;

            if (!(range > 0.0f) || !(rolling_sum_atr > 0.0f)) {
                row_out[t] = NAN;
            } else {
                // One log: log2(sum_atr / range)
                const float ratio = rolling_sum_atr / range;
                const float y = scale_over_log2p * log2f(ratio);
                row_out[t] = y;
            }
        }
    }
}

// --------------- Many-series × one param (time-major) ---------------
// Each thread handles one series and scans time sequentially.
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

    // Base-2 scaling: scalar / log2(period)
    const float inv_log2p = 1.0f / log2f((float)period);
    const float scale_over_log2p = scalar * inv_log2p;

    const int first = first_valids[s];
    if (UNLIKELY(first < 0 || first >= rows)) {
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
        // Sum ATR over the last `period` via prefix sums
        const float sum_atr = atr_psum_tm[(size_t)(r + 1) * cols + s]
                            - atr_psum_tm[(size_t)(r + 1 - period) * cols + s];
        if (!(sum_atr > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }

        // Window H/L extremes via direct scan
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

        if (nan_in_window) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }

        const float range = hmax - lmin;
        if (!(range > 0.0f)) {
            out_tm[(size_t)r * cols + s] = NAN;
            continue;
        }

        // One log: log2(sum_atr / range)
        const float y = scale_over_log2p * log2f(sum_atr / range);
        out_tm[(size_t)r * cols + s] = y;
    }
}

