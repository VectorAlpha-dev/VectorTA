// CUDA kernels for Percentile Nearest Rank (PNR)
//
// Math pattern: order statistic (nearest-rank) within a sliding window.
// We implement a reliable baseline that matches scalar semantics:
// - Warmup/NaN parity (warm = first_valid + length - 1).
// - NaNs inside a window are ignored; all-NaN window -> NaN.
// - Rank index k = round(p * wl) - 1 clamped to [0, wl-1], where wl is
//   the number of valid (non-NaN) elements in the current window.
//
// Design notes:
// - Baseline kernels use one thread per parameter row (combo) / series.
//   Each thread maintains a sorted window inside a per-row scratch slice.
// - Scratch buffer is provided by the host wrapper and sized as
//   (n_rows * max_length). This avoids device-side malloc and keeps
//   semantics simple and robust across toolchains.
// - Arithmetic is FP32; outputs are exact window values (no accumulation).
//
// Perf notes:
// - This baseline focuses on correctness and parity. For large windows
//   and long series, further optimizations (block-level merge, bucketed
//   selection) can be explored without changing the public ABI.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

static __device__ __forceinline__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

// Insert v into sorted[0..wl) keeping ascending order; returns new wl
static __device__ __forceinline__ int insert_sorted(float* sorted, int wl, float v) {
    // Binary search for insertion point
    int lo = 0, hi = wl;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float mv = sorted[mid];
        if (!(v < mv)) { // v >= mv (handles NaN case by placing at end; but v is guaranteed non-NaN)
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // shift right
    for (int i = wl; i > lo; --i) sorted[i] = sorted[i - 1];
    sorted[lo] = v;
    return wl + 1;
}

// Remove one instance of v from sorted[0..wl); returns new wl
static __device__ __forceinline__ int erase_sorted(float* sorted, int wl, float v) {
    // Binary search for equal
    int lo = 0, hi = wl;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float mv = sorted[mid];
        if (mv < v) {
            lo = mid + 1;
        } else if (v < mv) {
            hi = mid;
        } else {
            // found equal at mid; shift left
            for (int i = mid; i + 1 < wl; ++i) sorted[i] = sorted[i + 1];
            return wl - 1;
        }
    }
    return wl; // not found (can happen with NaNs ignored)
}

static __device__ __forceinline__ int nearest_rank_index(float p_wl, int wl) {
    // k = floor(p*wl + 0.5) - 1, clamped (matches scalar nearest_rank_index_fast)
    double raw_f = floor((double)p_wl + 0.5) - 1.0; // half-up
    int raw = (int)raw_f;
    if (raw <= 0) return 0;
    if (raw >= wl) return wl - 1;
    return raw;
}

extern "C" __global__
void percentile_nearest_rank_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ lengths,
    const float* __restrict__ percentages,  // in [0,100]
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out,                // [n_combos][series_len]
    float* __restrict__ scratch,            // [n_combos][max_length]
    int max_length
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int length = lengths[combo];
    const float perc = percentages[combo];
    float* out_row = out + (size_t)combo * series_len;
    float* sorted = scratch + (size_t)combo * max_length;

    if (UNLIKELY(length <= 0 || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_row[i] = qnan32();
        return;
    }

    const int warm = first_valid + length - 1;
    // Prefill warmup with NaNs
    for (int i = 0; i < warm && i < series_len; ++i) out_row[i] = qnan32();
    if (warm >= series_len) return;

    const float p_frac = perc * 0.01f;
    const int window_start0 = warm + 1 - length;

    // Build initial sorted window at index = warm
    int wl = 0;
    for (int idx = window_start0; idx <= warm; ++idx) {
        float v = prices[idx];
        if (!isnan(v)) wl = insert_sorted(sorted, wl, v);
    }
    const int k_full = nearest_rank_index(p_frac * (float)length, length);

    int i = warm;
    while (true) {
        if (wl == 0) {
            out_row[i] = qnan32();
        } else {
            int k = (wl == length) ? k_full : nearest_rank_index(p_frac * (float)wl, wl);
            out_row[i] = sorted[k];
        }

        if (i + 1 >= series_len) break;
        // Slide window: remove outgoing, insert incoming
        int out_idx = i + 1 - length;
        float v_out = prices[out_idx];
        if (!isnan(v_out)) wl = erase_sorted(sorted, wl, v_out);
        float v_in = prices[i + 1];
        if (!isnan(v_in)) wl = insert_sorted(sorted, wl, v_in);
        i += 1;
    }
}

// Many-series × one-param (time-major)
extern "C" __global__
void percentile_nearest_rank_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm, // [rows][cols]
    int cols,
    int rows,
    int length,
    float percentage,
    const int* __restrict__ first_valids, // per series
    float* __restrict__ out_tm,           // [rows][cols]
    float* __restrict__ scratch_cols,     // [cols][length]
    int max_length
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int fv = first_valids[series];
    if (UNLIKELY(length <= 0 || fv < 0 || fv >= rows)) {
        for (int t = 0; t < rows; ++t) out_tm[(size_t)t * cols + series] = qnan32();
        return;
    }
    const int warm = fv + length - 1;
    for (int t = 0; t < warm && t < rows; ++t) out_tm[(size_t)t * cols + series] = qnan32();
    if (warm >= rows) return;

    float* sorted = scratch_cols + (size_t)series * max_length;
    int wl = 0;
    const float p_frac = percentage * 0.01f;
    const int k_full = nearest_rank_index(p_frac * (float)length, length);

    auto load_tm = [&](int t) -> float { return prices_tm[(size_t)t * cols + series]; };
    auto store_tm = [&](int t, float v) { out_tm[(size_t)t * cols + series] = v; };

    // Build initial sorted window at warm
    const int w0 = warm + 1 - length;
    for (int idx = w0; idx <= warm; ++idx) {
        float v = load_tm(idx);
        if (!isnan(v)) wl = insert_sorted(sorted, wl, v);
    }

    int i = warm;
    while (true) {
        if (wl == 0) {
            store_tm(i, qnan32());
        } else {
            int k = (wl == length) ? k_full : nearest_rank_index(p_frac * (float)wl, wl);
            store_tm(i, sorted[k]);
        }
        if (i + 1 >= rows) break;
        float v_out = load_tm(i + 1 - length);
        if (!isnan(v_out)) wl = erase_sorted(sorted, wl, v_out);
        float v_in = load_tm(i + 1);
        if (!isnan(v_in)) wl = insert_sorted(sorted, wl, v_in);
        i += 1;
    }
}
