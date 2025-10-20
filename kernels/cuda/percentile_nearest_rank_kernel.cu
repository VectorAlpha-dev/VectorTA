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

// --- Helpers (FP32, NaN-handling, bitonic primitives) ------------------------

static __device__ __forceinline__ bool is_nan(float x) { return !(x == x); }

static __device__ __forceinline__ int next_pow2(int x) {
    x = x - 1; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

// Compare-exchange that places NaNs at the end in ascending order.
// If 'up' is true => ascending, else descending (used by bitonic).
static __device__ __forceinline__ void cx_nan_last(float &a, float &b, bool up) {
    float aa = a, bb = b;
    const bool a_nan = is_nan(aa);
    const bool b_nan = is_nan(bb);

    float lo, hi;
    if (a_nan & b_nan) {           // both NaN: preserve order
        lo = aa; hi = bb;
    } else if (a_nan) {            // only a is NaN: b is smaller
        lo = bb; hi = aa;
    } else if (b_nan) {            // only b is NaN: a is smaller
        lo = aa; hi = bb;
    } else {                       // numeric-numeric
        if (aa <= bb) { lo = aa; hi = bb; }
        else           { lo = bb; hi = aa; }
    }
    if (up) { a = lo; b = hi; } else { a = hi; b = lo; }
}

// Cooperative bitonic sort operating in-place on shared memory buffer `buf`.
// Handles arbitrary sizes by padding with NaNs (which are forced to the end).
// Works even when blockDim.x < size via striding; pairs guarded with idx < ixj.
static __device__ void bitonic_sort_shared_nan_last(float* buf, int size) {
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    // k: current bitonic sequence length
    for (int k = 2; k <= size; k <<= 1) {
        // j: distance of compare-exchange partner
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < size; idx += nthreads) {
                int ixj = idx ^ j;
                if (ixj > idx) {
                    bool up = ((idx & k) == 0);  // ascending in this region
                    float a = buf[idx];
                    float b = buf[ixj];
                    cx_nan_last(a, b, up);
                    buf[idx]  = a;
                    buf[ixj]  = b;
                }
            }
            __syncthreads();
        }
    }
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
    // FP32 half-up rounding; avoids FP64 cost on consumer GPUs
    float raw_f = floorf(p_wl + 0.5f) - 1.0f;
    int   raw   = (int)raw_f;
    if (raw <= 0) return 0;
    if (raw >= wl) return wl - 1;
    return raw;
}

static __device__ __forceinline__ int nearest_rank_index_from_frac(float p_frac, int wl) {
    // p_frac in [0,1]; same semantics as nearest_rank_index
    float raw_f = floorf(fmaf(p_frac, (float)wl, 0.5f)) - 1.0f;
    int   raw   = (int)raw_f;
    if (raw <= 0)     return 0;
    if (raw >= wl)    return wl - 1;
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

// One price series, many params (all share the same window length).
// Time-major writes: out[combo][t]. Prices are a single vector.
// Shared memory usage ~ next_pow2(length) * sizeof(float).
//
// Heuristic use: when #params is sizeable for a given 'length' group.
extern "C" __global__
void percentile_nearest_rank_one_series_many_params_same_len_f32(
    const float* __restrict__ prices,      // [series_len]
    int series_len,
    int length,                            // shared by all combos in this launch
    const float* __restrict__ percentages, // [n_combos] in [0,100]
    int n_combos,
    int first_valid,
    float* __restrict__ out                // [n_combos][series_len]
) {
    // Fast reject + warmup fill
    if (UNLIKELY(length <= 0 || first_valid < 0 || first_valid >= series_len)) {
        // Only 1 block needs to fill; others exit.
        if (blockIdx.x == 0) {
            for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
                float* row = out + (size_t)c * series_len;
                for (int t = 0; t < series_len; ++t) row[t] = qnan32();
            }
        }
        return;
    }

    const int warm = first_valid + length - 1;
    if (blockIdx.x == 0) {
        // Fill warmup prefix with NaN once
        for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
            float* row = out + (size_t)c * series_len;
            for (int t = 0; t < warm && t < series_len; ++t) row[t] = qnan32();
        }
    }
    __syncthreads();
    if (warm >= series_len) return;

    // Shared buffer for the current window (padded to pow2)
    extern __shared__ float s_win[];
    const int S = next_pow2(length);

    // Process time steps assigned to this block (grid-stride across t)
    for (int t = warm + blockIdx.x; t < series_len; t += gridDim.x) {
        // Load window [t-length+1 .. t] into shared memory; pad [length..S) with NaN.
        const int start = t + 1 - length;

        // Count valid (non-NaN) elements while loading
        int local_valid = 0;
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
            float v = prices[start + i];
            s_win[i] = v;
            if (!is_nan(v)) local_valid += 1;
        }
        for (int i = threadIdx.x + length; i < S; i += blockDim.x) {
            s_win[i] = qnan32(); // padding as NaN; NaNs sort to the end
        }

        // Block-wide reduction for wl
        __shared__ int wl_shared;
        if (threadIdx.x == 0) wl_shared = 0;
        __syncthreads();

        int sum = local_valid;
        // warp reduce:
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if ((threadIdx.x & 31) == 0) atomicAdd(&wl_shared, sum);
        __syncthreads();

        const int wl = wl_shared;

        // Edge case: all-NaN window
        if (wl == 0) {
            for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
                out[(size_t)c * series_len + t] = qnan32();
            }
            __syncthreads();
            continue;
        }

        // Sort window once (NaNs will be at the end)
        bitonic_sort_shared_nan_last(s_win, S);

        // Emit all combos for this time step
        for (int c = threadIdx.x; c < n_combos; c += blockDim.x) {
            float p = percentages[c] * 0.01f;
            int   k = (wl == length)
                    ? nearest_rank_index_from_frac(p, length)
                    : nearest_rank_index_from_frac(p, wl);
            out[(size_t)c * series_len + t] = s_win[k];
        }
        __syncthreads();
    }
}
