// CUDA kernels for the Slope Adaptive Moving Average (SAMA).
//
// Each block covers a single parameter combination because the SAMA recurrence
// is inherently sequential. Threads cooperate to initialise the output row, and
// lane 0 evaluates the adaptive smoothing in-order to mirror the scalar CPU
// semantics exactly. A companion kernel handles the time-major many-series
// entry point that the Rust and Python bindings expose.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__
void sama_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ lengths,
                    const float* __restrict__ min_alphas,
                    const float* __restrict__ maj_alphas,
                    const int* __restrict__ first_valids,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) {
        return;
    }

    const int row_offset = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const float price = prices[t];
        if (!isfinite(price)) {
            out[row_offset + t] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }
        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices[j];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out[row_offset + t] = prev;
    }
}

extern "C" __global__
void sama_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                    const int* __restrict__ first_valids,
                                    int length,
                                    float min_alpha,
                                    float maj_alpha,
                                    int num_series,
                                    int series_len,
                                    float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (length < 0 || num_series <= 0 || series_len <= 0) {
        return;
    }

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float prev = NAN;

    for (int t = first_valid; t < series_len; ++t) {
        const int offset = t * stride + series_idx;
        const float price = prices_tm[offset];
        if (!isfinite(price)) {
            out_tm[offset] = NAN;
            continue;
        }

        int start = t - length;
        if (start < 0) {
            start = 0;
        }

        float hh = -CUDART_INF_F;
        float ll = CUDART_INF_F;
        for (int j = start; j <= t; ++j) {
            const float v = prices_tm[j * stride + series_idx];
            if (!isfinite(v)) {
                continue;
            }
            if (v > hh) {
                hh = v;
            }
            if (v < ll) {
                ll = v;
            }
        }

        float mult = 0.0f;
        if (hh != ll) {
            const float numer = fabsf(2.0f * price - ll - hh);
            const float denom = hh - ll;
            if (denom != 0.0f) {
                mult = numer / denom;
            }
        }
        float alpha = (mult * (min_alpha - maj_alpha) + maj_alpha);
        alpha = alpha * alpha;

        if (!isfinite(prev)) {
            prev = price;
        } else {
            prev = __fmaf_rn(price - prev, alpha, prev);
        }

        out_tm[offset] = prev;
    }
}

// ============================================================================
// Optimized variants (CUDA 13+, Ada-ready): O(series_len) deque-based HH/LL
// These keep the original kernels intact for ABI compatibility. The optimized
// versions add a `max_window` parameter and use dynamic shared memory to hold
// two deques per block. Callers must pass sharedMemBytes =
//   2 * (max_window + 1) * sizeof(int)
// at launch. If insufficient, kernels fall back to the O(length) scan, but
// with improved NaN-window handling to avoid propagating NaNs into the EMA.
// ============================================================================

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#if !defined(CUDA_HAS_LDG_WRAPPER)
#define CUDA_HAS_LDG_WRAPPER
// Read-only cached load wrapper (falls back to normal load on older arch)
static __device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
#endif

// --------------------------- Utilities ---------------------------

static __device__ __forceinline__ int clamp_start(int t, int length) {
    int s = t - length;
    return s < 0 ? 0 : s;
}

// Pop from front while indices are older than "start" (circular deque).
static __device__ __forceinline__
void pop_outdated_front(int*& q, int& front, int& size, const int cap, int start) {
    while (size > 0) {
        int idx = q[front];
        if (idx >= start) break;
        front = (front + 1);
        if (front == cap) front = 0;
        --size;
    }
}

// Push 'k' into a max deque (drop smaller/equal from the back).
static __device__ __forceinline__
void push_max_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return; // only store finite elements
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);
        // Match CPU deque policy: pop while last <= new (strictly decreasing deque)
        if (vb > vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}

// Push 'k' into a min deque (drop larger/equal from the back).
static __device__ __forceinline__
void push_min_idx(const float* base, int*& q, int& back, int& size, const int cap, int k) {
    float vk = ldgf(base + k);
    if (!isfinite(vk)) return; // only store finite elements
    while (size > 0) {
        int back_pos = (back == 0 ? cap - 1 : back - 1);
        float vb = ldgf(base + q[back_pos]);
        // Match CPU deque policy: pop while last >= new (strictly increasing deque)
        if (vb < vk) break;
        back = back_pos;
        --size;
    }
    q[back] = k;
    back = (back + 1);
    if (back == cap) back = 0;
    ++size;
}

// ----------------------- Batch/combos kernel (optimized) ---------------------

extern "C" __global__
void sama_batch_f32_opt(const float* __restrict__ prices,     // length: series_len
                        const int*   __restrict__ lengths,     // length: n_combos
                        const float* __restrict__ min_alphas,  // length: n_combos
                        const float* __restrict__ maj_alphas,  // length: n_combos
                        const int*   __restrict__ first_valids,// length: n_combos
                        int series_len,
                        int n_combos,
                        int max_window,                        // NEW: max window across combos in this launch
                        float* __restrict__ out)               // shape: [n_combos, series_len]
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int length      = lengths[combo];
    const float min_alpha = min_alphas[combo];
    const float maj_alpha = maj_alphas[combo];
    const int first_valid = first_valids[combo];

    if (length < 0 || first_valid >= series_len || series_len <= 0) return;

    const int row_offset = combo * series_len;
    // Phase 1: init row to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[row_offset + idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    // Precompute once
    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;

    // If we don't have enough shared memory capacity, fall back to safe O(length)
    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const float price_t = ldgf(prices + t);
            if (!isfinite(price_t)) {
                out[row_offset + t] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);

            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices + j);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out[row_offset + t] = prev;
        }
        return;
    }

    // Deque path: O(series_len)
    extern __shared__ int shmem[];
    const int cap = max_window + 1;  // capacity for one window
    int* dq_max = shmem;             // [cap]
    int* dq_min = shmem + cap;       // [cap]

    int fmax = 0, bmax = 0, szmax = 0;  // front, back, size
    int fmin = 0, bmin = 0, szmin = 0;

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const float price_t = ldgf(prices + t);

        // Drop outdated indices from the front using current window start
        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out[row_offset + t] = NAN;
            continue;
        }

        // Push current index into deques (match CPU equality policy)
        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = ldgf(prices + dq_max[back_pos]);
            if (vb > price_t) break; // pop while <=
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = ldgf(prices + dq_min[back_pos]);
            if (vb < price_t) break; // pop while >=
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;

        // Compute mult safely
        const float hh = ldgf(prices + dq_max[fmax]);
        const float ll = ldgf(prices + dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out[row_offset + t] = prev;
    }
}

// -------------------- Time-major, many-series kernel (optimized) --------------------

extern "C" __global__
void sama_many_series_one_param_f32_opt(const float* __restrict__ prices_tm, // [t][series]
                                        const int*   __restrict__ first_valids, // [series]
                                        int length,
                                        float min_alpha,
                                        float maj_alpha,
                                        int num_series,
                                        int series_len,
                                        int max_window,                    // NEW: >= length
                                        float* __restrict__ out_tm)        // [t][series]
{
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) return;
    if (length < 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int first_valid = first_valids[series_idx];

    // Phase 1: init series column to NaN
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float dalpha = min_alpha - maj_alpha;

    float prev = NAN;

    // Fallback if shared memory capacity is insufficient
    const bool use_deque = (max_window >= length);
    if (!use_deque) {
        for (int t = first_valid; t < series_len; ++t) {
            const int off = t * stride + series_idx;
            const float price_t = ldgf(prices_tm + off);
            if (!isfinite(price_t)) {
                out_tm[off] = NAN;
                continue;
            }
            const int start = clamp_start(t, length);
            float hh = -CUDART_INF_F, ll = CUDART_INF_F;
            bool any = false;
            #pragma unroll 1
            for (int j = start; j <= t; ++j) {
                const float v = ldgf(prices_tm + j * stride + series_idx);
                if (!isfinite(v)) continue;
                any = true;
                if (v > hh) hh = v;
                if (v < ll) ll = v;
            }
            float mult = 0.0f;
            if (any) {
                const float denom = hh - ll;
                if (denom != 0.0f) {
                    const float numer = fabsf(2.0f * price_t - ll - hh);
                    mult = numer / denom;
                }
            }
            float alpha = __fmaf_rn(mult, dalpha, maj_alpha);
            alpha = alpha * alpha;

            prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
            out_tm[off] = prev;
        }
        return;
    }

    // Deque path
    extern __shared__ int shmem[];
    const int cap = max_window + 1;
    int* dq_max = shmem;       // [cap]
    int* dq_min = shmem + cap; // [cap]

    int fmax = 0, bmax = 0, szmax = 0;
    int fmin = 0, bmin = 0, szmin = 0;

    auto load_tm = [&](int t)->float {
        return ldgf(prices_tm + t * stride + series_idx);
    };

    for (int t = first_valid; t < series_len; ++t) {
        const int start = clamp_start(t, length);
        const int off   = t * stride + series_idx;
        const float price_t = load_tm(t);

        // Drop outdated
        pop_outdated_front(dq_max, fmax, szmax, cap, start);
        pop_outdated_front(dq_min, fmin, szmin, cap, start);

        if (!isfinite(price_t)) {
            out_tm[off] = NAN;
            continue;
        }

        // Push current t into deques (match CPU policy)
        while (szmax > 0) {
            int back_pos = (bmax == 0 ? cap - 1 : bmax - 1);
            float vb = load_tm(dq_max[back_pos]);
            if (vb > price_t) break; // pop while <=
            bmax = back_pos; --szmax;
        }
        dq_max[bmax] = t; bmax = (bmax + 1 == cap ? 0 : bmax + 1); ++szmax;

        while (szmin > 0) {
            int back_pos = (bmin == 0 ? cap - 1 : bmin - 1);
            float vb = load_tm(dq_min[back_pos]);
            if (vb < price_t) break; // pop while >=
            bmin = back_pos; --szmin;
        }
        dq_min[bmin] = t; bmin = (bmin + 1 == cap ? 0 : bmin + 1); ++szmin;

        const float hh = load_tm(dq_max[fmax]);
        const float ll = load_tm(dq_min[fmin]);
        const float denom = hh - ll;
        float mult = 0.0f;
        if (denom != 0.0f) {
            const float numer = fabsf(2.0f * price_t - ll - hh);
            mult = numer / denom;
        }

        float alpha = __fmaf_rn(mult, (min_alpha - maj_alpha), maj_alpha);
        alpha = alpha * alpha;

        prev = isfinite(prev) ? __fmaf_rn(price_t - prev, alpha, prev) : price_t;
        out_tm[off] = prev;
    }
}
