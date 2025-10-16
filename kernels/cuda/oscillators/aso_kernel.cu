// CUDA kernels for Average Sentiment Oscillator (ASO)
//
// Two entry points:
//  - aso_batch_f32:     one series × many params (period, mode per combo)
//                        Uses precomputed sparse tables for rolling min/max.
//  - aso_many_series_one_param_f32: many series × one param (time-major layout)
//                        Single block per series; sequential scan with monotonic deques.
//
// Semantics:
//  - FP32 compute. Warmup indices are NaN-initialized. Division-by-zero in
//    intrabar/group scales uses 1.0 as reciprocal factor (i.e., scale=50.0).
//  - NaN propagation follows arithmetic results; we do not force NaNs beyond
//    what the scalar path would naturally produce.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// -------------- Helpers --------------
__device__ __forceinline__ float inv_or_one(const float x) {
    return (x != 0.0f) ? (1.0f / x) : 1.0f;
}

// -------------- Batch: one series × many params --------------
// Dynamic shared memory layout per block:
//   [0..period-1]   -> ring_b (float)
//   [period..2P-1]  -> ring_e (float)
extern "C" __global__ void aso_batch_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int* __restrict__ periods,
    const int* __restrict__ modes,
    const int* __restrict__ log2_tbl,
    const int* __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    float* __restrict__ out_bulls,
    float* __restrict__ out_bears) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;
    const int mode = modes[combo];

    const int base = combo * series_len;

    // Initialize warmup with NaNs in parallel.
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out_bulls[base + i] = NAN;
        out_bears[base + i] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane sequential scan per combo

    const int warm = first_valid + period - 1;
    if (warm >= series_len) return;

    extern __shared__ float smem[];
    float* ring_b = smem;
    float* ring_e = smem + period;
    int head = 0; int filled = 0;
    float sum_b = 0.0f, sum_e = 0.0f;

    // Zero the first `period` elements we will touch to keep defined behavior.
    for (int k = 0; k < period; ++k) { ring_b[k] = 0.0f; ring_e[k] = 0.0f; }

    for (int t = warm; t < series_len; ++t) {
        const float o = open[t];
        const float h = high[t];
        const float l = low[t];
        const float c = close[t];

        // Intrabar component
        const float intrarange = h - l;
        const float scale1 = 50.0f * inv_or_one(intrarange);
        const float intrabarbulls = ((c - l) + (h - o)) * scale1;
        const float intrabarbears = ((h - c) + (o - l)) * scale1;

        // Group component via sparse tables
        const int start = t - period + 1;
        int k = log2_tbl[period];
        if (k < 0 || k >= level_count) { continue; }
        const int offset = 1 << k;
        const int lvl_base = level_offsets[k];
        const int idx_a = lvl_base + start;
        const int idx_b = lvl_base + (t + 1 - offset);
        const float gh = fmaxf(st_max[idx_a], st_max[idx_b]);
        const float gl = fminf(st_min[idx_a], st_min[idx_b]);
        const float gopen = open[start];
        const float gr = gh - gl;
        const float scale2 = 50.0f * inv_or_one(gr);
        const float groupbulls = ((c - gl) + (gh - gopen)) * scale2;
        const float groupbears = ((gh - c) + (gopen - gl)) * scale2;

        float b = intrabarbulls;
        float e = intrabarbears;
        if (mode == 0) {
            b = 0.5f * (intrabarbulls + groupbulls);
            e = 0.5f * (intrabarbears + groupbears);
        } else if (mode == 2) {
            b = groupbulls;
            e = groupbears;
        }

        const float old_b = (filled == period) ? ring_b[head] : 0.0f;
        const float old_e = (filled == period) ? ring_e[head] : 0.0f;
        sum_b += b - old_b;
        sum_e += e - old_e;
        ring_b[head] = b;
        ring_e[head] = e;
        head += 1; if (head == period) head = 0;
        if (filled < period) filled += 1;

        const float n = (float)filled;
        out_bulls[base + t] = sum_b / n;
        out_bears[base + t] = sum_e / n;
    }
}

// -------------- Many series × one param (time-major) --------------
// Dynamic shared memory layout per block:
//   float ring_b[period], ring_e[period];
//   int   dq_min_idx[period], dq_max_idx[period];
extern "C" __global__ void aso_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int* __restrict__ first_valids, // per series
    int cols,
    int rows,
    int period,
    int mode,
    float* __restrict__ out_bulls_tm,
    float* __restrict__ out_bears_tm) {
    const int s = blockIdx.x; // series index (column)
    if (s >= cols || period <= 0) return;

    // Initialize outputs to NaN in parallel across threads in the block
    for (int t = threadIdx.x; t < rows; t += blockDim.x) {
        const int idx = t * cols + s;
        out_bulls_tm[idx] = NAN;
        out_bears_tm[idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single-lane per series

    const int fv = first_valids[s];
    const int warm = fv + period - 1;
    if (warm >= rows) return;

    extern __shared__ unsigned char smem_uc[];
    float* ring_b = reinterpret_cast<float*>(smem_uc);
    float* ring_e = ring_b + period;
    int* dq_min_idx = reinterpret_cast<int*>(ring_e + period);
    int* dq_max_idx = dq_min_idx + period;

    for (int i = 0; i < period; ++i) {
        ring_b[i] = 0.0f; ring_e[i] = 0.0f;
        dq_min_idx[i] = 0; dq_max_idx[i] = 0;
    }

    int head = 0; int filled = 0;
    float sum_b = 0.0f, sum_e = 0.0f;

    // Monotonic deques (store indices into the time-major arrays)
    int min_head = 0, min_tail = 0, min_len = 0;
    int max_head = 0, max_tail = 0, max_len = 0;

    for (int t = fv; t < rows; ++t) {
        const int idx = t * cols + s;
        const float o = open_tm[idx];
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];

        // push low into monotonic-min deque
        while (min_len > 0) {
            int back = (min_tail == 0) ? (period - 1) : (min_tail - 1);
            int j = dq_min_idx[back];
            float lj = low_tm[j * cols + s];
            if (l <= lj) { min_tail = back; min_len -= 1; } else { break; }
        }
        if (min_len == period) { min_head = (min_head + 1) % period; min_len -= 1; }
        dq_min_idx[min_tail] = t; min_tail = (min_tail + 1) % period; min_len += 1;

        // push high into monotonic-max deque
        while (max_len > 0) {
            int back = (max_tail == 0) ? (period - 1) : (max_tail - 1);
            int j = dq_max_idx[back];
            float hj = high_tm[j * cols + s];
            if (h >= hj) { max_tail = back; max_len -= 1; } else { break; }
        }
        if (max_len == period) { max_head = (max_head + 1) % period; max_len -= 1; }
        dq_max_idx[max_tail] = t; max_tail = (max_tail + 1) % period; max_len += 1;

        if (t >= warm) {
            const int start = t - period + 1;
            while (min_len > 0 && dq_min_idx[min_head] < start) {
                min_head = (min_head + 1) % period; min_len -= 1;
            }
            while (max_len > 0 && dq_max_idx[max_head] < start) {
                max_head = (max_head + 1) % period; max_len -= 1;
            }

            const float gl = low_tm[dq_min_idx[min_head] * cols + s];
            const float gh = high_tm[dq_max_idx[max_head] * cols + s];
            const float gopen = open_tm[start * cols + s];

            const float intrarange = h - l;
            const float scale1 = 50.0f * inv_or_one(intrarange);
            const float intrabarbulls = ((c - l) + (h - o)) * scale1;
            const float intrabarbears = ((h - c) + (o - l)) * scale1;

            const float gr = gh - gl;
            const float scale2 = 50.0f * inv_or_one(gr);
            const float groupbulls = ((c - gl) + (gh - gopen)) * scale2;
            const float groupbears = ((gh - c) + (gopen - gl)) * scale2;

            float b = intrabarbulls;
            float e = intrabarbears;
            if (mode == 0) {
                b = 0.5f * (intrabarbulls + groupbulls);
                e = 0.5f * (intrabarbears + groupbears);
            } else if (mode == 2) {
                b = groupbulls; e = groupbears;
            }

            const float old_b = (filled == period) ? ring_b[head] : 0.0f;
            const float old_e = (filled == period) ? ring_e[head] : 0.0f;
            sum_b += b - old_b; sum_e += e - old_e;
            ring_b[head] = b; ring_e[head] = e;
            head += 1; if (head == period) head = 0;
            if (filled < period) filled += 1;

            const float n = (float)filled;
            const int out_idx = idx;
            out_bulls_tm[out_idx] = sum_b / n;
            out_bears_tm[out_idx] = sum_e / n;
        }
    }
}

