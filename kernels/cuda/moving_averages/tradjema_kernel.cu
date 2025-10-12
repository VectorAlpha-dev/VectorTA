// CUDA kernels for the Trend Adjusted EMA (TrAdjEMA).
// FP32 I/O for API compatibility; promote arithmetic to FP64 for CPU parity.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Force-inline for tiny hot helper
__device__ __forceinline__ double compute_true_range(
    double high, double low, double prev_close, bool first_bar)
{
    if (first_bar) {
        return high - low;
    }
    const double hl = high - low;
    const double hc = fabs(high - prev_close);
    const double lc = fabs(low - prev_close);
    return fmax(hl, fmax(hc, lc));
}

// Small helpers for circular indexing without %
__device__ __forceinline__ int inc_wrap(int x, int n) {
    ++x; return (x == n) ? 0 : x;
}
__device__ __forceinline__ int add_wrap(int head, int add, int n) {
    int s = head + add;
    return (s >= n) ? s - n : s;
}

//------------------------------------------------------------------------------
// Kernel 1: Many (length, mult) combos over a single series (contiguous time)
// Each block handles one combo; thread 0 runs the sequential EMA loop.
// Other threads only help with the initial NaN prefix fill.
//------------------------------------------------------------------------------

extern "C" __global__
void tradjema_batch_f32(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const float* __restrict__ close,
                        const int*   __restrict__ lengths,
                        const float* __restrict__ mults,
                        int series_len,
                        int n_combos,
                        int first_valid,
                        float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int   length   = lengths[combo];
    const float mult_f32 = mults[combo];
    const double mult    = (double)mult_f32;

    const int base = combo * series_len;

    // Invalid parameter set: write full NaN range and return
    if (length <= 1 || length > series_len || !isfinite(mult_f32) || mult_f32 <= 0.0f) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    const int warm  = first_valid + length - 1;
    const double alpha = 2.0 / ((double)length + 1.0);

    // Only write the necessary NaN prefix: [0, warm-1]
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    // Nothing to compute (window never fully warms)
    if (warm >= series_len || threadIdx.x != 0) return;

    // Shared memory layout:
    // [ double tr_buf[length] ][ int dq_min[length] ][ int dq_max[length] ]
    extern __shared__ double smem[];
    double* tr_buf = smem;
    int* dq_min = reinterpret_cast<int*>(tr_buf + length);
    int* dq_max = dq_min + length;

    // Deque bookkeeping (store absolute indices; ring index = abs % length)
    int min_head = 0, min_count = 0;
    int max_head = 0, max_count = 0;

    auto back_pos = [length](int head, int count) {
        int pos = head + count - 1;
        return (pos >= length) ? pos - length : pos;
    };
    auto get_ring = [length](int abs_idx) { return abs_idx % length; };

    // Initialize TR ring and deques over the warmup window [first_valid, first_valid+length-1]
    for (int k = 0; k < length; ++k) {
        const int idx = first_valid + k;
        const double prev_close = (idx == 0) ? 0.0 : (double)close[idx - 1];
        const double high_d = (double)high[idx];
        const double low_d  = (double)low[idx];
        const double tr = compute_true_range(high_d, low_d, prev_close, idx == first_valid);

        tr_buf[k] = tr;

        // push k to min deque
        while (min_count > 0) {
            const int bp = back_pos(min_head, min_count);
            const double vback = tr_buf[get_ring(dq_min[bp])];
            if (vback >= tr) { --min_count; } else { break; }
        }
        dq_min[add_wrap(min_head, min_count, length)] = k;
        ++min_count;

        // push k to max deque
        while (max_count > 0) {
            const int bp = back_pos(max_head, max_count);
            const double vback = tr_buf[get_ring(dq_max[bp])];
            if (vback <= tr) { --max_count; } else { break; }
        }
        dq_max[add_wrap(max_head, max_count, length)] = k;
        ++max_count;
    }

    int abs_idx = length - 1;               // absolute index of current TR in ring
    const double current_tr = tr_buf[abs_idx % length];
    const double tr_low  = tr_buf[get_ring(dq_min[min_head])];
    const double tr_high = tr_buf[get_ring(dq_max[max_head])];

    double tr_adj = (tr_high != tr_low) ? ((current_tr - tr_low) / (tr_high - tr_low)) : 0.0;

    // Initial EMA output at 'warm' (matches CPU reference form)
    const double src0 = (double)close[warm - 1];
    const double a0   = alpha * fma(tr_adj, mult, 1.0);   // alpha * (1 + tr_adj*mult)
    double y = a0 * src0;
    out[base + warm] = (float)y;

    // Main sequential loop
    for (int i = warm + 1; i < series_len; ++i) {
        ++abs_idx;
        const int pos = abs_idx % length;

        const double prev_close = (double)close[i - 1];
        const double tr_new = compute_true_range(
            (double)high[i], (double)low[i], prev_close, false);

        // expire old indices (<= abs_idx - length)
        while (min_count > 0 && dq_min[min_head] <= abs_idx - length) {
            min_head = inc_wrap(min_head, length);
            --min_count;
        }
        while (max_count > 0 && dq_max[max_head] <= abs_idx - length) {
            max_head = inc_wrap(max_head, length);
            --max_count;
        }

        // push new into deques
        while (min_count > 0) {
            const int bp = back_pos(min_head, min_count);
            const double vback = tr_buf[get_ring(dq_min[bp])];
            if (vback >= tr_new) { --min_count; } else { break; }
        }
        dq_min[add_wrap(min_head, min_count, length)] = abs_idx;
        ++min_count;

        while (max_count > 0) {
            const int bp = back_pos(max_head, max_count);
            const double vback = tr_buf[get_ring(dq_max[bp])];
            if (vback <= tr_new) { --max_count; } else { break; }
        }
        dq_max[add_wrap(max_head, max_count, length)] = abs_idx;
        ++max_count;

        // update ring after using old values
        tr_buf[pos] = tr_new;

        const double low_w  = tr_buf[get_ring(dq_min[min_head])];
        const double high_w = tr_buf[get_ring(dq_max[max_head])];
        tr_adj = (high_w != low_w) ? ((tr_new - low_w) / (high_w - low_w)) : 0.0;

        const double a = alpha * fma(tr_adj, mult, 1.0);
        const double src = (double)close[i - 1];
        y = fma(a, (src - y), y);  // one DFMA, stable update
        out[base + i] = (float)y;
    }
}

//------------------------------------------------------------------------------
// Kernel 2: Many series, one (length, mult), time-major storage.
// Same core improvements as above.
//------------------------------------------------------------------------------

extern "C" __global__
void tradjema_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int num_series,
    int series_len,
    int length,
    float mult_f32,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    if (length <= 1 || length > series_len || !isfinite(mult_f32) || mult_f32 <= 0.0f) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out_tm[t * num_series + series] = NAN;
        }
        return;
    }

    const int first_valid = first_valids[series];
    const int warm = first_valid + length - 1;

    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out_tm[t * num_series + series] = NAN;
    }
    __syncthreads();

    if (warm >= series_len || threadIdx.x != 0) {
        return;
    }

    extern __shared__ double tr_buf[];
    const double mult = static_cast<double>(mult_f32);
    const double alpha = 2.0 / (static_cast<double>(length) + 1.0);

    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    for (int k = 0; k < length; ++k) {
        const int idx = first_valid + k;
        const double prev_close = (idx == 0) ? 0.0 : static_cast<double>(at(close_tm, idx - 1, series));
        const double high_d = static_cast<double>(at(high_tm, idx, series));
        const double low_d = static_cast<double>(at(low_tm, idx, series));
        tr_buf[k] = compute_true_range(high_d, low_d, prev_close, idx == first_valid);
    }

    int head = length - 1;
    double tr_low = tr_buf[0];
    double tr_high = tr_buf[0];
    for (int k = 1; k < length; ++k) {
        const double v = tr_buf[k];
        tr_low = fmin(tr_low, v);
        tr_high = fmax(tr_high, v);
    }

    double current_tr = tr_buf[head];
    double tr_adj = (tr_high != tr_low) ? ((current_tr - tr_low) / (tr_high - tr_low)) : 0.0;
    const double src0 = static_cast<double>(at(close_tm, warm - 1, series));
    double y = alpha * (1.0 + tr_adj * mult) * (src0 - 0.0);
    out_tm[warm * num_series + series] = static_cast<float>(y);

    for (int i = warm + 1; i < series_len; ++i) {
        const double prev_close = static_cast<double>(at(close_tm, i - 1, series));
        const double tr_new = compute_true_range(
            static_cast<double>(at(high_tm, i, series)),
            static_cast<double>(at(low_tm, i, series)),
            prev_close,
            false
        );

        head = (head + 1) % length;
        const double tr_old = tr_buf[head];
        tr_buf[head] = tr_new;

        if (tr_old <= tr_low || tr_old >= tr_high) {
            tr_low = tr_buf[0];
            tr_high = tr_buf[0];
            for (int k = 1; k < length; ++k) {
                const double v = tr_buf[k];
                tr_low = fmin(tr_low, v);
                tr_high = fmax(tr_high, v);
            }
        } else {
            tr_low = fmin(tr_low, tr_new);
            tr_high = fmax(tr_high, tr_new);
        }

        tr_adj = (tr_high != tr_low) ? ((tr_new - tr_low) / (tr_high - tr_low)) : 0.0;
        const double a = alpha * (1.0 + tr_adj * mult);
        const double src = static_cast<double>(at(close_tm, i - 1, series));
        y += a * (src - y);
        out_tm[i * num_series + series] = static_cast<float>(y);
    }
}
