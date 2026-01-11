// CUDA kernels for the Trend Adjusted EMA (TrAdjEMA).
// FP32 I/O for API compatibility; prefer FP32 arithmetic for throughput.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Precise FP64 division even when compiled with --use_fast_math.
__device__ __forceinline__ double div_rn_f64(double num, double den) {
    return __ddiv_rn(num, den);
}

__device__ __forceinline__ double compute_true_range_f64(
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

// Force-inline for tiny hot helper
__device__ __forceinline__ float compute_true_range_f32(
    float high, float low, float prev_close, bool first_bar)
{
    if (first_bar) {
        return high - low;
    }
    const float hl = high - low;
    const float hc = fabsf(high - prev_close);
    const float lc = fabsf(low - prev_close);
    return fmaxf(hl, fmaxf(hc, lc));
}

// Small helpers for circular indexing without %
__device__ __forceinline__ int inc_wrap(int x, int n) {
    ++x; return (x == n) ? 0 : x;
}
__device__ __forceinline__ int dec_wrap(int x, int n) {
    return (x == 0) ? (n - 1) : (x - 1);
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
    const float mult     = mult_f32;

    const int base = combo * series_len;

    // Invalid parameter set: write full NaN range and return
    if (length <= 1 || length > series_len || !isfinite(mult_f32) || mult_f32 <= 0.0f) {
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            out[base + t] = NAN;
        }
        return;
    }

    const int warm  = first_valid + length - 1;
    const float alpha = 2.0f / (static_cast<float>(length) + 1.0f);

    // Only write the necessary NaN prefix: [0, warm-1]
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    // Nothing to compute (window never fully warms)
    if (warm >= series_len || threadIdx.x != 0) return;

    // Shared memory layout (mirrors the scalar ring-deque behavior):
    // [ f32 min_vals[length] ][ f32 max_vals[length] ][ i32 min_idx[length] ][ i32 max_idx[length] ]
    extern __shared__ __align__(16) unsigned char smem[];
    float* min_vals = reinterpret_cast<float*>(smem);
    float* max_vals = min_vals + length;
    int* min_idx = reinterpret_cast<int*>(max_vals + length);
    int* max_idx = min_idx + length;

    int min_head = 0, min_tail = 0;
    int max_head = 0, max_tail = 0;

    auto minq_push = [&](float v, int idx) {
        int back = dec_wrap(min_tail, length);
        // strict ">" to preserve older equal elements (matches scalar path)
        while (min_tail != min_head && min_vals[back] > v) {
            min_tail = back;
            back = dec_wrap(min_tail, length);
        }
        min_vals[min_tail] = v;
        min_idx[min_tail] = idx;
        min_tail = inc_wrap(min_tail, length);
    };
    auto maxq_push = [&](float v, int idx) {
        int back = dec_wrap(max_tail, length);
        // strict "<" to preserve older equal elements (matches scalar path)
        while (max_tail != max_head && max_vals[back] < v) {
            max_tail = back;
            back = dec_wrap(max_tail, length);
        }
        max_vals[max_tail] = v;
        max_idx[max_tail] = idx;
        max_tail = inc_wrap(max_tail, length);
    };

    // Seed window: [first_valid .. warm]
    float last_tr = high[first_valid] - low[first_valid];
    minq_push(last_tr, first_valid);
    maxq_push(last_tr, first_valid);

    for (int i = first_valid + 1; i <= warm; ++i) {
        const float prev_close = close[i - 1];
        const float tr = compute_true_range_f32(high[i], low[i], prev_close, false);
        minq_push(tr, i);
        maxq_push(tr, i);
        last_tr = tr;
    }

    const float tr_low  = min_vals[min_head];
    const float tr_high = max_vals[max_head];
    const float denom0 = tr_high - tr_low;
    const float tr_adj0 = (denom0 != 0.0f) ? ((last_tr - tr_low) / denom0) : 0.0f;

    // Initial EMA output at 'warm'
    const float src0 = close[warm - 1];
    const float a0 = alpha * (1.0f + tr_adj0 * mult);
    float y = src0 * a0;
    out[base + warm] = y;

    // Main sequential loop
    for (int i = warm + 1; i < series_len; ++i) {
        const int lim = i - length;
        while (min_head != min_tail && min_idx[min_head] <= lim) {
            min_head = inc_wrap(min_head, length);
        }
        while (max_head != max_tail && max_idx[max_head] <= lim) {
            max_head = inc_wrap(max_head, length);
        }

        const float prev_close = close[i - 1];
        const float tr = compute_true_range_f32(high[i], low[i], prev_close, false);

        minq_push(tr, i);
        maxq_push(tr, i);

        const float lo_tr = min_vals[min_head];
        const float hi_tr = max_vals[max_head];
        const float den = hi_tr - lo_tr;
        const float tr_adj = (den != 0.0f) ? ((tr - lo_tr) / den) : 0.0f;
        const float a = alpha * (1.0f + tr_adj * mult);

        const float src = prev_close; // src is close[i-1]
        y = fmaf(a, (src - y), y);
        out[base + i] = y;
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

    const double mult = static_cast<double>(mult_f32);
    const double alpha = div_rn_f64(2.0, static_cast<double>(length) + 1.0);

    auto at = [num_series](const float* buf, int row, int col) {
        return buf[row * num_series + col];
    };

    // Shared memory layout (mirrors the scalar ring-deque behavior):
    // [ f64 min_vals[length] ][ f64 max_vals[length] ][ i32 min_idx[length] ][ i32 max_idx[length] ]
    extern __shared__ __align__(16) unsigned char smem[];
    double* min_vals = reinterpret_cast<double*>(smem);
    double* max_vals = min_vals + length;
    int* min_idx = reinterpret_cast<int*>(max_vals + length);
    int* max_idx = min_idx + length;

    int min_head = 0, min_tail = 0;
    int max_head = 0, max_tail = 0;

    auto minq_push = [&](double v, int idx) {
        int back = dec_wrap(min_tail, length);
        while (min_tail != min_head && min_vals[back] > v) {
            min_tail = back;
            back = dec_wrap(min_tail, length);
        }
        min_vals[min_tail] = v;
        min_idx[min_tail] = idx;
        min_tail = inc_wrap(min_tail, length);
    };
    auto maxq_push = [&](double v, int idx) {
        int back = dec_wrap(max_tail, length);
        while (max_tail != max_head && max_vals[back] < v) {
            max_tail = back;
            back = dec_wrap(max_tail, length);
        }
        max_vals[max_tail] = v;
        max_idx[max_tail] = idx;
        max_tail = inc_wrap(max_tail, length);
    };

    // Seed window: [first_valid .. warm]
    double last_tr =
        static_cast<double>(at(high_tm, first_valid, series))
        - static_cast<double>(at(low_tm, first_valid, series));
    minq_push(last_tr, first_valid);
    maxq_push(last_tr, first_valid);

    for (int i = first_valid + 1; i <= warm; ++i) {
        const double prev_close = static_cast<double>(at(close_tm, i - 1, series));
        const double tr = compute_true_range_f64(
            static_cast<double>(at(high_tm, i, series)),
            static_cast<double>(at(low_tm, i, series)),
            prev_close,
            false
        );
        minq_push(tr, i);
        maxq_push(tr, i);
        last_tr = tr;
    }

    const double tr_low  = min_vals[min_head];
    const double tr_high = max_vals[max_head];
    const double denom0 = tr_high - tr_low;
    const double tr_adj0 = (denom0 != 0.0) ? div_rn_f64(last_tr - tr_low, denom0) : 0.0;

    const double src0 = static_cast<double>(at(close_tm, warm - 1, series));
    const double a0 = alpha * (1.0 + tr_adj0 * mult);
    double y = fma(src0, a0, 0.0);
    out_tm[warm * num_series + series] = static_cast<float>(y);

    for (int i = warm + 1; i < series_len; ++i) {
        const int lim = i - length;
        while (min_head != min_tail && min_idx[min_head] <= lim) {
            min_head = inc_wrap(min_head, length);
        }
        while (max_head != max_tail && max_idx[max_head] <= lim) {
            max_head = inc_wrap(max_head, length);
        }

        const double prev_close = static_cast<double>(at(close_tm, i - 1, series));
        const double tr = compute_true_range_f64(
            static_cast<double>(at(high_tm, i, series)),
            static_cast<double>(at(low_tm, i, series)),
            prev_close,
            false
        );

        minq_push(tr, i);
        maxq_push(tr, i);

        const double lo_tr = min_vals[min_head];
        const double hi_tr = max_vals[max_head];
        const double den = hi_tr - lo_tr;
        const double tr_adj = (den != 0.0) ? div_rn_f64(tr - lo_tr, den) : 0.0;
        const double a = alpha * (1.0 + tr_adj * mult);

        const double src = prev_close;
        y = fma(a, (src - y), y);
        out_tm[i * num_series + series] = static_cast<float>(y);
    }
}
