// CUDA kernels for QStick indicator using prefix sums of (close - open).
//
// Category: Sliding-sum average over the differences d[t] = close[t] - open[t].
// We precompute a prefix array P where P[0]=0 and P[t+1] = P[t] + d[t].
// Then each output is:
//   out[t] = (P[t+1] - P[t+1-period]) / period, for t >= warm,
// with warm = first_valid + period - 1. Indices before warm are filled with NaN
// to match scalar semantics.
//
// Provided kernels:
// - qstick_batch_prefix_f32                         : one-series × many-params (plain 1D time, grid.y = combos)
// - qstick_batch_prefix_tiled_f32_tile128/256      : optional tiled variants
// - qstick_many_series_one_param_f32               : many-series × one-param (time-major), 1D

#include <cuda_runtime.h>

#ifndef QS_NAN
#define QS_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

extern "C" __global__ void qstick_batch_prefix_f32(
    const float* __restrict__ prefix_diff, // length = len + 1
    int len,
    int first_valid,
    const int* __restrict__ periods,       // [n_combos]
    int n_combos,
    float* __restrict__ out                // [n_combos * len]
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float inv_p = 1.0f / static_cast<float>(period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        if (t < warm) {
            out[row_off + t] = QS_NAN;
        } else {
            const int t1 = t + 1;
            int start = t1 - period; if (start < 0) start = 0; // clamp for safety
            const float sum = prefix_diff[t1] - prefix_diff[start];
            out[row_off + t] = sum * inv_p;
        }
        t += stride;
    }
}

template<int TILE>
__device__ __forceinline__ void qstick_batch_prefix_tiled_impl(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out
) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;
    const int period = periods[combo];
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valid + period - 1;
    const int row_off = combo * len;
    const float inv_p = 1.0f / static_cast<float>(period);

    const int t0 = blockIdx.x * TILE;
    const int t = t0 + threadIdx.x;
    if (t >= len) return;

    if (t < warm) {
        out[row_off + t] = QS_NAN;
        return;
    }
    const int t1 = t + 1;
    int start = t1 - period; if (start < 0) start = 0;
    const float sum = prefix_diff[t1] - prefix_diff[start];
    out[row_off + t] = sum * inv_p;
}

extern "C" __global__ void qstick_batch_prefix_tiled_f32_tile128(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    qstick_batch_prefix_tiled_impl<128>(prefix_diff, len, first_valid, periods, n_combos, out);
}

extern "C" __global__ void qstick_batch_prefix_tiled_f32_tile256(
    const float* __restrict__ prefix_diff,
    int len,
    int first_valid,
    const int* __restrict__ periods,
    int n_combos,
    float* __restrict__ out) {
    qstick_batch_prefix_tiled_impl<256>(prefix_diff, len, first_valid, periods, n_combos, out);
}

// ---------------- Many-series, one param (time-major) ----------------------
// prefix_tm: (rows+1) x cols, time-major. out_tm: rows x cols, time-major.
extern "C" __global__ void qstick_many_series_one_param_f32(
    const float* __restrict__ prefix_tm, // (rows+1) x cols
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids, // [num_series]
    float* __restrict__ out_tm             // rows x cols
) {
    const int series = blockIdx.y;
    if (series >= num_series) return;
    if (UNLIKELY(period <= 0)) return;

    const int warm = first_valids[series] + period - 1;
    const int stride = num_series;
    const float inv_p = 1.0f / static_cast<float>(period);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * stride + series;
        if (t < warm) {
            out_tm[out_idx] = QS_NAN;
        } else {
            const int t1 = t + 1;
            int start = t1 - period; if (start < 0) start = 0;
            const int p_idx = t1 * stride + series;
            const int s_idx = start * stride + series;
            const float sum = prefix_tm[p_idx] - prefix_tm[s_idx];
            out_tm[out_idx] = sum * inv_p;
        }
        t += step;
    }
}

