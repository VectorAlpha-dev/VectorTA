













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

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    while (t < len) {
        if (t < warm) {
            out[row_off + t] = QS_NAN;
        } else {
            const int t1 = t + 1;
            int start = t1 - period; if (start < 0) start = 0;
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



extern "C" __global__ void qstick_many_series_one_param_f32(
    const float* __restrict__ prefix_tm,
    int period,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm
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

