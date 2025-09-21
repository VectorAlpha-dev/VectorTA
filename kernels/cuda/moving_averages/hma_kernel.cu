// CUDA kernels for the Hull Moving Average (HMA).
//
// Each batch kernel thread processes one parameter combination sequentially.
// We reuse a per-combo scratch ring buffer (supplied by the host) to maintain
// the \u221a(n) weighted window without recomputing history. The many-series
// variant mirrors the same logic for time-major inputs.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef HMA_NAN
#define HMA_NAN (__int_as_float(0x7fffffff))
#endif

static __device__ __forceinline__ int clamp_positive(int v) {
    return v > 0 ? v : 0;
}

extern "C" __global__ void hma_batch_f32(const float* __restrict__ prices,
                                          const int* __restrict__ periods,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          int max_sqrt_len,
                                          float* __restrict__ x_buf,
                                          float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    const int half = period / 2;
    if (period <= 1 || half < 1) {
        return;
    }

    int sqrt_len = clamp_positive(static_cast<int>(floorf(sqrtf(static_cast<float>(period)))));
    if (sqrt_len < 1 || sqrt_len > max_sqrt_len) {
        return;
    }

    const int base = combo * series_len;
    for (int idx = 0; idx < series_len; ++idx) {
        out[base + idx] = HMA_NAN;
    }

    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period || tail_len < period + sqrt_len - 1) {
        return;
    }

    float* ring = x_buf + combo * max_sqrt_len;
    for (int i = 0; i < sqrt_len; ++i) {
        ring[i] = 0.0f;
    }

    const float ws_half = 0.5f * static_cast<float>(half) * static_cast<float>(half + 1);
    const float ws_full = 0.5f * static_cast<float>(period) * static_cast<float>(period + 1);
    const float ws_sqrt = 0.5f * static_cast<float>(sqrt_len) * static_cast<float>(sqrt_len + 1);

    float sum_half = 0.0f;
    float wsum_half = 0.0f;
    float sum_full = 0.0f;
    float wsum_full = 0.0f;

    float sum_x = 0.0f;
    float wsum_x = 0.0f;
    int ring_head = 0;
    int ring_count = 0;

    for (int j = 0; j < tail_len; ++j) {
        const int idx = first_valid + j;
        const float val = prices[idx];

        if (j < period) {
            sum_full += val;
            wsum_full += (static_cast<float>(j) + 1.0f) * val;
        } else {
            const float old = prices[idx - period];
            const float prev_sum = sum_full;
            sum_full = prev_sum + val - old;
            wsum_full = wsum_full - prev_sum + static_cast<float>(period) * val;
        }

        if (j < half) {
            sum_half += val;
            wsum_half += (static_cast<float>(j) + 1.0f) * val;
        } else {
            const float old = prices[idx - half];
            const float prev_sum = sum_half;
            sum_half = prev_sum + val - old;
            wsum_half = wsum_half - prev_sum + static_cast<float>(half) * val;
        }

        if (j + 1 < period) {
            continue;
        }

        const float wma_full = wsum_full / ws_full;
        const float wma_half = wsum_half / ws_half;
        const float x_val = 2.0f * wma_half - wma_full;

        if (ring_count < sqrt_len) {
            ring[ring_count] = x_val;
            sum_x += x_val;
            wsum_x += (static_cast<float>(ring_count) + 1.0f) * x_val;
            ++ring_count;

            if (ring_count == sqrt_len) {
                out[base + idx] = wsum_x / ws_sqrt;
            }
        } else {
            const float old_x = ring[ring_head];
            ring[ring_head] = x_val;
            ring_head = (ring_head + 1) % sqrt_len;

            const float prev_sum = sum_x;
            sum_x = prev_sum + x_val - old_x;
            wsum_x = wsum_x - prev_sum + static_cast<float>(sqrt_len) * x_val;

            out[base + idx] = wsum_x / ws_sqrt;
        }
    }
}

extern "C" __global__ void hma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    int max_sqrt_len,
    float* __restrict__ x_buf,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    auto idx = [num_series, series](int row) { return row * num_series + series; };

    for (int row = 0; row < series_len; ++row) {
        out_tm[idx(row)] = HMA_NAN;
    }

    const int half = period / 2;
    if (period <= 1 || half < 1) {
        return;
    }

    int sqrt_len = clamp_positive(static_cast<int>(floorf(sqrtf(static_cast<float>(period)))));
    if (sqrt_len < 1 || sqrt_len > max_sqrt_len) {
        return;
    }

    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < period || tail_len < period + sqrt_len - 1) {
        return;
    }

    float* ring = x_buf + series * max_sqrt_len;
    for (int i = 0; i < sqrt_len; ++i) {
        ring[i] = 0.0f;
    }

    const float ws_half = 0.5f * static_cast<float>(half) * static_cast<float>(half + 1);
    const float ws_full = 0.5f * static_cast<float>(period) * static_cast<float>(period + 1);
    const float ws_sqrt = 0.5f * static_cast<float>(sqrt_len) * static_cast<float>(sqrt_len + 1);

    float sum_half = 0.0f;
    float wsum_half = 0.0f;
    float sum_full = 0.0f;
    float wsum_full = 0.0f;

    float sum_x = 0.0f;
    float wsum_x = 0.0f;
    int ring_head = 0;
    int ring_count = 0;

    for (int j = 0; j < tail_len; ++j) {
        const int row = first_valid + j;
        const float val = prices_tm[idx(row)];

        if (j < period) {
            sum_full += val;
            wsum_full += (static_cast<float>(j) + 1.0f) * val;
        } else {
            const float old = prices_tm[idx(row - period)];
            const float prev_sum = sum_full;
            sum_full = prev_sum + val - old;
            wsum_full = wsum_full - prev_sum + static_cast<float>(period) * val;
        }

        if (j < half) {
            sum_half += val;
            wsum_half += (static_cast<float>(j) + 1.0f) * val;
        } else {
            const float old = prices_tm[idx(row - half)];
            const float prev_sum = sum_half;
            sum_half = prev_sum + val - old;
            wsum_half = wsum_half - prev_sum + static_cast<float>(half) * val;
        }

        if (j + 1 < period) {
            continue;
        }

        const float wma_full = wsum_full / ws_full;
        const float wma_half = wsum_half / ws_half;
        const float x_val = 2.0f * wma_half - wma_full;

        if (ring_count < sqrt_len) {
            ring[ring_count] = x_val;
            sum_x += x_val;
            wsum_x += (static_cast<float>(ring_count) + 1.0f) * x_val;
            ++ring_count;

            if (ring_count == sqrt_len) {
                out_tm[idx(row)] = wsum_x / ws_sqrt;
            }
        } else {
            const float old_x = ring[ring_head];
            ring[ring_head] = x_val;
            ring_head = (ring_head + 1) % sqrt_len;

            const float prev_sum = sum_x;
            sum_x = prev_sum + x_val - old_x;
            wsum_x = wsum_x - prev_sum + static_cast<float>(sqrt_len) * x_val;

            out_tm[idx(row)] = wsum_x / ws_sqrt;
        }
    }
}
