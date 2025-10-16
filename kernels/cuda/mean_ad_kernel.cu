// CUDA kernels for Mean Absolute Deviation (MeanAd)
//
// Math pattern: Recurrence over time per parameter/series
// - Rolling SMA via window sum (sequential in t)
// - Rolling mean of absolute residuals via a period-length ring buffer
//
// Semantics:
// - Warmup index = first_valid + 2*period - 2
// - Write NaN before warmup; write value at warmup and onward
// - Ignore NaN inputs (host enforces first_valid and sufficient window)

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __forceinline__ __device__ float qnan32() {
    return __int_as_float(0x7fffffff);
}

// One-series × many-params (batch). One block per combo.
extern "C" __global__
void mean_ad_batch_f32(const float* __restrict__ prices,
                       const int* __restrict__ periods,
                       const int* __restrict__ warm_indices,
                       int first_valid,
                       int series_len,
                       int n_combos,
                       float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    const int warm   = warm_indices[combo];
    if (period <= 0) return;

    const int base = combo * series_len;

    // 1) Fill row with NaNs cooperatively
    for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
        out[base + t] = qnan32();
    }
    __syncthreads();

    if (warm >= series_len) return; // nothing more to write

    // 2) Initial SMA over [first_valid, first_valid+period)
    float local = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const int idx = first_valid + k;
        if (idx < series_len) local += prices[idx];
    }
    // Block-wide reduction using shared memory (one slot per warp)
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    // warp reduce
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        local += __shfl_down_sync(mask, local, ofs);
    }
    if (lane == 0) warp_sums[wid] = local;
    __syncthreads();
    float sum = 0.0f;
    if (wid == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            v += __shfl_down_sync(mask, v, ofs);
        }
        if (lane == 0) sum = v;
    }
    // Broadcast from lane 0 of warp 0
    sum = __shfl_sync(mask, sum, 0);

    if (first_valid + period > series_len) return;

    const float inv_p = 1.0f / (float)period;
    float sma = sum * inv_p;

    // 3) Residual ring buffer in dynamic shared memory (size == launch-time max_period)
    extern __shared__ float s_ring[];
    // Only one thread maintains the sequential ring/scan; others are idle after init
    if (threadIdx.x == 0) {
        // Fill residuals for first `period` SMAs
        const int start_t = first_valid + period - 1;
        const int fill_end = min(start_t + period - 1, series_len - 1);
        float residual_sum = 0.0f;
        int head = 0;
        for (int t = start_t; t <= fill_end; ++t) {
            const float r = fabsf(prices[t] - sma);
            s_ring[head++] = r;
            if (head == period) head = 0;
            residual_sum += r;
            if (t + 1 < series_len) {
                sum += prices[t + 1] - prices[t + 1 - period];
                sma = sum * inv_p;
            }
        }

        // First output (at warm index)
        out[base + warm] = residual_sum * inv_p;

        // Main loop
        int t = start_t + period;
        int idx = head;
        while (t < series_len) {
            const float r = fabsf(prices[t] - sma);
            const float old = s_ring[idx];
            s_ring[idx] = r;
            idx += 1; if (idx == period) idx = 0;
            residual_sum += r - old;
            out[base + t] = residual_sum * inv_p;
            if (t + 1 < series_len) {
                sum += prices[t + 1] - prices[t + 1 - period];
                sma = sum * inv_p;
            }
            ++t;
        }
    }
}

// Many-series × one-param (time-major). One thread per series with a per-thread ring in shared memory.
extern "C" __global__
void mean_ad_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                       int period,
                                       int num_series,
                                       int series_len,
                                       const int* __restrict__ first_valids,
                                       float* __restrict__ out_tm) {
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int first = first_valids[series_idx];
    if (first < 0 || first >= series_len) return;

    const int warm = first + 2 * period - 2;
    const int stride = num_series;

    // Initialize warmup prefix with NaNs for this series
    const int warm_clamped = warm < series_len ? warm : series_len;
    for (int t = 0; t < warm_clamped; ++t) {
        out_tm[t * stride + series_idx] = qnan32();
    }
    if (warm >= series_len) return;

    // Rolling SMA over prices[first .. first+period)
    float sum = 0.0f;
    size_t p = (size_t)first * (size_t)stride + (size_t)series_idx;
    for (int k = 0; k < period; ++k) {
        sum += prices_tm[p];
        p += (size_t)stride;
    }
    const float inv_p = 1.0f / (float)period;
    float sma = sum * inv_p;

    // Per-thread ring buffer in dynamic shared memory
    extern __shared__ float smem[];
    float* ring = smem + (size_t)threadIdx.x * (size_t)period;
    int head = 0;
    float residual_sum = 0.0f;

    const int start_t = first + period - 1;
    const int fill_end = min(start_t + period - 1, series_len - 1);
    for (int t = start_t; t <= fill_end; ++t) {
        const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
        const float r = fabsf(price_t - sma);
        ring[head++] = r; if (head == period) head = 0;
        residual_sum += r;
        if (t + 1 < series_len) {
            const float in_next = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
            const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
            sum += in_next - out_prev;
            sma = sum * inv_p;
        }
    }
    out_tm[(size_t)warm * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

    // Main loop
    int t = start_t + period;
    int idx = head;
    while (t < series_len) {
        const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
        const float r = fabsf(price_t - sma);
        const float old = ring[idx];
        ring[idx] = r;
        idx += 1; if (idx == period) idx = 0;
        residual_sum += r - old;
        out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;
        if (t + 1 < series_len) {
            const float in_next = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
            const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
            sum += in_next - out_prev;
            sma = sum * inv_p;
        }
        ++t;
    }
}

