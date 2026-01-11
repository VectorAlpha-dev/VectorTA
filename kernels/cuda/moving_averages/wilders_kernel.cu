// CUDA kernels for Wilder's Moving Average (Wilders).
// Drop-in optimized rewrite using warp-shuffle reductions and warp-per-series mapping.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---- Common helpers ---------------------------------------------------------
static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    // One shared slot per warp (max 32 for 1024 threads)
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5; // / warpSize

    // Reduce within each warp.
    v = warp_reduce_sum(v);

    // Warp leaders write to shared.
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    // First warp reads warp sums and reduces them.
    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum; // valid in lane 0 of warp 0
}

extern "C" __global__
void wilders_batch_f32(const float* __restrict__ prices,
                       const int* __restrict__ periods,
                       const float* __restrict__ alphas,
                       const int* __restrict__ warm_indices,
                       int series_len,
                       int first_valid,
                       int n_combos,
                       float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];

    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;

    // 1) Fill entire output row with NaNs cooperatively.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // 2) Cooperatively accumulate the first window sum with a block-wide reduction.
    const int start      = first_valid;
    const int window_end = start + period;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < period; i += blockDim.x) {
        const int idx = start + i;
        if (idx < series_len) {
            local_sum += prices[idx];
        }
    }

    // All threads participate; result valid in lane 0 of warp 0.
    const float sum = block_reduce_sum(local_sum);

    // Only one thread needs to continue (no further barriers).
    if (threadIdx.x != 0) return;

    if (window_end > series_len) return;

    const float inv_period = 1.0f / static_cast<float>(period);
    float value = sum * inv_period;
    out[base + warm] = value;

    // 3) Sequential Wilder recurrence (FMA, round-to-nearest)
    for (int t = warm + 1; t < series_len; ++t) {
        const float price = prices[t];
        value = __fmaf_rn(price - value, alpha, value);
        out[base + t] = value;
    }
}

// Batch warp-scan kernel: one warp computes one combo (row) and emits 32 timesteps
// per iteration via an inclusive scan over the affine Wilder transform:
//   y_t = (1-alpha) * y_{t-1} + alpha * x_t
//
// - blockDim.x must be exactly 32
// - output is written once: warmup prefix is NaN, then all t>=warm are computed
extern "C" __global__
void wilders_batch_warp_scan_f32(const float* __restrict__ prices,
                                 const int* __restrict__ periods,
                                 const float* __restrict__ alphas,
                                 const int* __restrict__ warm_indices,
                                 int series_len,
                                 int first_valid,
                                 int n_combos,
                                 float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0 || first_valid < 0 || first_valid >= series_len) return;
    if (threadIdx.x >= 32) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len) return;

    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)combo * (size_t)series_len;

    // Warmup prefix NaNs (indices < warm)
    for (int t = lane; t < warm; t += 32) {
        out[base + (size_t)t] = NAN;
    }
    if (warm < 0 || warm >= series_len) return;

    // Seed at warm: mean over [first_valid, first_valid+period)
    float y_prev = 0.0f;
    if (lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < period; ++i) {
            sum += prices[first_valid + i];
        }
        y_prev = sum / (float)period;
        out[base + (size_t)warm] = y_prev;
    }
    y_prev = __shfl_sync(mask, y_prev, 0);

    int t0 = warm + 1;
    if (t0 >= series_len) return;

    const float one_m_alpha = 1.0f - alpha;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);

        float A = valid ? one_m_alpha : 1.0f;
        float B = valid ? (alpha * prices[t]) : 0.0f;

        // Inclusive scan of composed affine transforms (A, B)
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }

        const float y = fmaf(A, y_prev, B);
        if (valid) {
            out[base + (size_t)t] = y;
        }

        const int remaining = series_len - tile;
        const int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        y_prev = __shfl_sync(mask, y, last_lane);
    }
}

// Many-series × one-param (time-major) kernel for Wilder's MA.
//
// - prices are time-major: prices_tm[t * num_series + series]
// - first_valids[series] marks the first finite sample per series
// - warmup is the simple average over the first full `period` window
// - recurrence uses FMA and propagates non-finite per IEEE-754 semantics
extern "C" __global__
void wilders_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                       const int* __restrict__ first_valids,
                                       int period,
                                       float alpha,
                                       int num_series,
                                       int series_len,
                                       float* __restrict__ out_tm) {
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    // Warp identification
    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5; // / warpSize
    const int warps_per_block = blockDim.x >> 5;  // / warpSize
    if (warps_per_block == 0) return;            // require at least one warp

    // Global warp index and grid-stride over warps
    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int series_idx = warp_idx; series_idx < num_series; series_idx += wstep) {
        const int first_valid = first_valids[series_idx];

        // Initialize output with NaNs cooperatively by lanes
        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + series_idx] = NAN;
        }

        if (first_valid < 0 || first_valid >= series_len) {
            continue; // whole series remains NaN
        }

        const int warm_end = first_valid + period;
        if (warm_end > series_len) {
            continue; // insufficient samples; leave NaNs
        }

        // Initial mean over the first full window [first_valid, warm_end)
        float local = 0.0f;
        for (int k = lane; k < period; k += warpSize) {
            const int idx = (first_valid + k) * stride + series_idx;
            local += prices_tm[idx];
        }
        float sum = warp_reduce_sum(local);

        // Lane 0 writes warm value and runs the sequential recurrence
        if (lane == 0) {
            const float inv_period = 1.0f / static_cast<float>(period);
            float y = sum * inv_period;
            const int warm = warm_end - 1;
            out_tm[warm * stride + series_idx] = y;

            for (int t = warm + 1; t < series_len; ++t) {
                const float x = prices_tm[t * stride + series_idx];
                y = __fmaf_rn(x - y, alpha, y);
                out_tm[t * stride + series_idx] = y;
            }
        }
        // No block-wide sync needed; warp-scope ops are sufficient
    }
}
