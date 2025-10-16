// CUDA kernels for Average True Range (ATR) using Wilder's RMA smoothing.
//
// Pattern mirrors wilders_kernel.cu: a block cooperatively seeds the initial
// window via reduction, then a single lane performs the sequential recurrence.
// True Range (TR) per time t is max(high-low, |high-prev_close|, |low-prev_close|),
// with t==first_valid using (high-low) as the seed per scalar behavior.

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

// Compute True Range at index t given high/low/close and prev close.
static __forceinline__ __device__ float tr_at(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int t,
    int first_valid)
{
    const float hi = high[t];
    const float lo = low[t];
    if (t == first_valid) {
        return hi - lo; // seed behavior matches scalar
    }
    const float pc = close[t - 1];
    float tr = hi - lo;
    float hc = fabsf(hi - pc);
    if (hc > tr) tr = hc;
    float lc = fabsf(lo - pc);
    if (lc > tr) tr = lc;
    return tr;
}

extern "C" __global__
void atr_batch_f32(const float* __restrict__ high,
                   const float* __restrict__ low,
                   const float* __restrict__ close,
                   const int* __restrict__ periods,
                   const float* __restrict__ alphas,
                   const int* __restrict__ warm_indices,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;

    // 1) Initialize NaN prefix and clear whole row to NaN cooperatively.
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    // 2) Cooperatively accumulate the first window TR sum with a block-wide reduction.
    const int start      = first_valid;
    const int window_end = start + period; // exclusive
    if (window_end > series_len) return;

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const int t = start + k;
        local_sum += tr_at(high, low, close, t, first_valid);
    }
    const float sum = block_reduce_sum(local_sum); // valid in lane 0 of warp 0

    // 3) Single-lane sequential RMA update across time
    if (threadIdx.x != 0) return;

    float y = sum / (float)period;
    out[base + (warm)] = y;
    for (int t = warm + 1; t < series_len; ++t) {
        const float tri = tr_at(high, low, close, t, first_valid);
        y = __fmaf_rn(tri - y, alpha, y); // y += alpha * (tri - y)
        out[base + t] = y;
    }
}

// Optimized batch kernel using shared precomputed TR and its prefix sums.
extern "C" __global__
void atr_batch_from_tr_prefix_f32(const float* __restrict__ tr,
                                  const double* __restrict__ prefix_tr,
                                  const int* __restrict__ periods,
                                  const float* __restrict__ alphas,
                                  const int* __restrict__ warm_indices,
                                  int series_len,
                                  int first_valid,
                                  int n_combos,
                                  float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    const int   period = periods[combo];
    const float alpha  = alphas[combo];
    const int   warm   = warm_indices[combo];
    if (period <= 0 || warm >= series_len || first_valid >= series_len) return;

    const int base = combo * series_len;
    // Initialize whole row to NaN cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const double sum = prefix_tr[warm + 1] - prefix_tr[first_valid];
        float y = (float)(sum / (double)period);
        out[base + warm] = y;
        for (int t = warm + 1; t < series_len; ++t) {
            const float tri = tr[t];
            y = __fmaf_rn(tri - y, alpha, y);
            out[base + t] = y;
        }
    }
}

// Many-series × one-param (time-major) kernel for ATR.
// Inputs are time-major: X_tm[t * num_series + s]. Each warp processes one series.
extern "C" __global__
void atr_many_series_one_param_f32(const float* __restrict__ high_tm,
                                   const float* __restrict__ low_tm,
                                   const float* __restrict__ close_tm,
                                   const int* __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;
    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5; // / warpSize
    const int warps_per_block = blockDim.x >> 5;  // / warpSize
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int first_valid = first_valids[s];
        // Initialize entire column to NaN cooperatively by lanes
        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + s] = NAN;
        }

        if (first_valid < 0 || first_valid >= series_len) continue;
        const int warm_end = first_valid + period; // exclusive
        if (warm_end > series_len) continue; // insufficient samples
        const int warm = warm_end - 1;

        // Seed via parallel reduction across the first window
        float local = 0.0f;
        for (int k = lane; k < period; k += warpSize) {
            const int t = first_valid + k;
            const float hi = high_tm[t * stride + s];
            const float lo = low_tm[t * stride + s];
            float tri;
            if (t == first_valid) {
                tri = hi - lo;
            } else {
                const float pc = close_tm[(t - 1) * stride + s];
                float tr = hi - lo;
                float hc = fabsf(hi - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(lo - pc);
                if (lc > tr) tr = lc;
                tri = tr;
            }
            local += tri;
        }
        float sum = warp_reduce_sum(local);

        if (lane == 0) {
            float y = sum / (float)period;
            out_tm[warm * stride + s] = y;
            for (int t = warm + 1; t < series_len; ++t) {
                const float hi = high_tm[t * stride + s];
                const float lo = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                float tr = hi - lo;
                float hc = fabsf(hi - pc);
                if (hc > tr) tr = hc;
                float lc = fabsf(lo - pc);
                if (lc > tr) tr = lc;
                y = __fmaf_rn(tr - y, alpha, y);
                out_tm[t * stride + s] = y;
            }
        }
    }
}
