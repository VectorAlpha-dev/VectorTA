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
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32]; // up to 1024 threads
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    float block_sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum; // valid in lane 0 of warp 0
}

// Branchless True Range at t. Seed rule: t==first_valid uses hi-lo.
static __forceinline__ __device__ float tr_at_branchless(
    const float hi, const float lo, const float pc, int t, int first_valid)
{
    const float seed = hi - lo;
    const float alt1 = fabsf(hi - pc);
    const float alt2 = fabsf(lo - pc);
    float m = fmaxf(seed, fmaxf(alt1, alt2));
    return (t == first_valid) ? seed : m;
}

// ---- Double-fp32 compensated sums (float2 hi/lo) ---------------------------
// Neumaier/Kahan style compensated accumulate: (hi, lo) += x
static __forceinline__ __device__ void acc_add(float &hi, float &lo, float x) {
    float s = hi + x;
    float bp = s - hi;
    float err = (hi - (s - bp)) + (x - bp);
    hi = s;
    lo += err;
}

// (hi, lo) += (bhi, blo)
static __forceinline__ __device__ void acc_add2(float &hi, float &lo, float bhi, float blo) {
    acc_add(hi, lo, bhi);
    acc_add(hi, lo, blo);
}

// r = a - b for float2 accumulators, stored back into (rhi, rlo)
static __forceinline__ __device__ void acc_sub2(float ahi, float alo, float bhi, float blo,
                                                float &rhi, float &rlo) {
    rhi = 0.0f; rlo = 0.0f;
    acc_add(rhi, rlo, ahi);
    acc_add(rhi, rlo, alo);
    acc_add(rhi, rlo, -bhi);
    acc_add(rhi, rlo, -blo);
}

// Precompute TR from H/L/C (optional helper)
extern "C" __global__
void tr_from_hlc_f32(const float* __restrict__ high,
                     const float* __restrict__ low,
                     const float* __restrict__ close,
                     int series_len,
                     int first_valid,
                     float* __restrict__ tr_out)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < series_len;
         t += blockDim.x * gridDim.x)
    {
        float tri = 0.0f;
        if (t >= first_valid) {
            const float hi = high[t];
            const float lo = low[t];
            const float pc = (t == first_valid) ? 0.0f : close[t - 1];
            tri = tr_at_branchless(hi, lo, pc, t, first_valid);
        }
        tr_out[t] = tri;
    }
}

// Exclusive prefix using double-fp32 accumulator (size N+1)
extern "C" __global__
void exclusive_prefix_float2_from_tr(const float* __restrict__ tr,
                                     int series_len,
                                     float2* __restrict__ prefix2)
{
    if (blockIdx.x != 0) return;
    float hi = 0.0f, lo = 0.0f;
    if (threadIdx.x == 0) {
        prefix2[0] = make_float2(0.0f, 0.0f);
        for (int t = 0; t < series_len; ++t) {
            acc_add(hi, lo, tr[t]);
            prefix2[t + 1] = make_float2(hi, lo);
        }
    }
}

// Unified batch kernel (one series × many params) with runtime seeding path
extern "C" __global__
void atr_batch_unified_f32(const float* __restrict__ high,
                           const float* __restrict__ low,
                           const float* __restrict__ close,
                           const float* __restrict__ tr,
                           const float2* __restrict__ prefix2,
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
    const int base  = combo * series_len;
    const int start = first_valid;

    // Only NaN-fill the pre-warm range
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    // Seed
    float seed_mean = 0.0f;
    if (prefix2 != nullptr) {
        if (threadIdx.x == 0) {
            float2 a = prefix2[warm + 1];
            float2 b = prefix2[start];
            float shi, slo;
            acc_sub2(a.x, a.y, b.x, b.y, shi, slo);
            seed_mean = (shi + slo) / (float)period;
        }
        __syncthreads();
    } else {
        float local = 0.0f;
        const int end = start + period; // exclusive
        for (int k = threadIdx.x; k < period; k += blockDim.x) {
            const int t = start + k;
            float tri;
            if (tr != nullptr) {
                tri = tr[t];
            } else {
                const float hi = high[t];
                const float lo = low[t];
                const float pc = (t == start) ? 0.0f : close[t - 1];
                tri = tr_at_branchless(hi, lo, pc, t, start);
            }
            local += tri;
        }
        const float sum = block_reduce_sum(local);
        if (threadIdx.x == 0) seed_mean = sum / (float)period;
        __syncthreads();
    }

    // Sequential RMA per combo
    if (threadIdx.x == 0) {
        float y = seed_mean;
        out[base + warm] = y;
        for (int t = warm + 1; t < series_len; ++t) {
            float tri;
            if (tr != nullptr) {
                tri = tr[t];
            } else {
                const float hi = high[t];
                const float lo = low[t];
                const float pc = close[t - 1];
                tri = tr_at_branchless(hi, lo, pc, t, start);
            }
            y = __fmaf_rn(tri - y, alpha, y);
            out[base + t] = y;
        }
    }
}

// Backward-compatible symbol: one-series × many-params, on-the-fly TR path
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

    const int base  = combo * series_len;
    const int start = first_valid;

    // Only pre-warm NaNs
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    // Seed via reduction over first window
    float local = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const int t = start + k;
        const float hi = high[t];
        const float lo = low[t];
        const float pc = (t == start) ? 0.0f : close[t - 1];
        local += tr_at_branchless(hi, lo, pc, t, start);
    }
    const float sum = block_reduce_sum(local);

    if (threadIdx.x == 0) {
        float y = sum / (float)period;
        out[base + warm] = y;
        for (int t = warm + 1; t < series_len; ++t) {
            const float hi = high[t];
            const float lo = low[t];
            const float pc = close[t - 1];
            const float tri = tr_at_branchless(hi, lo, pc, t, start);
            y = __fmaf_rn(tri - y, alpha, y);
            out[base + t] = y;
        }
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

    const int base  = combo * series_len;
    const int start = first_valid;

    // Only pre-warm NaNs
    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        out[base + t] = NAN;
    }
    __syncthreads();

    // Prefer prefix path if provided, else reduce over TR
    float seed_mean = 0.0f;
    if (prefix_tr != nullptr) {
        if (threadIdx.x == 0) {
            // Avoid FP64 math: cast each endpoint to f32 before subtracting
            const float a = (float)prefix_tr[warm + 1];
            const float b = (float)prefix_tr[start];
            seed_mean = (a - b) / (float)period;
        }
        __syncthreads();
    } else {
        float local = 0.0f;
        for (int k = threadIdx.x; k < period; k += blockDim.x) {
            const int t = start + k;
            local += tr[t];
        }
        const float sum = block_reduce_sum(local);
        if (threadIdx.x == 0) seed_mean = sum / (float)period;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float y = seed_mean;
        out[base + warm] = y;
        for (int t = warm + 1; t < series_len; ++t) {
            const float tri = tr[t];
            y = __fmaf_rn(tri - y, alpha, y);
            out[base + t] = y;
        }
    }
}

// Many-series × one-param, time-major, coalesced across series per warp
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
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;

    for (int s_base = warp_global * warpSize; s_base < num_series; s_base += warps_per_block * gridDim.x * warpSize) {
        const int s = s_base + lane;
        if (s >= num_series) continue;

        const int first_valid = first_valids[s];
        if (first_valid < 0 || first_valid >= series_len) continue;
        const int warm_end = first_valid + period; // exclusive
        if (warm_end > series_len) continue;
        const int warm = warm_end - 1;

        // Pre-warm NaNs for this series
        for (int t = 0; t < warm; ++t) {
            out_tm[t * stride + s] = NAN;
        }

        // Seed over first window with coalesced loads
        float sum = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < period; ++k) {
            const int t = first_valid + k;
            const float hi = high_tm[t * stride + s];
            const float lo = low_tm[t * stride + s];
            const float pc = (t == first_valid) ? 0.0f : close_tm[(t - 1) * stride + s];
            sum += tr_at_branchless(hi, lo, pc, t, first_valid);
        }

        float y = sum / (float)period;
        out_tm[warm * stride + s] = y;

        // Sequential RMA per series across time (coalesced streaming)
        for (int t = warm + 1; t < series_len; ++t) {
            const float hi = high_tm[t * stride + s];
            const float lo = low_tm[t * stride + s];
            const float pc = close_tm[(t - 1) * stride + s];
            const float tri = tr_at_branchless(hi, lo, pc, t, first_valid);
            y = __fmaf_rn(tri - y, alpha, y);
            out_tm[t * stride + s] = y;
        }
    }
}
