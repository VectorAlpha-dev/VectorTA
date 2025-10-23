// CUDA kernels for NATR (Normalized Average True Range)
// Optimized for FP32 throughput with compensated arithmetic (no FP64 in hot path)
//
// Math: NATR = (ATR / close) * 100, where ATR uses Wilder smoothing
// with warmup initialized as the arithmetic mean of True Range over the first
// full window [first_valid, first_valid + period - 1].
//
// Batch kernel (one series × many params):
// - Each block handles one parameter row (period)
// - Threads cooperatively compute the warmup TR sum using a block-wide reduction
//   with intrathread compensated partials
// - A single thread runs the sequential Wilder recurrence (compensated FP32)
// - Optional helper kernel can precompute inv_close100[t] = 100/close[t]
//   for sharing across rows (not required for correctness)
//
// Many-series kernel (time‑major):
// - Warp-per-series mapping, identical structure to wilders_many_series
// - TR is computed on-the-fly from H/L/close
// - Sequential Wilder recurrence is compensated FP32 in lane 0

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// -------------------- Reductions (borrowed from Wilder kernels) --------------------
static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    float block_sum = 0.0f;
    if (wid == 0 && lane == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        // Kahan-style accumulation across warp partials for improved stability
        float c = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < num_warps; ++i) {
            float y = warp_sums[i] - c;
            float t = block_sum + y;
            c = (t - block_sum) - y;
            block_sum = t;
        }
        block_sum += c;
    }
    // Only wid==0,lane==0 carries a valid result in our callers
    return (wid == 0 && lane == 0) ? block_sum : 0.0f;
}

// -------------------- Helpers --------------------
__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }

// Return 100.0f / close, or NaN if close invalid/zero.
__device__ __forceinline__ float safe_scale_100_over_close(float c) {
    return (isfinite(c) && c != 0.0f) ? (100.0f / c) : dev_nan();
}

// Compensated EMA update for: atr <- atr + alpha*(x - atr)
__device__ __forceinline__ void ema_update_kahan(float& atr, float& c, float alpha, float x) {
    // y = alpha * (x - (atr + c)) ; t = atr + y ; c = (t - atr) - y ; atr = t
    float y = __fmaf_rn(alpha, x - (atr + c), 0.0f);
    float t = atr + y;
    c = (t - atr) - y;
    atr = t;
}

// -------------------- (Optional) Precompute inv_close100 --------------------
// inv_close100[t] = 100.0f / close[t], or NaN if close[t] invalid/zero.
extern "C" __global__ void natr_make_inv_close100(
    const float* __restrict__ close, int len, float* __restrict__ inv_close100)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len) {
        inv_close100[t] = safe_scale_100_over_close(close[t]);
    }
}

// -------------------- Batch: one series × many params --------------------

// Inputs:
//  - tr:    True Range per time (length=len). Precomputed on host.
//  - close: Close price per time (length=len).
//  - periods: array of periods (length=n_combos)
//  - series_len, first_valid, n_combos
// Output:
//  - out: row-major [n_combos, len]
// Backward-compatible signature used by Rust wrapper.
// This variant computes 100/close on the fly inside the kernel.
extern "C" __global__ void natr_batch_f32(
    const float* __restrict__ tr,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    // 1) Fill output row with NaNs cooperatively
    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = dev_nan();
    }
    __syncthreads();

    if (first_valid >= series_len || warm >= series_len) return;

    // 2) Accumulate warmup TR sum across threads (intrathread Kahan, then block reduce)
    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c; // fold compensation back
    const float sum_f = block_reduce_sum(local_sum);

    if (threadIdx.x != 0) return; // single-thread sequential recurrence

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = static_cast<double>(sum_f) * inv_p;

    // 3) Emit warm value and run sequential Wilder recurrence
    {
        float c = close[warm];
        float scale = safe_scale_100_over_close(c);
        out[base + warm] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = (trv - atr) * inv_p + atr; // FMA-friendly form
        float c = close[t];
        float scale = safe_scale_100_over_close(c);
        out[base + t] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }
}

// Optimized batch variant that uses a precomputed, shared scale vector
// inv_close100[t] = 100.0f / close[t]. This saves per-row divisions.
extern "C" __global__ void natr_batch_f32_with_inv(
    const float* __restrict__ tr,
    const float* __restrict__ inv_close100, // length = series_len
    const int*   __restrict__ periods,
    int series_len,
    int first_valid,
    int n_combos,
    float*       __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0) return;

    const int warm = first_valid + period - 1;
    const int base = combo * series_len;

    for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
        out[base + idx] = dev_nan();
    }
    __syncthreads();

    if (first_valid >= series_len || warm >= series_len) return;

    const int start = first_valid;
    float local_sum = 0.0f;
    float local_c   = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const float v = tr[start + k];
        float y = v - local_c;
        float t = local_sum + y;
        local_c = (t - local_sum) - y;
        local_sum = t;
    }
    local_sum += local_c;
    const float sum_f = block_reduce_sum(local_sum);

    if (threadIdx.x != 0) return;

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = static_cast<double>(sum_f) * inv_p;

    {
        float scale = inv_close100[warm];
        out[base + warm] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = (trv - atr) * inv_p + atr;
        float scale = inv_close100[t];
        out[base + t] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
    }
}

// -------------------- Many-series × one param (time-major) --------------------

// time-major indexing: arr[t * num_series + s]
extern "C" __global__ void natr_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int period,
    int num_series,
    int series_len,
    const int*   __restrict__ first_valids,
    float*       __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;

    const int lane            = threadIdx.x & (warpSize - 1);
    const int warp_in_block   = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block == 0) return;

    int warp_idx    = blockIdx.x * warps_per_block + warp_in_block;
    const int wstep = gridDim.x * warps_per_block;

    for (int s = warp_idx; s < num_series; s += wstep) {
        const int fv = first_valids[s];

        // Initialize output with NaNs cooperatively
        for (int t = lane; t < series_len; t += warpSize) {
            out_tm[t * stride + s] = dev_nan();
        }

        if (fv < 0 || fv >= series_len) {
            continue;
        }

        const int warm_end = fv + period; // exclusive end
        if (warm_end > series_len) {
            continue;
        }

        // Warmup: sum TR over [fv, warm_end) with intrawarp Kahan partials
        float local = 0.0f, csum = 0.0f;
        #pragma unroll 1
        for (int k = lane; k < period; k += warpSize) {
            const int t = fv + k;
            const float h = high_tm[t * stride + s];
            const float l = low_tm[t * stride + s];
            float trv;
            if (t == fv) {
                trv = h - l;
            } else {
                const float pc = close_tm[(t - 1) * stride + s];
                const float hl = h - l;
                const float hc = fabsf(h - pc);
                const float lc = fabsf(l - pc);
                trv = fmaxf(hl, fmaxf(hc, lc));
            }
            float y = trv - csum;
            float tmp = local + y;
            csum = (tmp - local) - y;
            local = tmp;
        }
        local += csum;
        float sum = warp_reduce_sum(local);

        if (lane == 0) {
            const int warm = warm_end - 1;
            const double inv_p = 1.0 / static_cast<double>(period);
            double atr = static_cast<double>(sum) * inv_p;

            {
                float c = close_tm[warm * stride + s];
                float scale = safe_scale_100_over_close(c);
                out_tm[warm * stride + s] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
            }

            for (int t = warm + 1; t < series_len; ++t) {
                const float h = high_tm[t * stride + s];
                const float l = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                const float hl = h - l;
                const float hc = fabsf(h - pc);
                const float lc = fabsf(l - pc);
                const double trv = static_cast<double>(fmaxf(hl, fmaxf(hc, lc)));

                atr = (trv - atr) * inv_p + atr;

                float c = close_tm[t * stride + s];
                float scale = safe_scale_100_over_close(c);
                out_tm[t * stride + s] = (scale == scale) ? static_cast<float>(atr * static_cast<double>(scale)) : dev_nan();
            }
        }
    }
}

