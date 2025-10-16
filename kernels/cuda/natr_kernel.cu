// CUDA kernels for NATR (Normalized Average True Range)
//
// Math: NATR = (ATR / close) * 100, where ATR uses Wilder smoothing
// with warmup initialized as the arithmetic mean of True Range over the first
// full window [first_valid, first_valid + period - 1].
//
// Batch kernel (one series × many params):
// - Each block handles one parameter row (period)
// - Threads cooperatively compute the warmup TR sum using a block-wide reduction
// - A single thread runs the sequential Wilder recurrence for that row
//
// Many-series kernel (time‑major):
// - Warp-per-series mapping, identical structure to wilders_many_series
// - TR is computed on-the-fly from H/L/close

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
    if (wid == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum; // valid in lane 0 of warp 0
}

// -------------------- Helpers --------------------
__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }

// -------------------- Batch: one series × many params --------------------

// Inputs:
//  - tr:    True Range per time (length=len). Precomputed on host.
//  - close: Close price per time (length=len).
//  - periods: array of periods (length=n_combos)
//  - series_len, first_valid, n_combos
// Output:
//  - out: row-major [n_combos, len]
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

    // 2) Accumulate warmup TR sum across threads
    const int start = first_valid;
    float local = 0.0f;
    for (int k = threadIdx.x; k < period; k += blockDim.x) {
        const int i = start + k;
        local += tr[i];
    }
    const float sum_f = block_reduce_sum(local);

    if (threadIdx.x != 0) return; // single-thread sequential recurrence

    const double inv_p = 1.0 / static_cast<double>(period);
    double atr = static_cast<double>(sum_f) * inv_p;

    // 3) Emit warm value and run sequential Wilder recurrence
    float c = close[warm];
    out[base + warm] = (isfinite(c) && c != 0.0f) ? static_cast<float>((atr / static_cast<double>(c)) * 100.0) : dev_nan();

    for (int t = warm + 1; t < series_len; ++t) {
        const double trv = static_cast<double>(tr[t]);
        atr = (trv - atr) * inv_p + atr; // FMA-friendly form
        c = close[t];
        out[base + t] = (isfinite(c) && c != 0.0f) ? static_cast<float>((atr / static_cast<double>(c)) * 100.0) : dev_nan();
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

        // Warmup: sum TR over [fv, warm_end)
        float local = 0.0f;
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
            local += trv;
        }
        float sum = warp_reduce_sum(local);

        if (lane == 0) {
            const int warm = warm_end - 1;
            const double inv_p = 1.0 / static_cast<double>(period);
            double atr = static_cast<double>(sum) * inv_p;

            float c = close_tm[warm * stride + s];
            out_tm[warm * stride + s] = (isfinite(c) && c != 0.0f)
                ? static_cast<float>((atr / static_cast<double>(c)) * 100.0)
                : dev_nan();

            for (int t = warm + 1; t < series_len; ++t) {
                const float h = high_tm[t * stride + s];
                const float l = low_tm[t * stride + s];
                const float pc = close_tm[(t - 1) * stride + s];
                const float hl = h - l;
                const float hc = fabsf(h - pc);
                const float lc = fabsf(l - pc);
                const double trv = static_cast<double>(fmaxf(hl, fmaxf(hc, lc)));
                atr = (trv - atr) * inv_p + atr;
                c = close_tm[t * stride + s];
                out_tm[t * stride + s] = (isfinite(c) && c != 0.0f)
                    ? static_cast<float>((atr / static_cast<double>(c)) * 100.0)
                    : dev_nan();
            }
        }
    }
}

