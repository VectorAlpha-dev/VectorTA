// CUDA kernels for Commodity Channel Index (CCI)
//
// Semantics mirror the scalar Rust implementation in src/indicators/cci.rs:
// - Warmup prefix of NaNs at indices [0 .. first_valid + period - 1)
// - NaN inputs propagate (any NaN in the active window yields NaN via FP ops)
// - Denominator zero -> 0.0 (when mean absolute deviation is exactly 0)
// - FP32 arithmetic for throughput and interoperability with existing wrappers
//
// Optimization notes (FP32-only):
// - One-series × many-params path uses a per-block sliding window cached in
//   shared memory and block-wide parallel reductions for the seed/MAD.
// - Rolling sum is updated with Kahan compensation to reduce drift when series
//   is long, avoiding FP64. For very large periods that exceed a conservative
//   shared-memory static buffer, we safely fall back to the prior global-memory
//   scan per step (still correct; performance-lean).

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// ------------------------------
// Warp- and block-wide reductions
// ------------------------------
__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_partials[32]; // supports up to 32 warps per block
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_partials[wid] = v;
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        out = (lane < nwarps) ? warp_partials[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

// Kahan compensated streaming add: sum += x with carry c
__inline__ __device__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// Conservative static shared window bound for CCI.
// If period <= CCI_SMEM_MAX, we use the optimized sliding-window path in shared
// memory. Otherwise we safely fall back to a plain global-memory scan.
#ifndef CCI_SMEM_MAX
#define CCI_SMEM_MAX 2048
#endif
// Compile-time toggle for the shared-memory sliding-window optimization path.
// Set to 1 to enable the optimized path (requires tight numeric parity checks
// in your environment/tests). Default 0 to keep legacy parity by default.
#ifndef USE_CCI_SMEM_OPT
#define USE_CCI_SMEM_OPT 1
#endif

// ----------------------------------------------------------
// One-series × many-params (batch). Each block processes one param.
// Optimized with a shared-memory sliding window when period <= CCI_SMEM_MAX.
// ----------------------------------------------------------
extern "C" __global__ void cci_batch_f32(const float* __restrict__ prices,
                                          const int*   __restrict__ periods,
                                          int series_len,
                                          int n_combos,
                                          int first_valid,
                                          float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;

    if (UNLIKELY(period <= 0 || period > series_len) ||
        UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) out[base + i] = NAN;
        return;
    }
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) out[base + i] = NAN;
        return;
    }

    const float inv_p = 1.0f / static_cast<float>(period);
    const int warm = first_valid + period - 1;

    // Warmup prefix [0 .. warm) = NaN.
    for (int i = threadIdx.x; i < warm; i += blockDim.x) out[base + i] = NAN;
    __syncthreads();

    // Fast path: keep rolling window in shared memory; FP32 arithmetic for throughput.
    if (USE_CCI_SMEM_OPT && LIKELY(period <= CCI_SMEM_MAX)) {
        __shared__ float s_win_static[CCI_SMEM_MAX];
        __shared__ float s_sma;
        float* s_win = s_win_static;

        // Seed window [first_valid .. first_valid+period-1]
        {
            const float* p0 = prices + first_valid;
            for (int i = threadIdx.x; i < period; i += blockDim.x) {
                s_win[i] = p0[i];
            }
        }
        __syncthreads();

        // Initial rolling sum (parallel) -> SMA
        float sum_local = 0.0f;
        for (int i = threadIdx.x; i < period; i += blockDim.x) sum_local += s_win[i];
        float sum_total = block_reduce_sum(sum_local);

        float sum = sum_total;
        float csum = 0.0f; // Kahan compensation for rolling sum updates
        if (threadIdx.x == 0) s_sma = sum_total * inv_p;
        __syncthreads();

        // First MAD / CCI at warm (parallel MAD reduction)
        {
            const float sma = s_sma;
            float sum_abs_local = 0.0f;
            for (int i = threadIdx.x; i < period; i += blockDim.x) {
                sum_abs_local += fabsf(s_win[i] - sma);
            }
            float sum_abs = block_reduce_sum(sum_abs_local);
            if (threadIdx.x == 0) {
                float denom = 0.015f * (sum_abs * inv_p);
                float px = prices[warm];
                out[base + warm] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
            }
        }
        __syncthreads();

        // Rolling with circular buffer in shared memory
        int head = 0;
        for (int t = warm + 1; t < series_len; ++t) {
            if (threadIdx.x == 0) {
                const float newv = prices[t];
                const float oldv = s_win[head];
                s_win[head] = newv;
                head++; if (head == period) head = 0;
                kahan_add(newv - oldv, sum, csum);
                s_sma = sum * inv_p;
            }
            __syncthreads();

            const float sma = s_sma;
            float sum_abs_local = 0.0f;
            for (int i = threadIdx.x; i < period; i += blockDim.x) {
                sum_abs_local += fabsf(s_win[i] - sma);
            }
            float sum_abs = block_reduce_sum(sum_abs_local);
            if (threadIdx.x == 0) {
                float denom = 0.015f * (sum_abs * inv_p);
                float px = prices[t];
                out[base + t] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
            }
            __syncthreads();
        }
        return;
    }

    // Fallback path for very large periods: plain global-memory scan per step
    if (threadIdx.x != 0) return; // keep simple/robust when not using shared window

    // Fallback path uses pure FP32 arithmetic to match expected tolerances
    float sum = 0.0f;
    const float* p0 = prices + first_valid;
    for (int k = 0; k < period; ++k) sum += p0[k];
    float sma = sum * inv_p;

    {
        float sum_abs = 0.0f, cabs = 0.0f;
        const float* wptr = prices + (warm - period + 1);
        for (int k = 0; k < period; ++k) {
            float ai = fabsf(wptr[k] - sma);
            kahan_add(ai, sum_abs, cabs);
        }
        float denom = fmaf(sum_abs, (0.015f / static_cast<float>(period)), 0.0f);
        float px = prices[warm];
        out[base + warm] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }
    for (int t = warm + 1; t < series_len; ++t) {
        sum += prices[t];
        sum -= prices[t - period];
        sma = sum * inv_p;

        float sum_abs = 0.0f, cabs = 0.0f;
        const float* wptr = prices + (t - period + 1);
        for (int k = 0; k < period; ++k) {
            float ai = fabsf(wptr[k] - sma);
            kahan_add(ai, sum_abs, cabs);
        }
        float denom = fmaf(sum_abs, (0.015f / static_cast<float>(period)), 0.0f);
        float px = prices[t];
        out[base + t] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }
}

// ----------------------------------------------------------
// Many-series × one-param (time-major). Each thread handles one series.
// Minor cleanup: precompute inv_p and mad_scale.
// ----------------------------------------------------------
extern "C" __global__ void cci_many_series_one_param_f32(
    const float* __restrict__ prices_tm,   // [row * num_series + series]
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)            // time-major output
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* col_in  = prices_tm + series;
    float*       col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) {
        for (int r = 0; r < series_len; ++r) col_out[r * num_series] = NAN;
        return;
    }

    // Warmup prefix
    const int warm = first_valid + period - 1;
    for (int r = 0; r < warm; ++r) col_out[r * num_series] = NAN;

    const float inv_p     = 1.0f / static_cast<float>(period);
    const float* p = col_in + static_cast<size_t>(first_valid) * num_series;
    float sum = 0.0f, csum = 0.0f;
    for (int k = 0; k < period; ++k, p += num_series) kahan_add(*p, sum, csum);
    float sma = sum * inv_p;

    // First MAD / CCI at warm
    {
        float sum_abs = 0.0f, cabs = 0.0f;
        const float* w = col_in + static_cast<size_t>(warm - period + 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            kahan_add(fabsf((*w) - sma), sum_abs, cabs);
        }
        float denom = 0.015f * (sum_abs * inv_p);
        float px = *(col_in + static_cast<size_t>(warm) * num_series);
        *(col_out + static_cast<size_t>(warm) * num_series) = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
    }

    // Rolling
    const float* cur = col_in + static_cast<size_t>(warm + 1) * num_series;
    const float* old = col_in + static_cast<size_t>(first_valid) * num_series;
    float* dst       = col_out + static_cast<size_t>(warm + 1) * num_series;
    for (int r = warm + 1; r < series_len; ++r) {
        kahan_add((*cur) - (*old), sum, csum);
        sma = sum * inv_p;

        float sum_abs = 0.0f, cabs = 0.0f;
        const float* w = cur - static_cast<size_t>(period - 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            kahan_add(fabsf((*w) - sma), sum_abs, cabs);
        }
        float denom = 0.015f * (sum_abs * inv_p);
        *dst = (denom == 0.0f) ? 0.0f : ((*cur) - sma) / denom;
        cur += num_series;
        old += num_series;
        dst += num_series;
    }
}

