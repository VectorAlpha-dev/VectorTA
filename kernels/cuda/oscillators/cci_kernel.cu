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
#define USE_CCI_SMEM_OPT 0
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

    // Fill row with NaN up-front (parallel across threads).
    for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
        out[base + i] = NAN;
    }
    __syncthreads();

    if (UNLIKELY(period <= 0 || period > series_len)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < period)) return;

    const float inv_p     = 1.0f / static_cast<float>(period);
    const float mad_scale = 0.015f * inv_p; // denom = sum_abs * mad_scale
    const int   warm      = first_valid + period - 1;

    // Fast path: shared-memory sliding window when period fits our static buffer
    if (USE_CCI_SMEM_OPT && LIKELY(period <= CCI_SMEM_MAX)) {
        __shared__ float s_win_static[CCI_SMEM_MAX];
        float* s_win = s_win_static; // alias for clarity

        // Seed window [first_valid .. first_valid+period-1]
        {
            const float* p0 = prices + first_valid;
            for (int i = threadIdx.x; i < period; i += blockDim.x) {
                s_win[i] = p0[i];
            }
        }
        __syncthreads();

        // Initial rolling sum (sequential for numeric parity with legacy path)
        float sum0 = 0.0f;
        if (threadIdx.x == 0) {
            for (int i = 0; i < period; ++i) sum0 += s_win[i];
        }
        __shared__ float s_sma;
        if (threadIdx.x == 0) s_sma = sum0 * inv_p;
        __syncthreads();

        // First MAD / CCI at warm (sequential; compensated abs-sum for stability)
        if (threadIdx.x == 0) {
            const float sma = s_sma;
            float sum_abs = 0.0f, cabs = 0.0f;
            for (int i = 0; i < period; ++i) {
                float ai = fabsf(s_win[i] - sma);
                kahan_add(ai, sum_abs, cabs);
            }
            float denom = 0.015f * (sum_abs * inv_p);
            float px    = prices[warm];
            out[base + warm] = (denom == 0.0f) ? 0.0f : (px - sma) / denom;
        }
        __syncthreads();

        // Rolling with circular buffer in shared memory
        int   head = 0;
        float sum  = sum0;
        float csum = 0.0f; // Kahan compensation for rolling sum
        for (int t = warm + 1; t < series_len; ++t) {
            if (threadIdx.x == 0) {
                const float newv = prices[t];
                const float oldv = s_win[head];
                s_win[head] = newv;
                head++; if (head == period) head = 0;
                // compensated update for better numerical stability
                kahan_add(newv - oldv, sum, csum);
                s_sma = sum * inv_p;

                const float sma = s_sma;
                float sum_abs = 0.0f, cabs = 0.0f;
                for (int i = 0; i < period; ++i) {
                    float ai = fabsf(s_win[i] - sma);
                    kahan_add(ai, sum_abs, cabs);
                }
                float denom = 0.015f * (sum_abs * inv_p);
                float px    = prices[t];
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

    // Initial rolling sum for SMA (FP64 accumulate)
    double sum_d = 0.0;
    const float inv_p     = 1.0f / static_cast<float>(period);
    const double inv_p_d  = 1.0 / (double)period;
    const float* p = col_in + static_cast<size_t>(first_valid) * num_series;
    for (int k = 0; k < period; ++k, p += num_series) sum_d += (double)(*p);
    double sma_d = sum_d * inv_p_d;
    float sma = (float)sma_d;

    // First MAD / CCI at warm
    {
        double sum_abs_d = 0.0;
        const float* w = col_in + static_cast<size_t>(warm - period + 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            double d = (double)(*w) - sma_d;
            sum_abs_d += fabs(d);
        }
        double denom_d = 0.015 * (sum_abs_d * inv_p_d);
        double px_d = (double)(*(col_in + static_cast<size_t>(warm) * num_series));
        *(col_out + static_cast<size_t>(warm) * num_series) = (float)((denom_d == 0.0) ? 0.0 : (px_d - sma_d) / denom_d);
    }

    // Rolling
    const float* cur = col_in + static_cast<size_t>(warm + 1) * num_series;
    const float* old = col_in + static_cast<size_t>(first_valid) * num_series;
    float* dst       = col_out + static_cast<size_t>(warm + 1) * num_series;
    for (int r = warm + 1; r < series_len; ++r) {
        sum_d += (double)(*cur);
        sum_d -= (double)(*old);
        sma_d = sum_d * inv_p_d;
        sma = (float)sma_d;

        double sum_abs_d2 = 0.0;
        const float* w = cur - static_cast<size_t>(period - 1) * num_series;
        for (int k = 0; k < period; ++k, w += num_series) {
            double d = (double)(*w) - sma_d;
            sum_abs_d2 += fabs(d);
        }
        double denom_d = 0.015 * (sum_abs_d2 * inv_p_d);
        *dst = (float)((denom_d == 0.0) ? 0.0 : (((double)(*cur)) - sma_d) / denom_d);
        cur += num_series;
        old += num_series;
        dst += num_series;
    }
}

