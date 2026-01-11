// CUDA kernels for Stochastic Oscillator (Stoch)
//
// Kernels:
// - stoch_k_raw_from_hhll_f32: raw %K from precomputed HH/LL for one series.
//   Optimized: fully parallel via grid‑stride loop; single‑pass warmup fill + compute.
// - stoch_many_series_one_param_f32: raw %K for many series (time‑major), shared
//   fastk_period. Optimized memory access (time‑major), hoisted invariants, partial
//   unrolling of the inner O(period) loop.
// - stoch_one_series_many_params_f32 (optional): one series × many params path where
//   each thread owns one parameter. Time‑major outputs; same policy.
//
// Notes:
// - FP32 only. Warmup semantics: indices < first_valid + fastk - 1 → NaN.
// - NaNs in inputs propagate to NaN. Near‑zero denom → 50.0f.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCH_NAN
#define STOCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

// Small absolute epsilon for denom checks (policy‑compatible)
#ifndef STOCH_EPS
#define STOCH_EPS (1e-12f)
#endif

// Single place for %K computation when (C, H, L) are known
static __device__ __forceinline__
float stoch_k_from_chl(float c, float h, float l) {
    if (!(isfinite(c) && isfinite(h) && isfinite(l))) return STOCH_NAN;
    const float denom = h - l;
    return (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - l) * (100.0f / denom);
}

extern "C" __global__ __launch_bounds__(256, 2)
void stoch_k_raw_from_hhll_f32(const float* __restrict__ close,
                               const float* __restrict__ hh,
                               const float* __restrict__ ll,
                               int series_len,
                               int first_valid,
                               int fastk_period,
                               float* __restrict__ out) {
    if (UNLIKELY(series_len <= 0 || fastk_period <= 0)) return;
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;

    const int warm = first_valid + fastk_period - 1;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (UNLIKELY(warm >= series_len)) {
        // Entire range is warmup → fill NaNs in parallel
        for (int t = tid; t < series_len; t += stride) out[t] = STOCH_NAN;
        return;
    }

    // One parallel pass: fill warmup with NaN; compute %K for the rest
    for (int t = tid; t < series_len; t += stride) {
        if (t < warm) {
            out[t] = STOCH_NAN;
        } else {
            const float c = close[t];
            const float h = hh[t];
            const float l = ll[t];
            out[t] = stoch_k_from_chl(c, h, l);
        }
    }
}

// Time-major many-series kernel (shared fastk, naive O(period) window scan per step)
// prices are laid out time-major: idx = row * num_series + series
extern "C" __global__ __launch_bounds__(256, 2)
void stoch_many_series_one_param_f32(const float* __restrict__ high_tm,
                                     const float* __restrict__ low_tm,
                                     const float* __restrict__ close_tm,
                                     const int*   __restrict__ first_valids,
                                     int num_series,
                                     int series_len,
                                     int fastk_period,
                                     float* __restrict__ k_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; // series id
    if (s >= num_series) return;

    // Validate once per series
    if (UNLIKELY(fastk_period <= 0 || fastk_period > series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }

    const int first_valid = first_valids[s];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        float* out_col = k_tm + s;
        for (int row = 0; row < series_len; ++row, out_col += num_series) *out_col = STOCH_NAN;
        return;
    }

    const int S = num_series;
    const int warm = first_valid + fastk_period - 1;

    // Prefill NaN up to warm-1 (time-major stride S)
    {
        float* out_col = k_tm + s;
        const int limit = (warm < series_len) ? warm : series_len;
        for (int row = 0; row < limit; ++row, out_col += S) *out_col = STOCH_NAN;
        if (warm >= series_len) return;
    }

    // Fast path for period == 1
    if (fastk_period == 1) {
        const float* hptr = high_tm  + ((size_t)first_valid) * S + s;
        const float* lptr = low_tm   + ((size_t)first_valid) * S + s;
        const float* cptr = close_tm + ((size_t)first_valid) * S + s;
        float*       optr = k_tm     + ((size_t)first_valid) * S + s;
        for (int row = first_valid; row < series_len; ++row) {
            const float h = *hptr; const float l = *lptr; const float c = *cptr;
            *optr = stoch_k_from_chl(c, h, l);
            hptr += S; lptr += S; cptr += S; optr += S;
        }
        return;
    }

    // General O(period) loop per row (partially unrolled by 4)
    for (int row = warm; row < series_len; ++row) {
        const int start = row - fastk_period + 1;

        const float* hptr = high_tm + ((size_t)start) * S + s;
        const float* lptr = low_tm  + ((size_t)start) * S + s;

        float hmax = -CUDART_INF_F;
        float lmin =  CUDART_INF_F;
        bool any_nan = false;

        int k = 0;
        // Unrolled body: 4-at-a-time
        for (; k + 3 < fastk_period; k += 4) {
            const float h0 = hptr[0];  const float l0 = lptr[0];
            const float h1 = hptr[S];  const float l1 = lptr[S];
            const float h2 = hptr[S*2];const float l2 = lptr[S*2];
            const float h3 = hptr[S*3];const float l3 = lptr[S*3];

            any_nan |= !(isfinite(h0) && isfinite(l0));
            any_nan |= !(isfinite(h1) && isfinite(l1));
            any_nan |= !(isfinite(h2) && isfinite(l2));
            any_nan |= !(isfinite(h3) && isfinite(l3));

            hmax = fmaxf(hmax, fmaxf(fmaxf(h0, h1), fmaxf(h2, h3)));
            lmin = fminf(lmin, fminf(fminf(l0, l1), fminf(l2, l3)));

            hptr += S * 4; lptr += S * 4;
        }
        // Remainder
        for (; k < fastk_period; ++k) {
            const float hv = *hptr; const float lv = *lptr;
            any_nan |= !(isfinite(hv) && isfinite(lv));
            hmax = fmaxf(hmax, hv);
            lmin = fminf(lmin, lv);
            hptr += S; lptr += S;
        }

        float* outp = k_tm + ((size_t)row) * S + s;
        const float c = close_tm[((size_t)row) * S + s];

        if (any_nan || !isfinite(c) || !isfinite(hmax) || !isfinite(lmin)) {
            *outp = STOCH_NAN;
        } else {
            const float denom = hmax - lmin;
            *outp = (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - lmin) * (100.0f / denom);
        }
    }
}

// Optional: one series × many params path (time‑major output: [series_len, num_params])
extern "C" __global__ __launch_bounds__(256, 2)
void stoch_one_series_many_params_f32(const float* __restrict__ high,
                                      const float* __restrict__ low,
                                      const float* __restrict__ close,
                                      const int*   __restrict__ fastk_periods,   // [num_params]
                                      const int*   __restrict__ first_valids,    // [num_params]
                                      int series_len,
                                      int num_params,
                                      float* __restrict__ k_tm) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x; // parameter id
    if (p >= num_params) return;

    const int fastk = fastk_periods[p];
    const int first_valid = first_valids[p];

    if (UNLIKELY(series_len <= 0 || fastk <= 0 || fastk > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int t = 0; t < series_len; ++t) k_tm[((size_t)t) * num_params + p] = STOCH_NAN;
        return;
    }

    const int warm = first_valid + fastk - 1;

    // Warmup prefill
    for (int t = 0; t < warm; ++t) k_tm[((size_t)t) * num_params + p] = STOCH_NAN;

    // Fast path: window size == 1
    if (fastk == 1) {
        for (int t = first_valid; t < series_len; ++t) {
            const float h = high[t];
            const float l = low[t];
            const float c = close[t];
            k_tm[((size_t)t) * num_params + p] = stoch_k_from_chl(c, h, l);
        }
        return;
    }

    // General path, partially unrolled by 4
    for (int t = warm; t < series_len; ++t) {
        const int start = t - fastk + 1;

        float hmax = -CUDART_INF_F;
        float lmin =  CUDART_INF_F;
        bool any_nan = false;

        int k = 0;
        // Unrolled body
        for (; k + 3 < fastk; k += 4) {
            const float h0 = high[start + k + 0]; const float l0 = low[start + k + 0];
            const float h1 = high[start + k + 1]; const float l1 = low[start + k + 1];
            const float h2 = high[start + k + 2]; const float l2 = low[start + k + 2];
            const float h3 = high[start + k + 3]; const float l3 = low[start + k + 3];

            any_nan |= !(isfinite(h0) && isfinite(l0));
            any_nan |= !(isfinite(h1) && isfinite(l1));
            any_nan |= !(isfinite(h2) && isfinite(l2));
            any_nan |= !(isfinite(h3) && isfinite(l3));

            hmax = fmaxf(hmax, fmaxf(fmaxf(h0, h1), fmaxf(h2, h3)));
            lmin = fminf(lmin, fminf(fminf(l0, l1), fminf(l2, l3)));
        }
        for (; k < fastk; ++k) {
            const float hv = high[start + k];
            const float lv = low[start + k];
            any_nan |= !(isfinite(hv) && isfinite(lv));
            hmax = fmaxf(hmax, hv);
            lmin = fminf(lmin, lv);
        }

        const float c = close[t];
        float* outp = &k_tm[((size_t)t) * num_params + p];
        if (any_nan || !isfinite(c) || !isfinite(hmax) || !isfinite(lmin)) {
            *outp = STOCH_NAN;
        } else {
            const float denom = hmax - lmin;
            *outp = (fabsf(denom) < STOCH_EPS) ? 50.0f : (c - lmin) * (100.0f / denom);
        }
    }
}

// Tiny helper: broadcast a single row vector into multiple rows of a row‑major matrix
extern "C" __global__ __launch_bounds__(256, 2)
void pack_row_broadcast_rowmajor_f32(const float* __restrict__ src,
                                     int len,
                                     const int* __restrict__ rows_idx, // length nrows
                                     int nrows,
                                     float* __restrict__ dst,          // [num_rows_total, len] row-major
                                     int row_stride)                   // == len
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = t; i < len; i += stride) {
        const float v = src[i];
        #pragma unroll 4
        for (int j = 0; j < nrows; ++j) {
            const int row = rows_idx[j];
            dst[(size_t)row * (size_t)row_stride + (size_t)i] = v;
        }
    }
}

// Utility: 32x32 tiled transpose (time-major -> row-major)
// Input in_tm: [rows=time][cols=series] (row-major)
// Output out_rm: [rows=series][cols=time] (row-major)
extern "C" __global__
void transpose_tm_to_rm_f32(const float* __restrict__ in_tm,
                            int rows,
                            int cols,
                            float* __restrict__ out_rm) {
    // Match the common "tiled transpose" pattern (avoid shared-memory bank conflicts).
    // Use 32x8 threads and unroll over 32 rows.
    __shared__ float tile[32][33];

    const int x0 = blockIdx.x * 32 + threadIdx.x; // col in input
    const int y0 = blockIdx.y * 32 + threadIdx.y; // row in input

    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int y = y0 + j;
        if (x0 < cols && y < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in_tm[(size_t)y * (size_t)cols + (size_t)x0];
        }
    }

    __syncthreads();

    const int x1 = blockIdx.y * 32 + threadIdx.x; // col in output (time)
    const int y1 = blockIdx.x * 32 + threadIdx.y; // row in output (series)
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        const int y = y1 + j;
        if (x1 < rows && y < cols) {
            out_rm[(size_t)y * (size_t)rows + (size_t)x1] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
