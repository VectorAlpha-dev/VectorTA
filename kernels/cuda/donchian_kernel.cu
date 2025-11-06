// CUDA kernels for Donchian Channels (upper/middle/lower) in FP32.
//
// Semantics:
// - Warmup: indices < (first_valid + period - 1) are NaN
// - Any NaN in the window gates the entire output at that index to NaN
// - Upper = max(high), Lower = min(low), Middle = (Upper + Lower)/2
//
// Batch variant: one series × many params (periods)
// Many-series variant: time-major layout × one param

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef DCH_NAN
#define DCH_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// read-only load hint; maps to texture/RO cache where beneficial on many archs
#if __CUDA_ARCH__ >= 350
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

__device__ __forceinline__ int floor_log2_u32(unsigned int x) {
    return 31 - __clz(x);
}

// ------------------------ RMQ builders (sparse tables) ------------------------

// level-0 init (float copy)
extern "C" __global__ void rmq_init_level0_f32(const float* __restrict__ in,
                                               float* __restrict__ st_lvl0,
                                               int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) st_lvl0[i] = in[i];
}

// nan mask: OR(high_isnan || low_isnan), but ignore indices < first_valid
extern "C" __global__ void rmq_init_nan_mask_u8(const float* __restrict__ high,
                                                const float* __restrict__ low,
                                                int N,
                                                int first_valid,
                                                unsigned char* __restrict__ mask_lvl0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (i >= first_valid) {
            const float h = high[i];
            const float l = low[i];
            mask_lvl0[i] = (isnan(h) || isnan(l)) ? 1u : 0u;
        } else {
            mask_lvl0[i] = 0u;
        }
    }
}

// Build next level for MAX (float)
extern "C" __global__ void rmq_build_level_max_f32(const float* __restrict__ prev,
                                                   float* __restrict__ curr,
                                                   int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        float a = prev[i];
        float b = (j < N) ? prev[j] : -CUDART_INF_F;
        curr[i] = (i < limit) ? fmaxf(a, b) : -CUDART_INF_F;
    }
}

// Build next level for MIN (float)
extern "C" __global__ void rmq_build_level_min_f32(const float* __restrict__ prev,
                                                   float* __restrict__ curr,
                                                   int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        float a = prev[i];
        float b = (j < N) ? prev[j] : CUDART_INF_F;
        curr[i] = (i < limit) ? fminf(a, b) : CUDART_INF_F;
    }
}

// Build next level for OR over uint8 mask
extern "C" __global__ void rmq_build_level_or_u8(const unsigned char* __restrict__ prev,
                                                 unsigned char* __restrict__ curr,
                                                 int N, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int j = i + offset;
        const int limit = N - (offset << 1) + 1;
        const unsigned char a = prev[i];
        const unsigned char b = (j < N) ? prev[j] : 0u;
        curr[i] = (i < limit) ? (unsigned char)(a | b) : (unsigned char)0u;
    }
}

extern "C" __global__ void donchian_batch_f32(const float* __restrict__ high,
                                               const float* __restrict__ low,
                                               const int*   __restrict__ periods,
                                               int series_len,
                                               int n_combos,
                                               int first_valid,
                                               float* __restrict__ out_upper,
                                               float* __restrict__ out_middle,
                                               float* __restrict__ out_lower) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* uo = out_upper + base;
    float* mo = out_middle + base;
    float* lo = out_lower + base;

    // Guard invalid inputs by writing NaNs
    if (UNLIKELY(period <= 0 || period > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) {
            uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN;
        }
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < series_len; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }

    if (period == 1) {
        for (int i = first_valid; i < series_len; ++i) {
            const float h = high[i];
            const float l = low[i];
            if (isnan(h) || isnan(l)) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
            else { uo[i] = h; lo[i] = l; mo[i] = 0.5f * (h + l); }
        }
        return;
    }

    // Naive window scan per output index (kept simple for correctness parity)
    for (int i = warm; i < series_len; ++i) {
        const int start = i + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;
        for (int k = 0; k < period; ++k) {
            const float h = high[start + k];
            const float l = low[start + k];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (any_nan) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        else { uo[i] = maxv; lo[i] = minv; mo[i] = 0.5f * (maxv + minv); }
    }
}

// One series × many params using sparse-tables (RMQ)
// st_high, st_low: [K * N] with stride N per level; st_nan: [K * N] as uint8
extern "C" __global__ void donchian_batch_from_rmq_f32(
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    const float* __restrict__ st_high,
    const float* __restrict__ st_low,
    const unsigned char* __restrict__ st_nan,
    float* __restrict__ out_upper,
    float* __restrict__ out_middle,
    float* __restrict__ out_lower) {

    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int N = series_len;
    const int period = periods[combo];
    const int base   = combo * N;

    float* __restrict__ uo = out_upper  + base;
    float* __restrict__ mo = out_middle + base;
    float* __restrict__ lo = out_lower  + base;

    if (UNLIKELY(period <= 0 || period > N || first_valid < 0 || first_valid >= N)) {
        for (int i = 0; i < N; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }
    const int tail_len = N - first_valid;
    if (UNLIKELY(tail_len < period)) {
        for (int i = 0; i < N; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }
        return;
    }

    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm; ++i) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; }

    const int k    = floor_log2_u32((unsigned)period);
    const int len2 = 1 << k;
    const size_t off = (size_t)k * (size_t)N;

    const float* __restrict__ hi_lvl = st_high + off;
    const float* __restrict__ lo_lvl = st_low  + off;
    const unsigned char* __restrict__ nm_lvl = st_nan + off;

    for (int i = warm; i < N; ++i) {
        const int L = i + 1 - period;
        const int R = i;
        const int R2 = R - len2 + 1;

        const unsigned char nm = (unsigned char)(LDG(nm_lvl + L) | LDG(nm_lvl + R2));
        if (UNLIKELY(nm)) { uo[i] = DCH_NAN; mo[i] = DCH_NAN; lo[i] = DCH_NAN; continue; }

        const float uh = fmaxf(LDG(hi_lvl + L),  LDG(hi_lvl + R2));
        const float ll = fminf(LDG(lo_lvl + L),  LDG(lo_lvl + R2));
        uo[i] = uh; lo[i] = ll; mo[i] = 0.5f * (uh + ll);
    }
}

// Many series × one param (time-major I/O)
extern "C" __global__ void donchian_many_series_one_param_f32(
    const float* __restrict__ high_tm,   // time-major: [row * num_series + series]
    const float* __restrict__ low_tm,
    const int*   __restrict__ first_valids, // per series (column)
    int num_series,
    int series_len,
    int period,
    float* __restrict__ upper_tm,
    float* __restrict__ middle_tm,
    float* __restrict__ lower_tm) {

    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len || period <= 0 || period > series_len || (series_len - first_valid) < period) {
        // All NaN outputs
        int idx = series;
        for (int row = 0; row < series_len; ++row, idx += num_series) {
            upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
        }
        return;
    }

    const int warm = first_valid + period - 1;
    int idx = series;
    for (int row = 0; row < warm; ++row, idx += num_series) {
        upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN;
    }

    if (period == 1) {
        for (int row = first_valid; row < series_len; ++row, idx += num_series) {
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            if (UNLIKELY(isnan(h) || isnan(l))) { upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN; }
            else { upper_tm[idx] = h; lower_tm[idx] = l; middle_tm[idx] = 0.5f * (h + l); }
        }
        return;
    }

    for (int row = warm; row < series_len; ++row, idx += num_series) {
        const int start = row + 1 - period;
        float maxv = -CUDART_INF_F;
        float minv =  CUDART_INF_F;
        bool any_nan = false;
        // strided access in time-major layout
        int idxk = (start * num_series) + series;
        for (int k = 0; k < period; ++k, idxk += num_series) {
            const float h = high_tm[idxk];
            const float l = low_tm[idxk];
            if (UNLIKELY(isnan(h) || isnan(l))) { any_nan = true; break; }
            if (h > maxv) maxv = h;
            if (l < minv) minv = l;
        }
        if (any_nan) { upper_tm[idx] = DCH_NAN; middle_tm[idx] = DCH_NAN; lower_tm[idx] = DCH_NAN; }
        else { upper_tm[idx] = maxv; lower_tm[idx] = minv; middle_tm[idx] = 0.5f * (maxv + minv); }
    }
}
