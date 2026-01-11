














#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef STOCHF_QNAN
#define STOCHF_QNAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float stoch_from_tables(
    int t,
    int fast_k,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum
) {
    const int start = t - fast_k + 1;

    
    if (nan_psum[t + 1] - nan_psum[start]) return STOCHF_QNAN;

    const int k           = log2_tbl[fast_k];
    const int offset      = 1 << k;
    const int level_base  = level_offsets[k];
    const int idx_a       = level_base + start;
    const int idx_b       = level_base + (t + 1 - offset);

    const float h = fmaxf(st_max[idx_a], st_max[idx_b]);
    const float l = fminf(st_min[idx_a], st_min[idx_b]);
    const float c = close[t];

    
    if (!(h == h) || !(l == l) || !(c == c)) return STOCHF_QNAN;

    const float den = h - l;
    if (den == 0.0f) {
        
        return (c == h) ? 100.0f : 0.0f;
    }
    return 100.0f * ((c - l) / den);
}

extern "C" __global__ void stochf_batch_f32(
    
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    const int*   __restrict__ nan_psum,
    const int*   __restrict__ fastk_arr,
    const int*   __restrict__ fastd_arr,
    const int*   __restrict__ matype_arr, 
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    
    float* __restrict__ out_k,
    float* __restrict__ out_d
) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int fk = fastk_arr[combo];
    const int fd = fastd_arr[combo];
    const int mt = matype_arr[combo];

    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) return;
    if (UNLIKELY(level_count <= 0 || fk <= 0 || fd <= 0))       return;

    const int base   = combo * series_len;
    const int k_warm = first_valid + fk - 1;
    const int d_warm = k_warm + fd - 1;

    if (UNLIKELY(k_warm >= series_len)) {
        
        for (int t = threadIdx.x; t < min(series_len, d_warm); t += blockDim.x)
            out_d[base + t] = STOCHF_QNAN;
        return;
    }

    
    for (int t = threadIdx.x; t < k_warm; t += blockDim.x) out_k[base + t] = STOCHF_QNAN;
    for (int t = threadIdx.x; t < min(series_len, d_warm); t += blockDim.x) out_d[base + t] = STOCHF_QNAN;

    __syncthreads();

    
    for (int t = k_warm + threadIdx.x; t < series_len; t += blockDim.x) {
        out_k[base + t] = stoch_from_tables(t, fk, close, log2_tbl, level_offsets, st_max, st_min, nan_psum);
    }

    __syncthreads(); 

    
    
    
    if (mt == 0) {
        if (fd == 1) {
            
            for (int t = k_warm + threadIdx.x; t < series_len; t += blockDim.x)
                out_d[base + t] = out_k[base + t];
        } else {
            for (int t = d_warm + threadIdx.x; t < series_len; t += blockDim.x) {
                float sum = 0.0f;
                bool ok = true;
                const int start = t - fd + 1;
                for (int j = start; j <= t; ++j) {
                    const float kv = out_k[base + j];
                    if (UNLIKELY(!(kv == kv))) { ok = false; break; }
                    sum += kv;
                }
                out_d[base + t] = ok ? (sum / (float)fd) : STOCHF_QNAN;
            }
        }
    } else {
        
    }
}



extern "C" __global__ void stochf_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids, 
    int num_series,
    int series_len,
    int fast_k,
    int fast_d,
    int matype, 
    float* __restrict__ k_out_tm,
    float* __restrict__ d_out_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const int fv = first_valids[series];

    
    for (int t = 0; t < series_len; ++t) {
        *(k_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
        *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
    }
    if (UNLIKELY(fv < 0 || fv >= series_len || fast_k <= 0 || fast_d <= 0)) return;

    const int k_warm = fv + fast_k - 1;
    const int d_warm = k_warm + fast_d - 1;
    if (UNLIKELY(k_warm >= series_len)) return;

    auto load_tm = [num_series, series](const float* base, int t)->float {
        return *(base + (size_t)t * num_series + series);
    };

    auto stoch_naive = [&](int t)->float {
        const int start = t - fast_k + 1;

        
        float h = load_tm(high_tm, start);
        float l = load_tm(low_tm,  start);
        if (!(h == h) || !(l == l)) return STOCHF_QNAN;

        for (int i = start + 1; i <= t; ++i) {
            const float hi = load_tm(high_tm, i);
            const float lo = load_tm(low_tm,  i);
            if (!(hi == hi) || !(lo == lo)) return STOCHF_QNAN;
            h = fmaxf(h, hi);
            l = fminf(l, lo);
        }
        const float c = load_tm(close_tm, t);
        if (!(c == c)) return STOCHF_QNAN;

        const float den = h - l;
        if (den == 0.0f) return (c == h) ? 100.0f : 0.0f;
        return 100.0f * ((c - l) / den);
    };

    
    for (int t = k_warm; t < series_len; ++t) {
        float kv = stoch_naive(t);
        *(k_out_tm + (size_t)t * num_series + series) = kv;
    }

    
    if (matype == 0) {
        float sum = 0.0f, comp = 0.0f; int consec = 0;
        auto kahan_add = [](float &s, float x, float &c){ float y=x-c; float t=s+y; c=(t-s)-y; s=t; };

        for (int t = k_warm; t < series_len; ++t) {
            const float kv = *(k_out_tm + (size_t)t * num_series + series);
            if (kv == kv) {
                kahan_add(sum, kv, comp); ++consec;
                if (consec < fast_d) {
                    *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
                } else if (consec == fast_d) {
                    *(d_out_tm + (size_t)t * num_series + series) = sum / (float)fast_d;
                } else {
                    const float oldk = *(k_out_tm + (size_t)(t - fast_d) * num_series + series);
                    kahan_add(sum, -oldk, comp);
                    *(d_out_tm + (size_t)t * num_series + series) = sum / (float)fast_d;
                }
            } else {
                *(d_out_tm + (size_t)t * num_series + series) = STOCHF_QNAN;
                sum = 0.0f; comp = 0.0f; consec = 0;
            }
        }
    }
}
