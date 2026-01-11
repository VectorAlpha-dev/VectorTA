

















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>




#if !defined(EMA_USE_L2_PREFETCH) && !defined(EMA_DISABLE_L2_PREFETCH)
#  if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
#    define EMA_USE_L2_PREFETCH 1
#  endif
#endif

#if defined(EMA_USE_L2_PREFETCH)
__device__ __forceinline__ void prefetch_L2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

extern "C" __global__
void ema_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ periods,
                   const float* __restrict__ alphas,
                   int series_len,
                   int first_valid,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) return;

    const float alpha = alphas[combo];
    const float one_minus_alpha = 1.0f - alpha;
    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[base + static_cast<size_t>(idx)] = NAN;
    }

    
    if (blockDim.x < 32) {
        if (threadIdx.x != 0) return;

        
        int warm_end = first_valid + period;
        if (warm_end > series_len) warm_end = series_len;

        
        float mean = prices[first_valid];
        out[base + static_cast<size_t>(first_valid)] = mean;
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float x = prices[i];
            if (isfinite(x)) {
                ++valid_count;
                const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
                mean = __fmaf_rn(x - mean, inv, mean);
            }
            out[base + static_cast<size_t>(i)] = mean;
        }
        
        float prev = mean;
#if defined(EMA_USE_L2_PREFETCH)
        constexpr int PREFETCH_DIST = 64; 
#endif
        for (int i = warm_end; i < series_len; ++i) {
#if defined(EMA_USE_L2_PREFETCH)
            if (i + PREFETCH_DIST < series_len) prefetch_L2(&prices[i + PREFETCH_DIST]);
#endif
            const float x = prices[i];
            if (isfinite(x)) {
                
                prev = __fmaf_rn(x - prev, alpha, prev);
            }
            out[base + static_cast<size_t>(i)] = prev;
        }
        return;
    }

    
    
    if (threadIdx.x >= 32) return;

    const unsigned lane = static_cast<unsigned>(threadIdx.x); 
    const unsigned mask = 0xffffffffu;

    
    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    float prev = 0.0f;
    if (lane == 0) {
        
        float mean = prices[first_valid];
        out[base + static_cast<size_t>(first_valid)] = mean;
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float x = prices[i];
            if (isfinite(x)) {
                ++valid_count;
                const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
                mean = __fmaf_rn(x - mean, inv, mean);
            }
            out[base + static_cast<size_t>(i)] = mean;
        }
        prev = mean;
    }
    
    prev = __shfl_sync(mask, prev, 0);

#if defined(EMA_USE_L2_PREFETCH)
    constexpr int PREFETCH_DIST = 256; 
#endif
    for (int t0 = warm_end; t0 < series_len; t0 += 32) {
#if defined(EMA_USE_L2_PREFETCH)
        if (lane == 0) {
            const int pf = t0 + PREFETCH_DIST;
            if (pf < series_len) prefetch_L2(&prices[pf]);
        }
#endif
        const int t = t0 + static_cast<int>(lane);

        
        float A = 1.0f;
        float B = 0.0f;
        if (t < series_len) {
            const float x = prices[t];
            if (isfinite(x)) {
                A = one_minus_alpha;
                B = alpha * x;
            }
        }

        
        
        
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = __fmaf_rn(A_cur, B_prev, B_cur);
            }
        }

        const float y = __fmaf_rn(A, prev, B);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = y;
        }

        
        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        prev = __shfl_sync(mask, y, last_lane);
    }
}





extern "C" __global__
void ema_batch_f64_to_f32(const float* __restrict__ prices,
                          const int*   __restrict__ periods,
                          int series_len,
                          int first_valid,
                          int n_combos,
                          float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos || series_len <= 0) return;

    const int period = periods[combo];
    if (period <= 0 || first_valid >= series_len) return;

    const double alpha = 2.0 / (static_cast<double>(period) + 1.0);
    const double one_minus_alpha = 1.0 - alpha;
    const size_t base = static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    
    for (int idx = threadIdx.x; idx < first_valid; idx += blockDim.x) {
        out[base + static_cast<size_t>(idx)] = NAN;
    }

    
    if (blockDim.x < 32) {
        if (threadIdx.x != 0) return;

        int warm_end = first_valid + period;
        if (warm_end > series_len) warm_end = series_len;

        double mean = static_cast<double>(prices[first_valid]);
        out[base + static_cast<size_t>(first_valid)] = static_cast<float>(mean);

        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                ++valid_count;
                const double x = static_cast<double>(xf);
                const double vc = static_cast<double>(valid_count);
                
                mean = ((vc - 1.0) * mean + x) / vc;
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(mean);
        }

        double prev = mean;
        for (int i = warm_end; i < series_len; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                const double x = static_cast<double>(xf);
                
                prev = (one_minus_alpha * prev) + (alpha * x);
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(prev);
        }
        return;
    }

    
    if (threadIdx.x >= 32) return;
    const unsigned lane = static_cast<unsigned>(threadIdx.x);
    const unsigned mask = 0xffffffffu;

    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    double prev = 0.0;
    if (lane == 0) {
        double mean = static_cast<double>(prices[first_valid]);
        out[base + static_cast<size_t>(first_valid)] = static_cast<float>(mean);
        int valid_count = 1;
        for (int i = first_valid + 1; i < warm_end; ++i) {
            const float xf = prices[i];
            if (isfinite(xf)) {
                ++valid_count;
                const double x = static_cast<double>(xf);
                const double vc = static_cast<double>(valid_count);
                mean = ((vc - 1.0) * mean + x) / vc;
            }
            out[base + static_cast<size_t>(i)] = static_cast<float>(mean);
        }
        prev = mean;
    }
    prev = __shfl_sync(mask, prev, 0);

    for (int t0 = warm_end; t0 < series_len; t0 += 32) {
        const int t = t0 + static_cast<int>(lane);

        
        double A = 1.0;
        double B = 0.0;
        if (t < series_len) {
            const float xf = prices[t];
            if (isfinite(xf)) {
                A = one_minus_alpha;
                B = alpha * static_cast<double>(xf);
            }
        }

        
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double A_prev = __shfl_up_sync(mask, A, offset);
            const double B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= static_cast<unsigned>(offset)) {
                const double A_cur = A;
                const double B_cur = B;
                A = A_cur * A_prev;
                B = fma(A_cur, B_prev, B_cur);
            }
        }

        const double y = fma(A, prev, B);
        if (t < series_len) {
            out[base + static_cast<size_t>(t)] = static_cast<float>(y);
        }

        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        prev = __shfl_sync(mask, y, last_lane);
    }
}

extern "C" __global__
void ema_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   const int*   __restrict__ first_valids,
                                   int period,
                                   float alpha,
                                   int num_series,
                                   int series_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x; 
    if (series_idx >= num_series || period <= 0 || series_len <= 0) return;

    const int stride = num_series; 
    int first_valid  = first_valids[series_idx];
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len)    return;

    
    for (int t = threadIdx.x; t < first_valid; t += blockDim.x) {
        out_tm[t * stride + series_idx] = NAN;
    }

    if (threadIdx.x != 0) return;

    int warm_end = first_valid + period;
    if (warm_end > series_len) warm_end = series_len;

    float mean = prices_tm[first_valid * stride + series_idx];
    out_tm[first_valid * stride + series_idx] = mean;

    int valid_count = 1;
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            ++valid_count;
            const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
            mean = __fmaf_rn(x - mean, inv, mean);
        }
        out_tm[t * stride + series_idx] = mean;
    }

    float prev = mean;
    for (int t = warm_end; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out_tm[t * stride + series_idx] = prev;
    }
}




extern "C" __global__
void ema_many_series_one_param_f32_coalesced(const float* __restrict__ prices_tm,
                                             const int*   __restrict__ first_valids,
                                             int period,
                                             float alpha,
                                             int num_series,
                                             int series_len,
                                             float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series || period <= 0 || series_len <= 0) return;

    const int stride      = num_series; 
    const int first_valid = max(0, first_valids[series_idx]);
    const int warm_end    = min(series_len, first_valid + period);

    
    for (int t = 0; t < first_valid; ++t) {
        out_tm[t * stride + series_idx] = NAN;
    }

    float mean = prices_tm[first_valid * stride + series_idx];
    out_tm[first_valid * stride + series_idx] = mean;

    int valid_count = 1;
    for (int t = first_valid + 1; t < warm_end; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            ++valid_count;
            const float inv = __fdividef(1.0f, static_cast<float>(valid_count));
            mean = __fmaf_rn(x - mean, inv, mean);
        }
        out_tm[t * stride + series_idx] = mean;
    }

    float prev = mean;
    for (int t = warm_end; t < series_len; ++t) {
        const float x = prices_tm[t * stride + series_idx];
        if (isfinite(x)) {
            prev = __fmaf_rn(x - prev, alpha, prev);
        }
        out_tm[t * stride + series_idx] = prev;
    }
}
