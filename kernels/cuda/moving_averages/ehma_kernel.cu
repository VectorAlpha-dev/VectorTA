












#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#ifndef EHMA_USE_ASYNC
#define EHMA_USE_ASYNC 1
#endif

__device__ __forceinline__ float ehma_hann_weight(int period, int idx) {
    
    
    const float i = static_cast<float>(period - idx);      
    const float x = i / (static_cast<float>(period) + 1.0f);
    const float s = sinpif(x);                              
    return 2.0f * s * s;                                    
}


extern "C" __global__
void ehma_batch_f32(const float* __restrict__ prices,
                    const int* __restrict__ periods,
                    const int* __restrict__ warm_indices,
                    int series_len,
                    int n_combos,
                    int max_period,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) {
        return;
    }

    const int period = periods[combo];
    if (period <= 0 || period > max_period) {
        return;
    }

    extern __shared__ float weights[];

    
    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        weights[idx] = ehma_hann_weight(period, idx);
    }
    __syncthreads();

    const float inv_norm = 1.0f / (static_cast<float>(period) + 1.0f); 

    const int warm = warm_indices[combo];
    const int first = warm - period + 1;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm || (t - period + 1) < first) {
            out[base_out + t] = NAN;
        } else {
            const int start = t - period + 1;
            
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            #pragma unroll 4
            for (; k + 3 < period; k += 4) {
                s0 = __fmaf_rn(prices[start + k],     weights[k],     s0);
                s1 = __fmaf_rn(prices[start + k + 1], weights[k + 1], s1);
                s2 = __fmaf_rn(prices[start + k + 2], weights[k + 2], s2);
                s3 = __fmaf_rn(prices[start + k + 3], weights[k + 3], s3);
            }
            float acc = (s0 + s1) + (s2 + s3);
            for (; k < period; ++k) {
                acc = __fmaf_rn(prices[start + k], weights[k], acc);
            }
            out[base_out + t] = acc * inv_norm;
        }
        t += stride;
    }
}


extern "C" __global__
void ehma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) {
        return;
    }

    const int first = first_valids[series_idx];
    const int warm = first + period - 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            const int start = t - period + 1;
            
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            #pragma unroll 4
            for (; k + 3 < period; k += 4) {
                int idx0 = (start + k) * num_series + series_idx;
                int idx1 = (start + k + 1) * num_series + series_idx;
                int idx2 = (start + k + 2) * num_series + series_idx;
                int idx3 = (start + k + 3) * num_series + series_idx;
                s0 = __fmaf_rn(prices_tm[idx0], shared_weights[k],     s0);
                s1 = __fmaf_rn(prices_tm[idx1], shared_weights[k + 1], s1);
                s2 = __fmaf_rn(prices_tm[idx2], shared_weights[k + 2], s2);
                s3 = __fmaf_rn(prices_tm[idx3], shared_weights[k + 3], s3);
            }
            float acc = (s0 + s1) + (s2 + s3);
            for (; k < period; ++k) {
                int in_idx = (start + k) * num_series + series_idx;
                acc = __fmaf_rn(prices_tm[in_idx], shared_weights[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}






__device__ __forceinline__ size_t ehma_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

__device__ __forceinline__ float ehma_dot_uncomp(const float* __restrict__ x,
                                                 const float* __restrict__ w,
                                                 int n) {
    
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        s0 = __fmaf_rn(x[i],     w[i],     s0);
        s1 = __fmaf_rn(x[i + 1], w[i + 1], s1);
        s2 = __fmaf_rn(x[i + 2], w[i + 2], s2);
        s3 = __fmaf_rn(x[i + 3], w[i + 3], s3);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s = __fmaf_rn(x[i], w[i], s);
    return s;
}

__device__ __forceinline__ void ehma_dot2_shared(const float* __restrict__ buf,
                                                  int b,
                                                  const float* __restrict__ w,
                                                  int n,
                                                  float& s0_out,
                                                  float& s1_out) {
    
    float s00 = 0.f, s01 = 0.f, s02 = 0.f, s03 = 0.f;
    float s10 = 0.f, s11 = 0.f, s12 = 0.f, s13 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        float w0 = w[i];     float w1 = w[i + 1];
        float w2 = w[i + 2]; float w3 = w[i + 3];
        s00 = __fmaf_rn(buf[b + i],     w0, s00);
        s10 = __fmaf_rn(buf[b + i + 1], w0, s10);
        s01 = __fmaf_rn(buf[b + i + 1], w1, s01);
        s11 = __fmaf_rn(buf[b + i + 2], w1, s11);
        s02 = __fmaf_rn(buf[b + i + 2], w2, s02);
        s12 = __fmaf_rn(buf[b + i + 3], w2, s12);
        s03 = __fmaf_rn(buf[b + i + 3], w3, s03);
        s13 = __fmaf_rn(buf[b + i + 4], w3, s13);
    }
    float s0 = (s00 + s01) + (s02 + s03);
    float s1 = (s10 + s11) + (s12 + s13);
    for (; i < n; ++i) {
        float wi = w[i];
        s0 = __fmaf_rn(buf[b + i],     wi, s0);
        s1 = __fmaf_rn(buf[b + i + 1], wi, s1);
    }
    s0_out = s0; s1_out = s1;
}

template<int TILE>
struct EhmaBatchTiledPrecomputed2X {
    static __device__ __forceinline__
    void run(const float* __restrict__ prices,
             const float* __restrict__ weights_flat,
             const int*   __restrict__ periods,
             const float* __restrict__ inv_norms, 
             int max_period,
             int series_len,
             int n_combos,
             int first_valid,
             float* __restrict__ out) {
        const int THREADS = TILE / 2; 
        if (blockDim.x != THREADS) return;

        const int combo = blockIdx.y;
        if (combo >= n_combos) return;
        const int period = periods[combo];

        const int t0 = blockIdx.x * TILE;
        if (t0 >= series_len) return;

        const int total = TILE + period - 1;
        extern __shared__ __align__(16) unsigned char shraw[];
        size_t off = 0;
        float* w = reinterpret_cast<float*>(shraw + off);
        off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
        float* buf = reinterpret_cast<float*>(shraw + off); 

        
        const float* wsrc = weights_flat + combo * max_period;
        uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
        if ((waddr & 0xF) == 0) {
            int ve = period >> 2; 
            for (int vi = threadIdx.x; vi < ve; vi += THREADS) {
                reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
            }
            if ((threadIdx.x == 0) && ((period & 3) != 0)) {
                int base = ve << 2;
                for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
            }
        } else {
            for (int i = threadIdx.x; i < period; i += THREADS) w[i] = wsrc[i];
        }
        __syncthreads();

        
        const int p_base0 = t0 - (period - 1);
        bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);
        if (in_bounds) {
            const float* src = prices + p_base0;
            uintptr_t addr = reinterpret_cast<uintptr_t>(src);
            if ((addr & 0xF) == 0) {
                int vec_elems = total >> 2;
                for (int vi = threadIdx.x; vi < vec_elems; vi += THREADS) {
                    reinterpret_cast<float4*>(buf)[vi] = reinterpret_cast<const float4*>(src)[vi];
                }
                if ((threadIdx.x == 0) && ((total & 3) != 0)) {
                    int base = vec_elems << 2;
                    for (int r = 0; r < (total & 3); ++r) buf[base + r] = src[base + r];
                }
            } else {
                for (int i = threadIdx.x; i < total; i += THREADS) buf[i] = src[i];
            }
        } else {
            for (int i = threadIdx.x; i < total; i += THREADS) {
                int idx = p_base0 + i;
                buf[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
            }
        }
        __syncthreads();

        const int warm = first_valid + period - 1;
        const int combo_base = combo * series_len;

        
        int b = 2 * threadIdx.x; 
        int t = t0 + b;
        float out0 = NAN, out1 = NAN;
        if (t < series_len) {
            const bool can0 = (t >= warm);
            const bool can1 = ((t + 1) < series_len) && ((t + 1) >= warm);
            if (can0 && can1) {
                float s0, s1;
                ehma_dot2_shared(buf, b, w, period, s0, s1);
                out0 = s0;
                out1 = s1;
            } else if (can0) {
                out0 = ehma_dot_uncomp(&buf[b], w, period);
            } else if (can1) {
                out1 = ehma_dot_uncomp(&buf[b + 1], w, period);
            }
            out[combo_base + t] = out0;
            if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
        }
    }
};

#define DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(NAME, TILE_OUT)                         \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices,                                               \
  const float* __restrict__ weights_flat,                                         \
  const int*   __restrict__ periods,                                              \
  const float* __restrict__ inv_norms,                                            \
  int max_period, int series_len, int n_combos, int first_valid,                  \
  float* __restrict__ out) {                                                      \
  EhmaBatchTiledPrecomputed2X<TILE_OUT>::run(                                     \
    prices, weights_flat, periods, inv_norms, max_period,                         \
    series_len, n_combos, first_valid, out);                                      \
}

DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile128, 128)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile256, 256)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X(ehma_batch_tiled_f32_2x_tile512, 512)







template<int TILE>
struct EhmaBatchTiledPrecomputed2X_Async {
    static __device__ __forceinline__
    void run(const float* __restrict__ prices,
             const float* __restrict__ weights_flat,
             const int*   __restrict__ periods,
             const float* __restrict__ inv_norms, 
             int max_period, int series_len, int n_combos, int first_valid,
             float* __restrict__ out) {

        const int THREADS = TILE / 2;    
        if (blockDim.x != THREADS) return;

        const int combo = blockIdx.y;
        if (combo >= n_combos) return;

        const int period = periods[combo];
        if (period <= 0 || period > max_period) return;

        
        extern __shared__ __align__(16) unsigned char shraw[];
        size_t off = 0;
        float* w = reinterpret_cast<float*>(shraw + off);
        off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
        
        float* buf = reinterpret_cast<float*>(shraw + off);
        const int total = TILE + period - 1;  

        
        const float* wsrc = weights_flat + combo * max_period;
        uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
        if ((waddr & 0xF) == 0) {
            int ve = period >> 2;
            for (int vi = threadIdx.x; vi < ve; vi += THREADS) {
                reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
            }
            if ((threadIdx.x == 0) && ((period & 3) != 0)) {
                int base = ve << 2;
                for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
            }
        } else {
            for (int i = threadIdx.x; i < period; i += THREADS) w[i] = wsrc[i];
        }
        __syncthreads();

        const int warm = first_valid + period - 1;
        const int combo_base = combo * series_len;

        
        for (int t0 = blockIdx.x * TILE; t0 < series_len; t0 += gridDim.x * TILE) {

            
            const int p_base0 = t0 - (period - 1);
            const bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);

#if EHMA_USE_ASYNC && (__CUDA_ARCH__ >= 800)
            if (in_bounds) {
                
                auto block = cg::this_thread_block();
                cg::memcpy_async(block, buf, prices + p_base0, sizeof(float) * total);
                cg::wait(block);   
                __syncthreads();
            } else
#endif
            {
                
                for (int i = threadIdx.x; i < total; i += THREADS) {
                    int idx = p_base0 + i;
                    buf[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
                }
                __syncthreads();
            }

            
            int b = 2 * threadIdx.x;     
            int t = t0 + b;

            if (t < series_len) {
                float out0 = NAN, out1 = NAN;
                const bool can0 = (t >= warm);
                const bool can1 = ((t + 1) < series_len) && ((t + 1) >= warm);
                if (can0 && can1) {
                    float s0, s1;
                    ehma_dot2_shared(buf, b, w, period, s0, s1);
                    out0 = s0; out1 = s1;
                } else if (can0) {
                    out0 = ehma_dot_uncomp(&buf[b], w, period);
                } else if (can1) {
                    out1 = ehma_dot_uncomp(&buf[b + 1], w, period);
                }
                out[combo_base + t] = out0;
                if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
            }
            __syncthreads();
        } 
    }
};

#define DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(NAME, TILE_OUT)                               \
extern "C" __global__ void NAME(                                                               \
  const float* __restrict__ prices, const float* __restrict__ weights_flat,                    \
  const int*   __restrict__ periods, const float* __restrict__ inv_norms,                      \
  int max_period, int series_len, int n_combos, int first_valid,                               \
  float* __restrict__ out) {                                                                   \
  EhmaBatchTiledPrecomputed2X_Async<TILE_OUT>::run(                                            \
    prices, weights_flat, periods, inv_norms, max_period,                                      \
    series_len, n_combos, first_valid, out);                                                   \
}

DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile128_async, 128)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile256_async, 256)
DEFINE_EHMA_BATCH_TILED_PRECOMP_2X_ASYNC(ehma_batch_tiled_f32_2x_tile512_async, 512)




__device__ __forceinline__
float ehma_dot_stride_uncomp(const float* __restrict__ x, int stride,
                             const float* __restrict__ w, int n) {
    
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    #pragma unroll 4
    for (; i + 3 < n; i += 4) {
        s0 = __fmaf_rn(x[(i + 0) * stride], w[i + 0], s0);
        s1 = __fmaf_rn(x[(i + 1) * stride], w[i + 1], s1);
        s2 = __fmaf_rn(x[(i + 2) * stride], w[i + 2], s2);
        s3 = __fmaf_rn(x[(i + 3) * stride], w[i + 3], s3);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s = __fmaf_rn(x[i * stride], w[i], s);
    return s;
}

template<int TX, int TY>
__device__ __forceinline__
void ehma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float inv_norm, 
                          int num_series,
                          int series_len,
                          const int* __restrict__ first_valids,
                          float* __restrict__ out_tm) {
    const int t0 = blockIdx.x * TX;
    const int s0 = blockIdx.y * TY;
    if (t0 >= series_len || s0 >= num_series) return;

    
    const int total = TX + period - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);

    
    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    const int THREADS = blockDim.x * blockDim.y;
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2;
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += THREADS) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += THREADS) {
            w[i] = weights[i];
        }
    }
    __syncthreads();

    
    const bool vec_ok = (TY == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);
    const int p0 = t0 - (period - 1);
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            if (vec_ok && threadIdx.y == 0) {
                const float4* src4 = reinterpret_cast<const float4*>(&prices_tm[t * num_series + s0]);
                float4 v = src4[0];
                tile[dt * TY + 0] = v.x;
                tile[dt * TY + 1] = v.y;
                tile[dt * TY + 2] = v.z;
                tile[dt * TY + 3] = v.w;
            } else {
                int s = s0 + threadIdx.y;
                float val = 0.f;
                if (s < num_series) val = prices_tm[t * num_series + s];
                tile[dt * TY + threadIdx.y] = val;
            }
        } else {
            int idx = dt * TY + threadIdx.y;
            if (idx < total * TY) tile[idx] = 0.f;
        }
    }
    __syncthreads();

    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;
    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) { out_tm[out_idx] = NAN; return; }

    int start = threadIdx.x; 
    const float* xptr = &tile[start * TY + threadIdx.y];
    float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
    out_tm[out_idx] = acc;
}

#define DEFINE_EHMA_MS1P_TILED(NAME, TX, TY)                                       \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices_tm,                                            \
  const float* __restrict__ weights,                                              \
  int period, float inv_norm, int num_series, int series_len,                     \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {             \
  ehma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,              \
                               num_series, series_len, first_valids, out_tm);     \
}

DEFINE_EHMA_MS1P_TILED(ehma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_EHMA_MS1P_TILED(ehma_ms1p_tiled_f32_tx128_ty4, 128, 4)




template<int TX, int TY>
__device__ __forceinline__
void ehma_ms1p_tiled_core_async(const float* __restrict__ prices_tm,
                                const float* __restrict__ weights,
                                int period, float inv_norm,    
                                int num_series, int series_len,
                                const int* __restrict__ first_valids,
                                float* __restrict__ out_tm) {
    const int t0 = blockIdx.x * TX;
    const int s0 = blockIdx.y * TY;
    if (t0 >= series_len || s0 >= num_series) return;

    const int total = TX + period - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = ehma_align_up(off + size_t(period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off); 

    
    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    const int THREADS = blockDim.x * blockDim.y;
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2;
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += THREADS) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if (threadIdx.x == 0 && threadIdx.y == 0 && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += THREADS) w[i] = weights[i];
    }
    __syncthreads();

    
    const int p0 = t0 - (period - 1);
#if EHMA_USE_ASYNC && (__CUDA_ARCH__ >= 800)
    
    
    const bool vec_ok = (TY == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);
    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pss;
    auto pipe = cuda::make_pipeline(block, &pss);

    pipe.producer_acquire();
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            if (vec_ok) {
                if (threadIdx.y == 0) {
                    const float* src = &prices_tm[t * num_series + s0];
                    float* dst = &tile[dt * TY];
                    cuda::memcpy_async(dst, src, sizeof(float4), pipe);
                }
            } else {
                int s = s0 + threadIdx.y;
                float* dst = &tile[dt * TY + threadIdx.y];
                if (s < num_series) {
                    const float* src = &prices_tm[t * num_series + s];
                    cuda::memcpy_async(dst, src, sizeof(float), pipe);
                } else {
                    *dst = 0.f;
                }
            }
        } else {
            int idx = dt * TY + threadIdx.y;
            if (idx < total * TY) tile[idx] = 0.f;
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();

    
    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s < num_series && t < series_len) {
        int warm = first_valids[s] + period - 1;
        int out_idx = t * num_series + s;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            int start = threadIdx.x; 
            const float* xptr = &tile[start * TY + threadIdx.y];
            float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
            out_tm[out_idx] = acc;
        }
    }
    __syncthreads();
    pipe.consumer_release();
#else
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
        int t = p0 + dt;
        if (t >= 0 && t < series_len) {
            const float* src = &prices_tm[t * num_series + s0];
            float* dst = &tile[dt * TY];
            for (int j = threadIdx.y; j < TY; j += blockDim.y) {
                int s = s0 + j;
                dst[j] = (s < num_series) ? src[j] : 0.f;
            }
        } else {
            for (int j = threadIdx.y; j < TY; j += blockDim.y) {
                int idx = dt * TY + j;
                if (idx < total * TY) tile[idx] = 0.f;
            }
        }
    }
    __syncthreads();

    
    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;
    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) { out_tm[out_idx] = NAN; return; }

    int start = threadIdx.x; 
    const float* xptr = &tile[start * TY + threadIdx.y];
    float acc = ehma_dot_stride_uncomp(xptr, TY, w, period);
    out_tm[out_idx] = acc;
#endif
}

#define DEFINE_EHMA_MS1P_TILED_ASYNC(NAME, TX, TY)                                            \
extern "C" __global__ void NAME(                                                              \
  const float* __restrict__ prices_tm, const float* __restrict__ weights,                     \
  int period, float inv_norm, int num_series, int series_len,                                 \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {                         \
  ehma_ms1p_tiled_core_async<TX, TY>(prices_tm, weights, period, inv_norm,                    \
                                     num_series, series_len, first_valids, out_tm);           \
}

DEFINE_EHMA_MS1P_TILED_ASYNC(ehma_ms1p_tiled_f32_tx128_ty2_async, 128, 2)
DEFINE_EHMA_MS1P_TILED_ASYNC(ehma_ms1p_tiled_f32_tx128_ty4_async, 128, 4)
