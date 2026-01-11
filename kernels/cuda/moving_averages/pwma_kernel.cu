








#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#ifndef PWMA_TILE_TX
#define PWMA_TILE_TX 128
#endif

__device__ __forceinline__ size_t pwma_align_up_sz(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

extern "C" __global__
void pwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
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

    extern __shared__ float shared_weights[];

    for (int idx = threadIdx.x; idx < period; idx += blockDim.x) {
        shared_weights[idx] = weights_flat[combo * max_period + idx];
    }
    __syncthreads();

    const int warm = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        if (t < warm) {
            out[base_out + t] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices[start + k], shared_weights[k], acc);
            }
            out[base_out + t] = acc;
        }
        t += stride;
    }
}

extern "C" __global__
void pwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     
                                     
                                     float /*inv_norm*/,
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

    const int warm = first_valids[series_idx] + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
            const int in_idx = (start + k) * num_series + series_idx;
            acc = fmaf(prices_tm[in_idx], shared_weights[k], acc);
        }
        out_tm[out_idx] = acc;
    }
    t += stride;
}
}







extern "C" __global__
void pwma_batch_tiled_async_f32(const float* __restrict__ prices,
                                const float* __restrict__ weights_flat,
                                const int* __restrict__ periods,
                                const int* __restrict__ warm_indices,
                                int series_len,
                                int n_combos,
                                int max_period,
                                float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    if (period <= 0 || period > max_period) return;

    const int TILE = PWMA_TILE_TX;             
    const int wlen = period;                   
    const int total = TILE + wlen - 1;         

    const int warm = warm_indices[combo];
    const int base_out = combo * series_len;
    const float nan_f = __int_as_float(0x7fffffff);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);           
    off = pwma_align_up_sz(off + size_t(max_period) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);         

    
    const float* wsrc = weights_flat + combo * max_period;
    for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
        w[i] = wsrc[i];
    }
    __syncthreads();

    auto cta = cg::this_thread_block();
    constexpr int STAGES = 2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto pipe = cuda::make_pipeline(cta, &pss);

    const int lane = threadIdx.x;
    const int grid_tile_stride = gridDim.x * TILE;

    int t_base = blockIdx.x * TILE;
    int stage  = 0;

    
    for (int s = 0; s < STAGES; ++s) {
        pipe.producer_acquire();
        const int t0 = t_base + s * grid_tile_stride;
        const int p0 = t0 - (wlen - 1);
        for (int dt = lane; dt < total; dt += blockDim.x) {
            const int tcur = p0 + dt;
            if (tcur >= 0 && tcur < series_len) {
                cuda::memcpy_async(&tile[s * total + dt], &prices[tcur], sizeof(float), pipe);
            } else {
                tile[s * total + dt] = 0.0f;
            }
        }
        pipe.producer_commit();
    }

    
    while (t_base < series_len) {
        pipe.consumer_wait();
        __syncthreads();

        
        const float* tbuf = &tile[stage * total];
        const int t = t_base + lane;
        if (t < series_len) {
            if (t < warm) {
                out[base_out + t] = nan_f;
            } else {
                int start = lane; 
                const float* xptr = &tbuf[start];
                float acc = 0.0f;
#pragma unroll 8
                for (int k = 0; k < wlen; ++k) {
                    acc = fmaf(xptr[k], w[k], acc);
                }
                out[base_out + t] = acc;
            }
        }

        __syncthreads();
        pipe.consumer_release();

        
        pipe.producer_acquire();
        const int next_t0 = t_base + STAGES * grid_tile_stride;
        const int next_p0 = next_t0 - (wlen - 1);
        const int next_stage = stage; 

        for (int dt = lane; dt < total; dt += blockDim.x) {
            const int tcur = next_p0 + dt;
            if (tcur >= 0 && tcur < series_len) {
                cuda::memcpy_async(&tile[next_stage * total + dt], &prices[tcur], sizeof(float), pipe);
            } else {
                tile[next_stage * total + dt] = 0.0f;
            }
        }
        pipe.producer_commit();

        t_base += grid_tile_stride;
        stage = (stage + 1) % STAGES;
    }
}




__device__ __forceinline__ size_t pwma_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

template<int TX, int TY>
__device__ __forceinline__
void pwma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float /*inv_norm_unused*/, 
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
    off = pwma_align_up(off + size_t(period) * sizeof(float), 16);
    const int LD = TY + 1; 
    float* tile = reinterpret_cast<float*>(shraw + off);

    
    uintptr_t waddr = reinterpret_cast<uintptr_t>(weights);
    if ((waddr & 0xF) == 0) {
        int ve = period >> 2; 
        for (int vi = threadIdx.y * blockDim.x + threadIdx.x; vi < ve; vi += blockDim.x * blockDim.y) {
            reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(weights)[vi];
        }
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && ((period & 3) != 0)) {
            int base = ve << 2;
            for (int r = 0; r < (period & 3); ++r) w[base + r] = weights[base + r];
        }
    } else {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < period; i += blockDim.x * blockDim.y) {
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
                tile[dt * LD + 0] = v.x;
                tile[dt * LD + 1] = v.y;
                tile[dt * LD + 2] = v.z;
                tile[dt * LD + 3] = v.w;
            } else {
                int s = s0 + threadIdx.y;
                float val = 0.f;
                if (s < num_series) val = prices_tm[t * num_series + s];
                tile[dt * LD + threadIdx.y] = val;
            }
        } else {
            int idx = dt * LD + threadIdx.y;
            if (idx < total * LD) tile[idx] = 0.f;
        }
    }
    __syncthreads();

    
    int s = s0 + threadIdx.y;
    int t = t0 + threadIdx.x;
    if (s >= num_series || t >= series_len) return;

    int warm = first_valids[s] + period - 1;
    int out_idx = t * num_series + s;
    if (t < warm) {
        out_tm[out_idx] = __int_as_float(0x7fffffff);
        return;
    }

    int start = threadIdx.x; 
    const float* xptr = &tile[start * LD + threadIdx.y];
    float acc = 0.f;
#pragma unroll 8
    for (int i = 0; i < period; ++i) {
        acc = fmaf(xptr[i * LD], w[i], acc);
    }
    
    out_tm[out_idx] = acc;
}

#define DEFINE_PWMA_MS1P_TILED(NAME, TX, TY)                                    \
extern "C" __global__ void NAME(                                                \
  const float* __restrict__ prices_tm,                                          \
  const float* __restrict__ weights,                                            \
  int period, float inv_norm, int num_series, int series_len,                   \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {           \
  pwma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,            \
                               num_series, series_len, first_valids, out_tm);   \
}


DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_PWMA_MS1P_TILED(pwma_ms1p_tiled_f32_tx128_ty4, 128, 4)


#ifndef PWMA_MAX_PERIOD_CONST
#define PWMA_MAX_PERIOD_CONST 4096
#endif

__constant__ float pwma_const_w[PWMA_MAX_PERIOD_CONST];

extern "C" __global__
void pwma_ms1p_const_f32(const float* __restrict__ prices_tm,
                         int period,
                         int num_series, int series_len,
                         const int* __restrict__ first_valids,
                         float* __restrict__ out_tm) {
    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;
    const int warm = first_valids[series_idx] + period - 1;
    const float nan_f = __int_as_float(0x7fffffff);

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = nan_f;
        } else {
            const int start = t - period + 1;
            float acc = 0.f;
#pragma unroll 8
            for (int k = 0; k < period; ++k) {
                acc = fmaf(prices_tm[(start + k) * num_series + series_idx], pwma_const_w[k], acc);
            }
            out_tm[out_idx] = acc;
        }
        t += stride;
    }
}
