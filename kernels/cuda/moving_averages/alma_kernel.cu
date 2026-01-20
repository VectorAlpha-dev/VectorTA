






















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>



#ifdef ALMA_USE_CUB_REDUCE
  #include <cub/cub.cuh>
#endif



#ifndef ALMA_UNROLL
  #define ALMA_UNROLL 4
#endif






#ifndef ALMA_ASSUME
#  if defined(__CUDA_ARCH__)
#    define ALMA_ASSUME(x) if (!(x)) __trap();
#  else
#    define ALMA_ASSUME(x) ((void)0)
#  endif
#endif

__device__ __forceinline__ size_t alma_align_up(size_t x, size_t a) {
  return (x + (a - 1)) & ~(a - 1);
}


__device__ __forceinline__ float alma_warp_sum(float v) {
  unsigned m = 0xffffffffu;
  v += __shfl_down_sync(m, v, 16);
  v += __shfl_down_sync(m, v,  8);
  v += __shfl_down_sync(m, v,  4);
  v += __shfl_down_sync(m, v,  2);
  v += __shfl_down_sync(m, v,  1);
  return v;
}



#ifdef ALMA_USE_CUB_REDUCE
template<int BLOCK_THREADS>
__device__ __forceinline__ float alma_block_sum_cub(float v) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp;
  return BlockReduce(temp).Sum(v);
}
#endif

__device__ __forceinline__ float alma_block_sum(float v) {
#ifdef ALMA_USE_CUB_REDUCE
  switch (blockDim.x) {
    case  64: return alma_block_sum_cub< 64>(v);
    case 128: return alma_block_sum_cub<128>(v);
    case 256: return alma_block_sum_cub<256>(v);
    case 512: return alma_block_sum_cub<512>(v);
    case 1024:return alma_block_sum_cub<1024>(v);
    default:  break;
  }
#endif
  __shared__ float warp_buf[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  float wsum = alma_warp_sum(v);
  if (lane == 0) warp_buf[wid] = wsum;
  __syncthreads();

  float out = 0.f;
  if (wid == 0) {
    out = (lane < (blockDim.x + 31) / 32) ? warp_buf[lane] : 0.f;
    out = alma_warp_sum(out);
  }
  return out;
}


__device__ __forceinline__
float alma_dot_uncomp(const float* __restrict__ x,
                      const float* __restrict__ w, int n) {
  float s = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) s = __fmaf_rn(x[i], w[i], s);
  return s;
}

__device__ __forceinline__
float alma_dot(const float* __restrict__ x,
              const float* __restrict__ w, int n) {
  return alma_dot_uncomp(x, w, n);
}



__device__ __forceinline__
void alma_dot2_shared(const float* __restrict__ buf, int b,
                      const float* __restrict__ w, int n,
                      float& s0_out, float& s1_out) {
  float s0 = 0.f, s1 = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) {
    float wi = w[i];
    s0 = __fmaf_rn(buf[b + i],     wi, s0);
    s1 = __fmaf_rn(buf[b + i + 1], wi, s1);
  }
  s0_out = s0; s1_out = s1;
}


__device__ __forceinline__
float alma_dot_stride(const float* __restrict__ x,
                      int stride,
                      const float* __restrict__ w, int n) {
  float s = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) {
    s = __fmaf_rn(x[i * stride], w[i], s);
  }
  return s;
}



__device__ __forceinline__
void alma_compute_weights_and_invnorm(int period, float m, float s2,
                                      float* __restrict__ weights,
                                      float* __restrict__ inv_norm_s) {
  float local = 0.f;
  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    float d  = float(i) - m;

    float wi = __expf(-(d * d) / s2);
    weights[i] = wi;
    local     += wi;
  }
  float norm = alma_block_sum(local);
  if (threadIdx.x == 0) {
    norm = fmaxf(norm, 1e-20f);
    *inv_norm_s = 1.0f / norm;
  }
  __syncthreads();
}



extern "C" __global__
void alma_batch_f32_onthefly(const float* __restrict__ prices,
                             const int*   __restrict__ periods,
                             const float* __restrict__ offsets,
                             const float* __restrict__ sigmas,
                             int series_len,
                             int n_combos,
                             int first_valid,
                             float* __restrict__ out) {
  const int combo = blockIdx.y;
  if (combo >= n_combos) return;

  __shared__ int   period_s;
  __shared__ float offset_s, sigma_s;
  if (threadIdx.x == 0) {
    period_s = periods[combo];
    offset_s = offsets[combo];
    sigma_s  = sigmas[combo];
  }
  __syncthreads();

  const int   period = period_s;
  const float m      = offset_s * float(period - 1);
  const float s      = float(period) / fmaxf(sigma_s, 1e-6f);
  const float s2     = 2.0f * s * s;
  const int   warm   = first_valid + period - 1;
  const int   base_o = combo * series_len;

  extern __shared__ float sh[];
  float* weights = sh;

  __shared__ float inv_norm_s;
  alma_compute_weights_and_invnorm(period, m, s2, weights, &inv_norm_s);

  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    weights[i] *= inv_norm_s;
  }
  __syncthreads();

  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x  * blockDim.x;

  while (t < series_len) {
    float outv = NAN;
    if (t >= warm) {
      int start = t - period + 1;
      outv = alma_dot(&prices[start], weights, period);
    }
    out[base_o + t] = outv;
    t += stride;
  }
}



extern "C" __global__
void alma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int*   __restrict__ periods,
                    const float* __restrict__ inv_norms,
                    int max_period,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
  const int combo = blockIdx.y;
  if (combo >= n_combos) return;

  const int   period   = periods[combo];

  extern __shared__ float sh[];
  float* w = sh;
  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    w[i] = weights_flat[combo * max_period + i];
  }
  __syncthreads();

  const int warm   = first_valid + period - 1;
  const int base_o = combo * series_len;

  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x  * blockDim.x;

  while (t < series_len) {
    float outv = NAN;
    if (t >= warm) {
      int start = t - period + 1;
      outv = alma_dot(&prices[start], w, period);
    }
    out[base_o + t] = outv;
    t += stride;
  }
}

extern "C" __global__
void alma_batch_f32_tm(const float* __restrict__ prices,
                       const float* __restrict__ weights_flat,
                       const int*   __restrict__ periods,
                       const float* __restrict__ inv_norms,
                       int max_period,
                       int series_len,
                       int n_combos,
                       int first_valid,
                       float* __restrict__ out_tm) {
  const int combo = blockIdx.y;
  if (combo >= n_combos) return;

  const int   period   = periods[combo];

  extern __shared__ float sh[];
  float* w = sh;
  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    w[i] = weights_flat[combo * max_period + i];
  }
  __syncthreads();

  const int warm = first_valid + period - 1;

  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x  * blockDim.x;

  while (t < series_len) {
    float outv = NAN;
    if (t >= warm) {
      int start = t - period + 1;
      outv = alma_dot(&prices[start], w, period);
    }
    out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = outv;
    t += stride;
  }
}









template<int TILE>
struct AlmaBatchTiledPrecomputed {
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
    static_assert(TILE > 0, "TILE must be positive");
    if (blockDim.x != TILE) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int   period   = periods[combo];

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    const int total = TILE + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
    float* buf    = reinterpret_cast<float*>(shraw + off);


    const float* wsrc = weights_flat + combo * max_period;
    uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
    if ((waddr & 0xF) == 0) {
      int ve = period >> 2;
      for (int vi = threadIdx.x; vi < ve; vi += TILE) {
        reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
      }
      if ((threadIdx.x == 0) && ((period & 3) != 0)) {
        int base = ve << 2;
        for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
      }
    } else {
      for (int i = threadIdx.x; i < period; i += TILE) w[i] = wsrc[i];
    }
    __syncthreads();

    const int warm = first_valid + period - 1;
    const int combo_base = combo * series_len;


    const int p_base0 = t0 - (period - 1);
    bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);
    if (in_bounds) {
      const float* src = prices + p_base0;
      uintptr_t addr = reinterpret_cast<uintptr_t>(src);
      if ((addr & 0xF) == 0) {
        int vec_elems = total >> 2;
        int vec_idx = threadIdx.x;
        float4* dst4 = reinterpret_cast<float4*>(buf);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        while (vec_idx < vec_elems) {
          dst4[vec_idx] = src4[vec_idx];
          vec_idx += TILE;
        }
        if ((threadIdx.x == 0) && ((total & 3) != 0)) {
          int base = vec_elems << 2;
          for (int r = 0; r < (total & 3); ++r) buf[base + r] = src[base + r];
        }
      } else {
        for (int i = threadIdx.x; i < total; i += TILE) buf[i] = src[i];
      }
    } else {
      for (int i = threadIdx.x; i < total; i += TILE) {
        int idx = p_base0 + i;
        buf[i]  = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
      }
    }
    __syncthreads();

    int t = t0 + threadIdx.x;
    if (t < series_len) {
      float outv = NAN;
      if (t >= warm) {
        int start = threadIdx.x;
        outv = alma_dot(&buf[start], w, period);
      }
      out[combo_base + t] = outv;
    }
  }
};

template<int TILE>
struct AlmaBatchTiledPrecomputedTM {
  static __device__ __forceinline__
  void run(const float* __restrict__ prices,
           const float* __restrict__ weights_flat,
           const int*   __restrict__ periods,
           const float* __restrict__ inv_norms,
           int max_period,
           int series_len,
           int n_combos,
           int first_valid,
           float* __restrict__ out_tm) {
    static_assert(TILE > 0, "TILE must be positive");
    if (blockDim.x != TILE) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int   period   = periods[combo];

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    const int total = TILE + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
    float* buf    = reinterpret_cast<float*>(shraw + off);


    const float* wsrc = weights_flat + combo * max_period;
    uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
    if ((waddr & 0xF) == 0) {
      int ve = period >> 2;
      for (int vi = threadIdx.x; vi < ve; vi += TILE) {
        reinterpret_cast<float4*>(w)[vi] = reinterpret_cast<const float4*>(wsrc)[vi];
      }
      if ((threadIdx.x == 0) && ((period & 3) != 0)) {
        int base = ve << 2;
        for (int r = 0; r < (period & 3); ++r) w[base + r] = wsrc[base + r];
      }
    } else {
      for (int i = threadIdx.x; i < period; i += TILE) w[i] = wsrc[i];
    }
    __syncthreads();

    const int warm = first_valid + period - 1;


    const int p_base0 = t0 - (period - 1);
    bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);
    if (in_bounds) {
      const float* src = prices + p_base0;
      uintptr_t addr = reinterpret_cast<uintptr_t>(src);
      if ((addr & 0xF) == 0) {
        int vec_elems = total >> 2;
        int vec_idx = threadIdx.x;
        float4* dst4 = reinterpret_cast<float4*>(buf);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        while (vec_idx < vec_elems) {
          dst4[vec_idx] = src4[vec_idx];
          vec_idx += TILE;
        }
        if ((threadIdx.x == 0) && ((total & 3) != 0)) {
          int base = vec_elems << 2;
          for (int r = 0; r < (total & 3); ++r) buf[base + r] = src[base + r];
        }
      } else {
        for (int i = threadIdx.x; i < total; i += TILE) buf[i] = src[i];
      }
    } else {
      for (int i = threadIdx.x; i < total; i += TILE) {
        int idx = p_base0 + i;
        buf[i]  = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
      }
    }
    __syncthreads();

    int t = t0 + threadIdx.x;
    if (t < series_len) {
      float outv = NAN;
      if (t >= warm) {
        int start = threadIdx.x;
        outv = alma_dot(&buf[start], w, period);
      }
      out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = outv;
    }
  }
};


#define DEFINE_ALMA_BATCH_TILED_PRECOMP(NAME, TILE)                              \
extern "C" __global__ void NAME(                                                 \
  const float* __restrict__ prices,                                              \
  const float* __restrict__ weights_flat,                                        \
  const int*   __restrict__ periods,                                             \
  const float* __restrict__ inv_norms,                                           \
  int max_period, int series_len, int n_combos, int first_valid,                 \
  float* __restrict__ out) {                                                     \
  AlmaBatchTiledPrecomputed<TILE>::run(                                          \
    prices, weights_flat, periods, inv_norms, max_period,                        \
    series_len, n_combos, first_valid, out);                                     \
}

DEFINE_ALMA_BATCH_TILED_PRECOMP(alma_batch_tiled_f32_tile128, 128)
DEFINE_ALMA_BATCH_TILED_PRECOMP(alma_batch_tiled_f32_tile256, 256)
DEFINE_ALMA_BATCH_TILED_PRECOMP(alma_batch_tiled_f32_tile512, 512)

#define DEFINE_ALMA_BATCH_TILED_PRECOMP_TM(NAME, TILE)                           \
extern "C" __global__ void NAME(                                                 \
  const float* __restrict__ prices,                                              \
  const float* __restrict__ weights_flat,                                        \
  const int*   __restrict__ periods,                                             \
  const float* __restrict__ inv_norms,                                           \
  int max_period, int series_len, int n_combos, int first_valid,                 \
  float* __restrict__ out_tm) {                                                  \
  AlmaBatchTiledPrecomputedTM<TILE>::run(                                        \
    prices, weights_flat, periods, inv_norms, max_period,                        \
    series_len, n_combos, first_valid, out_tm);                                  \
}

DEFINE_ALMA_BATCH_TILED_PRECOMP_TM(alma_batch_tiled_f32_tile128_tm, 128)
DEFINE_ALMA_BATCH_TILED_PRECOMP_TM(alma_batch_tiled_f32_tile256_tm, 256)
DEFINE_ALMA_BATCH_TILED_PRECOMP_TM(alma_batch_tiled_f32_tile512_tm, 512)



template<int TILE_OUT>
struct AlmaBatchTiledPrecomputed2X {
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
    static_assert(TILE_OUT % 2 == 0, "TILE_OUT must be even");
    constexpr int THREADS = TILE_OUT / 2;
    if (blockDim.x != THREADS) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int   period   = periods[combo];

    const int t0 = blockIdx.x * TILE_OUT;
    if (t0 >= series_len) return;

    const int total = TILE_OUT + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
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
        int vec_idx = threadIdx.x;
        float4* dst4 = reinterpret_cast<float4*>(buf);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        while (vec_idx < vec_elems) {
          dst4[vec_idx] = src4[vec_idx];
          vec_idx += THREADS;
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
        buf[i]  = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
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
        alma_dot2_shared(buf, b, w, period, s0, s1);
        out0 = s0;
        out1 = s1;
      } else if (can0) {

        out0 = alma_dot(&buf[b], w, period);
      } else if (can1) {

        out1 = alma_dot(&buf[b + 1], w, period);
      }
      out[combo_base + t] = out0;
      if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
    }
  }
};

template<int TILE_OUT>
struct AlmaBatchTiledPrecomputed2XTM {
  static __device__ __forceinline__
  void run(const float* __restrict__ prices,
           const float* __restrict__ weights_flat,
           const int*   __restrict__ periods,
           const float* __restrict__ inv_norms,
           int max_period,
           int series_len,
           int n_combos,
           int first_valid,
           float* __restrict__ out_tm) {
    static_assert(TILE_OUT % 2 == 0, "TILE_OUT must be even");
    constexpr int THREADS = TILE_OUT / 2;
    if (blockDim.x != THREADS) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int   period   = periods[combo];

    const int t0 = blockIdx.x * TILE_OUT;
    if (t0 >= series_len) return;

    const int total = TILE_OUT + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
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
        int vec_idx = threadIdx.x;
        float4* dst4 = reinterpret_cast<float4*>(buf);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        while (vec_idx < vec_elems) {
          dst4[vec_idx] = src4[vec_idx];
          vec_idx += THREADS;
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
        buf[i]  = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
      }
    }
    __syncthreads();

    const int warm = first_valid + period - 1;


    int b = 2 * threadIdx.x;
    int t = t0 + b;
    float out0 = NAN, out1 = NAN;
    if (t < series_len) {
      const bool can0 = (t >= warm);
      const bool can1 = ((t + 1) < series_len) && ((t + 1) >= warm);
      if (can0 && can1) {
        float s0, s1;
        alma_dot2_shared(buf, b, w, period, s0, s1);
        out0 = s0;
        out1 = s1;
      } else if (can0) {

        out0 = alma_dot(&buf[b], w, period);
      } else if (can1) {

        out1 = alma_dot(&buf[b + 1], w, period);
      }
      out_tm[(size_t)t * (size_t)n_combos + (size_t)combo] = out0;
      if ((t + 1) < series_len) {
        out_tm[(size_t)(t + 1) * (size_t)n_combos + (size_t)combo] = out1;
      }
    }
  }
};

#define DEFINE_ALMA_BATCH_TILED_PRECOMP_2X(NAME, TILE_OUT)                        \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices,                                              \
  const float* __restrict__ weights_flat,                                        \
  const int*   __restrict__ periods,                                             \
  const float* __restrict__ inv_norms,                                           \
  int max_period, int series_len, int n_combos, int first_valid,                 \
  float* __restrict__ out) {                                                     \
  AlmaBatchTiledPrecomputed2X<TILE_OUT>::run(                                    \
    prices, weights_flat, periods, inv_norms, max_period,                        \
    series_len, n_combos, first_valid, out);                                     \
}

DEFINE_ALMA_BATCH_TILED_PRECOMP_2X(alma_batch_tiled_f32_2x_tile128, 128)
DEFINE_ALMA_BATCH_TILED_PRECOMP_2X(alma_batch_tiled_f32_2x_tile256, 256)
DEFINE_ALMA_BATCH_TILED_PRECOMP_2X(alma_batch_tiled_f32_2x_tile512, 512)

#define DEFINE_ALMA_BATCH_TILED_PRECOMP_2X_TM(NAME, TILE_OUT)                     \
extern "C" __global__ void NAME(                                                  \
  const float* __restrict__ prices,                                              \
  const float* __restrict__ weights_flat,                                        \
  const int*   __restrict__ periods,                                             \
  const float* __restrict__ inv_norms,                                           \
  int max_period, int series_len, int n_combos, int first_valid,                 \
  float* __restrict__ out_tm) {                                                  \
  AlmaBatchTiledPrecomputed2XTM<TILE_OUT>::run(                                  \
    prices, weights_flat, periods, inv_norms, max_period,                        \
    series_len, n_combos, first_valid, out_tm);                                  \
}

DEFINE_ALMA_BATCH_TILED_PRECOMP_2X_TM(alma_batch_tiled_f32_2x_tile128_tm, 128)
DEFINE_ALMA_BATCH_TILED_PRECOMP_2X_TM(alma_batch_tiled_f32_2x_tile256_tm, 256)
DEFINE_ALMA_BATCH_TILED_PRECOMP_2X_TM(alma_batch_tiled_f32_2x_tile512_tm, 512)







extern "C" __global__
void alma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights,
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
  const int TX = blockDim.x;
  const int SY = blockDim.y;

  int t = blockIdx.x * TX + threadIdx.x;
  int s = blockIdx.y * SY + threadIdx.y;
  if (s >= num_series || t >= series_len) return;

  extern __shared__ float sh[];
  float* w = sh;
  for (int i = threadIdx.y * TX + threadIdx.x; i < period; i += TX * SY) {
    w[i] = weights[i];
  }
  __syncthreads();

  const int warm = first_valids[s] + period - 1;
  const int out_idx = t * num_series + s;

  if (t < warm) {
    out_tm[out_idx] = NAN;
    return;
  }

  int start = t - period + 1;
  const float* xptr = &prices_tm[start * num_series + s];
  float acc = alma_dot_stride(xptr, num_series, w, period);

  out_tm[out_idx] = acc;
}



extern "C" __global__
void alma_precompute_weights_f32(const int*   __restrict__ periods,
                                 const float* __restrict__ offsets,
                                 const float* __restrict__ sigmas,
                                 int n_combos,
                                 int max_period,
                                 float* __restrict__ weights_flat,
                                 float* __restrict__ inv_norms) {
  const int combo = blockIdx.x;
  if (combo >= n_combos) return;

  const int   period = periods[combo];
  const float m      = offsets[combo] * float(period - 1);
  const float s      = float(period) / fmaxf(sigmas[combo], 1e-6f);
  const float s2     = 2.0f * s * s;

  extern __shared__ float sh[];
  float* w = sh;

  __shared__ float inv_norm_s;
  alma_compute_weights_and_invnorm(period, m, s2, w, &inv_norm_s);

  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    weights_flat[combo * max_period + i] = w[i] * inv_norm_s;
  }
  if (threadIdx.x == 0) inv_norms[combo] = 1.0f;
}



template<int TX, int TY>
__device__ __forceinline__
void alma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float inv_norm,
                          int num_series,
                          int series_len,
                          const int* __restrict__ first_valids,
                          float* __restrict__ out_tm) {
  const int TX_ = TX;
  const int TY_ = TY;
  const int t0 = blockIdx.x * TX_;
  const int s0 = blockIdx.y * TY_;

  if (t0 >= series_len || s0 >= num_series) return;


  const int total = TX_ + period - 1;
  extern __shared__ __align__(16) unsigned char shraw[];
  size_t off = 0;
  float* w = reinterpret_cast<float*>(shraw + off);
  off = alma_align_up(off + size_t(period) * sizeof(float), 16);
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


  const bool vec_ok = (TY_ == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);

  const int p0 = t0 - (period - 1);
  for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
    int t = p0 + dt;
    if (t >= 0 && t < series_len) {
      if (vec_ok && threadIdx.y == 0) {

        const float4* src4 = reinterpret_cast<const float4*>(&prices_tm[t * num_series + s0]);
        float4 v = src4[0];
        tile[dt * TY_ + 0] = v.x;
        tile[dt * TY_ + 1] = v.y;
        tile[dt * TY_ + 2] = v.z;
        tile[dt * TY_ + 3] = v.w;
      } else {
        int s = s0 + threadIdx.y;
        float val = 0.f;
        if (s < num_series) val = prices_tm[t * num_series + s];
        tile[dt * TY_ + threadIdx.y] = val;
      }
    } else {
      int idx = dt * TY_ + threadIdx.y;
      if (idx < total * TY_) tile[idx] = 0.f;
    }
  }
  __syncthreads();


  int s = s0 + threadIdx.y;
  int t = t0 + threadIdx.x;
  if (s >= num_series || t >= series_len) return;

  int warm = first_valids[s] + period - 1;
  int out_idx = t * num_series + s;

  if (t < warm) {
    out_tm[out_idx] = NAN;
    return;
  }

  int start = threadIdx.x;
  const float* xptr = &tile[start * TY_ + threadIdx.y];
  float acc = alma_dot_stride(xptr, TY_, w, period);

  out_tm[out_idx] = acc;
}

#define DEFINE_ALMA_MS1P_TILED(NAME, TX, TY)                                     \
extern "C" __global__ void NAME(                                                 \
  const float* __restrict__ prices_tm,                                           \
  const float* __restrict__ weights,                                             \
  int period, float inv_norm, int num_series, int series_len,                    \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {            \
  alma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,             \
                               num_series, series_len, first_valids, out_tm);    \
}

DEFINE_ALMA_MS1P_TILED(alma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_ALMA_MS1P_TILED(alma_ms1p_tiled_f32_tx128_ty4, 128, 4)
