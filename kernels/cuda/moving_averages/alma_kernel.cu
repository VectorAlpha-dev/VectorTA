// alma_kernels.cu
// CUDA 13-ready ALMA kernels with:
// - Two-stage cuda::pipeline tiling + cooperative cuda::memcpy_async
// - 16B-aligned dynamic shared memory layout
// - Optional compensated summation for improved FP32 accuracy
// - Optional CUB BlockReduce for normalization
//
// Build example (Ada / RTX 4090):
//   nvcc -O3 -std=c++17 -arch=sm_89 -Xptxas -v -lineinfo alma_kernels.cu -c
//
// Runtime note: To overlap copy/compute you must launch sufficient CTAs.
// Shared memory per CTA (tiled kernel):
//   bytes = align16(period*4)
//           + stages * align16((TILE + period - 1) * 4)
//   where stages = 2.
//
// References:
// - cuda::pipeline + memcpy_async (CUDA 13 guide)  [memcpy_async, staging].
// - libcudacxx barrier API (proper init).          [init(&bar,...)].
// - CUB BlockReduce usage.                         [BlockReduce].
// - RTX 4090 FP64 ≈ 1/64 FP32 (tradeoff for accuracy).
//
// (c) Drop-in for your existing names and signatures.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>

#ifdef ALMA_USE_CUB_REDUCE
  #include <cub/cub.cuh>
#endif

namespace cg = cooperative_groups;

// ------------------------- Tunables & feature flags -------------------------

#ifndef ALMA_UNROLL
  #define ALMA_UNROLL 4
#endif

// Set to 1 to enable Kahan-Neumaier compensated accumulation in dot-products.
#ifndef ALMA_COMPENSATED_DOT
  #define ALMA_COMPENSATED_DOT 0
#endif

// Enable double-buffered pipeline in tiled async kernel when SM80+ available.
#ifndef ALMA_PIPELINE_ENABLED
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    #define ALMA_PIPELINE_ENABLED 1
  #else
    #define ALMA_PIPELINE_ENABLED 0
  #endif
#endif

// ------------------------------- Utilities ---------------------------------

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

// Warp reduce sum
__device__ __forceinline__ float alma_warp_sum(float v) {
  unsigned m = 0xffffffffu;
  v += __shfl_down_sync(m, v, 16);
  v += __shfl_down_sync(m, v,  8);
  v += __shfl_down_sync(m, v,  4);
  v += __shfl_down_sync(m, v,  2);
  v += __shfl_down_sync(m, v,  1);
  return v;
}

// Block reduce sum with optional CUB fast path.
// Falls back to dynamic warp tree when blockDim.x is not one of the common sizes.
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

// Dot product helpers
__device__ __forceinline__
float alma_dot_uncomp(const float* __restrict__ x,
                      const float* __restrict__ w, int n) {
  float s = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) s = __fmaf_rn(x[i], w[i], s);
  return s;
}

__device__ __forceinline__
float alma_dot_comp(const float* __restrict__ x,
                    const float* __restrict__ w, int n) {
  float s = 0.f, c = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) {
    float term = __fmaf_rn(x[i], w[i], 0.f);
    float y = term - c;
    float t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

__device__ __forceinline__
float alma_dot(const float* __restrict__ x,
               const float* __restrict__ w, int n) {
#if ALMA_COMPENSATED_DOT
  return alma_dot_comp(x, w, n);
#else
  return alma_dot_uncomp(x, w, n);
#endif
}

// Compute Gaussian weights in shared and store 1/norm to *inv_norm_s.
// weights[0..period-1] must be addressable.
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

// ---------------------- 1) On-the-fly batched kernel -----------------------

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
  const float inv_norm = inv_norm_s;

  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x  * blockDim.x;

  while (t < series_len) {
    float outv = NAN;
    if (t >= warm) {
      int start = t - period + 1;
      outv = alma_dot(&prices[start], weights, period) * inv_norm;
    }
    out[base_o + t] = outv;
    t += stride;
  }
}

// ---------------- 2) Precomputed-weight batched kernel ---------------------

extern "C" __global__
void alma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat, // [n_combos * max_period]
                    const int*   __restrict__ periods,
                    const float* __restrict__ inv_norms,     // [n_combos]
                    int max_period,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
  const int combo = blockIdx.y;
  if (combo >= n_combos) return;

  const int   period   = periods[combo];
  const float inv_norm = inv_norms[combo];

  extern __shared__ float sh[];
  float* w = sh; // [period]
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
      outv = alma_dot(&prices[start], w, period) * inv_norm;
    }
    out[base_o + t] = outv;
    t += stride;
  }
}

// -------- 3) Async-tiled on-the-fly kernel with two-stage pipeline ---------
// Dynamic shared layout (16B aligned):
//   [weights: period*4] [pad->16] [stage0: (TILE+period-1)*4] [pad->16] [stage1: ...]
extern "C" __global__
void alma_batch_tiled_async_f32(const float* __restrict__ prices,
                                const int*   __restrict__ periods,
                                const float* __restrict__ offsets,
                                const float* __restrict__ sigmas,
                                int series_len,
                                int n_combos,
                                int first_valid,
                                float* __restrict__ out) {
  const int combo = blockIdx.y;
  if (combo >= n_combos) return;

  // Broadcast params
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

  const int TILE  = blockDim.x;
  const int total = TILE + period - 1;
  const size_t weights_bytes = size_t(period) * sizeof(float);
  const size_t tile_bytes    = size_t(total)  * sizeof(float);

  // 16B-aligned dynamic shared memory partition
  extern __shared__ __align__(16) unsigned char shmem_raw[];
  size_t off = 0;
  float* weights = reinterpret_cast<float*>(shmem_raw + off);
  off = alma_align_up(off + weights_bytes, 16);
  float* stage0  = reinterpret_cast<float*>(shmem_raw + off);
  off = alma_align_up(off + tile_bytes, 16);
  float* stage1  = reinterpret_cast<float*>(shmem_raw + off);

  __shared__ float inv_norm_s;
  alma_compute_weights_and_invnorm(period, m, s2, weights, &inv_norm_s);
  const float inv_norm = inv_norm_s;

  const int combo_base = combo * series_len;

#if ALMA_PIPELINE_ENABLED
  auto block = cg::this_thread_block();
  constexpr int stages = 2;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> pipe_state;
  auto pipe = cuda::make_pipeline(block, &pipe_state);

  // Stride tiles across grid.x
  int t_base = blockIdx.x * TILE;
  if (t_base >= series_len) return;

  // Kickstart stage0
  int p_base = t_base - (period - 1);
  pipe.producer_acquire();
  if (p_base >= 0 && (p_base + total) <= series_len) {
    cuda::memcpy_async(block, stage0, prices + p_base, tile_bytes, pipe);  // cooperative copy
  } else {
    for (int i = threadIdx.x; i < total; i += TILE) {
      int idx = p_base + i;
      stage0[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
    }
  }
  pipe.producer_commit();

  while (true) {
    // Prefetch next tile into stage1
    int t_next = t_base + gridDim.x * TILE;
    int p_next = t_next - (period - 1);
    if (t_next < series_len) {
      pipe.producer_acquire();
      if (p_next >= 0 && (p_next + total) <= series_len) {
        cuda::memcpy_async(block, stage1, prices + p_next, tile_bytes, pipe);
      } else {
        for (int i = threadIdx.x; i < total; i += TILE) {
          int idx = p_next + i;
          stage1[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
        }
      }
      pipe.producer_commit();
    }

    // Wait for current stage to be ready and compute it
    pipe.consumer_wait();
    {
      int t = t_base + threadIdx.x;
      if (t < series_len) {
        float outv = NAN;
        if (t >= warm) {
          int start = threadIdx.x;
          outv = alma_dot(&stage0[start], weights, period) * inv_norm;
        }
        out[combo_base + t] = outv;
      }
    }
    pipe.consumer_release();

    if (t_next >= series_len) break;

    // Advance and swap buffers
    t_base = t_next;
    float* tmp = stage0; stage0 = stage1; stage1 = tmp;
  }
#else
  // Fallback: synchronous cooperative load and compute
  int t0    = blockIdx.x * TILE;
  if (t0 >= series_len) return;
  int pbase = t0 - (period - 1);

  // load tile into shared
  for (int i = threadIdx.x; i < total; i += TILE) {
    int idx = pbase + i;
    stage0[i] = (0 <= idx && idx < series_len) ? prices[idx] : 0.f;
  }
  __syncthreads();

  int t = t0 + threadIdx.x;
  if (t < series_len) {
    float outv = NAN;
    if (t >= warm) {
      int start = threadIdx.x;
      outv = alma_dot(&stage0[start], weights, period) * inv_norm;
    }
    out[combo_base + t] = outv;
  }
#endif
}

// ------------- 4) Precomputed-weight tiled (template TILE) ------------------

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
    const float inv_norm = inv_norms[combo];

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    const int total = TILE + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);           // [period]
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
    float* buf    = reinterpret_cast<float*>(shraw + off);        // [TILE+period-1]

    // Load weights into shared (vectorized when aligned)
    const float* wsrc = weights_flat + combo * max_period;
    uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
    if ((waddr & 0xF) == 0) {
      int ve = period >> 2; // period / 4
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

    // Synchronous cooperative load and compute for one tile
    const int p_base0 = t0 - (period - 1);
    bool in_bounds = (p_base0 >= 0) && ((p_base0 + total) <= series_len);
    if (in_bounds) {
      const float* src = prices + p_base0;
      uintptr_t addr = reinterpret_cast<uintptr_t>(src);
      if ((addr & 0xF) == 0) {
        int vec_elems = total >> 2; // total / 4
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
        outv = alma_dot(&buf[start], w, period) * inv_norm;
      }
      out[combo_base + t] = outv;
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

// ------------- 4b) Precomputed tiled with 2 outputs/thread ------------------

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
    const float inv_norm = inv_norms[combo];

    const int t0 = blockIdx.x * TILE_OUT;
    if (t0 >= series_len) return;

    const int total = TILE_OUT + period - 1;
    const size_t tile_bytes = size_t(total) * sizeof(float);

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w   = reinterpret_cast<float*>(shraw + off);           // [period]
    off = alma_align_up(off + size_t(period)*sizeof(float), 16);
    float* buf = reinterpret_cast<float*>(shraw + off);           // [TILE_OUT+period-1]

    // Load weights into shared (vectorized when aligned)
    const float* wsrc = weights_flat + combo * max_period;
    uintptr_t waddr = reinterpret_cast<uintptr_t>(wsrc);
    if ((waddr & 0xF) == 0) {
      int ve = period >> 2; // period / 4
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
        int vec_elems = total >> 2; // total / 4
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

    // Each thread computes two consecutive outputs using a sliding update.
    int b = 2 * threadIdx.x; // start offset within tile
    int t = t0 + b;
    // First output
    float out0 = NAN, out1 = NAN;
    if (t < series_len) {
      if (t >= warm) {
        float s0 = alma_dot(&buf[b], w, period);
        out0 = s0 * inv_norm;
        // Second output via sliding update using two FMAs
        if ((t + 1) < series_len) {
          if ((t + 1) >= warm) {
            float s1 = __fmaf_rn(w[period - 1], buf[b + period], s0);
            s1 = __fmaf_rn(-w[0], buf[b], s1);
            out1 = s1 * inv_norm;
          } else {
            out1 = NAN;
          }
        }
      } else {
        out0 = NAN;
        if ((t + 1) < series_len) {
          if ((t + 1) >= warm) {
            float s1 = alma_dot(&buf[b + 1], w, period);
            out1 = s1 * inv_norm;
          } else {
            out1 = NAN;
          }
        }
      }
      // Write results
      out[combo_base + t] = out0;
      if ((t + 1) < series_len) out[combo_base + t + 1] = out1;
    }
  }
};

#define DEFINE_ALMA_BATCH_TILED_PRECOMP_2X(NAME, TILE_OUT)                        \
extern "C" __global__ void NAME(                                                 \
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


// ------------- 5) Many-series, one-parameter (time-major) -------------------

extern "C" __global__
void alma_multi_series_one_param_onthefly_f32(const float* __restrict__ prices_tm,
                                              const int*   __restrict__ first_valids, // [num_series]
                                              int period,
                                              float offset,
                                              float sigma,
                                              int num_series,
                                              int series_len,
                                              float* __restrict__ out_tm) {
  const int TX = blockDim.x;
  const int SY = blockDim.y;

  int t = blockIdx.x * TX + threadIdx.x;  // time
  int s = blockIdx.y * SY + threadIdx.y;  // series
  if (s >= num_series || t >= series_len) return;

  extern __shared__ float sh[];
  float* weights = sh;

  const float m  = offset * float(period - 1);
  const float s1 = float(period) / fmaxf(sigma, 1e-6f);
  const float s2 = 2.0f * s1 * s1;

  __shared__ float inv_norm_s;
  alma_compute_weights_and_invnorm(period, m, s2, weights, &inv_norm_s);
  const float inv_norm = inv_norm_s;

  const int warm = first_valids[s] + period - 1;
  const int out_idx = t * num_series + s;

  if (t < warm) {
    out_tm[out_idx] = NAN;
    return;
  }

  int start = t - period + 1;
  float acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < period; ++k) {
    int in_idx = (start + k) * num_series + s;
    acc = __fmaf_rn(prices_tm[in_idx], weights[k], acc);
  }
  out_tm[out_idx] = acc * inv_norm;
}

extern "C" __global__
void alma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     const float* __restrict__ weights, // [period]
                                     int period,
                                     float inv_norm,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm) {
  const int TX = blockDim.x;
  const int SY = blockDim.y;

  int t = blockIdx.x * TX + threadIdx.x;  // time
  int s = blockIdx.y * SY + threadIdx.y;  // series
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
  float acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < period; ++k) {
    int in_idx = (start + k) * num_series + s;
    acc = __fmaf_rn(prices_tm[in_idx], w[k], acc);
  }
  out_tm[out_idx] = acc * inv_norm;
}

// ------------- 6) Device-side precompute (weights + inv_norm) ---------------

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
  const float inv_norm = inv_norm_s;

  for (int i = threadIdx.x; i < period; i += blockDim.x) {
    weights_flat[combo * max_period + i] = w[i];
  }
  if (threadIdx.x == 0) inv_norms[combo] = inv_norm;
}
