// CUDA kernels for the Cubic Weighted Moving Average (CWMA).
//
// This file provides a plain batched kernel, tiled batched variants (1x and
// 2x-per-thread), and many-series kernels (1D and 2D tiled) following the
// performance and determinism patterns used by the ALMA CUDA implementation.
//
// Notes
// - All kernels are FP32.
// - Weights are expected to be precomputed on host as w[k] = (period-k)^3 for
//   k in [0, period-2], i.e., weight_len = period-1, and optionally pre-scaled
//   by the inverse normalization to eliminate a per-output multiply. The
//   kernels below do not rely on inv_norm when weights are already scaled.
// - Outputs prior to warm = first_valid + (period-1) are NaN.
// - Dynamic shared memory regions are 16B aligned and we use float4 vectorized
//   global loads when alignment allows.

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---- Feature toggles -------------------------------------------------------
#ifndef CWMA_USE_ASYNC_COPY       // SM80+ cp.async via libcu++ pipeline
#define CWMA_USE_ASYNC_COPY 1
#endif

#ifndef CWMA_WEIGHTS_OLDEST_FIRST // Set to 1 if weights[] are oldest->newest
#define CWMA_WEIGHTS_OLDEST_FIRST 1
#endif

// Optional padding for 2D many-series tiles to avoid bank conflicts
#ifndef CWMA_PAD_2D
#define CWMA_PAD_2D 1
#endif

// ---- Asynchronous copy/pipeline (Ampere+) ----------------------------------
#if CWMA_USE_ASYNC_COPY
#  if defined(__CUDACC__)
#    include <cooperative_groups.h>
#    include <cooperative_groups/memcpy_async.h>
#    include <cuda/pipeline>
     namespace cg = cooperative_groups;
#  endif
#endif

// ------------------------------- Utilities ---------------------------------

#ifndef CWMA_ASSUME
#  if defined(__CUDA_ARCH__)
#    define CWMA_ASSUME(x) if (!(x)) __trap();
#  else
#    define CWMA_ASSUME(x) ((void)0)
#  endif
#endif

__device__ __forceinline__ size_t cwma_align_up(size_t x, size_t a) {
  return (x + (a - 1)) & ~(a - 1);
}

// Warp reduce sum (optional helper when needed, not used in current kernels)
__device__ __forceinline__ float cwma_warp_sum(float v) {
  unsigned m = 0xffffffffu;
  v += __shfl_down_sync(m, v, 16);
  v += __shfl_down_sync(m, v,  8);
  v += __shfl_down_sync(m, v,  4);
  v += __shfl_down_sync(m, v,  2);
  v += __shfl_down_sync(m, v,  1);
  return v;
}

// Dot product helpers (Kahan-Neumaier compensated accumulation by default)
#ifndef CWMA_UNROLL
#  define CWMA_UNROLL 4
#endif
#ifndef CWMA_COMPENSATED_DOT
#  define CWMA_COMPENSATED_DOT 1
#endif

__device__ __forceinline__
float cwma_dot_uncomp(const float* __restrict__ x,
                      const float* __restrict__ w, int n) {
  float s = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) s = __fmaf_rn(x[i], w[i], s);
  return s;
}

__device__ __forceinline__
float cwma_dot_comp(const float* __restrict__ x,
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
float cwma_dot(const float* __restrict__ x,
               const float* __restrict__ w, int n) {
#if CWMA_COMPENSATED_DOT
  return cwma_dot_comp(x, w, n);
#else
  return cwma_dot_uncomp(x, w, n);
#endif
}

// Fused two-output dot product using the same weights. Computes outputs at
// consecutive positions (b and b+1) from a shared buffer `buf`.
__device__ __forceinline__
void cwma_dot2_shared(const float* __restrict__ buf, int b,
                      const float* __restrict__ w, int n,
                      float& s0_out, float& s1_out) {
#if CWMA_COMPENSATED_DOT
  float s0 = 0.f, c0 = 0.f;
  float s1 = 0.f, c1 = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) {
    float wi = w[i];
    float t0 = __fmaf_rn(buf[b + i],     wi, 0.f);
    float y0 = t0 - c0;
    float u0 = s0 + y0;
    c0 = (u0 - s0) - y0;
    s0 = u0;

    float t1 = __fmaf_rn(buf[b + i + 1], wi, 0.f);
    float y1 = t1 - c1;
    float u1 = s1 + y1;
    c1 = (u1 - s1) - y1;
    s1 = u1;
  }
  s0_out = s0; s1_out = s1;
#else
  float s0 = 0.f, s1 = 0.f;
  #pragma unroll 4
  for (int i = 0; i < n; ++i) {
    float wi = w[i];
    s0 = __fmaf_rn(buf[b + i],     wi, s0);
    s1 = __fmaf_rn(buf[b + i + 1], wi, s1);
  }
  s0_out = s0; s1_out = s1;
#endif
}

// ---------------------- 1) Plain batched kernel ----------------------------

extern "C" __global__
void cwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ weights_flat,
                    const int* __restrict__ periods,
                    const float* __restrict__ inv_norms,
                    int max_period,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out) {
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int weight_len = (period > 0) ? (period - 1) : 0;
    const float inv_norm = inv_norms[combo];

    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < weight_len; i += blockDim.x) {
        shared_weights[i] = weights_flat[combo * max_period + i];
    }
    __syncthreads();

    const int warm = first_valid + weight_len;
    const int base_out = combo * series_len;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = base_out + t;
        if (t < warm) {
            out[out_idx] = NAN;
        } else {
            float s = 0.0f, c = 0.0f;
#if CWMA_WEIGHTS_OLDEST_FIRST
            // Process window in chronological order: [t - wlen + 1 .. t]
            const int start = t - weight_len + 1;
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                float term = __fmaf_rn(prices[start + k], shared_weights[k], 0.0f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
#else
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                float term = __fmaf_rn(prices[t - k], shared_weights[k], 0.0f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
#endif
            out[out_idx] = __fmul_rn(s, inv_norm);
        }
        t += stride;
    }
}

// -------- 2) Precomputed-weight tiled (template TILE), 1x/thread -----------

template<int TILE>
struct CwmaBatchTiledPrecomputed1x {
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

    const int period = periods[combo];
    const int wlen   = max(0, period - 1);

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    // Tile covers [t0 .. t0+TILE-1] and needs (wlen-1) history samples
    const int total = TILE + wlen - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);            // [wlen]
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);         // [TILE + wlen]

    // Load weights into shared (vectorized when aligned)
    const float* wsrc = weights_flat + combo * max_period;
    // Load weights, then optionally reverse so w[0] multiplies the oldest sample
    for (int i = threadIdx.x; i < wlen; i += TILE) { w[i] = wsrc[i]; }
    __syncthreads();
#if !CWMA_WEIGHTS_OLDEST_FIRST
    // Reverse to oldest-first so w[0] multiplies the oldest sample in the tile
    for (int i = threadIdx.x; i < (wlen >> 1); i += TILE) {
      float tmp = w[i];
      int j = wlen - 1 - i;
      w[i] = w[j];
      w[j] = tmp;
    }
    __syncthreads();
#endif

    const int warm = first_valid + wlen;
    const int combo_base = combo * series_len;

    // Synchronous cooperative load and compute for one tile
    const int p0 = t0 - (wlen - 1); // earliest needed sample for t0
    for (int dt = threadIdx.x; dt < total; dt += TILE) {
      int t = p0 + dt;
      float val = 0.f;
      if (t >= 0 && t < series_len) val = prices[t];
      tile[dt] = val; // zero-fill OOB
    }
    __syncthreads();

    int t = t0 + threadIdx.x;
    if (t >= series_len) return;
    int out_idx = combo_base + t;
    if (t < warm) {
      out[out_idx] = NAN;
      return;
    }

    // Compute dot on shared slice [t-wlen .. t]
    int start = threadIdx.x; // within tile
    float acc = cwma_dot(&tile[start], w, wlen);
    // If weights are pre-scaled, inv_norms[combo] should be 1.0
    out[out_idx] = __fmul_rn(acc, inv_norms[combo]);
  }
};

// -------- 3) Precomputed-weight tiled (template TILE), 2x/thread ------------

template<int TILE>
struct CwmaBatchTiledPrecomputed2x {
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
    if ((blockDim.x * 2) != TILE) return; // half threads, two outputs each

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int wlen   = max(0, period - 1);

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    const int total = TILE + wlen - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);            // [wlen]
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);         // [TILE + wlen]

    // Load weights into shared (vectorized when aligned)
    const float* wsrc = weights_flat + combo * max_period;
    for (int i = threadIdx.x; i < wlen; i += blockDim.x) w[i] = wsrc[i];
    __syncthreads();
#if !CWMA_WEIGHTS_OLDEST_FIRST
    // Reverse to oldest-first so w[0] multiplies the oldest sample in the tile
    for (int i = threadIdx.x; i < (wlen >> 1); i += blockDim.x) {
      float tmp = w[i];
      int j = wlen - 1 - i;
      w[i] = w[j];
      w[j] = tmp;
    }
    __syncthreads();
#endif

    const int warm = first_valid + wlen;
    const int combo_base = combo * series_len;

    // Load tile into shared (simple cooperative loop to avoid edge holes)
    const int p0 = t0 - (wlen - 1);
    for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
      int tcur = p0 + dt;
      float v = 0.f;
      if (tcur >= 0 && tcur < series_len) v = prices[tcur];
      tile[dt] = v;
    }
    __syncthreads();

    int lane = threadIdx.x;
    int t_even = t0 + (lane * 2);
    int t_odd  = t_even + 1;
    if (t_even >= series_len) return;
    int out_even = combo_base + t_even;
    int out_odd  = combo_base + t_odd;

    // Compute two outputs with proper frontier handling
    int start = lane * 2; // within tile (earliest sample)
    float s0 = 0.f, s1 = 0.f;
    cwma_dot2_shared(tile, start, w, wlen, s0, s1);
    float out0 = NAN, out1 = NAN;
    if (t_even >= warm) {
      out0 = __fmul_rn(s0, inv_norms[combo]);
    }
    if (t_odd < series_len && t_odd >= warm) {
      out1 = __fmul_rn(s1, inv_norms[combo]);
    }
    out[out_even] = out0;
    if (t_odd < series_len) out[out_odd] = out1;
  }
};

#define DEFINE_CWMA_BATCH_TILED_1X(NAME, TILE)                                       \
extern "C" __global__ void NAME(                                                     \
  const float* __restrict__ prices,                                                  \
  const float* __restrict__ weights_flat,                                            \
  const int*   __restrict__ periods,                                                 \
  const float* __restrict__ inv_norms,                                               \
  int max_period, int series_len, int n_combos, int first_valid,                     \
  float* __restrict__ out) {                                                         \
  CwmaBatchTiledPrecomputed1x<TILE>::run(prices, weights_flat, periods, inv_norms,   \
                                         max_period, series_len, n_combos,           \
                                         first_valid, out);                          \
}

#define DEFINE_CWMA_BATCH_TILED_2X(NAME, TILE)                                       \
extern "C" __global__ void NAME(                                                     \
  const float* __restrict__ prices,                                                  \
  const float* __restrict__ weights_flat,                                            \
  const int*   __restrict__ periods,                                                 \
  const float* __restrict__ inv_norms,                                               \
  int max_period, int series_len, int n_combos, int first_valid,                     \
  float* __restrict__ out) {                                                         \
  CwmaBatchTiledPrecomputed2x<TILE>::run(prices, weights_flat, periods, inv_norms,   \
                                         max_period, series_len, n_combos,           \
                                         first_valid, out);                          \
}

DEFINE_CWMA_BATCH_TILED_1X(cwma_batch_tiled_f32_tile128, 128)
DEFINE_CWMA_BATCH_TILED_1X(cwma_batch_tiled_f32_tile256, 256)
DEFINE_CWMA_BATCH_TILED_2X(cwma_batch_tiled_f32_2x_tile128, 128)
DEFINE_CWMA_BATCH_TILED_2X(cwma_batch_tiled_f32_2x_tile256, 256)

// -------- 3b) Async double-buffered 2x/thread kernel (Ampere/Ada+) ---------

template<int TILE, int STAGES /*2 recommended*/>
struct CwmaBatchTiledPrecomputed2xAsync {
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

#if !CWMA_USE_ASYNC_COPY || (__CUDA_ARCH__ < 800)
    // Fallback to the synchronous 2x implementation
    CwmaBatchTiledPrecomputed2x<TILE>::run(prices, weights_flat, periods, inv_norms,
                                           max_period, series_len, n_combos, first_valid, out);
    return;
#else
    static_assert(TILE % 2 == 0, "TILE must be even (2 outputs/thread)");
    if ((blockDim.x * 2) != TILE) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int wlen   = max(0, period - 1);
    const int total  = TILE + wlen - 1;

    const int combo_base = combo * series_len;
    const int warm = first_valid + wlen;

    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);               // [wlen]
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
    // Two stage double buffer for tiles
    float* tile = reinterpret_cast<float*>(shraw + off);            // [STAGES * (TILE + wlen)]
    const int tile_span = total;

    // Load weights (and conditionally reverse)
    const float* wsrc = weights_flat + combo * max_period;
    for (int i = threadIdx.x; i < wlen; i += blockDim.x) w[i] = wsrc[i];
    __syncthreads();
#if !CWMA_WEIGHTS_OLDEST_FIRST
    for (int i = threadIdx.x; i < (wlen >> 1); i += blockDim.x) {
      float tmp = w[i]; int j = wlen - 1 - i; w[i] = w[j]; w[j] = tmp;
    }
    __syncthreads();
#endif

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto pipe = cuda::make_pipeline(block, &pss);

    const int lane = threadIdx.x;
    const int grid_tile_stride = gridDim.x * TILE;

    int t_base = blockIdx.x * TILE;
    int stage  = 0;

    // Preload up to STAGES tiles
    for (int s = 0; s < STAGES; ++s) {
      pipe.producer_acquire();
      const int t0 = t_base + s * grid_tile_stride;
      const int p0 = t0 - (wlen - 1);
      // Cooperative copy of the tile into tile[s]
      for (int dt = lane; dt < tile_span; dt += blockDim.x) {
        const int tcur = p0 + dt;
        if (tcur >= 0 && tcur < series_len) {
          cuda::memcpy_async(&tile[s * tile_span + dt], &prices[tcur], sizeof(float), pipe);
        } else {
          // OOB zero-fill (no async copy for invalid pointers)
          tile[s * tile_span + dt] = 0.f;
        }
      }
      pipe.producer_commit();
    }

    // Main loop over this CTA's tiles (grid-stride by TILE)
    while (t_base < series_len) {
      // Ensure current stage is ready
      pipe.consumer_wait();
      __syncthreads();

      // Compute the 2x outputs per thread from tile[stage]
      const float* tbuf = &tile[stage * tile_span];
      const int t_even  = t_base + (lane * 2);
      const int t_odd   = t_even + 1;
      if (t_even < series_len) {
        int start = lane * 2;  // within tile
        float s0 = 0.f, s1 = 0.f;
        cwma_dot2_shared(tbuf, start, w, wlen, s0, s1);

        float out0 = NAN, out1 = NAN;
        if (t_even >= warm) out0 = __fmul_rn(s0, inv_norms[combo]);
        if (t_odd  <  series_len && t_odd >= warm) out1 = __fmul_rn(s1, inv_norms[combo]);

        out[combo_base + t_even] = out0;
        if (t_odd < series_len) out[combo_base + t_odd] = out1;
      }

      __syncthreads();
      pipe.consumer_release();  // free current stage buffer

      // Preload the next tile STAGES*stride ahead to keep pipeline full
      pipe.producer_acquire();
      const int next_t0 = t_base + STAGES * grid_tile_stride;
      const int next_p0 = next_t0 - (wlen - 1);
      const int next_stage = stage;  // we rotate over same slot just released

      for (int dt = lane; dt < tile_span; dt += blockDim.x) {
        const int tcur = next_p0 + dt;
        if (tcur >= 0 && tcur < series_len) {
          cuda::memcpy_async(&tile[next_stage * tile_span + dt], &prices[tcur], sizeof(float), pipe);
        } else {
          tile[next_stage * tile_span + dt] = 0.f;
        }
      }
      pipe.producer_commit();

      // Advance to next tile in this CTA's grid-stride walk
      t_base += grid_tile_stride;
      stage   = (stage + 1) % STAGES;
    }
#endif
  }
};

#define DEFINE_CWMA_BATCH_TILED_2X_ASYNC(NAME, TILE)                                   \
extern "C" __global__ void NAME(                                                       \
  const float* __restrict__ prices,                                                    \
  const float* __restrict__ weights_flat,                                              \
  const int*   __restrict__ periods,                                                   \
  const float* __restrict__ inv_norms,                                                 \
  int max_period, int series_len, int n_combos, int first_valid,                       \
  float* __restrict__ out) {                                                           \
  CwmaBatchTiledPrecomputed2xAsync<TILE, 2>::run(prices, weights_flat, periods,        \
                                                 inv_norms, max_period, series_len,    \
                                                 n_combos, first_valid, out);          \
}

DEFINE_CWMA_BATCH_TILED_2X_ASYNC(cwma_batch_tiled_async_f32_2x_tile128, 128)
DEFINE_CWMA_BATCH_TILED_2X_ASYNC(cwma_batch_tiled_async_f32_2x_tile256, 256)

// ---------------------- 4) Many-series 1D kernel ---------------------------

extern "C" __global__
void cwma_multi_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const float* __restrict__ weights,
    int period,
    float inv_norm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int weight_len = (period > 0) ? (period - 1) : 0;

    extern __shared__ float shared_weights[];
    for (int i = threadIdx.x; i < weight_len; i += blockDim.x) {
        shared_weights[i] = weights[i];
    }
    __syncthreads();

    const int series_idx = blockIdx.y;
    if (series_idx >= num_series) return;

    const int warm = first_valids[series_idx] + weight_len;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int out_idx = t * num_series + series_idx;
        if (t < warm) {
            out_tm[out_idx] = NAN;
        } else {
            float s = 0.0f, c = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                const int in_idx = (t - k) * num_series + series_idx;
                float term = __fmaf_rn(prices_tm[in_idx], shared_weights[k], 0.0f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
            out_tm[out_idx] = __fmul_rn(s, inv_norm);
        }
        t += stride;
    }
}

// -------- 5) Many-series 2D tiled kernels (precomputed weights) ------------

template<int TX, int TY>
__device__ __forceinline__
void cwma_ms1p_tiled_core(const float* __restrict__ prices_tm,
                          const float* __restrict__ weights,
                          int period,
                          float inv_norm,
                          int num_series,
                          int series_len,
                          const int* __restrict__ first_valids,
                          float* __restrict__ out_tm) {
  const int TX_ = TX;
  const int TY_ = TY;
  const int wlen = max(0, period - 1);
  const int t0 = blockIdx.x * TX_;
  const int s0 = blockIdx.y * TY_;

  if (t0 >= series_len || s0 >= num_series) return;

  // Shared: weights + tile [ (TX + wlen) x (TY + PAD) ] if padding enabled
  const int total = TX_ + wlen - 1;
  extern __shared__ __align__(16) unsigned char shraw[];
  size_t off = 0;
  float* w = reinterpret_cast<float*>(shraw + off);
  off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
  // Add 1-column padding if enabled and TY_ divides 32 banks to avoid worst-case conflicts
  constexpr int PAD = (CWMA_PAD_2D && (32 % TY_ == 0)) ? 1 : 0;
  const int STRIDE = TY_ + PAD;
  float* tile = reinterpret_cast<float*>(shraw + off); // [total * STRIDE]

  // Load weights into shared (vectorized if aligned)
  // Load weights then (optionally) reverse so w[0] corresponds to oldest sample
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < wlen; i += blockDim.x * blockDim.y) {
    w[i] = weights[i];
  }
  __syncthreads();
  
#if !CWMA_WEIGHTS_OLDEST_FIRST
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < (wlen >> 1); i += blockDim.x * blockDim.y) {
    float tmp = w[i];
    int j = wlen - 1 - i;
    w[i] = w[j];
    w[j] = tmp;
  }
  __syncthreads();
#endif

  // Cooperative load of tile across series and time
  const bool vec_ok = (TY_ == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);

  const int p0 = t0 - (wlen - 1); // earliest sample needed for t0
  for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
    int t = p0 + dt;
    if (t >= 0 && t < series_len) {
      if (vec_ok && threadIdx.y == 0) {
        // Vectorized load across 4 contiguous series columns
        const float4* src4 = reinterpret_cast<const float4*>(&prices_tm[t * num_series + s0]);
        float4 v = src4[0];
        tile[dt * STRIDE + 0] = v.x;
        tile[dt * STRIDE + 1] = v.y;
        tile[dt * STRIDE + 2] = v.z;
        tile[dt * STRIDE + 3] = v.w;
      } else {
        int s = s0 + threadIdx.y;
        float val = 0.f;
        if (s < num_series) val = prices_tm[t * num_series + s];
        tile[dt * STRIDE + threadIdx.y] = val;
      }
    } else {
      int idx = dt * STRIDE + threadIdx.y;
      if (idx < total * STRIDE) tile[idx] = 0.f;
    }
  }
  __syncthreads();

  // Compute outputs for this CTA
  int s = s0 + threadIdx.y;
  int t = t0 + threadIdx.x;
  if (s >= num_series || t >= series_len) return;

  int warm = first_valids[s] + wlen;
  int out_idx = t * num_series + s;

  if (t < warm) {
    out_tm[out_idx] = NAN;
    return;
  }

  int start = threadIdx.x; // within tile
  const float* xptr = &tile[start * STRIDE + threadIdx.y];
  // strided dot across series dimension (stride = STRIDE) with compensation
  float s_acc = 0.f, c_acc = 0.f;
  #pragma unroll 4
  for (int i = 0; i < wlen; ++i) {
    float term = __fmaf_rn(xptr[i * STRIDE], w[i], 0.f);
    float y = term - c_acc;
    float u = s_acc + y;
    c_acc = (u - s_acc) - y;
    s_acc = u;
  }
  out_tm[out_idx] = __fmul_rn(s_acc, inv_norm);
}

#define DEFINE_CWMA_MS1P_TILED(NAME, TX, TY)                                         \
extern "C" __global__ void NAME(                                                     \
  const float* __restrict__ prices_tm,                                               \
  const float* __restrict__ weights,                                                 \
  int period, float inv_norm, int num_series, int series_len,                        \
  const int* __restrict__ first_valids, float* __restrict__ out_tm) {                \
  cwma_ms1p_tiled_core<TX, TY>(prices_tm, weights, period, inv_norm,                 \
                               num_series, series_len, first_valids, out_tm);        \
}

DEFINE_CWMA_MS1P_TILED(cwma_ms1p_tiled_f32_tx128_ty2, 128, 2)
DEFINE_CWMA_MS1P_TILED(cwma_ms1p_tiled_f32_tx128_ty4, 128, 4)

// -------- 6) On-device precompute (optional VRAM-first path) ----------------

extern "C" __global__
void cwma_precompute_weights_f32(const int* __restrict__ periods,
                                 int n_combos,
                                 int max_period,
                                 float* __restrict__ weights_flat,
                                 float* __restrict__ inv_norms) {
  const int combo = blockIdx.x;
  if (combo >= n_combos) return;

  const int period = periods[combo];
  const int wlen   = max(0, period - 1);
  const int off    = combo * max_period;

  // Build cubic weights without powf
  for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
    float t = float(period - i);
    float w = t * t * t;
    weights_flat[off + i] = w;
  }
  __syncthreads();

  // Block-serial normalization with Neumaier + exact-sum fix
  if (threadIdx.x == 0) {
    float s = 0.f, c = 0.f;
    for (int i = 0; i < wlen; ++i) {
      float y = weights_flat[off + i] - c;
      float u = s + y;
      c = (u - s) - y;
      s = u;
    }
    s = fmaxf(s, 1e-30f);
    float inv = __frcp_rn(s);
    for (int i = 0; i < wlen; ++i) {
      weights_flat[off + i] = __fmul_rn(weights_flat[off + i], inv);
    }

    // Make FP32 sum exactly 1.0f by correcting the largest tap
    if (wlen > 0) {
      float s2 = 0.f;
      for (int i = 0; i < wlen; ++i) s2 = __fadd_rn(s2, weights_flat[off + i]);
      weights_flat[off + 0] = __fadd_rn(weights_flat[off + 0], __fsub_rn(1.0f, s2));
    }

    inv_norms[combo] = 1.0f; // pre-scaled
  }
}

// -------- 7) On-device precompute (oldest-first layout variant) ------------

extern "C" __global__
void cwma_precompute_weights_oldest_first_f32(const int* __restrict__ periods,
                                              int n_combos,
                                              int max_period,
                                              float* __restrict__ weights_flat,
                                              float* __restrict__ inv_norms) {
  const int combo = blockIdx.x;
  if (combo >= n_combos) return;

  const int period = periods[combo];
  const int wlen   = max(0, period - 1);
  const int off    = combo * max_period;

  // Emit weights in oldest->newest order: w[i] = (i+2)^3, i in [0,wlen-1]
  for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
    float t = float(i + 2);
    float w = t * t * t;
    weights_flat[off + i] = w;
  }
  __syncthreads();

  // Normalize in FP32 using Neumaier compensation; force exact 1.0 sum.
  if (threadIdx.x == 0) {
    float s = 0.f, c = 0.f;
    for (int i = 0; i < wlen; ++i) {
      float y = weights_flat[off + i] - c;
      float u = s + y; c = (u - s) - y; s = u;
    }
    s = fmaxf(s, 1e-30f);
    float inv = __frcp_rn(s);
    for (int i = 0; i < wlen; ++i) weights_flat[off + i] = __fmul_rn(weights_flat[off + i], inv);

    // Exact sum fix: adjust the largest tap (last entry == newest sample)
    if (wlen > 0) {
      float s2 = 0.f; for (int i = 0; i < wlen; ++i) s2 = __fadd_rn(s2, weights_flat[off + i]);
      weights_flat[off + (wlen - 1)] = __fadd_rn(weights_flat[off + (wlen - 1)], __fsub_rn(1.0f, s2));
    }
    inv_norms[combo] = 1.0f; // already normalized
  }
}
