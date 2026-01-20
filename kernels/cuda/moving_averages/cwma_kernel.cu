















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


#ifndef CWMA_USE_ASYNC_COPY
#define CWMA_USE_ASYNC_COPY 1
#endif

#ifndef CWMA_WEIGHTS_OLDEST_FIRST
#define CWMA_WEIGHTS_OLDEST_FIRST 1
#endif


#ifndef CWMA_PAD_2D
#define CWMA_PAD_2D 1
#endif


#if CWMA_USE_ASYNC_COPY
#  if defined(__CUDACC__)
#    include <cooperative_groups.h>
#    include <cooperative_groups/memcpy_async.h>
#    include <cuda/pipeline>
     namespace cg = cooperative_groups;
#  endif
#endif



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


__device__ __forceinline__ float cwma_warp_sum(float v) {
  unsigned m = 0xffffffffu;
  v += __shfl_down_sync(m, v, 16);
  v += __shfl_down_sync(m, v,  8);
  v += __shfl_down_sync(m, v,  4);
  v += __shfl_down_sync(m, v,  2);
  v += __shfl_down_sync(m, v,  1);
  return v;
}


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


    const int total = TILE + wlen - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);


    const float* wsrc = weights_flat + combo * max_period;

    for (int i = threadIdx.x; i < wlen; i += TILE) { w[i] = wsrc[i]; }
    __syncthreads();
#if !CWMA_WEIGHTS_OLDEST_FIRST

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


    const int p0 = t0 - (wlen - 1);
    for (int dt = threadIdx.x; dt < total; dt += TILE) {
      int t = p0 + dt;
      float val = 0.f;
      if (t >= 0 && t < series_len) val = prices[t];
      tile[dt] = val;
    }
    __syncthreads();

    int t = t0 + threadIdx.x;
    if (t >= series_len) return;
    int out_idx = combo_base + t;
    if (t < warm) {
      out[out_idx] = NAN;
      return;
    }


    int start = threadIdx.x;
    float acc = cwma_dot(&tile[start], w, wlen);

    out[out_idx] = __fmul_rn(acc, inv_norms[combo]);
  }
};



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
    if ((blockDim.x * 2) != TILE) return;

    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int wlen   = max(0, period - 1);

    const int t0 = blockIdx.x * TILE;
    if (t0 >= series_len) return;

    const int total = TILE + wlen - 1;
    extern __shared__ __align__(16) unsigned char shraw[];
    size_t off = 0;
    float* w = reinterpret_cast<float*>(shraw + off);
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);
    float* tile = reinterpret_cast<float*>(shraw + off);


    const float* wsrc = weights_flat + combo * max_period;
    for (int i = threadIdx.x; i < wlen; i += blockDim.x) w[i] = wsrc[i];
    __syncthreads();
#if !CWMA_WEIGHTS_OLDEST_FIRST

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


    int start = lane * 2;
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



template<int TILE, int STAGES >
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
    float* w = reinterpret_cast<float*>(shraw + off);
    off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);

    float* tile = reinterpret_cast<float*>(shraw + off);
    const int tile_span = total;


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


    for (int s = 0; s < STAGES; ++s) {
      pipe.producer_acquire();
      const int t0 = t_base + s * grid_tile_stride;
      const int p0 = t0 - (wlen - 1);

      for (int dt = lane; dt < tile_span; dt += blockDim.x) {
        const int tcur = p0 + dt;
        if (tcur >= 0 && tcur < series_len) {
          cuda::memcpy_async(&tile[s * tile_span + dt], &prices[tcur], sizeof(float), pipe);
        } else {

          tile[s * tile_span + dt] = 0.f;
        }
      }
      pipe.producer_commit();
    }


    while (t_base < series_len) {

      pipe.consumer_wait();
      __syncthreads();


      const float* tbuf = &tile[stage * tile_span];
      const int t_even  = t_base + (lane * 2);
      const int t_odd   = t_even + 1;
      if (t_even < series_len) {
        int start = lane * 2;
        float s0 = 0.f, s1 = 0.f;
        cwma_dot2_shared(tbuf, start, w, wlen, s0, s1);

        float out0 = NAN, out1 = NAN;
        if (t_even >= warm) out0 = __fmul_rn(s0, inv_norms[combo]);
        if (t_odd  <  series_len && t_odd >= warm) out1 = __fmul_rn(s1, inv_norms[combo]);

        out[combo_base + t_even] = out0;
        if (t_odd < series_len) out[combo_base + t_odd] = out1;
      }

      __syncthreads();
      pipe.consumer_release();


      pipe.producer_acquire();
      const int next_t0 = t_base + STAGES * grid_tile_stride;
      const int next_p0 = next_t0 - (wlen - 1);
      const int next_stage = stage;

      for (int dt = lane; dt < tile_span; dt += blockDim.x) {
        const int tcur = next_p0 + dt;
        if (tcur >= 0 && tcur < series_len) {
          cuda::memcpy_async(&tile[next_stage * tile_span + dt], &prices[tcur], sizeof(float), pipe);
        } else {
          tile[next_stage * tile_span + dt] = 0.f;
        }
      }
      pipe.producer_commit();


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
#if CWMA_WEIGHTS_OLDEST_FIRST

            const int start = t - weight_len + 1;
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                const int in_idx = (start + k) * num_series + series_idx;
                float term = __fmaf_rn(prices_tm[in_idx], shared_weights[k], 0.0f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
#else
            #pragma unroll 4
            for (int k = 0; k < weight_len; ++k) {
                const int in_idx = (t - k) * num_series + series_idx;
                float term = __fmaf_rn(prices_tm[in_idx], shared_weights[k], 0.0f);
                float y = term - c;
                float u = s + y;
                c = (u - s) - y;
                s = u;
            }
#endif
            out_tm[out_idx] = __fmul_rn(s, inv_norm);
        }
        t += stride;
    }
}



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


  const int total = TX_ + wlen - 1;
  extern __shared__ __align__(16) unsigned char shraw[];
  size_t off = 0;
  float* w = reinterpret_cast<float*>(shraw + off);
  off = cwma_align_up(off + size_t(wlen) * sizeof(float), 16);

  constexpr int PAD = (CWMA_PAD_2D && (32 % TY_ == 0)) ? 1 : 0;
  const int STRIDE = TY_ + PAD;
  float* tile = reinterpret_cast<float*>(shraw + off);



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


  const bool vec_ok = (TY_ == 4) && ((num_series & 3) == 0) && ((s0 & 3) == 0);

  const int p0 = t0 - (wlen - 1);
  for (int dt = threadIdx.x; dt < total; dt += blockDim.x) {
    int t = p0 + dt;
    if (t >= 0 && t < series_len) {
      if (vec_ok && threadIdx.y == 0) {

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


  int s = s0 + threadIdx.y;
  int t = t0 + threadIdx.x;
  if (s >= num_series || t >= series_len) return;

  int warm = first_valids[s] + wlen;
  int out_idx = t * num_series + s;

  if (t < warm) {
    out_tm[out_idx] = NAN;
    return;
  }

  int start = threadIdx.x;
  const float* xptr = &tile[start * STRIDE + threadIdx.y];

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


  for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
    float t = float(period - i);
    float w = t * t * t;
    weights_flat[off + i] = w;
  }
  __syncthreads();


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


    if (wlen > 0) {
      float s2 = 0.f;
      for (int i = 0; i < wlen; ++i) s2 = __fadd_rn(s2, weights_flat[off + i]);
      weights_flat[off + 0] = __fadd_rn(weights_flat[off + 0], __fsub_rn(1.0f, s2));
    }

    inv_norms[combo] = 1.0f;
  }
}



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


  for (int i = threadIdx.x; i < wlen; i += blockDim.x) {
    float t = float(i + 2);
    float w = t * t * t;
    weights_flat[off + i] = w;
  }
  __syncthreads();


  if (threadIdx.x == 0) {
    float s = 0.f, c = 0.f;
    for (int i = 0; i < wlen; ++i) {
      float y = weights_flat[off + i] - c;
      float u = s + y; c = (u - s) - y; s = u;
    }
    s = fmaxf(s, 1e-30f);
    float inv = __frcp_rn(s);
    for (int i = 0; i < wlen; ++i) weights_flat[off + i] = __fmul_rn(weights_flat[off + i], inv);


    if (wlen > 0) {
      float s2 = 0.f; for (int i = 0; i < wlen; ++i) s2 = __fadd_rn(s2, weights_flat[off + i]);
      weights_flat[off + (wlen - 1)] = __fadd_rn(weights_flat[off + (wlen - 1)], __fsub_rn(1.0f, s2));
    }
    inv_norms[combo] = 1.0f;
  }
}
