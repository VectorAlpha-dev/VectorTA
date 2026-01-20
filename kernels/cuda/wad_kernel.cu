











#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>








struct KBNAcc32 {
  float sum;
  float c;
  __device__ __forceinline__ KBNAcc32() : sum(0.f), c(0.f) {}

  __device__ __forceinline__ void add(float x) {
    float t = sum + x;

    float e = (fabsf(sum) >= fabsf(x)) ? (sum - t) + x : (x - t) + sum;
    c += e;
    sum = t;
  }

  __device__ __forceinline__ float value() const { return sum + c; }
};



__device__ __forceinline__ float wad_step(float hi, float lo, float c, float pc) {
  const float trh = (pc > hi) ? pc : hi;
  const float trl = (pc < lo) ? pc : lo;

  float ad = 0.0f;
  if (c > pc)       ad = c - trl;
  else if (c < pc)  ad = c - trh;
  return ad;
}






extern "C" __global__ void wad_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    int n_combos,
    float* __restrict__ out) {

  if (series_len <= 0 || n_combos <= 0) return;

  const int tpb = blockDim.x * gridDim.x;
  int combo = blockIdx.x * blockDim.x + threadIdx.x;

  for (; combo < n_combos; combo += tpb) {
    float* __restrict__ out_row = out + combo * series_len;


    out_row[0] = 0.0f;
    KBNAcc32 acc;
    float pc = close[0];


    #pragma unroll 1
    for (int i = 1; i < series_len; ++i) {
      const float ad = wad_step(high[i], low[i], close[i], pc);
      acc.add(ad);
      out_row[i] = acc.value();
      pc = close[i];
    }
  }
}





extern "C" __global__ void wad_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    float* __restrict__ out_tm) {

  if (rows <= 0 || cols <= 0) return;


  const int stride_series = blockDim.x * gridDim.x;
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride_series) {
    const int stride = cols;

    out_tm[0 * stride + s] = 0.0f;

    KBNAcc32 acc;
    float pc = close_tm[0 * stride + s];

    #pragma unroll 1
    for (int t = 1; t < rows; ++t) {
      const int idx = t * stride + s;
      const float ad = wad_step(high_tm[idx], low_tm[idx], close_tm[idx], pc);
      acc.add(ad);
      out_tm[idx] = acc.value();
      pc = close_tm[idx];
    }
  }
}





extern "C" __global__ void wad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int n_series,
    float* __restrict__ out) {

  if (len <= 0 || n_series <= 0) return;

  const int stride_series = blockDim.x * gridDim.x;
  for (int series = blockIdx.x * blockDim.x + threadIdx.x; series < n_series; series += stride_series) {
    const int offset = series * len;
    const float* high_row  = high  + offset;
    const float* low_row   = low   + offset;
    const float* close_row = close + offset;
    float* out_row         = out   + offset;

    out_row[0] = 0.0f;
    KBNAcc32 acc;
    float pc = close_row[0];

    #pragma unroll 1
    for (int i = 1; i < len; ++i) {
      const float ad = wad_step(high_row[i], low_row[i], close_row[i], pc);
      acc.add(ad);
      out_row[i] = acc.value();
      pc = close_row[i];
    }
  }
}










extern "C" __global__ void wad_compute_single_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    float* __restrict__ out_row) {
  if (series_len <= 0) return;
  out_row[0] = 0.0f;
  KBNAcc32 acc;
  float pc = close[0];
  #pragma unroll 1
  for (int i = 1; i < series_len; ++i) {
    const float ad = wad_step(high[i], low[i], close[i], pc);
    acc.add(ad);
    out_row[i] = acc.value();
    pc = close[i];
  }
}


extern "C" __global__ void broadcast_row_f32(
    const float* __restrict__ row,
    int series_len,
    int n_combos,
    float* __restrict__ out) {
  if (series_len <= 0 || n_combos <= 0) return;
  const int n = series_len * n_combos;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int j = idx % series_len;
    out[idx] = row[j];
  }
}
