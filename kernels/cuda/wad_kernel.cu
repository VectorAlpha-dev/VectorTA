// Optimized CUDA kernels for Williams Accumulation/Distribution (WAD)
//
// Semantics:
// - No parameters; cumulative running sum with warmup value 0.0 at index 0
// - NaNs propagate naturally via arithmetic (no special casing)
// - True Range High/Low and branch logic match scalar reference
//
// Design changes vs. original:
// - Remove FP64 from hot path; use compensated FP32 (KBN) accumulator
// - Grid-stride mapping for batch kernel to raise occupancy
// - Same public kernel names; two optional helpers at bottom for "compute once + broadcast"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// -------------------------------
// Numeric helper: KBN accumulator
// -------------------------------
// Kahan–Babuška–Neumaier compensated summation in FP32.
// Provides better accumulation accuracy than naive FP32 without FP64 cost.
// Ref: Wikipedia "Kahan summation algorithm" (Neumaier improvement).
// (We use a branch on |sum| vs |x|; per-thread branch divergence is negligible.)
struct KBNAcc32 {
  float sum;
  float c;
  __device__ __forceinline__ KBNAcc32() : sum(0.f), c(0.f) {}

  __device__ __forceinline__ void add(float x) {
    float t = sum + x;
    // Neumaier: pick correction based on magnitudes
    float e = (fabsf(sum) >= fabsf(x)) ? (sum - t) + x : (x - t) + sum;
    c += e;
    sum = t;
  }

  __device__ __forceinline__ float value() const { return sum + c; }
};

// Core per-step WAD delta, preserving the original branch and NaN semantics.
// NOTE: use ternary instead of fmaxf/fminf to mirror the "NaNs don't dominate" scalar behavior.
__device__ __forceinline__ float wad_step(float hi, float lo, float c, float pc) {
  const float trh = (pc > hi) ? pc : hi;
  const float trl = (pc < lo) ? pc : lo;

  float ad = 0.0f;
  if (c > pc)       ad = c - trl;
  else if (c < pc)  ad = c - trh;
  return ad;
}

// -----------------------------------------------------------
// One-series × many-params (batch). Each thread handles a row
// via a grid-stride loop over combos. This preserves sequential
// dependency within a row and offers high occupancy across rows.
// -----------------------------------------------------------
extern "C" __global__ void wad_batch_f32(
    const float* __restrict__ high,   // [series_len]
    const float* __restrict__ low,    // [series_len]
    const float* __restrict__ close,  // [series_len]
    int series_len,
    int n_combos,
    float* __restrict__ out) {        // [n_combos * series_len]

  if (series_len <= 0 || n_combos <= 0) return;

  const int tpb = blockDim.x * gridDim.x;
  int combo = blockIdx.x * blockDim.x + threadIdx.x;

  for (; combo < n_combos; combo += tpb) {
    float* __restrict__ out_row = out + combo * series_len;

    // Warmup
    out_row[0] = 0.0f;
    KBNAcc32 acc;
    float pc = close[0];

    // Sequential scan over the single price series
    #pragma unroll 1
    for (int i = 1; i < series_len; ++i) {
      const float ad = wad_step(high[i], low[i], close[i], pc);
      acc.add(ad);
      out_row[i] = acc.value();
      pc = close[i];
    }
  }
}

// -------------------------------------------------------------
// Many-series × one-param (time-major): thread-per-series.
// Accesses at each t are coalesced across threads (stride=cols).
// -------------------------------------------------------------
extern "C" __global__ void wad_many_series_one_param_f32(
    const float* __restrict__ high_tm,   // [rows * cols], rows=time
    const float* __restrict__ low_tm,    // [rows * cols]
    const float* __restrict__ close_tm,  // [rows * cols]
    int cols,   // number of series
    int rows,   // series length
    float* __restrict__ out_tm) {        // [rows * cols]

  if (rows <= 0 || cols <= 0) return;

  // Grid-stride over series to cover any cols
  const int stride_series = blockDim.x * gridDim.x;
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride_series) {
    const int stride = cols;
    // Warmup
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

// ----------------------------------------------------------
// Back-compat: series-major helper (row-major per series).
// Each thread processes one contiguous series.
// ----------------------------------------------------------
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

// -------------------------------------------------------------------
// OPTIONAL helpers (for wrapper optimization):
// 1) Compute the single WAD row once
// 2) Broadcast that row to all combos (for WAD, rows are identical)
//
// These let you precompute once and then replicate cheaply when
// n_combos>1, which avoids redundant work across combos.
// -------------------------------------------------------------------

extern "C" __global__ void wad_compute_single_row_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int series_len,
    float* __restrict__ out_row) { // [series_len]
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

// Broadcast one row into [n_combos * series_len] (row-major).
extern "C" __global__ void broadcast_row_f32(
    const float* __restrict__ row, // [series_len]
    int series_len,
    int n_combos,
    float* __restrict__ out) {     // [n_combos * series_len]
  if (series_len <= 0 || n_combos <= 0) return;
  const int n = series_len * n_combos;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int j = idx % series_len;      // position within the row
    out[idx] = row[j];
  }
}
