// CUDA kernels for Bill Williams Alligator (three SMMA lines with forward shift)
//
// Category: Recurrence/IIR. Each output line is a smoothed moving average (SMMA)
// with its own period and forward shift (offset). We compute the three recurrences
// in a single pass over prices to minimize global loads. Warmup/NaN semantics
// match the scalar Rust implementation in src/indicators/alligator.rs:
//   - Warm index per line = first_valid + period - 1 + offset
//   - Write NaN before warm; at warm write the initial average; thereafter use
//     the SMMA recurrence prev = (1-1/p)*prev + (1/p)*price[i]
//   - Writes occur at shifted indices t = i + offset

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

static __device__ __forceinline__ unsigned lane_id() {
  return threadIdx.x & (WARP_SIZE - 1);
}

static __device__ __forceinline__ int warp_reduce_max(int v) {
  const unsigned full = 0xFFFFFFFFu;
  for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1) {
    int other = __shfl_down_sync(full, v, ofs);
    v = max(v, other);
  }
  return v;
}

extern "C" __global__
void alligator_batch_f32(const float* __restrict__ prices,
                         const int*   __restrict__ jaw_periods,
                         const int*   __restrict__ jaw_offsets,
                         const int*   __restrict__ teeth_periods,
                         const int*   __restrict__ teeth_offsets,
                         const int*   __restrict__ lips_periods,
                         const int*   __restrict__ lips_offsets,
                         int first_valid,
                         int series_len,
                         int n_combos,
                         float* __restrict__ out_jaw,
                         float* __restrict__ out_teeth,
                         float* __restrict__ out_lips) {

  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned full = 0xFFFFFFFFu;
  const float nan_f = __int_as_float(0x7fffffff);

  // Per-thread parameters
  int pj = 0, pt = 0, pl = 0;
  int oj = 0, ot = 0, ol = 0;
  int base = 0;
  bool valid = false;
  float aj = 0.f, bj = 0.f; // jaw alpha/beta
  float at = 0.f, bt = 0.f; // teeth
  float al = 0.f, bl = 0.f; // lips

  if (combo < n_combos) {
    pj = jaw_periods[combo];
    pt = teeth_periods[combo];
    pl = lips_periods[combo];
    oj = jaw_offsets[combo];
    ot = teeth_offsets[combo];
    ol = lips_offsets[combo];
    base = combo * series_len;
    valid = (pj > 0) & (pt > 0) & (pl > 0);
    if (valid) {
      aj = 1.0f / float(pj); bj = 1.0f - aj;
      at = 1.0f / float(pt); bt = 1.0f - at;
      al = 1.0f / float(pl); bl = 1.0f - al;
    }
  }

  // Initialize outputs with NaN up to warm indices (clamped). Do this once
  // per thread to ensure prefixes are set even when writes are sparse due to offsets.
  if (combo < n_combos && valid) {
    const int warm_base_j = first_valid + pj - 1;
    const int warm_base_t = first_valid + pt - 1;
    const int warm_base_l = first_valid + pl - 1;
    const int warm_j = min(series_len, warm_base_j + oj);
    const int warm_t = min(series_len, warm_base_t + ot);
    const int warm_l = min(series_len, warm_base_l + ol);
    for (int i = 0; i < warm_j; ++i) out_jaw[base + i] = nan_f;
    for (int i = 0; i < warm_t; ++i) out_teeth[base + i] = nan_f;
    for (int i = 0; i < warm_l; ++i) out_lips[base + i] = nan_f;
    // For indices >= warm_* we will write progressively below; cells beyond last write
    // keep their initial value; callers generally compare only within valid window.
  }

  // Compute initial averages for each line at i = first_valid + period - 1, using a single
  // broadcasted pass up to max(pj, pt, pl).
  float prev_j = 0.f, prev_t = 0.f, prev_l = 0.f;
  if (combo < n_combos && valid) {
    const int warm_base_j = first_valid + pj - 1;
    const int warm_base_t = first_valid + pt - 1;
    const int warm_base_l = first_valid + pl - 1;
    const int maxP = warp_reduce_max(max(pj, max(pt, pl)));
    const int leader = 0; // broadcast lane
    float sum_j = 0.f, sum_t = 0.f, sum_l = 0.f;
    for (int k = 0; k < maxP; ++k) {
      float v = 0.f;
      if (lane_id() == (unsigned)leader) {
        v = prices[first_valid + k];
      }
      const float pk = __shfl_sync(full, v, leader);
      if (k < pj) sum_j += pk;
      if (k < pt) sum_t += pk;
      if (k < pl) sum_l += pk;
    }
    prev_j = sum_j * aj;
    prev_t = sum_t * at;
    prev_l = sum_l * al;

    // Write initial averages at shifted indices t = warm_base + offset (if in range)
    const int tj = warm_base_j + oj;
    const int tt = warm_base_t + ot;
    const int tl = warm_base_l + ol;
    if (tj < series_len) out_jaw[base + tj] = prev_j;
    if (tt < series_len) out_teeth[base + tt] = prev_t;
    if (tl < series_len) out_lips[base + tl] = prev_l;

    // Continue recurrence for i = warm_base+1.., writing at shifted indices.
    const int start_i = min(min(warm_base_j, warm_base_t), warm_base_l) + 1;
    for (int i = start_i; i < series_len; ++i) {
      float v2 = 0.f;
      if (lane_id() == (unsigned)leader) {
        v2 = prices[i];
      }
      const float px = __shfl_sync(full, v2, leader);
      // Update only after each line's warm_base
      if (i > warm_base_j) {
        prev_j = fmaf(prev_j, bj, px * aj);
      }
      if (i > warm_base_t) {
        prev_t = fmaf(prev_t, bt, px * at);
      }
      if (i > warm_base_l) {
        prev_l = fmaf(prev_l, bl, px * al);
      }
      const int wj = i + oj;
      const int wt = i + ot;
      const int wl = i + ol;
      if (wj < series_len && i >= warm_base_j) out_jaw[base + wj] = prev_j;
      if (wt < series_len && i >= warm_base_t) out_teeth[base + wt] = prev_t;
      if (wl < series_len && i >= warm_base_l) out_lips[base + wl] = prev_l;
    }
  }
}

extern "C" __global__
void alligator_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                         int jaw_period,
                                         int jaw_offset,
                                         int teeth_period,
                                         int teeth_offset,
                                         int lips_period,
                                         int lips_offset,
                                         int num_series,
                                         int series_len,
                                         const int* __restrict__ first_valids,
                                         float* __restrict__ out_jaw_tm,
                                         float* __restrict__ out_teeth_tm,
                                         float* __restrict__ out_lips_tm) {
  const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (series_idx >= num_series) return;

  const int first = first_valids[series_idx];
  if (first < 0 || first >= series_len) return;
  if (jaw_period <= 0 || teeth_period <= 0 || lips_period <= 0) return;

  const int warm_base_j = first + jaw_period - 1;
  const int warm_base_t = first + teeth_period - 1;
  const int warm_base_l = first + lips_period - 1;
  const int warm_j = warm_base_j + jaw_offset;
  const int warm_t = warm_base_t + teeth_offset;
  const int warm_l = warm_base_l + lips_offset;

  const float aj = 1.0f / float(jaw_period);
  const float bj = 1.0f - aj;
  const float at = 1.0f / float(teeth_period);
  const float bt = 1.0f - at;
  const float al = 1.0f / float(lips_period);
  const float bl = 1.0f - al;

  const size_t stride = size_t(num_series);
  const size_t col    = size_t(series_idx);
  const float nan_f = __int_as_float(0x7fffffff);

  // Fill NaN prefixes
  for (int t = 0; t < min(series_len, warm_j); ++t)
    out_jaw_tm[size_t(t) * stride + col] = nan_f;
  for (int t = 0; t < min(series_len, warm_t); ++t)
    out_teeth_tm[size_t(t) * stride + col] = nan_f;
  for (int t = 0; t < min(series_len, warm_l); ++t)
    out_lips_tm[size_t(t) * stride + col] = nan_f;

  if (warm_base_j >= series_len && warm_base_t >= series_len && warm_base_l >= series_len) return;

  // Initial sums over [first .. first+period)
  size_t idx = size_t(first) * stride + col;
  float sum_j = 0.f, sum_t = 0.f, sum_l = 0.f;
  for (int k = 0; k < max(max(jaw_period, teeth_period), lips_period); ++k) {
    const float px = prices_tm[idx];
    if (k < jaw_period)  sum_j += px;
    if (k < teeth_period) sum_t += px;
    if (k < lips_period)  sum_l += px;
    idx += stride;
  }
  float prev_j = sum_j * aj;
  float prev_t = sum_t * at;
  float prev_l = sum_l * al;
  if (warm_j < series_len) out_jaw_tm[size_t(warm_j) * stride + col] = prev_j;
  if (warm_t < series_len) out_teeth_tm[size_t(warm_t) * stride + col] = prev_t;
  if (warm_l < series_len) out_lips_tm[size_t(warm_l) * stride + col] = prev_l;

  // Continue recurrence over source time i; write at shifted t = i + offset
  int start_i = min(min(warm_base_j, warm_base_t), warm_base_l) + 1;
  size_t t_idx = size_t(start_i) * stride + col;
  for (int i = start_i; i < series_len; ++i, t_idx += stride) {
    const float px = prices_tm[t_idx];
    if (i > warm_base_j) prev_j = fmaf(prev_j, bj, px * aj);
    if (i > warm_base_t) prev_t = fmaf(prev_t, bt, px * at);
    if (i > warm_base_l) prev_l = fmaf(prev_l, bl, px * al);
    const int tj = i + jaw_offset;
    const int tt = i + teeth_offset;
    const int tl = i + lips_offset;
    if (tj < series_len) out_jaw_tm[size_t(tj) * stride + col] = prev_j;
    if (tt < series_len) out_teeth_tm[size_t(tt) * stride + col] = prev_t;
    if (tl < series_len) out_lips_tm[size_t(tl) * stride + col] = prev_l;
  }
}

