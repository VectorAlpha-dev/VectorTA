// CUDA kernels for Know Sure Thing (KST)
//
// Implements two entry points mirroring our Rust CUDA wrapper:
// - kst_batch_f32:      one series × many parameter combinations (rows = combos)
// - kst_many_series_one_param_f32: many series (time-major) × one parameter set
//
// Numeric policy:
// - FP32 arithmetic and storage
// - Warmup/NaN semantics identical to scalar implementation
// - safe_roc: returns 0 when inputs are non-finite or previous value is zero

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float kst_safe_roc(float curr, float prev) {
  if (prev != 0.0f && isfinite(curr) && isfinite(prev)) {
    return ((curr / prev) - 1.0f) * 100.0f;
  }
  return 0.0f;
}

extern "C" __global__
void kst_batch_f32(const float* __restrict__ prices,
                   const int*   __restrict__ s1s,
                   const int*   __restrict__ s2s,
                   const int*   __restrict__ s3s,
                   const int*   __restrict__ s4s,
                   const int*   __restrict__ r1s,
                   const int*   __restrict__ r2s,
                   const int*   __restrict__ r3s,
                   const int*   __restrict__ r4s,
                   const int*   __restrict__ sigs,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out_line,
                   float* __restrict__ out_signal) {
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float nn = nanf("");

  for (int combo = tid; combo < n_combos; combo += stride) {
    const int s1  = s1s[combo];
    const int s2  = s2s[combo];
    const int s3  = s3s[combo];
    const int s4  = s4s[combo];
    const int r1  = r1s[combo];
    const int r2  = r2s[combo];
    const int r3  = r3s[combo];
    const int r4  = r4s[combo];
    const int sig = sigs[combo];

    const float inv1 = (s1 > 0) ? (1.0f / float(s1)) : 0.0f;
    const float w2   = (s2 > 0) ? (2.0f / float(s2)) : 0.0f;
    const float w3   = (s3 > 0) ? (3.0f / float(s3)) : 0.0f;
    const float w4   = (s4 > 0) ? (4.0f / float(s4)) : 0.0f;

    const int start1 = first_valid + r1;
    const int start2 = first_valid + r2;
    const int start3 = first_valid + r3;
    const int start4 = first_valid + r4;
    const int warm_line = max(start1 + s1 - 1,
                          max(start2 + s2 - 1,
                          max(start3 + s3 - 1,
                              start4 + s4 - 1)));
    const int warm_sig  = warm_line + sig - 1;

    float* line_row   = out_line   + combo * series_len;
    float* signal_row = out_signal + combo * series_len;

    // Initialize NaN prefixes where required
    const int nan_end_line = (warm_line < series_len ? warm_line : series_len);
    for (int i = 0; i < nan_end_line; ++i) line_row[i] = nn;
    const int nan_end_sig = (warm_sig < series_len ? warm_sig : series_len);
    for (int i = 0; i < nan_end_sig; ++i) signal_row[i] = nn;

    float sum1 = 0.f, sum2 = 0.f, sum3 = 0.f, sum4 = 0.f;
    float ssum = 0.f;

    for (int i = first_valid; i < series_len; ++i) {
      const float x = prices[i];
      if (i >= start1) {
        const float v = kst_safe_roc(x, prices[i - r1]);
        if (i < start1 + s1) sum1 += v; else sum1 += v - kst_safe_roc(prices[i - s1], prices[i - s1 - r1]);
      }
      if (i >= start2) {
        const float v = kst_safe_roc(x, prices[i - r2]);
        if (i < start2 + s2) sum2 += v; else sum2 += v - kst_safe_roc(prices[i - s2], prices[i - s2 - r2]);
      }
      if (i >= start3) {
        const float v = kst_safe_roc(x, prices[i - r3]);
        if (i < start3 + s3) sum3 += v; else sum3 += v - kst_safe_roc(prices[i - s3], prices[i - s3 - r3]);
      }
      if (i >= start4) {
        const float v = kst_safe_roc(x, prices[i - r4]);
        if (i < start4 + s4) sum4 += v; else sum4 += v - kst_safe_roc(prices[i - s4], prices[i - s4 - r4]);
      }

      if (i >= warm_line) {
        const float k = sum1 * inv1 + sum2 * w2 + sum3 * w3 + sum4 * w4;
        line_row[i] = k;
        if (i < warm_sig) {
          ssum += k;
        } else {
          ssum += k - line_row[i - sig];
          signal_row[i] = ssum / float(sig);
        }
      }
    }
  }
}

extern "C" __global__
void kst_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int num_series,
                                   int series_len,
                                   int s1, int s2, int s3, int s4,
                                   int r1, int r2, int r3, int r4,
                                   int sig,
                                   const int* __restrict__ first_valids,
                                   float* __restrict__ out_line_tm,
                                   float* __restrict__ out_signal_tm) {
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float nn = nanf("");

  const float inv1 = (s1 > 0) ? (1.0f / float(s1)) : 0.0f;
  const float w2   = (s2 > 0) ? (2.0f / float(s2)) : 0.0f;
  const float w3   = (s3 > 0) ? (3.0f / float(s3)) : 0.0f;
  const float w4   = (s4 > 0) ? (4.0f / float(s4)) : 0.0f;

  for (int s = tid; s < num_series; s += stride) {
    int fv = first_valids[s];
    if (fv < 0) fv = 0;
    if (fv >= series_len) {
      for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + s;
        out_line_tm[idx] = nn;
        out_signal_tm[idx] = nn;
      }
      continue;
    }

    const int start1 = fv + r1;
    const int start2 = fv + r2;
    const int start3 = fv + r3;
    const int start4 = fv + r4;
    const int warm_line = max(start1 + s1 - 1,
                          max(start2 + s2 - 1,
                          max(start3 + s3 - 1,
                              start4 + s4 - 1)));
    const int warm_sig  = warm_line + sig - 1;

    for (int t = 0; t < warm_line && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_line_tm[idx] = nn;
    }
    for (int t = 0; t < warm_sig && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_signal_tm[idx] = nn;
    }

    float sum1 = 0.f, sum2 = 0.f, sum3 = 0.f, sum4 = 0.f;
    float ssum = 0.f;

    for (int t = fv; t < series_len; ++t) {
      const float x = prices_tm[t * num_series + s];
      if (t >= start1) {
        float prev = prices_tm[(t - r1) * num_series + s];
        float v = kst_safe_roc(x, prev);
        if (t < start1 + s1) sum1 += v; else {
          float old = kst_safe_roc(prices_tm[(t - s1) * num_series + s],
                                   prices_tm[(t - s1 - r1) * num_series + s]);
          sum1 += v - old;
        }
      }
      if (t >= start2) {
        float prev = prices_tm[(t - r2) * num_series + s];
        float v = kst_safe_roc(x, prev);
        if (t < start2 + s2) sum2 += v; else {
          float old = kst_safe_roc(prices_tm[(t - s2) * num_series + s],
                                   prices_tm[(t - s2 - r2) * num_series + s]);
          sum2 += v - old;
        }
      }
      if (t >= start3) {
        float prev = prices_tm[(t - r3) * num_series + s];
        float v = kst_safe_roc(x, prev);
        if (t < start3 + s3) sum3 += v; else {
          float old = kst_safe_roc(prices_tm[(t - s3) * num_series + s],
                                   prices_tm[(t - s3 - r3) * num_series + s]);
          sum3 += v - old;
        }
      }
      if (t >= start4) {
        float prev = prices_tm[(t - r4) * num_series + s];
        float v = kst_safe_roc(x, prev);
        if (t < start4 + s4) sum4 += v; else {
          float old = kst_safe_roc(prices_tm[(t - s4) * num_series + s],
                                   prices_tm[(t - s4 - r4) * num_series + s]);
          sum4 += v - old;
        }
      }

      if (t >= warm_line) {
        const float k = sum1 * inv1 + sum2 * w2 + sum3 * w3 + sum4 * w4;
        int idx = t * num_series + s;
        out_line_tm[idx] = k;
        if (t < warm_sig) {
          ssum += k;
        } else {
          ssum += k - out_line_tm[(t - sig) * num_series + s];
          out_signal_tm[idx] = ssum / float(sig);
        }
      }
    }
  }
}

