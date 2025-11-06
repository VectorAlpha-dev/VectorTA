// CUDA kernels for Know Sure Thing (KST)
// Optimized: FP32 + compensated sums, FMA, fast qNaN, signal first-sample fix.
// Entry points unchanged:
// - kst_batch_f32
// - kst_many_series_one_param_f32

#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// Numerics helpers

// Cheap quiet-NaN: bitcast canonical qNaN (0x7fffffff) instead of nanf("").
// Avoids string parsing and function call overhead.
static __device__ __forceinline__ float kst_qnan() {
  return __int_as_float(0x7fffffff);
}

// Compensated accumulator (Kahan-Babuška-Neumaier).
// Adds with small extra cost, markedly better precision in FP32.
struct CompSum {
  float sum;
  float c; // compensation
  __device__ __forceinline__ void init() { sum = 0.f; c = 0.f; }
  __device__ __forceinline__ void add(float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  __device__ __forceinline__ float val() const { return sum; }
};

// ROC (%) = (curr/prev - 1) * 100, safe for zeros/non-finites
// Use FMA form: fmaf(curr, 100/prev, -100) when valid.
// Semantics: return 0 if prev==0 or non-finite inputs (your policy).
__device__ __forceinline__ float kst_safe_roc(float curr, float prev) {
  if (prev != 0.0f && isfinite(curr) && isfinite(prev)) {  // CUDA device isfinite() OK
    const float inv100_prev = 100.0f / prev;
    return __fmaf_rn(curr, inv100_prev, -100.0f);
  }
  return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel: one series × many parameter combos (rows=combinations)

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
  const float nn   = kst_qnan();

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
    const float invSig = (sig > 0) ? (1.0f / float(sig)) : 0.0f;

    const int start1 = first_valid + r1;
    const int start2 = first_valid + r2;
    const int start3 = first_valid + r3;
    const int start4 = first_valid + r4;

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    float* __restrict__ line_row   = out_line   + combo * series_len;
    float* __restrict__ signal_row = out_signal + combo * series_len;

    // NaN prefixes
    const int nan_end_line = (warm_line < series_len ? warm_line : series_len);
    for (int i = 0; i < nan_end_line; ++i) line_row[i] = nn;
    const int nan_end_sig = (warm_sig < series_len ? warm_sig : series_len);
    for (int i = 0; i < nan_end_sig; ++i) signal_row[i] = nn;

    CompSum sum1, sum2, sum3, sum4, ssum;
    sum1.init(); sum2.init(); sum3.init(); sum4.init(); ssum.init();

    for (int i = first_valid; i < series_len; ++i) {
      const float x = prices[i];

      if (i >= start1) {
        const float v = kst_safe_roc(x, prices[i - r1]);
        if (i < start1 + s1) sum1.add(v);
        else                 sum1.add(v - kst_safe_roc(prices[i - s1], prices[i - s1 - r1]));
      }
      if (i >= start2) {
        const float v = kst_safe_roc(x, prices[i - r2]);
        if (i < start2 + s2) sum2.add(v);
        else                 sum2.add(v - kst_safe_roc(prices[i - s2], prices[i - s2 - r2]));
      }
      if (i >= start3) {
        const float v = kst_safe_roc(x, prices[i - r3]);
        if (i < start3 + s3) sum3.add(v);
        else                 sum3.add(v - kst_safe_roc(prices[i - s3], prices[i - s3 - r3]));
      }
      if (i >= start4) {
        const float v = kst_safe_roc(x, prices[i - r4]);
        if (i < start4 + s4) sum4.add(v);
        else                 sum4.add(v - kst_safe_roc(prices[i - s4], prices[i - s4 - r4]));
      }

      if (i >= warm_line) {
        // k = sum1*(1/s1) + sum2*(2/s2) + sum3*(3/s3) + sum4*(4/s4)
        float k = __fmaf_rn(sum4.val(), w4,
                  __fmaf_rn(sum3.val(), w3,
                  __fmaf_rn(sum2.val(), w2, sum1.val() * inv1)));

        line_row[i] = k;

        // --- Signal rolling sum with bugfix (see note below) ---
        // 1) Always add current k
        ssum.add(k);
        // 2) Subtract the element leaving the window *only if it exists*.
        //    i - sig is a valid K index only when >= warm_line.
        if (sig > 0 && (i - sig) >= warm_line) {
          ssum.add(-line_row[i - sig]);
        }
        // 3) Emit signal when i >= warm_sig (first valid window completed).
        if (i >= warm_sig) {
          signal_row[i] = ssum.val() * invSig;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel: many series (time-major) × one parameter set

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
  const float nn   = kst_qnan();

  const float inv1  = (s1 > 0) ? (1.0f / float(s1)) : 0.0f;
  const float w2    = (s2 > 0) ? (2.0f / float(s2)) : 0.0f;
  const float w3    = (s3 > 0) ? (3.0f / float(s3)) : 0.0f;
  const float w4    = (s4 > 0) ? (4.0f / float(s4)) : 0.0f;
  const float invSig = (sig > 0) ? (1.0f / float(sig)) : 0.0f;

  for (int s = tid; s < num_series; s += stride) {
    int fv = first_valids[s];
    if (fv < 0)       fv = 0;
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

    const int warm_line = max(max(start1 + s1 - 1, start2 + s2 - 1),
                              max(start3 + s3 - 1, start4 + s4 - 1));
    const int warm_sig  = warm_line + sig - 1;

    for (int t = 0; t < warm_line && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_line_tm[idx] = nn;
    }
    for (int t = 0; t < warm_sig && t < series_len; ++t) {
      int idx = t * num_series + s;
      out_signal_tm[idx] = nn;
    }

    CompSum sum1, sum2, sum3, sum4, ssum;
    sum1.init(); sum2.init(); sum3.init(); sum4.init(); ssum.init();

    for (int t = fv; t < series_len; ++t) {
      const int idx  = t * num_series + s;
      const float x  = prices_tm[idx];

      if (t >= start1) {
        const float v = kst_safe_roc(x, prices_tm[(t - r1) * num_series + s]);
        if (t < start1 + s1) sum1.add(v);
        else                 sum1.add(v - kst_safe_roc(prices_tm[(t - s1) * num_series + s],
                                                      prices_tm[(t - s1 - r1) * num_series + s]));
      }
      if (t >= start2) {
        const float v = kst_safe_roc(x, prices_tm[(t - r2) * num_series + s]);
        if (t < start2 + s2) sum2.add(v);
        else                 sum2.add(v - kst_safe_roc(prices_tm[(t - s2) * num_series + s],
                                                      prices_tm[(t - s2 - r2) * num_series + s]));
      }
      if (t >= start3) {
        const float v = kst_safe_roc(x, prices_tm[(t - r3) * num_series + s]);
        if (t < start3 + s3) sum3.add(v);
        else                 sum3.add(v - kst_safe_roc(prices_tm[(t - s3) * num_series + s],
                                                      prices_tm[(t - s3 - r3) * num_series + s]));
      }
      if (t >= start4) {
        const float v = kst_safe_roc(x, prices_tm[(t - r4) * num_series + s]);
        if (t < start4 + s4) sum4.add(v);
        else                 sum4.add(v - kst_safe_roc(prices_tm[(t - s4) * num_series + s],
                                                      prices_tm[(t - s4 - r4) * num_series + s]));
      }

      if (t >= warm_line) {
        float k = __fmaf_rn(sum4.val(), w4,
                  __fmaf_rn(sum3.val(), w3,
                  __fmaf_rn(sum2.val(), w2, sum1.val() * inv1)));
        out_line_tm[idx] = k;

        // Rolling signal with same bugfix logic:
        ssum.add(k);
        if (sig > 0 && (t - sig) >= warm_line) {
          ssum.add(-out_line_tm[(t - sig) * num_series + s]);
        }
        if (t >= warm_sig) {
          out_signal_tm[idx] = ssum.val() * invSig;
        }
      }
    }
  }
}
