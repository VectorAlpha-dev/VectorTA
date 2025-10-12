// Keep compatibility macros and headers
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

// Define to 1 if host pre-fills out buffers with qNaN via cuMemsetD32Async
#ifndef PREFILL_NAN_ON_HOST
#define PREFILL_NAN_ON_HOST 0
#endif

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fc00000); }

// ---------------------- tilson_batch_f32 ----------------------
// One thread computes one combo; still works with blockDim.x==1.
extern "C" __global__
void tilson_batch_f32(const float* __restrict__ prices,
                      const int*   __restrict__ periods,
                      const float* __restrict__ ks,
                      const float* __restrict__ c1s,
                      const float* __restrict__ c2s,
                      const float* __restrict__ c3s,
                      const float* __restrict__ c4s,
                      const int*   __restrict__ lookbacks,
                      int series_len,
                      int first_valid,
                      int n_combos,
                      float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int   period   = periods[combo];
    const int   lookback = lookbacks[combo];
    const float k        = ks[combo];
    const float one_m_k  = 1.0f - k;
    const float c1       = c1s[combo];
    const float c2       = c2s[combo];
    const float c3       = c3s[combo];
    const float c4       = c4s[combo];

    if (period <= 0 || lookback < 0 || series_len <= 0) return;
    if (first_valid < 0 || first_valid >= series_len)   return;

    const int base = combo * series_len;
    const int warm_index = first_valid + lookback;
    if (warm_index >= series_len) return;

    const int need_last = first_valid + (6*period - 6);
    if (need_last >= series_len) return;

#if !PREFILL_NAN_ON_HOST
    const float nanv = qnan_f32();
    for (int i = 0; i < series_len; ++i) out[base + i] = nanv;
#endif

    if (first_valid + period > series_len) return;

    const float invP = 1.0f / static_cast<float>(period);
    const float* __restrict__ P = prices + first_valid;

    int   today = 0;
    float sum   = 0.0f;

    // e1 seed (SMA)
    for (int i = 0; i < period; ++i) sum += P[i];
    float e1 = sum * invP;
    today += period;

    // e2 seed
    sum = e1;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        sum += e1;
    }
    float e2 = sum * invP;

    // e3 seed
    sum = e2;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        sum += e2;
    }
    float e3 = sum * invP;

    // e4 seed
    sum = e3;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        sum += e3;
    }
    float e4 = sum * invP;

    // e5 seed
    sum = e4;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        sum += e4;
    }
    float e5 = sum * invP;

    // e6 seed
    sum = e5;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        e5 = fmaf(k, e4,   one_m_k * e5);
        sum += e5;
    }
    float e6 = sum * invP;

    // First valid output
    out[base + warm_index] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));

    // Stream updates
    int out_idx = warm_index + 1;
    const int N = series_len - first_valid;
    while (today <= (N - 1)) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        e5 = fmaf(k, e4,   one_m_k * e5);
        e6 = fmaf(k, e5,   one_m_k * e6);

        if (out_idx < series_len) {
            out[base + out_idx] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));
        }
        ++out_idx;
    }
}

// ---------------------- tilson_many_series_one_param_f32 ----------------------
// One thread computes one series; works with any 1D/2D grid (flattened).
extern "C" __global__
void tilson_many_series_one_param_f32(const float* __restrict__ prices_tm, // time-major
                                      const int*   __restrict__ first_valids,
                                      int   period,
                                      float k,
                                      float c1,
                                      float c2,
                                      float c3,
                                      float c4,
                                      int   lookback,
                                      int   num_series,
                                      int   series_len,
                                      float* __restrict__ out_tm)
{
    long gtid = (long)blockIdx.y * (gridDim.x * blockDim.x)
              + (long)blockIdx.x * blockDim.x
              + (long)threadIdx.x;
    const int series = (int)gtid;
    if (series >= num_series) return;

    if (period <= 0 || lookback < 0 || num_series <= 0 || series_len <= 0) return;

    const int stride = num_series;
    const int fv = first_valids[series];
    if (fv < 0 || fv >= series_len) return;

    const int warm_index = fv + lookback;
    if (warm_index >= series_len) return;

    const float one_m_k = 1.0f - k;

#if !PREFILL_NAN_ON_HOST
    const float nanv = qnan_f32();
    for (int t = 0; t < series_len; ++t) out_tm[t * stride + series] = nanv;
#endif

    const int need_last = fv + (6*period - 6);
    if (need_last >= series_len) return;

    const float invP = 1.0f / static_cast<float>(period);

    auto P = [&](int t)->float { return prices_tm[t * stride + series]; };

    int   today = 0;
    float sum   = 0.0f;

    // e1 seed
    for (int i = 0; i < period; ++i) sum += P(fv + i);
    float e1 = sum * invP;
    today += period;

    // e2 seed
    sum = e1;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        sum += e1;
    }
    float e2 = sum * invP;

    // e3 seed
    sum = e2;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        sum += e2;
    }
    float e3 = sum * invP;

    // e4 seed
    sum = e3;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        sum += e3;
    }
    float e4 = sum * invP;

    // e5 seed
    sum = e4;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        sum += e4;
    }
    float e5 = sum * invP;

    // e6 seed
    sum = e5;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        e5 = fmaf(k, e4,   one_m_k * e5);
        sum += e5;
    }
    float e6 = sum * invP;

    out_tm[warm_index * stride + series] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));

    int out_idx = warm_index + 1;
    const int end_t = series_len - 1;
    while ((fv + today) <= end_t) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        e4 = fmaf(k, e3,   one_m_k * e4);
        e5 = fmaf(k, e4,   one_m_k * e5);
        e6 = fmaf(k, e5,   one_m_k * e6);

        if (out_idx < series_len) {
            out_tm[out_idx * stride + series] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));
        }
        ++out_idx;
    }
}
