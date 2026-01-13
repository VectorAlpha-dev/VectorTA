
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#ifndef PREFILL_NAN_ON_HOST
#define PREFILL_NAN_ON_HOST 0
#endif

__device__ __forceinline__ float qnan_f32() { return __int_as_float(0x7fc00000); }



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

    const int nan_end = (warm_index < series_len ? warm_index : series_len);
    for (int i = 0; i < nan_end; ++i) out[base + i] = nanv;
#endif

    if (first_valid + period > series_len) return;

    const float invP = 1.0f / static_cast<float>(period);
    const float* __restrict__ P = prices + first_valid;

    int   today = 0;
    float sum   = 0.0f;


    for (int i = 0; i < period; ++i) sum += P[i];
    float e1 = sum * invP;
    today += period;


    sum = e1;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        sum += e1;
    }
    float e2 = sum * invP;


    sum = e2;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        sum += e2;
    }
    float e3 = sum * invP;


    sum = e3;
    for (int i = 1; i < period; ++i) {
        const float price = P[today++];
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        sum += e3;
    }
    float e4 = sum * invP;


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


    out[base + warm_index] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));


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








extern "C" __global__
void tilson_batch_warp_scan_f32(const float* __restrict__ prices,
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
    if (series_len <= 0 || n_combos <= 0) return;
    if (first_valid < 0 || first_valid >= series_len) return;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    if (warps_per_block <= 0) return;

    const int combo = blockIdx.x * warps_per_block + warp_in_block;
    if (combo >= n_combos) return;

    const unsigned mask = 0xffffffffu;


    int period = 0;
    int lookback = 0;
    float k = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
    if (lane == 0) {
        period   = periods[combo];
        lookback = lookbacks[combo];
        k  = ks[combo];
        c1 = c1s[combo];
        c2 = c2s[combo];
        c3 = c3s[combo];
        c4 = c4s[combo];
    }
    period   = __shfl_sync(mask, period, 0);
    lookback = __shfl_sync(mask, lookback, 0);
    k  = __shfl_sync(mask, k, 0);
    c1 = __shfl_sync(mask, c1, 0);
    c2 = __shfl_sync(mask, c2, 0);
    c3 = __shfl_sync(mask, c3, 0);
    c4 = __shfl_sync(mask, c4, 0);

    if (period <= 0 || lookback < 0) return;

    const float one_m_k = 1.0f - k;
    const int warm_index = first_valid + lookback;
    const size_t base = (size_t)combo * (size_t)series_len;

    if (first_valid + period > series_len) return;
    const int need_last = first_valid + (6 * period - 6);
    if (need_last >= series_len) return;


    const float nanv = qnan_f32();
    for (int t = lane; t < warm_index && t < series_len; t += 32) {
        out[base + (size_t)t] = nanv;
    }
    if (warm_index >= series_len) return;


    float e1_prev = 0.0f, e2_prev = 0.0f, e3_prev = 0.0f, e4_prev = 0.0f, e5_prev = 0.0f, e6_prev = 0.0f;
    if (lane == 0) {
        const float invP = 1.0f / (float)period;
        const float* __restrict__ P = prices + first_valid;

        int today = 0;
        float sum = 0.0f;


        for (int i = 0; i < period; ++i) sum += P[i];
        float e1 = sum * invP;
        today += period;


        sum = e1;
        for (int i = 1; i < period; ++i) {
            const float price = P[today++];
            e1 = fmaf(k, price, one_m_k * e1);
            sum += e1;
        }
        float e2 = sum * invP;


        sum = e2;
        for (int i = 1; i < period; ++i) {
            const float price = P[today++];
            e1 = fmaf(k, price, one_m_k * e1);
            e2 = fmaf(k, e1,   one_m_k * e2);
            sum += e2;
        }
        float e3 = sum * invP;


        sum = e3;
        for (int i = 1; i < period; ++i) {
            const float price = P[today++];
            e1 = fmaf(k, price, one_m_k * e1);
            e2 = fmaf(k, e1,   one_m_k * e2);
            e3 = fmaf(k, e2,   one_m_k * e3);
            sum += e3;
        }
        float e4 = sum * invP;


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


        out[base + (size_t)warm_index] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));

        e1_prev = e1; e2_prev = e2; e3_prev = e3; e4_prev = e4; e5_prev = e5; e6_prev = e6;
    }


    e1_prev = __shfl_sync(mask, e1_prev, 0);
    e2_prev = __shfl_sync(mask, e2_prev, 0);
    e3_prev = __shfl_sync(mask, e3_prev, 0);
    e4_prev = __shfl_sync(mask, e4_prev, 0);
    e5_prev = __shfl_sync(mask, e5_prev, 0);
    e6_prev = __shfl_sync(mask, e6_prev, 0);

    int t0 = warm_index + 1;
    if (t0 >= series_len) return;

    for (int tile = t0; tile < series_len; tile += 32) {
        const int t = tile + lane;
        const bool valid = (t < series_len);


        float A = valid ? one_m_k : 1.0f;
        float B = valid ? (k * prices[t]) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e1 = fmaf(A, e1_prev, B);


        A = valid ? one_m_k : 1.0f;
        B = valid ? (k * e1) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e2 = fmaf(A, e2_prev, B);


        A = valid ? one_m_k : 1.0f;
        B = valid ? (k * e2) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e3 = fmaf(A, e3_prev, B);


        A = valid ? one_m_k : 1.0f;
        B = valid ? (k * e3) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e4 = fmaf(A, e4_prev, B);


        A = valid ? one_m_k : 1.0f;
        B = valid ? (k * e4) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e5 = fmaf(A, e5_prev, B);


        A = valid ? one_m_k : 1.0f;
        B = valid ? (k * e5) : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float A_prev = __shfl_up_sync(mask, A, offset);
            const float B_prev = __shfl_up_sync(mask, B, offset);
            if (lane >= offset) {
                const float A_cur = A;
                const float B_cur = B;
                A = A_cur * A_prev;
                B = fmaf(A_cur, B_prev, B_cur);
            }
        }
        const float e6 = fmaf(A, e6_prev, B);

        if (valid) {
            out[base + (size_t)t] = fmaf(c1, e6, fmaf(c2, e5, fmaf(c3, e4, c4 * e3)));
        }

        const int remaining = series_len - tile;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        e1_prev = __shfl_sync(mask, e1, last_lane);
        e2_prev = __shfl_sync(mask, e2, last_lane);
        e3_prev = __shfl_sync(mask, e3, last_lane);
        e4_prev = __shfl_sync(mask, e4, last_lane);
        e5_prev = __shfl_sync(mask, e5, last_lane);
        e6_prev = __shfl_sync(mask, e6, last_lane);
    }
}



extern "C" __global__
void tilson_many_series_one_param_f32(const float* __restrict__ prices_tm,
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

    const int nan_end = (warm_index < series_len ? warm_index : series_len);
    for (int t = 0; t < nan_end; ++t) out_tm[t * stride + series] = nanv;
#endif

    const int need_last = fv + (6*period - 6);
    if (need_last >= series_len) return;

    const float invP = 1.0f / static_cast<float>(period);

    auto P = [&](int t)->float { return prices_tm[t * stride + series]; };

    int   today = 0;
    float sum   = 0.0f;


    for (int i = 0; i < period; ++i) sum += P(fv + i);
    float e1 = sum * invP;
    today += period;


    sum = e1;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        sum += e1;
    }
    float e2 = sum * invP;


    sum = e2;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        sum += e2;
    }
    float e3 = sum * invP;


    sum = e3;
    for (int i = 1; i < period; ++i) {
        const float price = P(fv + today++);
        e1 = fmaf(k, price, one_m_k * e1);
        e2 = fmaf(k, e1,   one_m_k * e2);
        e3 = fmaf(k, e2,   one_m_k * e3);
        sum += e3;
    }
    float e4 = sum * invP;


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
