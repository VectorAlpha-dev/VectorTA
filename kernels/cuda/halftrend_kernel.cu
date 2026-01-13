





#include <cuda_runtime.h>
#include <math_constants.h>
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif


__device__ __forceinline__ float qnan_f() { return __int_as_float(0x7fc00000); }


__device__ __forceinline__ int ht_inc(int i, int cap) { int j = i + 1; return (j == cap) ? 0 : j; }
__device__ __forceinline__ int ht_dec(int i, int cap) { return (i == 0) ? (cap - 1) : (i - 1); }


#define AT_TM(t, rows, row) ((t) * (rows) + (row))


#define HT_THREADS_PER_BLOCK 256

#define HT_FUSED_MAX_AMP 64

extern "C" {





__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ atr_rows,
    const float* __restrict__ highma_rows,
    const float* __restrict__ lowma_rows,
    const float* __restrict__ roll_high_rows,
    const float* __restrict__ roll_low_rows,
    const int*   __restrict__ warms,
    const float* __restrict__ chdevs,
    int n,
    int rows,
    float* __restrict__ out_halftrend,
    float* __restrict__ out_trend,
    float* __restrict__ out_atr_high,
    float* __restrict__ out_atr_low,
    float* __restrict__ out_buy,
    float* __restrict__ out_sell)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < rows; row += stride) {
        const int base = row * n;
        const float* __restrict__ atr = atr_rows + base;
        const float* __restrict__ hma = highma_rows + base;
        const float* __restrict__ lma = lowma_rows + base;
        const float* __restrict__ rhi = roll_high_rows + base;
        const float* __restrict__ rlo = roll_low_rows + base;
        float* __restrict__ o_ht = out_halftrend + base;
        float* __restrict__ o_tr = out_trend + base;
        float* __restrict__ o_ah = out_atr_high + base;
        float* __restrict__ o_al = out_atr_low + base;
        float* __restrict__ o_bs = out_buy + base;
        float* __restrict__ o_ss = out_sell + base;

        const int warm_in = LDG(warms + row);
        const int warm    = (warm_in < n ? warm_in : n);
        const float ch_half = LDG(chdevs + row) * 0.5f;
        const float qnan = qnan_f();


        #pragma unroll 8
        for (int i = 0; i < warm; ++i) {
            o_ht[i] = qnan; o_tr[i] = qnan; o_ah[i] = qnan; o_al[i] = qnan; o_bs[i] = qnan; o_ss[i] = qnan;
        }
        if (warm >= n) continue;

        int   current_trend = 0;
        int   next_trend    = 0;
        int   prev_trend    = 0;
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1))  : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int i = warm; i < n; ++i) {

            o_bs[i] = qnan; o_ss[i] = qnan;

            const float high_price = LDG(rhi + i);
            const float low_price  = LDG(rlo + i);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (LDG(hma + i) < max_low_price && LDG(close + i) < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (LDG(lma + i) > min_high_price && LDG(close + i) > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = LDG(atr + i);
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;

            const bool flipped = (i > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {
                    up = down;
                    o_bs[i] = up - atr2;
                } else {
                    if (i == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                o_ht[i] = up;
                o_ah[i] = up + dev;
                o_al[i] = up - dev;
                o_tr[i] = 0.0f;
            } else {
                if (flipped) {
                    down = up;
                    o_ss[i] = down + atr2;
                } else {
                    if (i == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                o_ht[i] = down;
                o_ah[i] = down + dev;
                o_al[i] = down - dev;
                o_tr[i] = 1.0f;
            }

            prev_low   = LDG(low  + i);
            prev_high  = LDG(high + i);
            prev_trend = current_trend;
        }
    }
}







__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_fused_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ amps,
    const int*   __restrict__ atr_periods,
    const float* __restrict__ chdevs,
    int first,
    int n,
    int rows,
    float* __restrict__ out_halftrend,
    float* __restrict__ out_trend,
    float* __restrict__ out_atr_high,
    float* __restrict__ out_atr_low,
    float* __restrict__ out_buy,
    float* __restrict__ out_sell)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float qnan = qnan_f();

    for (int row = tid; row < rows; row += stride) {
        const int amp = LDG(amps + row);
        const int atr_p = LDG(atr_periods + row);
        const float ch_half = LDG(chdevs + row) * 0.5f;


        if (amp <= 0 || atr_p <= 0 || amp > HT_FUSED_MAX_AMP) {
            const int base = row * n;
            float* __restrict__ o_ht = out_halftrend + base;
            float* __restrict__ o_tr = out_trend + base;
            float* __restrict__ o_ah = out_atr_high + base;
            float* __restrict__ o_al = out_atr_low + base;
            float* __restrict__ o_bs = out_buy + base;
            float* __restrict__ o_ss = out_sell + base;
            for (int i = 0; i < n; ++i) {
                o_ht[i] = qnan; o_tr[i] = qnan; o_ah[i] = qnan; o_al[i] = qnan; o_bs[i] = qnan; o_ss[i] = qnan;
            }
            continue;
        }


        int warm = first + (amp > atr_p ? amp : atr_p) - 1;
        warm = (warm < n ? warm : n);

        const int base = row * n;
        float* __restrict__ o_ht = out_halftrend + base;
        float* __restrict__ o_tr = out_trend + base;
        float* __restrict__ o_ah = out_atr_high + base;
        float* __restrict__ o_al = out_atr_low + base;
        float* __restrict__ o_bs = out_buy + base;
        float* __restrict__ o_ss = out_sell + base;


        #pragma unroll 8
        for (int i = 0; i < warm; ++i) {
            o_ht[i] = qnan; o_tr[i] = qnan; o_ah[i] = qnan; o_al[i] = qnan; o_bs[i] = qnan; o_ss[i] = qnan;
        }
        if (warm >= n) continue;


        float sum_high = 0.0f;
        float sum_low  = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < amp; ++k) {
            sum_high += LDG(high + (first + k));
            sum_low  += LDG(low  + (first + k));
        }
        const float inv_amp = 1.0f / (float)amp;
        const int sma_warm = first + amp - 1;
        for (int i = sma_warm + 1; i <= warm; ++i) {
            sum_high = sum_high + LDG(high + i) - LDG(high + (i - amp));
            sum_low  = sum_low  + LDG(low  + i) - LDG(low  + (i - amp));
        }


        const float alpha = 1.0f / (float)atr_p;
        const int atr_warm = first + atr_p - 1;
        float sum_tr = LDG(high + first) - LDG(low + first);
        float prev_c = LDG(close + first);
        for (int i = first + 1; i <= atr_warm; ++i) {
            const float hi = LDG(high + i);
            const float lo = LDG(low + i);
            float tr = hi - lo;
            const float hc = fabsf(hi - prev_c);
            if (hc > tr) tr = hc;
            const float lc = fabsf(lo - prev_c);
            if (lc > tr) tr = lc;
            sum_tr += tr;
            prev_c = LDG(close + i);
        }
        float rma = sum_tr / (float)atr_p;
        for (int i = atr_warm + 1; i <= warm; ++i) {
            const float hi = LDG(high + i);
            const float lo = LDG(low + i);
            float tr = hi - lo;
            const float hc = fabsf(hi - prev_c);
            if (hc > tr) tr = hc;
            const float lc = fabsf(lo - prev_c);
            if (lc > tr) tr = lc;
            rma = fmaf(alpha, (tr - rma), rma);
            prev_c = LDG(close + i);
        }


        int max_idx[HT_FUSED_MAX_AMP];
        int min_idx[HT_FUSED_MAX_AMP];
        int max_head = 0, max_tail = 0, max_cnt = 0;
        int min_head = 0, min_tail = 0, min_cnt = 0;
        const int cap = amp;

        const int wstart0 = warm + 1 - cap;
        for (int k = wstart0; k <= warm; ++k) {
            const float hv = LDG(high + k);
            while (max_cnt > 0) {
                const int back = ht_dec(max_tail, cap);
                const float bv = LDG(high + max_idx[back]);
                if (bv <= hv) {
                    max_tail = back;
                    max_cnt -= 1;
                } else {
                    break;
                }
            }
            max_idx[max_tail] = k;
            max_tail = ht_inc(max_tail, cap);
            max_cnt += 1;

            const float lv = LDG(low + k);
            while (min_cnt > 0) {
                const int back = ht_dec(min_tail, cap);
                const float bv = LDG(low + min_idx[back]);
                if (bv >= lv) {
                    min_tail = back;
                    min_cnt -= 1;
                } else {
                    break;
                }
            }
            min_idx[min_tail] = k;
            min_tail = ht_inc(min_tail, cap);
            min_cnt += 1;
        }


        int   current_trend = 0;
        int   next_trend    = 0;
        int   prev_trend    = 0;
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1))  : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int t = warm; t < n; ++t) {

            o_bs[t] = qnan; o_ss[t] = qnan;


            if (t > warm) {
                const int wstart = t + 1 - cap;
                while (max_cnt > 0 && max_idx[max_head] < wstart) {
                    max_head = ht_inc(max_head, cap);
                    max_cnt -= 1;
                }
                while (min_cnt > 0 && min_idx[min_head] < wstart) {
                    min_head = ht_inc(min_head, cap);
                    min_cnt -= 1;
                }

                const float hv = LDG(high + t);
                while (max_cnt > 0) {
                    const int back = ht_dec(max_tail, cap);
                    const float bv = LDG(high + max_idx[back]);
                    if (bv <= hv) {
                        max_tail = back;
                        max_cnt -= 1;
                    } else {
                        break;
                    }
                }
                max_idx[max_tail] = t;
                max_tail = ht_inc(max_tail, cap);
                max_cnt += 1;

                const float lv = LDG(low + t);
                while (min_cnt > 0) {
                    const int back = ht_dec(min_tail, cap);
                    const float bv = LDG(low + min_idx[back]);
                    if (bv >= lv) {
                        min_tail = back;
                        min_cnt -= 1;
                    } else {
                        break;
                    }
                }
                min_idx[min_tail] = t;
                min_tail = ht_inc(min_tail, cap);
                min_cnt += 1;
            }

            const float high_price = LDG(high + max_idx[max_head]);
            const float low_price  = LDG(low  + min_idx[min_head]);

            const float highma_t = sum_high * inv_amp;
            const float lowma_t  = sum_low  * inv_amp;

            const float close_t = LDG(close + t);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (highma_t < max_low_price && close_t < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (lowma_t > min_high_price && close_t > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = rma;
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;

            const bool flipped = (t > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {
                    up = down;
                    o_bs[t] = up - atr2;
                } else {
                    if (t == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                o_ht[t] = up;
                o_ah[t] = up + dev;
                o_al[t] = up - dev;
                o_tr[t] = 0.0f;
            } else {
                if (flipped) {
                    down = up;
                    o_ss[t] = down + atr2;
                } else {
                    if (t == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                o_ht[t] = down;
                o_ah[t] = down + dev;
                o_al[t] = down - dev;
                o_tr[t] = 1.0f;
            }


            const int ni = t + 1;
            if (ni < n) {
                sum_high = sum_high + LDG(high + ni) - LDG(high + (ni - amp));
                sum_low  = sum_low  + LDG(low  + ni) - LDG(low  + (ni - amp));

                const float hi = LDG(high + ni);
                const float lo = LDG(low + ni);
                float tr = hi - lo;
                const float hc = fabsf(hi - close_t);
                if (hc > tr) tr = hc;
                const float lc = fabsf(lo - close_t);
                if (lc > tr) tr = lc;
                rma = fmaf(alpha, (tr - rma), rma);
            }

            prev_low   = LDG(low  + t);
            prev_high  = LDG(high + t);
            prev_trend = current_trend;
        }
    }
}





__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_time_major_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ atr_tm,
    const float* __restrict__ highma_tm,
    const float* __restrict__ lowma_tm,
    const float* __restrict__ roll_high_tm,
    const float* __restrict__ roll_low_tm,
    const int*   __restrict__ warms,
    const float* __restrict__ chdevs,
    int n,
    int rows,
    float* __restrict__ out_halftrend_tm,
    float* __restrict__ out_trend_tm,
    float* __restrict__ out_atr_high_tm,
    float* __restrict__ out_atr_low_tm,
    float* __restrict__ out_buy_tm,
    float* __restrict__ out_sell_tm)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride= blockDim.x * gridDim.x;

    for (int row = tid; row < rows; row += stride) {
        const float ch_half = LDG(chdevs + row) * 0.5f;
        const int   warm_in = LDG(warms  + row);
        const int   warm    = (warm_in < n ? warm_in : n);
        const float qnan = qnan_f();

        #pragma unroll 8
        for (int t = 0; t < warm; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_halftrend_tm[idx] = qnan;
            out_trend_tm[idx]     = qnan;
            out_atr_high_tm[idx]  = qnan;
            out_atr_low_tm[idx]   = qnan;
            out_buy_tm[idx]       = qnan;
            out_sell_tm[idx]      = qnan;
        }
        if (warm >= n) continue;

        int   current_trend = 0;
        int   next_trend    = 0;
        int   prev_trend    = 0;
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1)) : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int t = warm; t < n; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_buy_tm[idx]  = qnan;
            out_sell_tm[idx] = qnan;

            const float high_price = LDG(roll_high_tm + idx);
            const float low_price  = LDG(roll_low_tm  + idx);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (LDG(highma_tm + idx) < max_low_price && LDG(close + t) < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (LDG(lowma_tm + idx) > min_high_price && LDG(close + t) > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = LDG(atr_tm + idx);
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;

            const bool flipped = (t > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {
                    up = down;
                    out_buy_tm[idx] = up - atr2;
                } else {
                    if (t == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                out_halftrend_tm[idx] = up;
                out_atr_high_tm[idx]  = up + dev;
                out_atr_low_tm[idx]   = up - dev;
                out_trend_tm[idx]     = 0.0f;
            } else {
                if (flipped) {
                    down = up;
                    out_sell_tm[idx] = down + atr2;
                } else {
                    if (t == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                out_halftrend_tm[idx] = down;
                out_atr_high_tm[idx]  = down + dev;
                out_atr_low_tm[idx]   = down - dev;
                out_trend_tm[idx]     = 1.0f;
            }

            prev_low  = LDG(low  + t);
            prev_high = LDG(high + t);
            prev_trend = current_trend;
        }
    }
}





__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_batch_fused_time_major_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ amps,
    const int*   __restrict__ atr_periods,
    const float* __restrict__ chdevs,
    int first,
    int n,
    int rows,
    float* __restrict__ out_halftrend_tm,
    float* __restrict__ out_trend_tm,
    float* __restrict__ out_atr_high_tm,
    float* __restrict__ out_atr_low_tm,
    float* __restrict__ out_buy_tm,
    float* __restrict__ out_sell_tm)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float qnan = qnan_f();

    for (int row = tid; row < rows; row += stride) {
        const int amp = LDG(amps + row);
        const int atr_p = LDG(atr_periods + row);
        const float ch_half = LDG(chdevs + row) * 0.5f;

        if (amp <= 0 || atr_p <= 0 || amp > HT_FUSED_MAX_AMP) {
            for (int t = 0; t < n; ++t) {
                const int idx = AT_TM(t, rows, row);
                out_halftrend_tm[idx] = qnan;
                out_trend_tm[idx]     = qnan;
                out_atr_high_tm[idx]  = qnan;
                out_atr_low_tm[idx]   = qnan;
                out_buy_tm[idx]       = qnan;
                out_sell_tm[idx]      = qnan;
            }
            continue;
        }

        int warm = first + (amp > atr_p ? amp : atr_p) - 1;
        warm = (warm < n ? warm : n);

        #pragma unroll 8
        for (int t = 0; t < warm; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_halftrend_tm[idx] = qnan;
            out_trend_tm[idx]     = qnan;
            out_atr_high_tm[idx]  = qnan;
            out_atr_low_tm[idx]   = qnan;
            out_buy_tm[idx]       = qnan;
            out_sell_tm[idx]      = qnan;
        }
        if (warm >= n) continue;


        float sum_high = 0.0f;
        float sum_low  = 0.0f;
        for (int k = 0; k < amp; ++k) {
            sum_high += LDG(high + (first + k));
            sum_low  += LDG(low  + (first + k));
        }
        const float inv_amp = 1.0f / (float)amp;
        const int sma_warm = first + amp - 1;
        for (int t = sma_warm + 1; t <= warm; ++t) {
            sum_high = sum_high + LDG(high + t) - LDG(high + (t - amp));
            sum_low  = sum_low  + LDG(low  + t) - LDG(low  + (t - amp));
        }


        const float alpha = 1.0f / (float)atr_p;
        const int atr_warm = first + atr_p - 1;
        float sum_tr = LDG(high + first) - LDG(low + first);
        float prev_c = LDG(close + first);
        for (int t = first + 1; t <= atr_warm; ++t) {
            const float hi = LDG(high + t);
            const float lo = LDG(low + t);
            float tr = hi - lo;
            const float hc = fabsf(hi - prev_c);
            if (hc > tr) tr = hc;
            const float lc = fabsf(lo - prev_c);
            if (lc > tr) tr = lc;
            sum_tr += tr;
            prev_c = LDG(close + t);
        }
        float rma = sum_tr / (float)atr_p;
        for (int t = atr_warm + 1; t <= warm; ++t) {
            const float hi = LDG(high + t);
            const float lo = LDG(low + t);
            float tr = hi - lo;
            const float hc = fabsf(hi - prev_c);
            if (hc > tr) tr = hc;
            const float lc = fabsf(lo - prev_c);
            if (lc > tr) tr = lc;
            rma = fmaf(alpha, (tr - rma), rma);
            prev_c = LDG(close + t);
        }


        int max_idx[HT_FUSED_MAX_AMP];
        int min_idx[HT_FUSED_MAX_AMP];
        int max_head = 0, max_tail = 0, max_cnt = 0;
        int min_head = 0, min_tail = 0, min_cnt = 0;
        const int cap = amp;

        const int wstart0 = warm + 1 - cap;
        for (int k = wstart0; k <= warm; ++k) {
            const float hv = LDG(high + k);
            while (max_cnt > 0) {
                const int back = ht_dec(max_tail, cap);
                const float bv = LDG(high + max_idx[back]);
                if (bv <= hv) {
                    max_tail = back;
                    max_cnt -= 1;
                } else {
                    break;
                }
            }
            max_idx[max_tail] = k;
            max_tail = ht_inc(max_tail, cap);
            max_cnt += 1;

            const float lv = LDG(low + k);
            while (min_cnt > 0) {
                const int back = ht_dec(min_tail, cap);
                const float bv = LDG(low + min_idx[back]);
                if (bv >= lv) {
                    min_tail = back;
                    min_cnt -= 1;
                } else {
                    break;
                }
            }
            min_idx[min_tail] = k;
            min_tail = ht_inc(min_tail, cap);
            min_cnt += 1;
        }

        int   current_trend = 0;
        int   next_trend    = 0;
        int   prev_trend    = 0;
        float up   = 0.0f;
        float down = 0.0f;

        float prev_low  = (warm > 0) ? LDG(low  + (warm - 1))  : LDG(low  + 0);
        float prev_high = (warm > 0) ? LDG(high + (warm - 1)) : LDG(high + 0);
        float max_low_price  = prev_low;
        float min_high_price = prev_high;

        for (int t = warm; t < n; ++t) {
            const int idx = AT_TM(t, rows, row);
            out_buy_tm[idx]  = qnan;
            out_sell_tm[idx] = qnan;

            if (t > warm) {
                const int wstart = t + 1 - cap;
                while (max_cnt > 0 && max_idx[max_head] < wstart) {
                    max_head = ht_inc(max_head, cap);
                    max_cnt -= 1;
                }
                while (min_cnt > 0 && min_idx[min_head] < wstart) {
                    min_head = ht_inc(min_head, cap);
                    min_cnt -= 1;
                }

                const float hv = LDG(high + t);
                while (max_cnt > 0) {
                    const int back = ht_dec(max_tail, cap);
                    const float bv = LDG(high + max_idx[back]);
                    if (bv <= hv) {
                        max_tail = back;
                        max_cnt -= 1;
                    } else {
                        break;
                    }
                }
                max_idx[max_tail] = t;
                max_tail = ht_inc(max_tail, cap);
                max_cnt += 1;

                const float lv = LDG(low + t);
                while (min_cnt > 0) {
                    const int back = ht_dec(min_tail, cap);
                    const float bv = LDG(low + min_idx[back]);
                    if (bv >= lv) {
                        min_tail = back;
                        min_cnt -= 1;
                    } else {
                        break;
                    }
                }
                min_idx[min_tail] = t;
                min_tail = ht_inc(min_tail, cap);
                min_cnt += 1;
            }

            const float high_price = LDG(high + max_idx[max_head]);
            const float low_price  = LDG(low  + min_idx[min_head]);

            const float highma_t = sum_high * inv_amp;
            const float lowma_t  = sum_low  * inv_amp;

            const float close_t = LDG(close + t);

            if (next_trend == 1) {
                max_low_price = fmaxf(max_low_price, low_price);
                if (highma_t < max_low_price && close_t < prev_low) {
                    current_trend  = 1;
                    next_trend     = 0;
                    min_high_price = high_price;
                }
            } else {
                min_high_price = fminf(min_high_price, high_price);
                if (lowma_t > min_high_price && close_t > prev_high) {
                    current_trend  = 0;
                    next_trend     = 1;
                    max_low_price  = low_price;
                }
            }

            const float a    = rma;
            const float atr2 = 0.5f * a;
            const float dev  = a * ch_half;
            const bool flipped = (t > warm) && (prev_trend != current_trend);

            if (current_trend == 0) {
                if (flipped) {
                    up = down;
                    out_buy_tm[idx] = up - atr2;
                } else {
                    if (t == warm || up == 0.0f) up = max_low_price;
                    else if (max_low_price > up) up = max_low_price;
                }
                out_halftrend_tm[idx] = up;
                out_atr_high_tm[idx]  = up + dev;
                out_atr_low_tm[idx]   = up - dev;
                out_trend_tm[idx]     = 0.0f;
            } else {
                if (flipped) {
                    down = up;
                    out_sell_tm[idx] = down + atr2;
                } else {
                    if (t == warm || down == 0.0f) down = min_high_price;
                    else if (min_high_price < down) down = min_high_price;
                }
                out_halftrend_tm[idx] = down;
                out_atr_high_tm[idx]  = down + dev;
                out_atr_low_tm[idx]   = down - dev;
                out_trend_tm[idx]     = 1.0f;
            }

            const int ni = t + 1;
            if (ni < n) {
                sum_high = sum_high + LDG(high + ni) - LDG(high + (ni - amp));
                sum_low  = sum_low  + LDG(low  + ni) - LDG(low  + (ni - amp));

                const float hi = LDG(high + ni);
                const float lo = LDG(low + ni);
                float tr = hi - lo;
                const float hc = fabsf(hi - close_t);
                if (hc > tr) tr = hc;
                const float lc = fabsf(lo - close_t);
                if (lc > tr) tr = lc;
                rma = fmaf(alpha, (tr - rma), rma);
            }

            prev_low   = LDG(low  + t);
            prev_high  = LDG(high + t);
            prev_trend = current_trend;
        }
    }
}





__global__ __launch_bounds__(HT_THREADS_PER_BLOCK, 2)
void halftrend_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ atr_tm,
    const float* __restrict__ highma_tm,
    const float* __restrict__ lowma_tm,
    const float* __restrict__ roll_high_tm,
    const float* __restrict__ roll_low_tm,
    const int*   __restrict__ warms_cols,
    float ch_dev,
    int cols,
    int rows,
    float* __restrict__ out_halftrend_tm,
    float* __restrict__ out_trend_tm,
    float* __restrict__ out_atr_high_tm,
    float* __restrict__ out_atr_low_tm,
    float* __restrict__ out_buy_tm,
    float* __restrict__ out_sell_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const float ch_half = ch_dev * 0.5f;
    const int warm_in = LDG(warms_cols + s);
    const int warm    = (warm_in < rows ? warm_in : rows);

    const float qnan = qnan_f();
    #pragma unroll 8
    for (int t = 0; t < warm; ++t) {
        const int idx = t * cols + s;
        out_halftrend_tm[idx] = qnan;
        out_trend_tm[idx]     = qnan;
        out_atr_high_tm[idx]  = qnan;
        out_atr_low_tm[idx]   = qnan;
        out_buy_tm[idx]       = qnan;
        out_sell_tm[idx]      = qnan;
    }
    if (warm >= rows) return;

    int   current_trend = 0;
    int   next_trend    = 0;
    int   prev_trend    = 0;
    float up   = 0.0f;
    float down = 0.0f;

    float prev_low  = (warm > 0) ? LDG(low_tm  + ((warm - 1) * cols + s)) : LDG(low_tm  + (0 * cols + s));
    float prev_high = (warm > 0) ? LDG(high_tm + ((warm - 1) * cols + s)) : LDG(high_tm + (0 * cols + s));
    float max_low_price  = prev_low;
    float min_high_price = prev_high;

    for (int t = warm; t < rows; ++t) {
        const int idx = t * cols + s;
        out_buy_tm[idx]  = qnan;
        out_sell_tm[idx] = qnan;

        const float high_price = LDG(roll_high_tm + idx);
        const float low_price  = LDG(roll_low_tm  + idx);

        if (next_trend == 1) {
            max_low_price = fmaxf(max_low_price, low_price);
            if (LDG(highma_tm + idx) < max_low_price && LDG(close_tm + idx) < prev_low) {
                current_trend  = 1;
                next_trend     = 0;
                min_high_price = high_price;
            }
        } else {
            min_high_price = fminf(min_high_price, high_price);
            if (LDG(lowma_tm + idx) > min_high_price && LDG(close_tm + idx) > prev_high) {
                current_trend  = 0;
                next_trend     = 1;
                max_low_price  = low_price;
            }
        }

        const float a    = LDG(atr_tm + idx);
        const float atr2 = 0.5f * a;
        const float dev  = a * ch_half;

        const bool flipped = (t > warm) && (prev_trend != current_trend);

        if (current_trend == 0) {
            if (flipped) {
                up = down;
                out_buy_tm[idx] = up - atr2;
            } else {
                if (t == warm || up == 0.0f) up = max_low_price;
                else if (max_low_price > up) up = max_low_price;
            }
            out_halftrend_tm[idx] = up;
            out_atr_high_tm[idx]  = up + dev;
            out_atr_low_tm[idx]   = up - dev;
            out_trend_tm[idx]     = 0.0f;
        } else {
            if (flipped) {
                down = up;
                out_sell_tm[idx] = down + atr2;
            } else {
                if (t == warm || down == 0.0f) down = min_high_price;
                else if (min_high_price < down) down = min_high_price;
            }
            out_halftrend_tm[idx] = down;
            out_atr_high_tm[idx]  = down + dev;
            out_atr_low_tm[idx]   = down - dev;
            out_trend_tm[idx]     = 1.0f;
        }

        prev_low  = LDG(low_tm  + (t * cols + s));
        prev_high = LDG(high_tm + (t * cols + s));
        prev_trend = current_trend;
    }
}

}

#undef AT_TM
#undef HT_THREADS_PER_BLOCK
#undef LDG
