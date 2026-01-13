










#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

static __forceinline__ __device__ float qnan32() {
    return __int_as_float(0x7fffffff);
}


static __forceinline__ __device__ float warp_sum(float v) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) v += __shfl_down_sync(mask, v, ofs);
    return v;
}


static __forceinline__ __device__ void kahan_add(float &sum, float &c, float x) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}





extern "C" __global__
void mean_ad_batch_f32(const float* __restrict__ prices,
                       const int*   __restrict__ periods,
                       const int*   __restrict__ warm_indices,
                       int first_valid,
                       int series_len,
                       int n_combos,
                       int max_period,
                       float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;


    const int lane             = threadIdx.x & 31;
    const int warp_in_block    = threadIdx.x >> 5;
    const int warps_per_block  = blockDim.x >> 5;
    const int warp_global      = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps_grid = gridDim.x * warps_per_block;


    extern __shared__ float s_ring[];
    float* ring = s_ring + (size_t)warp_in_block * (size_t)max_period;

    for (int combo = warp_global; combo < n_combos; combo += total_warps_grid) {

        const int period = periods[combo];
        if (period <= 0) continue;

        const int warm = warm_indices[combo];
        const size_t base = (size_t)combo * (size_t)series_len;


        const int nan_end = (warm < series_len ? warm : series_len);
        for (int t = lane; t < nan_end; t += 32) {
            out[base + t] = qnan32();
        }

        if (warm >= series_len) continue;
        if (first_valid + period > series_len) continue;


        float partial = 0.0f;
        for (int k = lane; k < period; k += 32) {
            partial += prices[first_valid + k];
        }
        float sum = warp_sum(partial);
        sum = __shfl_sync(__activemask(), sum, 0);

        const float inv_p = 1.0f / (float)period;
        float sma = sum * inv_p;


        if (lane == 0) {
            int head = 0;
            float residual_sum = 0.0f, c_res = 0.0f;
            float c_sum = 0.0f;

            const int start_t = first_valid + period - 1;
            const int fill_end = min(start_t + period - 1, series_len - 1);


            for (int t = start_t; t <= fill_end; ++t) {
                const float r = fabsf(prices[t] - sma);
                ring[head++] = r; if (head == period) head = 0;
                kahan_add(residual_sum, c_res, r);

                if (t + 1 < series_len) {
                    const float in_next  = prices[t + 1];
                    const float out_prev = prices[t + 1 - period];
                    kahan_add(sum, c_sum,  in_next);
                    kahan_add(sum, c_sum, -out_prev);
                    sma = sum * inv_p;
                }
            }


            out[base + warm] = residual_sum * inv_p;


            int t = start_t + period;
            int idx = head;
            while (t < series_len) {
                const float r   = fabsf(prices[t] - sma);
                const float old = ring[idx];
                ring[idx] = r;
                idx += 1; if (idx == period) idx = 0;


                kahan_add(residual_sum, c_res,  r);
                kahan_add(residual_sum, c_res, -old);

                out[base + t] = residual_sum * inv_p;

                if (t + 1 < series_len) {
                    const float in_next  = prices[t + 1];
                    const float out_prev = prices[t + 1 - period];
                    kahan_add(sum, c_sum,  in_next);
                    kahan_add(sum, c_sum, -out_prev);
                    sma = sum * inv_p;
                }
                ++t;
            }
        }
    }
}






#ifndef SMALL_PERIOD_MAX
#define SMALL_PERIOD_MAX 64
#endif

extern "C" __global__
void mean_ad_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                       int period,
                                       int num_series,
                                       int series_len,
                                       const int* __restrict__ first_valids,
                                       float* __restrict__ out_tm)
{
    if (period <= 0 || num_series <= 0 || series_len <= 0) return;

    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (series_idx >= num_series) return;

    const int first = first_valids[series_idx];
    if (first < 0 || first >= series_len) return;

    const int warm = first + 2 * period - 2;
    const int stride = num_series;


    const int nan_end = (warm < series_len ? warm : series_len);
    for (int t = 0; t < nan_end; ++t) {
        out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = qnan32();
    }
    if (warm >= series_len) return;


    float sum = 0.0f, c_sum = 0.0f;
    size_t p = (size_t)first * (size_t)stride + (size_t)series_idx;
    for (int k = 0; k < period; ++k) {
        kahan_add(sum, c_sum, prices_tm[p]);
        p += (size_t)stride;
    }
    const float inv_p = 1.0f / (float)period;
    float sma = sum * inv_p;


    float residual_sum = 0.0f, c_res = 0.0f;
    int head = 0;


    float ring_reg[SMALL_PERIOD_MAX];
    float* ring = nullptr;

    if (period <= SMALL_PERIOD_MAX) {
        const int start_t = first + period - 1;
        const int fill_end = min(start_t + period - 1, series_len - 1);
        for (int t = start_t; t <= fill_end; ++t) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r = fabsf(price_t - sma);
            ring_reg[head++] = r; if (head == period) head = 0;
            kahan_add(residual_sum, c_res, r);

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
        }
        out_tm[(size_t)warm * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

        int t = first + 2 * period - 1;
        int idx = head;
        while (t < series_len) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r   = fabsf(price_t - sma);
            const float old = ring_reg[idx];
            ring_reg[idx] = r;
            idx += 1; if (idx == period) idx = 0;

            kahan_add(residual_sum, c_res,  r);
            kahan_add(residual_sum, c_res, -old);

            out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
            ++t;
        }
    } else {

        extern __shared__ float smem[];
        ring = smem + (size_t)threadIdx.x * (size_t)period;

        const int start_t = first + period - 1;
        const int fill_end = min(start_t + period - 1, series_len - 1);
        for (int t = start_t; t <= fill_end; ++t) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r = fabsf(price_t - sma);
            ring[head++] = r; if (head == period) head = 0;
            kahan_add(residual_sum, c_res, r);

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
        }
        out_tm[(size_t)warm * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

        int t = first + 2 * period - 1;
        int idx = head;
        while (t < series_len) {
            const float price_t = prices_tm[(size_t)t * (size_t)stride + (size_t)series_idx];
            const float r   = fabsf(price_t - sma);
            const float old = ring[idx];
            ring[idx] = r;
            idx += 1; if (idx == period) idx = 0;

            kahan_add(residual_sum, c_res,  r);
            kahan_add(residual_sum, c_res, -old);

            out_tm[(size_t)t * (size_t)stride + (size_t)series_idx] = residual_sum * inv_p;

            if (t + 1 < series_len) {
                const float in_next  = prices_tm[(size_t)(t + 1) * (size_t)stride + (size_t)series_idx];
                const float out_prev = prices_tm[(size_t)(t + 1 - period) * (size_t)stride + (size_t)series_idx];
                kahan_add(sum, c_sum,  in_next);
                kahan_add(sum, c_sum, -out_prev);
                sma = sum * inv_p;
            }
            ++t;
        }
    }
}

