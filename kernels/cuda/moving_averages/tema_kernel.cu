





#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef TEMA_WARPS_PER_BLOCK

#define TEMA_WARPS_PER_BLOCK 4
#endif



#ifndef TEMA_QNAN_U32
#define TEMA_QNAN_U32 0x7fc00000u
#endif

static __device__ __forceinline__ float tema_qnan() {
    return __int_as_float((int)TEMA_QNAN_U32);
}


static __device__ __forceinline__ float ema_step(float prev, float x, float alpha) {
    return fmaf(alpha, x - prev, prev);
}


static __device__ __forceinline__ float warp_broadcast0(float v) {
    unsigned m = __activemask();            
    return __shfl_sync(m, v, 0);            
}






extern "C" __global__
void tema_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    if (series_len <= 0 || n_combos <= 0) return;

    
    const int lane         = threadIdx.x & 31;
    const int warp_in_block= threadIdx.x >> 5; 
    const int warps_pb     = blockDim.x >> 5;
    if (warps_pb <= 0) return;
    const int combo        = blockIdx.x * warps_pb + warp_in_block;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const size_t base_out = static_cast<size_t>(combo) * static_cast<size_t>(series_len);

    
    if (period <= 0 || first_valid >= series_len) {
        if (lane == 0) {
            const float qn = tema_qnan();
            for (int i = 0; i < series_len; ++i) out[base_out + static_cast<size_t>(i)] = qn;
        }
        return;
    }

    
    const int lookback   = (period - 1) * 3;
    const int ema2_start = first_valid + (period - 1);
    const int ema3_start = first_valid + 2 * (period - 1);
    const int warm       = first_valid + lookback;

    
    if (lane == 0) {
        const int nan_to = warm < series_len ? warm : series_len;
        const float qn = tema_qnan();
        for (int i = 0; i < nan_to; ++i) out[base_out + static_cast<size_t>(i)] = qn;
    }
    if (warm >= series_len) {
        
        return;
    }

    
    const float alpha = 2.0f / (float(period) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    
    float ema1_prev = 0.0f;
    float ema2_prev = 0.0f;
    float ema3_prev = 0.0f;
    if (lane == 0) {
        float ema1 = prices[first_valid];
        float ema2 = 0.0f;
        float ema3 = 0.0f;

        
        int end0 = ema2_start;
        if (end0 > warm) end0 = warm;
        for (int t = first_valid; t < end0; ++t) {
            ema1 = ema_step(ema1, prices[t], alpha);
        }

        
        if (ema2_start < warm) {
            
            const float px = prices[ema2_start];
            ema1 = ema_step(ema1, px, alpha);
            ema2 = ema1;
            ema2 = ema_step(ema2, ema1, alpha);

            int end1 = ema3_start;
            if (end1 > warm) end1 = warm;
            for (int t = ema2_start + 1; t < end1; ++t) {
                const float p = prices[t];
                ema1 = ema_step(ema1, p, alpha);
                ema2 = ema_step(ema2, ema1, alpha);
            }

            
            if (ema3_start < warm) {
                
                const float p = prices[ema3_start];
                ema1 = ema_step(ema1, p, alpha);
                ema2 = ema_step(ema2, ema1, alpha);
                ema3 = ema2;
                ema3 = ema_step(ema3, ema2, alpha);

                for (int t = ema3_start + 1; t < warm; ++t) {
                    const float p2 = prices[t];
                    ema1 = ema_step(ema1, p2, alpha);
                    ema2 = ema_step(ema2, ema1, alpha);
                    ema3 = ema_step(ema3, ema2, alpha);
                }
            }
        }

        ema1_prev = ema1;
        ema2_prev = ema2;
        ema3_prev = ema3;
    }

    
    const unsigned mask = 0xffffffffu;
    ema1_prev = __shfl_sync(mask, ema1_prev, 0);
    ema2_prev = __shfl_sync(mask, ema2_prev, 0);
    ema3_prev = __shfl_sync(mask, ema3_prev, 0);

    
    for (int t0 = warm; t0 < series_len; t0 += 32) {
        const int t = t0 + lane;

        
        float A1 = 1.0f;
        float B1 = 0.0f;
        if (t < series_len) {
            A1 = one_minus_alpha;
            B1 = alpha * prices[t];
        }
        float A = A1;
        float B = B1;
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
        const float ema1 = fmaf(A, ema1_prev, B);

        
        float A2 = 1.0f;
        float B2 = 0.0f;
        if (t < series_len) {
            A2 = one_minus_alpha;
            B2 = alpha * ema1;
        }
        A = A2;
        B = B2;
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
        const float ema2 = fmaf(A, ema2_prev, B);

        
        float A3 = 1.0f;
        float B3 = 0.0f;
        if (t < series_len) {
            A3 = one_minus_alpha;
            B3 = alpha * ema2;
        }
        A = A3;
        B = B3;
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
        const float ema3 = fmaf(A, ema3_prev, B);

        if (t < series_len) {
            out[base_out + static_cast<size_t>(t)] = fmaf(3.0f, (ema1 - ema2), ema3);
        }

        
        const int remaining = series_len - t0;
        const int last_lane = remaining >= 32 ? 31 : (remaining - 1);
        ema1_prev = __shfl_sync(mask, ema1, last_lane);
        ema2_prev = __shfl_sync(mask, ema2, last_lane);
        ema3_prev = __shfl_sync(mask, ema3, last_lane);
    }
}





extern "C" __global__
__launch_bounds__(TEMA_WARPS_PER_BLOCK * 32, 2)
void tema_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     int period,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    if (series_len <= 0 || period <= 0 || num_series <= 0) return;

    const int lane     = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_global   = blockIdx.x * TEMA_WARPS_PER_BLOCK + warp_in_block;
    const int sidx          = warp_global * 32 + lane;
    if (sidx >= num_series) return;

    int fv = first_valids[sidx];
    if (fv < 0) fv = 0;

    const int lookback   = (period - 1) * 3;
    const int ema2_start = fv + (period - 1);
    const int ema3_start = fv + 2 * (period - 1);
    const int warm       = fv + lookback;

    
    if (fv >= series_len) {
        float qn = tema_qnan();
        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + sidx] = qn;
        }
        return;
    } else {
        float qn = tema_qnan();
        const int nan_to = warm < series_len ? warm : series_len;
        for (int t = 0; t < nan_to; ++t) {
            out_tm[t * num_series + sidx] = qn;
        }
    }

    const float alpha = 2.0f / (float(period) + 1.0f);

    
    float ema1 = prices_tm[fv * num_series + sidx];
    float ema2 = 0.0f;
    float ema3 = 0.0f;

    
    for (int t = fv; t < series_len; ++t) {
        const float px = prices_tm[t * num_series + sidx];

        ema1 = ema_step(ema1, px, alpha);

        if (t >= ema2_start) {
            if (t == ema2_start) ema2 = ema1;
            ema2 = ema_step(ema2, ema1, alpha);
        }
        if (t >= ema3_start) {
            if (t == ema3_start) ema3 = ema2;
            ema3 = ema_step(ema3, ema2, alpha);
        }
        if (t >= warm) {
            out_tm[t * num_series + sidx] = fmaf(3.0f, (ema1 - ema2), ema3);
        }
    }
}
