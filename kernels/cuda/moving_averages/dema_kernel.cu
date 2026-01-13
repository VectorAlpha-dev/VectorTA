












#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>




#ifndef USE_DEMA_COMPENSATION
#define USE_DEMA_COMPENSATION 0
#endif



#ifndef DEMA_INIT_NANS_IN_KERNEL
#define DEMA_INIT_NANS_IN_KERNEL 1
#endif





extern "C" __global__
void dema_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    int series_len,
                    int first_valid,
                    int n_combos,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (series_len <= 0)   return;

    const int base = combo * series_len;

    const int period = periods[combo];
    const bool invalid =
        (period <= 0) ||
        (first_valid < 0) ||
        (first_valid >= series_len) ||
        (period > (series_len - first_valid));

    if (invalid) {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out[base + i] = NAN;
        }
        return;
    }

    const float alpha = 2.0f / (static_cast<float>(period) + 1.0f);
    const int   warm  = first_valid + period - 1;

#if DEMA_INIT_NANS_IN_KERNEL

    const int nan_end = (warm < series_len ? warm : series_len);
    for (int i = threadIdx.x; i < nan_end; i += blockDim.x) {
        out[base + i] = NAN;
    }
#endif


    if (threadIdx.x != 0)  return;


    int t = first_valid;
    float ema  = prices[t];
    float ema2 = ema;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif

    if (t >= warm) {
        out[base + t] = 2.0f * ema - ema2;
    }


    for (++t; t < series_len; ++t) {
        const float x = prices[t];
#if USE_DEMA_COMPENSATION
        float inc1 = fmaf(alpha, x - ema, -c1);
        float tmp1 = ema + inc1;
        c1 = (tmp1 - ema) - inc1;
        ema = tmp1;

        float inc2 = fmaf(alpha, ema - ema2, -c2);
        float tmp2 = ema2 + inc2;
        c2 = (tmp2 - ema2) - inc2;
        ema2 = tmp2;
#else
        ema  = fmaf(alpha, x   - ema,    ema);
        ema2 = fmaf(alpha, ema - ema2,   ema2);
#endif
        if (t >= warm) {
            out[base + t] = fmaf(2.0f, ema, -ema2);
        }
    }
}







extern "C" __global__
void dema_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int period,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)
{
    if (period <= 0 || series_len <= 0) return;

    const int lane      = threadIdx.x & 31;
    const int warps_pb  = blockDim.x >> 5;
    const int warp_id   = threadIdx.x >> 5;
    const int warp_gbl  = blockIdx.x * warps_pb + warp_id;
    const int series0   = warp_gbl * 32;
    const int series_idx= series0 + lane;
    if (series_idx >= num_series) return;

    const int   stride  = num_series;
    const int   fv      = first_valids[series_idx];
    const float alpha   = 2.0f / (static_cast<float>(period) + 1.0f);
    const int   warm    = fv + period - 1;

#if DEMA_INIT_NANS_IN_KERNEL

    const int nan_end = (warm < series_len ? warm : series_len);


    for (int t = 0; t < nan_end; ++t) {
        out_tm[(size_t)t * stride + series_idx] = NAN;
    }
#endif

    if (fv >= series_len) {
        return;
    }



    bool  started = false;
    float ema = 0.0f, ema2 = 0.0f;
#if USE_DEMA_COMPENSATION
    float c1 = 0.0f, c2 = 0.0f;
#endif


    const float* x_ptr = prices_tm + series_idx;
    float*       y_ptr = out_tm    + series_idx;

    for (int t = 0; t < series_len; ++t) {
        const float x = *x_ptr;

        if (!started && t == fv) {
            started = true;
            ema  = x;
            ema2 = x;

            if (t >= warm) {
                *y_ptr = 2.0f * ema - ema2;
            }
        } else if (started) {
#if USE_DEMA_COMPENSATION
            float inc1 = fmaf(alpha, x - ema, -c1);
            float tmp1 = ema + inc1;
            c1 = (tmp1 - ema) - inc1;
            ema = tmp1;

            float inc2 = fmaf(alpha, ema - ema2, -c2);
            float tmp2 = ema2 + inc2;
            c2 = (tmp2 - ema2) - inc2;
            ema2 = tmp2;
#else
            ema  = fmaf(alpha, x   - ema,    ema);
            ema2 = fmaf(alpha, ema - ema2,   ema2);
#endif
            if (t >= warm) {
                *y_ptr = fmaf(2.0f, ema, -ema2);
            }
        }

        x_ptr += stride;
        y_ptr += stride;
    }
}
