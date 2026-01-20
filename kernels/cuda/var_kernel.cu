



#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>

__device__ __forceinline__ float dev_nan() { return __int_as_float(0x7fffffff); }


struct df32 { float hi, lo; };
__device__ __forceinline__ df32 make_df32(float2 a) { return {a.x, a.y}; }


__device__ __forceinline__ void two_diff(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - (b + bb);
}


__device__ __forceinline__ df32 df_sub(df32 a, df32 b) {
    float s, e;
    two_diff(a.hi, b.hi, s, e);
    e += a.lo - b.lo;

    float t1 = s + e;
    float t2 = e - (t1 - s);
    return {t1, t2};
}






#ifndef VAR_COMBO_TILE
#define VAR_COMBO_TILE 4
#endif


extern "C" __global__ void var_batch_f32(
    const float2* __restrict__ prefix_sum,
    const float2* __restrict__ prefix_sum_sq,
    const int*    __restrict__ prefix_nan,
    int len,
    int first_valid,
    const int*    __restrict__ periods,
    const float*  __restrict__ nbdev2,
    int n_combos,
    float*        __restrict__ out)
{
    const int group   = blockIdx.y;
    const int co_base = group * VAR_COMBO_TILE;


    __shared__ int   s_period[VAR_COMBO_TILE];
    __shared__ int   s_warm[VAR_COMBO_TILE];
    __shared__ float s_scale[VAR_COMBO_TILE];
    __shared__ float s_invden[VAR_COMBO_TILE];

    if (threadIdx.x < VAR_COMBO_TILE) {
        const int c = co_base + threadIdx.x;
        if (c < n_combos) {
            const int p = periods[c];
            s_period[threadIdx.x] = p;
            s_warm  [threadIdx.x] = first_valid + p - 1;
            s_scale [threadIdx.x] = nbdev2[c];
            s_invden[threadIdx.x] = 1.0f / (float)p;
        } else {
            s_period[threadIdx.x] = 0;
            s_warm  [threadIdx.x] = INT_MAX;
            s_scale [threadIdx.x] = 0.0f;
            s_invden[threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < len) {
        const int e_idx = t + 1;


        const df32 end_sum   = make_df32(prefix_sum   [e_idx]);
        const df32 end_sum2  = make_df32(prefix_sum_sq[e_idx]);
        const int  end_bad   = prefix_nan[e_idx];

        #pragma unroll
        for (int k = 0; k < VAR_COMBO_TILE; ++k) {
            const int combo = co_base + k;
            if (combo >= n_combos) break;

            float out_val = dev_nan();
            const int warm = s_warm[k];
            if (t >= warm) {
                const int p      = s_period[k];
                const int start  = e_idx - p;
                const int bad    = end_bad - prefix_nan[start];
                if (bad == 0) {

                    const df32 st_sum  = make_df32(prefix_sum   [start]);
                    const df32 st_sum2 = make_df32(prefix_sum_sq[start]);


                    const df32 win_sum_df  = df_sub(end_sum,  st_sum);
                    const df32 win_sum2_df = df_sub(end_sum2, st_sum2);
                    const float sum  = win_sum_df.hi  + win_sum_df.lo;
                    const float sum2 = win_sum2_df.hi + win_sum2_df.lo;


                    const float invden = s_invden[k];
                    const float mean   = sum * invden;
                    float var = fmaf(-sum, mean, sum2) * invden;
                    if (var < 0.0f) var = 0.0f;
                    out_val = var * s_scale[k];
                }
            }

            out[combo * len + t] = out_val;
        }

        t += stride;
    }
}





extern "C" __global__ void var_many_series_one_param_f32(
    const float2* __restrict__ prefix_sum_tm,
    const float2* __restrict__ prefix_sum_sq_tm,
    const int*    __restrict__ prefix_nan_tm,
    int period,
    int num_series,
    int series_len,
    const int*  __restrict__ first_valids,
    float nbdev2,
    float*      __restrict__ out_tm)
{
    const int series = blockIdx.y;
    if (series >= num_series) return;

    const int fv   = first_valids[series];
    const int warm = fv + period - 1;
    const float invden = 1.0f / (float)period;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (t < series_len) {
        const int idx = t * num_series + series;
        float out_val = dev_nan();

        if (t >= warm) {
            const int start_pre_t = t - period;
            const int w_end   = idx + 1;
            const int w_start = (start_pre_t >= 0) ? (start_pre_t * num_series + series + 1) : 0;

            const int bad = prefix_nan_tm[w_end] - prefix_nan_tm[w_start];
            if (bad == 0) {
                const df32 end_sum   = make_df32(prefix_sum_tm   [w_end]);
                const df32 end_sum2  = make_df32(prefix_sum_sq_tm[w_end]);
                const df32 st_sum    = make_df32(prefix_sum_tm   [w_start]);
                const df32 st_sum2   = make_df32(prefix_sum_sq_tm[w_start]);

                const df32 win_sum_df  = df_sub(end_sum,  st_sum);
                const df32 win_sum2_df = df_sub(end_sum2, st_sum2);

                const float sum  = win_sum_df.hi  + win_sum_df.lo;
                const float sum2 = win_sum2_df.hi + win_sum2_df.lo;

                const float mean = sum * invden;
                float var = fmaf(-sum, mean, sum2) * invden;
                if (var < 0.0f) var = 0.0f;
                out_val = var * nbdev2;
            }
        }
        out_tm[idx] = out_val;
        t += stride;
    }
}
