






#ifndef AD_ACCUM_MODE
#define AD_ACCUM_MODE 2
#endif

#ifndef AD_USE_FAST_DIV
#define AD_USE_FAST_DIV 0
#endif

#ifndef AD_BLOCK_SIZE_TM
#define AD_BLOCK_SIZE_TM 256
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#if AD_USE_FAST_DIV
  #define AD_DIV(x,y) __fdividef((x),(y))
#else
  #define AD_DIV(x,y) ((x)/(y))
#endif





__device__ __forceinline__ float ad_mfv_f32(float h, float l, float c, float v)
{
    float hl  = h - l;
    if (hl == 0.0f) return 0.0f;


    float num = fmaf(2.0f, c, -(h + l));
    float m   = AD_DIV(num, hl);
    return m * v;
}


struct Kahan32 {
    float s, c;
    __device__ __forceinline__ Kahan32() : s(0.f), c(0.f) {}
    __device__ __forceinline__ float add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
        return s;
    }
};


struct TwoSum32 {
    float hi, lo;
    __device__ __forceinline__ TwoSum32() : hi(0.f), lo(0.f) {}
    __device__ __forceinline__ void add_inplace(float x) {
        float s  = hi + x;
        float bp = s - hi;
        float err1 = (hi - (s - bp)) + (x - bp);
        float t  = lo + err1;
        float s2 = s + t;
        float bq = s2 - s;
        float err2 = (s - (s2 - bq)) + (t - bq);
        hi = s2;
        lo = err2;
    }
    __device__ __forceinline__ float value() const { return hi + lo; }
};



extern "C" __global__ void ad_series_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int n_series,
    float* __restrict__ out)
{
    int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= n_series || len <= 0) return;

    int offset = series * len;
    const float* __restrict__ h = high   + offset;
    const float* __restrict__ l = low    + offset;
    const float* __restrict__ c = close  + offset;
    const float* __restrict__ v = volume + offset;
    float* __restrict__ o       = out    + offset;



    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        double hl = (double)h[i] - (double)l[i];
        if (hl != 0.0) {
            double num = (double)2.0 * (double)c[i] - (double)h[i] - (double)l[i];
            double mfv = (num / hl) * (double)v[i];
            sum += mfv;
        }
        o[i] = (float)sum;
    }
}





extern "C" __global__ void ad_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int num_series,
    int series_len,
    float* __restrict__ out_tm)
{
    int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series || series_len <= 0) return;

#if AD_ACCUM_MODE == 3
    double sum = 0.0;
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series;
        double hl = (double)high_tm[idx] - (double)low_tm[idx];
        if (hl != 0.0) {
            double num = (double)2.0 * (double)close_tm[idx]
                       - (double)high_tm[idx] - (double)low_tm[idx];
            sum += (num / hl) * (double)volume_tm[idx];
        }
        out_tm[idx] = (float)sum;
    }
#elif AD_ACCUM_MODE == 2
    TwoSum32 acc;
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series;
        float mfv = ad_mfv_f32(high_tm[idx], low_tm[idx], close_tm[idx], volume_tm[idx]);
        acc.add_inplace(mfv);
        out_tm[idx] = acc.value();
    }
#elif AD_ACCUM_MODE == 1
    Kahan32 acc;
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series;
        float mfv = ad_mfv_f32(high_tm[idx], low_tm[idx], close_tm[idx], volume_tm[idx]);
        out_tm[idx] = acc.add(mfv);
    }
#else
    float sum = 0.f;
    for (int t = 0; t < series_len; ++t) {
        int idx = t * num_series + series;
        sum += ad_mfv_f32(high_tm[idx], low_tm[idx], close_tm[idx], volume_tm[idx]);
        out_tm[idx] = sum;
    }
#endif
}

