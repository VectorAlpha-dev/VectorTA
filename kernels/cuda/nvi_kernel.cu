




#include <cuda_runtime.h>
#include <math_constants.h>


struct dsfloat {
    float hi;
    float lo;
};

__device__ __forceinline__ dsfloat ds_make(float x) {
    dsfloat a; a.hi = x; a.lo = 0.0f; return a;
}


__device__ __forceinline__ void ds_renorm(dsfloat& a, float t) {
    float s = a.hi + t;
    a.lo    = t - (s - a.hi);
    a.hi    = s;
}


__device__ __forceinline__ dsfloat ds_add(dsfloat a, dsfloat b) {

    float s  = a.hi + b.hi;
    float bb = s - a.hi;
    float err = (a.hi - (s - bb)) + (b.hi - bb);

    float t = a.lo + b.lo + err;

    dsfloat r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}


__device__ __forceinline__ dsfloat ds_mul_scalar(dsfloat a, float b) {
    float p = a.hi * b;
    float e = fmaf(a.hi, b, -p);
    float t = a.lo * b + e;
    dsfloat r;
    r.hi = p + t;
    r.lo = t - (r.hi - p);
    return r;
}


__device__ __forceinline__ float ds_to_float(dsfloat a) {
    return a.hi + a.lo;
}


extern "C" __global__ void nvi_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{
    if (len <= 0) return;



    if (blockIdx.x != 0) return;


    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 16) return;
    const unsigned mask = 0x0000ffffu;

    const int fv = first_valid < 0 ? 0 : first_valid;


    const float nan_f = CUDART_NAN_F;
    for (int i = lane; i < fv && i < len; i += 16) out[i] = nan_f;
    if (fv >= len) return;


    if (lane == 0) out[fv] = 1000.0f;
    if (fv + 1 >= len) return;

    double nvi0 = 1000.0;

    for (int t0 = fv + 1; t0 < len; t0 += 16) {
        const int i = t0 + lane;
        double f = 1.0;
        if (i < len) {
            const float c = close[i];
            const float c0 = close[i - 1];
            const float v = volume[i];
            const float v0 = volume[i - 1];
            if (v < v0) {
                const float pct = (c - c0) / c0;
                f = 1.0 + (double)pct;
            }
        }


        double prefix = f;
        for (int offset = 1; offset < 16; offset <<= 1) {
            double other = __shfl_up_sync(mask, prefix, offset, 16);
            if (lane >= offset) prefix *= other;
        }

        double base = __shfl_sync(mask, nvi0, 0, 16);
        if (i < len) out[i] = (float)(base * prefix);

        double tile_prod = __shfl_sync(mask, prefix, 15, 16);
        if (lane == 0) nvi0 *= tile_prod;
    }
}


extern "C" __global__ void nvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    if (rows <= 0 || cols <= 0) return;
    const float nan_f = CUDART_NAN_F;


    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const int fv = first_valids[s] < 0 ? 0 : first_valids[s];


        if (fv >= rows) {
            for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = nan_f;
            continue;
        }


        for (int t = 0; t < fv; ++t) out_tm[t * cols + s] = nan_f;


        dsfloat nvi = ds_make(1000.0f);
        out_tm[fv * cols + s] = ds_to_float(nvi);
        if (fv + 1 >= rows) continue;

        float prev_close  = close_tm[fv * cols + s];
        float prev_volume = volume_tm[fv * cols + s];

        for (int t = fv + 1; t < rows; ++t) {
            const float c = close_tm[t * cols + s];
            const float v = volume_tm[t * cols + s];

            if (v < prev_volume) {
                const float pct = (c - prev_close) / prev_close;
                dsfloat prod = ds_mul_scalar(nvi, pct);
                nvi = ds_add(nvi, prod);
            }
            out_tm[t * cols + s] = ds_to_float(nvi);
            prev_close  = c;
            prev_volume = v;
        }
    }
}

