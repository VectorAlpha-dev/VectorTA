




#pragma once
#include <cuda_runtime.h>
#include <math.h>

struct dsf { float hi, lo; };

__device__ __forceinline__ dsf ds_set(float x)        { return {x, 0.0f}; }
__device__ __forceinline__ dsf ds_make(float hi, float lo){ return {hi, lo}; }

__device__ __forceinline__ dsf ds_norm(float s, float e) {

    float t  = s + e;
    float lo = e - (t - s);
    return {t, lo};
}


__device__ __forceinline__ dsf ds_add(dsf a, dsf b) {
    float s  = a.hi + b.hi;
    float z  = s - a.hi;
    float e  = (a.hi - (s - z)) + (b.hi - z);
    e += a.lo + b.lo;
    return ds_norm(s, e);
}

__device__ __forceinline__ dsf ds_neg(dsf a)          { return {-a.hi, -a.lo}; }
__device__ __forceinline__ dsf ds_sub(dsf a, dsf b)   { return ds_add(a, ds_neg(b)); }


__device__ __forceinline__ dsf ds_mul(dsf a, dsf b) {
    float p  = a.hi * b.hi;
    float e  = fmaf(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    e += a.lo * b.lo;
    return ds_norm(p, e);
}


__device__ __forceinline__ dsf ds_scale(dsf a, float s) {
    float p  = a.hi * s;
    float e  = fmaf(a.hi, s, -p) + a.lo * s;
    return ds_norm(p, e);
}


__device__ __forceinline__ dsf ds_fma(dsf a, dsf b, dsf c) {
    return ds_add(ds_mul(a, b), c);
}

__device__ __forceinline__ float ds_to_f(dsf a)       { return a.hi + a.lo; }
