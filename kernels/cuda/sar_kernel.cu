









#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>


static __forceinline__ __device__ float f32_nan() { return __int_as_float(0x7fffffff); }


struct dsf { float hi, lo; };

static __forceinline__ __device__ dsf ds_from_f32(float x) { return dsf{x, 0.0f}; }

static __forceinline__ __device__ void two_sum(float a, float b, float &s, float &e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);
}
static __forceinline__ __device__ void two_prod(float a, float b, float &p, float &e) {
    p = a * b;

    e = fmaf(a, b, -p);
}
static __forceinline__ __device__ dsf ds_add(dsf a, dsf b) {
    float s, e; two_sum(a.hi, b.hi, s, e);
    float t = a.lo + b.lo;
    float hi, lo; two_sum(s, e + t, hi, lo);
    return dsf{hi, lo};
}
static __forceinline__ __device__ dsf ds_sub(dsf a, dsf b) {
    float s, e; two_sum(a.hi, -b.hi, s, e);
    float t = a.lo - b.lo;
    float hi, lo; two_sum(s, e + t, hi, lo);
    return dsf{hi, lo};
}
static __forceinline__ __device__ dsf ds_mul(dsf a, dsf b) {
    float p, pe; two_prod(a.hi, b.hi, p, pe);

    float cross = fmaf(a.hi, b.lo, a.lo * b.hi);
    float s, e; two_sum(p, pe + cross, s, e);

    e += a.lo * b.lo;
    float hi, lo; two_sum(s, e, hi, lo);
    return dsf{hi, lo};
}
static __forceinline__ __device__ dsf ds_fma(dsf a, dsf b, dsf c) {
    dsf m = ds_mul(a, b);
    return ds_add(m, c);
}
static __forceinline__ __device__ float ds_to_f32(dsf a) { return a.hi + a.lo; }
static __forceinline__ __device__ bool float_lt_ds(float x, dsf a) {
    if (x < a.hi) return true;
    if (x > a.hi) return false;
    return 0.0f < a.lo;
}
static __forceinline__ __device__ bool float_gt_ds(float x, dsf a) {
    if (x > a.hi) return true;
    if (x < a.hi) return false;
    return 0.0f > a.lo;
}
static __forceinline__ __device__ bool ds_gt_float(dsf a, float x) {
    if (a.hi > x) return true;
    if (a.hi < x) return false;
    return a.lo > 0.0f;
}
static __forceinline__ __device__ bool ds_lt_float(dsf a, float x) {
    if (a.hi < x) return true;
    if (a.hi > x) return false;
    return a.lo < 0.0f;
}
static __forceinline__ __device__ dsf ds_min_float(dsf a, float x) {

    if (a.hi < x) return a;
    if (a.hi > x) return ds_from_f32(x);

    return (a.lo <= 0.0f) ? a : ds_from_f32(x);
}
static __forceinline__ __device__ dsf ds_max_float(dsf a, float x) {

    if (a.hi > x) return a;
    if (a.hi < x) return ds_from_f32(x);
    return (a.lo >= 0.0f) ? a : ds_from_f32(x);
}



static __forceinline__ __device__ void warp_broadcast_hilo(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int i,
    float &hi_out,
    float &lo_out)
{
    unsigned mask   = __activemask();
    int lane        = threadIdx.x & 31;
    int leader_lane = __ffs(mask) - 1;

    float hi_lane = 0.0f, lo_lane = 0.0f;
    if (lane == leader_lane) {
        hi_lane = high[i];
        lo_lane = low[i];
    }
    hi_out = __shfl_sync(mask, hi_lane, leader_lane);
    lo_out = __shfl_sync(mask, lo_lane, leader_lane);
}









extern "C" __global__ void sar_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    int len,
    int first_valid,
    const float* __restrict__ accs,
    const float* __restrict__ maxes,
    int n_rows,
    float* __restrict__ out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int base = row * len;
    const int warm = first_valid + 1;


    const float acceleration = accs[row];
    const float maximum      = maxes[row];


    const int warm_upto = (warm < len) ? warm : len;
    for (int t = 0; t < warm_upto; ++t) {
        out[base + t] = f32_nan();
    }
    if (warm >= len) return;


    if (acceleration <= 0.0f || maximum <= 0.0f) {
        for (int t = warm; t < len; ++t) out[base + t] = f32_nan();
        return;
    }


    const int i0 = first_valid;
    const int i1 = warm;

    const float h0 = high[i0];
    const float h1 = high[i1];
    const float l0 = low[i0];
    const float l1 = low[i1];

    bool  trend_up = (h1 > h0);

    dsf sar = ds_from_f32(trend_up ? l0 : h0);
    dsf ep  = ds_from_f32(trend_up ? h1 : l1);
    dsf acc = ds_from_f32(acceleration);
    const dsf acc_incr = ds_from_f32(acceleration);
    const dsf acc_max  = ds_from_f32(maximum);


    out[base + i1] = ds_to_f32(sar);


    float low_prev2  = l0;
    float low_prev   = l1;
    float high_prev2 = h0;
    float high_prev  = h1;


    #pragma unroll 1
    for (int i = i1 + 1; i < len; ++i) {
        float hi, lo;
        warp_broadcast_hilo(high, low, i, hi, lo);


        dsf diff = ds_sub(ep, sar);
        dsf prod = ds_mul(acc, diff);
        dsf next_sar = ds_add(prod, sar);


        if (trend_up) {

            if (float_lt_ds(lo, next_sar)) {

                trend_up = false;
                next_sar = ep;
                ep       = ds_from_f32(lo);
                acc      = acc_incr;
            } else {

                if (float_gt_ds(hi, ep)) {
                    ep  = ds_from_f32(hi);

                    dsf acc_plus = ds_add(acc, acc_incr);


                    if (ds_gt_float(acc_plus, maximum)) {
                        acc = acc_max;
                    } else {
                        acc = acc_plus;
                    }
                }

                next_sar = ds_min_float(next_sar, low_prev);
                next_sar = ds_min_float(next_sar, low_prev2);
            }
        } else {
            if (float_gt_ds(hi, next_sar)) {

                trend_up = true;
                next_sar = ep;
                ep       = ds_from_f32(hi);
                acc      = acc_incr;
            } else {

                if (float_lt_ds(lo, ep)) {
                    ep  = ds_from_f32(lo);
                    dsf acc_plus = ds_add(acc, acc_incr);
                    if (ds_gt_float(acc_plus, maximum)) {
                        acc = acc_max;
                    } else {
                        acc = acc_plus;
                    }
                }


                if (ds_lt_float(next_sar, high_prev)) next_sar = ds_from_f32(high_prev);
                if (ds_lt_float(next_sar, high_prev2)) next_sar = ds_from_f32(high_prev2);
            }
        }

        out[base + i] = ds_to_f32(next_sar);
        sar           = next_sar;


        low_prev2  = low_prev;   low_prev  = lo;
        high_prev2 = high_prev;  high_prev = hi;
    }
}









extern "C" __global__ void sar_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    float acceleration,
    float maximum,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            out_tm[t * cols + s] = f32_nan();
        }
        return;
    }

    const int warm = fv + 1;


    for (int t = threadIdx.x; t < min(warm, rows); t += blockDim.x) {
        out_tm[t * cols + s] = f32_nan();
    }

    if (threadIdx.x != 0) return;
    if (warm >= rows) return;


    const int i0 = fv;
    const int i1 = warm;
    float h0 = high_tm[i0 * cols + s];
    float h1 = high_tm[i1 * cols + s];
    float l0 = low_tm[i0 * cols + s];
    float l1 = low_tm[i1 * cols + s];

    bool trend_up = (h1 > h0);
    float sar = trend_up ? l0 : h0;
    float ep = trend_up ? h1 : l1;
    float acc = acceleration;

    out_tm[i1 * cols + s] = sar;

    float low_prev2 = l0;
    float low_prev  = l1;
    float high_prev2 = h0;
    float high_prev  = h1;

    for (int i = i1 + 1; i < rows; ++i) {
        const float hi = high_tm[i * cols + s];
        const float lo = low_tm[i * cols + s];

        float next_sar = fmaf(acc, (ep - sar), sar);
        if (trend_up) {
            if (lo < next_sar) {
                trend_up = false;
                next_sar = ep;
                ep = lo;
                acc = acceleration;
            } else {
                if (hi > ep) { ep = hi; acc = fminf(acc + acceleration, maximum); }
                next_sar = fminf(next_sar, fminf(low_prev, low_prev2));
            }
        } else {
            if (hi > next_sar) {
                trend_up = true;
                next_sar = ep;
                ep = hi;
                acc = acceleration;
            } else {
                if (lo < ep) { ep = lo; acc = fminf(acc + acceleration, maximum); }
                next_sar = fmaxf(next_sar, fmaxf(high_prev, high_prev2));
            }
        }
        out_tm[i * cols + s] = next_sar;
        sar = next_sar;

        low_prev2 = low_prev;
        low_prev = lo;
        high_prev2 = high_prev;
        high_prev = hi;
    }
}

