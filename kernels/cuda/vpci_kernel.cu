



#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include "ds_float2.cuh"

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float nan_f32() { return __int_as_float(0x7fffffff); }


__device__ __forceinline__ dsf load_dsf_f2(const float2* __restrict__ p, int idx) {
    float2 v = p[idx];
    return ds_make(v.x, v.y);
}


__device__ __forceinline__ dsf ds_div(dsf num, dsf den) {
    if (den.hi == 0.0f && den.lo == 0.0f) return ds_make(nan_f32(), 0.0f);
    float q1 = num.hi / den.hi;
    dsf t = ds_scale(den, q1);
    dsf r = ds_sub(num, t);
    float q2 = r.hi / den.hi;

    float s = q1 + q2;
    float e = q2 - (s - q1);
    return ds_norm(s, e);
}


__device__ __forceinline__ void kahan_add(float x, float& sum, float& c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


__device__ __forceinline__ float warp_bcast_f32_first(float v_any) {
    unsigned mask = __activemask();
    int first = __ffs(mask) - 1;
    return __shfl_sync(mask, v_any, first);
}
__device__ __forceinline__ dsf warp_bcast_dsf_first(dsf v_any) {
    unsigned mask = __activemask();
    int first = __ffs(mask) - 1;
    float hi = __shfl_sync(mask, v_any.hi, first);
    float lo = __shfl_sync(mask, v_any.lo, first);
    return ds_make(hi, lo);
}



extern "C" __global__ void vpci_batch_f32(
    const float2* __restrict__ pfx_c,
    const float2* __restrict__ pfx_v,
    const float2* __restrict__ pfx_cv,
    const float*  __restrict__ volume,
    const int*    __restrict__ shorts,
    const int*    __restrict__ longs,
    int series_len,
    int n_rows,
    int first_valid,
    float* __restrict__ out_vpci,
    float* __restrict__ out_vpcis
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int short_p = shorts[row];
    const int long_p  = longs[row];
    const int base    = row * series_len;
    float* __restrict__ y_vpci  = out_vpci  + base;
    float* __restrict__ y_vpcis = out_vpcis + base;

    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > series_len || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int tail = series_len - first_valid;
    if (UNLIKELY(tail < long_p)) {
        for (int i = 0; i < series_len; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }
        return;
    }

    const int warm = first_valid + long_p - 1;


    for (int i = 0; i < warm; ++i) { y_vpci[i] = nan_f32(); y_vpcis[i] = nan_f32(); }

    const float inv_long  = 1.0f / (float)long_p;
    const float inv_short = 1.0f / (float)short_p;

    float sum_vpci_vol_short = 0.0f;
    float sum_comp           = 0.0f;

    for (int i = warm; i < series_len; ++i) {
        const int idx_long_prev  = i - long_p;
        const int idx_short_prev = i - short_p;



        dsf c_cur  = load_dsf_f2(pfx_c,  i);
        dsf v_cur  = load_dsf_f2(pfx_v,  i);
        dsf cv_cur = load_dsf_f2(pfx_cv, i);
        float vol_i = volume[i];


        const dsf c_prev_l  = load_dsf_f2(pfx_c,  idx_long_prev);
        const dsf v_prev_l  = load_dsf_f2(pfx_v,  idx_long_prev);
        const dsf cv_prev_l = load_dsf_f2(pfx_cv, idx_long_prev);
        const dsf c_prev_s  = load_dsf_f2(pfx_c,  idx_short_prev);
        const dsf v_prev_s  = load_dsf_f2(pfx_v,  idx_short_prev);
        const dsf cv_prev_s = load_dsf_f2(pfx_cv, idx_short_prev);


        const dsf sc_l  = ds_sub(c_cur,  c_prev_l);
        const dsf sv_l  = ds_sub(v_cur,  v_prev_l);
        const dsf scv_l = ds_sub(cv_cur, cv_prev_l);
        const dsf sc_s  = ds_sub(c_cur,  c_prev_s);
        const dsf sv_s  = ds_sub(v_cur,  v_prev_s);
        const dsf scv_s = ds_sub(cv_cur, cv_prev_s);


        const dsf sma_l   = ds_scale(sc_l,  inv_long);
        const dsf sma_s   = ds_scale(sc_s,  inv_short);
        const dsf sma_v_l = ds_scale(sv_l,  inv_long);
        const dsf sma_v_s = ds_scale(sv_s,  inv_short);


        const dsf vwma_l = ds_div(scv_l, sv_l);
        const dsf vwma_s = ds_div(scv_s, sv_s);

        const dsf vpc_ds = ds_sub(vwma_l, sma_l);
        const dsf vpr_ds = ds_div(vwma_s, sma_s);
        const dsf vm_ds  = ds_div(sma_v_s, sma_v_l);

        const float vpc = ds_to_f(vpc_ds);
        const float vpr = ds_to_f(vpr_ds);
        const float vm  = ds_to_f(vm_ds);

        const float vpci = vpc * vpr * vm;

        y_vpci[i] = vpci;


        const float contrib = isfinite(vpci) ? (vpci * vol_i) : 0.0f;
        kahan_add(contrib, sum_vpci_vol_short, sum_comp);
        if (i >= warm + short_p) {
            const int rm = i - short_p;
            const float vpci_rm = y_vpci[rm];
            const float rm_contrib = isfinite(vpci_rm) ? (vpci_rm * volume[rm]) : 0.0f;
            kahan_add(-rm_contrib, sum_vpci_vol_short, sum_comp);
        }


        const float denom = ds_to_f(sma_v_s);
        if (denom != 0.0f && isfinite(denom)) {
            y_vpcis[i] = (sum_vpci_vol_short * inv_short) / denom;
        } else {
            y_vpcis[i] = nan_f32();
        }
    }
}



extern "C" __global__ void vpci_many_series_one_param_f32(
    const float2* __restrict__ pfx_c_tm,
    const float2* __restrict__ pfx_v_tm,
    const float2* __restrict__ pfx_cv_tm,
    const float*  __restrict__ volume_tm,
    const int*    __restrict__ first_valids,
    int cols,
    int rows,
    int short_p,
    int long_p,
    float* __restrict__ out_vpci_tm,
    float* __restrict__ out_vpcis_tm
) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols) return;

    const int first = first_valids[series];
    if (UNLIKELY(short_p <= 0 || long_p <= 0 || short_p > long_p ||
                 long_p > rows || first < 0 || first >= rows)) {
        for (int r = 0; r < rows; ++r) {
            const int idx = r * cols + series;
            out_vpci_tm[idx]  = nan_f32();
            out_vpcis_tm[idx] = nan_f32();
        }
        return;
    }

    const int warm = first + long_p - 1;
    for (int r = 0; r < warm; ++r) {
        const int idx = r * cols + series;
        out_vpci_tm[idx]  = nan_f32();
        out_vpcis_tm[idx] = nan_f32();
    }

    const float inv_long  = 1.0f / (float)long_p;
    const float inv_short = 1.0f / (float)short_p;

    float sum_vpci_vol_short = 0.0f;
    float sum_comp           = 0.0f;

    for (int r = warm; r < rows; ++r) {
        const int idx          = r * cols + series;
        const int idx_long_pr  = (r - long_p) * cols + series;
        const int idx_short_pr = (r - short_p) * cols + series;

        const dsf c_cur  = load_dsf_f2(pfx_c_tm,  idx);
        const dsf v_cur  = load_dsf_f2(pfx_v_tm,  idx);
        const dsf cv_cur = load_dsf_f2(pfx_cv_tm, idx);

        const dsf sc_l  = ds_sub(c_cur,  load_dsf_f2(pfx_c_tm,  idx_long_pr));
        const dsf sv_l  = ds_sub(v_cur,  load_dsf_f2(pfx_v_tm,  idx_long_pr));
        const dsf scv_l = ds_sub(cv_cur, load_dsf_f2(pfx_cv_tm, idx_long_pr));
        const dsf sc_s  = ds_sub(c_cur,  load_dsf_f2(pfx_c_tm,  idx_short_pr));
        const dsf sv_s  = ds_sub(v_cur,  load_dsf_f2(pfx_v_tm,  idx_short_pr));
        const dsf scv_s = ds_sub(cv_cur, load_dsf_f2(pfx_cv_tm, idx_short_pr));

        const dsf sma_l   = ds_scale(sc_l,  inv_long);
        const dsf sma_s   = ds_scale(sc_s,  inv_short);
        const dsf sma_v_l = ds_scale(sv_l,  inv_long);
        const dsf sma_v_s = ds_scale(sv_s,  inv_short);

        const dsf vwma_l = ds_div(scv_l, sv_l);
        const dsf vwma_s = ds_div(scv_s, sv_s);

        const dsf vpc_ds = ds_sub(vwma_l, sma_l);
        const dsf vpr_ds = ds_div(vwma_s, sma_s);
        const dsf vm_ds  = ds_div(sma_v_s, sma_v_l);

        const float vpci = ds_to_f(vpc_ds) * ds_to_f(vpr_ds) * ds_to_f(vm_ds);
        out_vpci_tm[idx] = vpci;

        float contrib = isfinite(vpci) ? (vpci * volume_tm[idx]) : 0.0f;
        kahan_add(contrib, sum_vpci_vol_short, sum_comp);

        if (r >= warm + short_p) {
            const int rm = (r - short_p) * cols + series;
            const float vpci_rm = out_vpci_tm[rm];
            const float rm_contrib = isfinite(vpci_rm) ? (vpci_rm * volume_tm[rm]) : 0.0f;
            kahan_add(-rm_contrib, sum_vpci_vol_short, sum_comp);
        }

        const float denom = ds_to_f(sma_v_s);
        out_vpcis_tm[idx] = (denom != 0.0f && isfinite(denom))
                          ? (sum_vpci_vol_short * inv_short) / denom
                          : nan_f32();
    }
}

