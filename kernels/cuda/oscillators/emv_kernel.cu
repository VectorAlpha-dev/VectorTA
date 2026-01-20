















#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef EMV_NAN
#define EMV_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif



static __device__ __forceinline__ void two_diff_f32(float a, float b, float &s, float &e) {
    s = a - b;
    float bb = s - a;
    e = (a - (s - bb)) - (b + bb);
}


extern "C" __global__ void emv_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    float* row = out + (size_t)combo * series_len;

    if (UNLIKELY(series_len <= 0 || first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) row[i] = EMV_NAN;
        return;
    }

    const int warm = first_valid + 1;


    for (int i = 0; i < warm && i < series_len; ++i) row[i] = EMV_NAN;


    const unsigned mask = __activemask();
    const int src_lane = __ffs(mask) - 1;


    float h0 = 0.0f, l0 = 0.0f;
    if ((threadIdx.x & 31) == src_lane) {
        h0 = high[first_valid];
        l0 = low[first_valid];
    }
    h0 = __shfl_sync(mask, h0, src_lane);
    l0 = __shfl_sync(mask, l0, src_lane);
    float last_mid = 0.5f * (h0 + l0);

    for (int i = warm; i < series_len; ++i) {

        float hf = 0.0f, lf = 0.0f, vf = 0.0f;
        if ((threadIdx.x & 31) == src_lane) {
            hf = high[i];
            lf = low[i];
            vf = volume[i];
        }
        hf = __shfl_sync(mask, hf, src_lane);
        lf = __shfl_sync(mask, lf, src_lane);
        vf = __shfl_sync(mask, vf, src_lane);

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            row[i] = EMV_NAN;
            continue;
        }

        const float range = hf - lf;
        const float current_mid = 0.5f * (hf + lf);

        if (UNLIKELY(range == 0.0f)) {
            row[i] = EMV_NAN;
            last_mid = current_mid;
            continue;
        }


        float s, e;
        two_diff_f32(current_mid, last_mid, s, e);


        const float k = range * (10000.0f / vf);


        row[i] = fmaf(s, k, e * k);

        last_mid = current_mid;
    }
}



extern "C" __global__ void emv_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int fv = first_valids[s];
    if (UNLIKELY(series_len <= 0 || fv < 0 || fv >= series_len)) {
        float* o = out_tm + s;
        for (int r = 0; r < series_len; ++r, o += num_series) *o = EMV_NAN;
        return;
    }

    const int warm = fv + 1;

    {
        float* o = out_tm + s;
        for (int r = 0; r < warm && r < series_len; ++r, o += num_series) *o = EMV_NAN;
    }


    const size_t idx0 = (size_t)fv * num_series + s;
    float last_mid = 0.5f * (high_tm[idx0] + low_tm[idx0]);

    for (int r = warm; r < series_len; ++r) {
        const size_t idx = (size_t)r * num_series + s;
        const float hf = high_tm[idx];
        const float lf = low_tm[idx];
        const float vf = volume_tm[idx];
        float* out_elem = out_tm + idx;

        if (UNLIKELY(isnan(hf) || isnan(lf) || isnan(vf))) {
            *out_elem = EMV_NAN;
            continue;
        }
        const float current_mid = 0.5f * (hf + lf);
        const float range = hf - lf;
        if (UNLIKELY(range == 0.0f)) {
            *out_elem = EMV_NAN;
            last_mid = current_mid;
            continue;
        }

        float s_hi, s_lo;
        two_diff_f32(current_mid, last_mid, s_hi, s_lo);

        const float k = range * (10000.0f / vf);
        *out_elem = fmaf(s_hi, k, s_lo * k);

        last_mid = current_mid;
    }
}

