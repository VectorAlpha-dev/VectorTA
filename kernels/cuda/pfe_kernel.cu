












#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>





static __device__ __forceinline__ void two_sumf(float a, float b, float &s, float &e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}


static __device__ __forceinline__ void quick_two_sumf(float a, float b, float &s, float &e) {
    s = a + b;
    e = b - (s - a);
}


static __device__ __forceinline__ void f2_add_scalar(float &a_hi, float &a_lo, float x) {
    float s, e1; two_sumf(a_hi, x, s, e1);
    float e = a_lo + e1;
    quick_two_sumf(s, e, a_hi, a_lo);
}


static __device__ __forceinline__ float sqrt1p_squaref(float d) {
    return sqrtf(fmaf(d, d, 1.0f));
}



extern "C" __global__
void pfe_batch_f32(const float* __restrict__ data,
                   int len,
                   int first_valid,
                   const int* __restrict__ periods,
                   const int* __restrict__ smoothings,
                   int n_combos,
                   float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;

    
    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    
    float denom = 0.0f;
    const bool use_ring = period <= 256;
    float ring[256];
    int head = 0;
    for (int j = start - period; j < start; ++j) {
        const float d = data[j + 1] - data[j];
        const float s = sqrt1p_squaref(d);
        denom += s;
        if (use_ring) { ring[head++] = s; }
    }
    head = 0;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));

        float raw = 0.0f;
        if (denom > 0.0f) raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);

        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;

        if (t + 1 == len) break;

        const float add_d = data[t + 1] - data[t];
        const float add_s = sqrt1p_squaref(add_d);
        float sub_s;
        if (use_ring) {
            sub_s = ring[head];
            ring[head] = add_s;
            head = (head + 1) % period;
        } else {
            const int oldest = t - period + 1;
            const float sd = data[oldest + 1] - data[oldest];
            sub_s = sqrt1p_squaref(sd);
        }
        denom += add_s - sub_s;
    }
}






extern "C" __global__
void pfe_batch_prefix_f32(const float* __restrict__ data,
                          const double* __restrict__ prefix,
                          int len,
                          int first_valid,
                          const int* __restrict__ periods,
                          const int* __restrict__ smoothings,
                          int n_combos,
                          float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;
    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    bool ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));

        
        const double denom_d = prefix[t] - prefix[t - period];
        const float denom = (float)denom_d;

        if (!(denom > 0.0f)) {
            out[row_off + t] = CUDART_NAN_F;
            continue;
        }

        const float raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);
        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;
    }
}



extern "C" __global__
void pfe_many_series_one_param_time_major_f32(const float* __restrict__ data_tm,
                                              const int*   __restrict__ first_valids,
                                              int cols,
                                              int rows,
                                              int period,
                                              int smoothing,
                                              float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s >= cols || period <= 0 || smoothing <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv >= rows) {
        for (int t = 0; t < rows; ++t) out_tm[t * cols + s] = CUDART_NAN_F;
        return;
    }

    const int start = fv + period;
    for (int t = 0; t < ((start < rows) ? start : rows); ++t) out_tm[t * cols + s] = CUDART_NAN_F;
    if (start >= rows) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    float denom = 0.0f;
    for (int j = fv; j < start; ++j) {
        const float d = data_tm[(j + 1) * cols + s] - data_tm[j * cols + s];
        denom += sqrt1p_squaref(d);
    }
    int oldest = fv;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < rows; ++t) {
        const float cur  = data_tm[t * cols + s];
        const float past = data_tm[(t - period) * cols + s];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));
        const float raw = (denom > 0.0f) ? (100.0f * (long_leg / denom)) : 0.0f;
        const float signed_val = copysignf(raw, diff);

        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out_tm[t * cols + s] = ema;

        if (t + 1 == rows) break;
        const float add_d = data_tm[(t + 1) * cols + s] - data_tm[t * cols + s];
        const float sub_d = data_tm[(oldest + 1) * cols + s] - data_tm[oldest * cols + s];
        denom += sqrt1p_squaref(add_d) - sqrt1p_squaref(sub_d);
        ++oldest;
    }
}









extern "C" __global__
void pfe_build_steps_f32(const float* __restrict__ data,
                         int len,
                         float* __restrict__ steps_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) { steps_out[0] = 0.0f; }
    for (int t = i + 1; t < len; t += gridDim.x * blockDim.x) {
        const float d = data[t] - data[t - 1];
        steps_out[t] = sqrt1p_squaref(d);
    }
}


extern "C" __global__
void pfe_build_prefix_float2_serial(const float* __restrict__ steps,
                                    int len,
                                    float* __restrict__ pref_hi,
                                    float* __restrict__ pref_lo) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s_hi = 0.0f, s_lo = 0.0f;
    pref_hi[0] = 0.0f; pref_lo[0] = 0.0f;
    for (int t = 1; t < len; ++t) {
        f2_add_scalar(s_hi, s_lo, steps[t]);
        pref_hi[t] = s_hi;
        pref_lo[t] = s_lo;
    }
}


extern "C" __global__
void pfe_many_params_prefix_f32(const float* __restrict__ data,
                                const float* __restrict__ pref_hi,
                                const float* __restrict__ pref_lo,
                                int len,
                                int first_valid,
                                const int* __restrict__ periods,
                                const int* __restrict__ smoothings,
                                int n_combos,
                                float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period    = periods[combo];
    const int smoothing = smoothings[combo];
    if (period <= 0 || smoothing <= 0 || period > len) return;

    const int row_off = combo * len;
    const int start = first_valid + period;
    for (int t = 0; t < ((start < len) ? start : len); ++t) out[row_off + t] = CUDART_NAN_F;
    if (start >= len) return;

    const float p2  = float(period) * float(period);
    const float alpha = 2.0f / (float(smoothing) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    bool  ema_started = false;
    float ema = 0.0f;

    #pragma unroll 1
    for (int t = start; t < len; ++t) {
        const float cur  = data[t];
        const float past = data[t - period];
        const float diff = cur - past;
        const float long_leg = sqrtf(fmaf(diff, diff, p2));

        const float d_hi = pref_hi[t] - pref_hi[t - period];
        const float d_lo = pref_lo[t] - pref_lo[t - period];
        const float denom = d_hi + d_lo;

        if (!(denom > 0.0f)) { out[row_off + t] = CUDART_NAN_F; continue; }

        const float raw = 100.0f * (long_leg / denom);
        const float signed_val = copysignf(raw, diff);
        if (!ema_started) { ema_started = true; ema = signed_val; }
        else { ema = fmaf(alpha, signed_val, one_minus_alpha * ema); }
        out[row_off + t] = ema;
    }
}
